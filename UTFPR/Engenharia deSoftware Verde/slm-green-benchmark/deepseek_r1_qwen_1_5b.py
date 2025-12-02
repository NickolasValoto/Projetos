import os
import time
import json

import psutil
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from codecarbon import EmissionsTracker

import gpt2

MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
DEVICE = "cpu"
QUERIES_FILE = "data/queries.txt"
RESULTS_FILE = "results/environmental_data_deepseek_r1_qwen_1_5b.json"

MAX_NEW_TOKENS = 64
MEASURE_POWER_SECS = 1

USE_RAG = True  # Altere para True para executar com RAG

# Fator médio de emissão da rede elétrica brasileira (~2021)
# Fonte: Climate Transparency Report / Climatiq (aprox. 0,1295 kg CO2/kWh)
EMISSION_FACTOR_KG_PER_KWH = 0.1295


def load_queries(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_model_and_tokenizer():
    print(f"\n=== Carregando modelo: {MODEL_NAME} (device={DEVICE}) ===")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.to(DEVICE)
    return model, tokenizer


def retrieve_documents(query: str):
    if not USE_RAG:
        return []
    return [
        "Contexto simulado de RAG para DeepSeek-R1-Distill-Qwen-1.5B.",
        "Substitua esta função por uma busca real em base vetorial/documental.",
    ]


def build_prompt(query: str, context_text: str):
    if not USE_RAG or not context_text:
        return (
            "Você é uma calculadora. Responda SOMENTE com o resultado numérico final.\n"
            f"Pergunta: {query}"
        )

    return (
        "Você é uma calculadora. Utilize o contexto abaixo apenas se for necessário "
        "para obter o resultado numérico final.\n\n"
        f"Contexto:\n{context_text}\n\n"
        f"Pergunta:\n{query}"
    )


def run_model_on_queries(model, tokenizer, queries):
    mode_str = "rag" if USE_RAG else "baseline"
    print(f"\n[Modo de execução] {mode_str.upper()}")

    print("\n[Warm-up] Executando algumas queries sem medir emissões...")
    warmup_n = min(5, len(queries))
    for q in queries[:warmup_n]:
        docs = retrieve_documents(q)
        context_text = "\n".join(docs) if docs else ""
        prompt = build_prompt(q, context_text)
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        _ = model.generate(**inputs, max_new_tokens=8)

    print("\n[Benchmark] Iniciando medição energética com CodeCarbon...")
    tracker = EmissionsTracker(
        measure_power_secs=MEASURE_POWER_SECS,
        log_level="error",
        save_to_file=False,
    )

    tracker.start()

    query_history = []
    wallclock_start = time.time()

    process = psutil.Process(os.getpid())
    max_ram_mb = 0.0

    for idx, query in enumerate(
        tqdm(queries, desc=f"Queries deepseek ({mode_str})"),
        start=1
    ):
        t0 = time.time()

        # Tempo de "retrieval" (RAG)
        retr_start = time.time()
        docs = retrieve_documents(query)
        retr_end = time.time()
        retrieval_time_s = retr_end - retr_start

        context_text = "\n".join(docs) if docs else ""
        context_tokens = 0
        if context_text:
            ctx_ids = tokenizer(
                context_text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )["input_ids"]
            context_tokens = int(ctx_ids.shape[-1])

        prompt = build_prompt(query, context_text)

        # Medição de RAM antes da geração
        mem_info_before = process.memory_info()
        ram_mb_before = mem_info_before.rss / (1024 * 1024)

        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        tokens_input = inputs["input_ids"].shape[-1]

        # Geração
        gen_start = time.time()
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            pad_token_id=tokenizer.eos_token_id,
        )
        gen_end = time.time()

        generation_time_s = gen_end - gen_start
        duration_s = gen_end - t0

        # Medição de RAM depois da geração
        mem_info_after = process.memory_info()
        ram_mb_after = mem_info_after.rss / (1024 * 1024)
        ram_mb = max(ram_mb_before, ram_mb_after)
        if ram_mb > max_ram_mb:
            max_ram_mb = ram_mb

        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        tokens_output = outputs.shape[-1] - tokens_input

        query_history.append(
            {
                "timestamp": idx,
                "mode": mode_str,
                "query": query,
                "response": decoded,
                "duration_s": float(duration_s),
                "retrieval_time_s": float(retrieval_time_s),
                "generation_time_s": float(generation_time_s),
                "context_tokens": int(context_tokens),
                "tokens_input": int(tokens_input),
                "tokens_output": int(tokens_output),
                "ram_mb": float(ram_mb),
                "emissions": None,
                "energy_kwh": None,
                "power_w": None,
                "co2_kg_per_token_output": None,
                "energy_kwh_per_token_output": None,
            }
        )

    # Para o rastreamento no CodeCarbon e obtém dados
    emissions_data = tracker.stop()
    wallclock_end = time.time()
    wallclock_duration_s = wallclock_end - wallclock_start

    # Inicializa agregados
    total_emissions_kg = None
    total_energy_kwh = None
    total_duration_s = None

    # Caso 1: versões em que stop() retorna um float (kg CO2)
    if isinstance(emissions_data, (int, float)):
        total_emissions_kg = float(emissions_data)
        # usamos o tempo de relógio como aproximação da duração total
        total_duration_s = wallclock_duration_s

    # Caso 2: versões em que stop() retorna um objeto EmissionsData
    elif emissions_data is not None:
        total_emissions_kg = getattr(emissions_data, "emissions", None)
        total_energy_kwh = getattr(emissions_data, "energy_consumed", None)
        total_duration_s = getattr(emissions_data, "duration", None)

    # Fallback adicional: tentar final_emissions_data, se existir
    if total_emissions_kg is None:
        fed = getattr(tracker, "final_emissions_data", None)
        if fed is not None:
            if total_emissions_kg is None:
                total_emissions_kg = getattr(fed, "emissions", None)
            if total_energy_kwh is None:
                total_energy_kwh = getattr(fed, "energy_consumed", None)
            if total_duration_s is None:
                total_duration_s = getattr(fed, "duration", None)

    # Se ainda não temos energia, estima a partir das emissões e do fator de emissão
    if total_energy_kwh is None and total_emissions_kg is not None:
        total_energy_kwh = total_emissions_kg / EMISSION_FACTOR_KG_PER_KWH

    # Potência média (só se houver energia e duração)
    avg_power_w = None
    if total_energy_kwh is not None and total_duration_s and total_duration_s > 0:
        avg_power_w = (total_energy_kwh * 3_600_000) / total_duration_s

    sum_durations = sum(q["duration_s"] for q in query_history) or 0.0

    # Distribuição de emissões/energia por query
    for q in query_history:
        if sum_durations > 0 and q["duration_s"] > 0:
            frac = q["duration_s"] / sum_durations
        else:
            frac = None

        q_energy_kwh = None
        q_emissions = None
        q_power_w = None

        if frac is not None:
            if total_energy_kwh is not None:
                q_energy_kwh = total_energy_kwh * frac
                q_power_w = (q_energy_kwh * 3_600_000) / q["duration_s"]
            if total_emissions_kg is not None:
                q_emissions = total_emissions_kg * frac

        q["energy_kwh"] = float(q_energy_kwh) if q_energy_kwh is not None else None
        q["emissions"] = float(q_emissions) if q_emissions is not None else None
        q["power_w"] = float(q_power_w) if q_power_w is not None else None

        if q["tokens_output"] > 0 and q_emissions is not None:
            q["co2_kg_per_token_output"] = float(q_emissions / q["tokens_output"])
        else:
            q["co2_kg_per_token_output"] = None

        if q["tokens_output"] > 0 and q_energy_kwh is not None:
            q["energy_kwh_per_token_output"] = float(
                q_energy_kwh / q["tokens_output"]
            )
        else:
            q["energy_kwh_per_token_output"] = None

    # Agregações finais
    total_tokens_input = sum(q["tokens_input"] for q in query_history)
    total_tokens_output = sum(q["tokens_output"] for q in query_history)
    total_tokens = total_tokens_input + total_tokens_output
    total_context_tokens = sum(q["context_tokens"] for q in query_history)

    avg_ram_mb = (
        sum(q["ram_mb"] for q in query_history) / len(query_history)
        if query_history
        else None
    )

    co2_per_token_output = (
        float(total_emissions_kg) / total_tokens_output
        if total_emissions_kg is not None and total_tokens_output > 0
        else None
    )

    co2_per_token_total = (
        float(total_emissions_kg) / total_tokens
        if total_emissions_kg is not None and total_tokens > 0
        else None
    )

    energy_kwh_per_token_total = (
        float(total_energy_kwh) / total_tokens
        if total_energy_kwh is not None and total_tokens > 0
        else None
    )

    results = {
        "model": MODEL_NAME,
        "device": DEVICE,
        "mode": mode_str,
        "total_queries": len(query_history),
        "total_emissions_kg": float(total_emissions_kg)
        if total_emissions_kg is not None
        else None,
        "total_energy_kwh": float(total_energy_kwh)
        if total_energy_kwh is not None
        else None,
        "total_time_s_codecarbon": float(total_duration_s)
        if total_duration_s is not None
        else None,
        "total_time_s_wallclock": float(wallclock_duration_s),
        "avg_emissions_kg_per_query": float(total_emissions_kg) / len(query_history)
        if total_emissions_kg is not None and len(query_history) > 0
        else None,
        "avg_time_s_per_query": float(sum_durations) / len(query_history)
        if len(query_history) > 0
        else None,
        "avg_power_w": float(avg_power_w) if avg_power_w is not None else None,
        "total_tokens_input": int(total_tokens_input),
        "total_tokens_output": int(total_tokens_output),
        "total_tokens": int(total_tokens),
        "total_context_tokens": int(total_context_tokens),
        "co2_kg_per_token_output": co2_per_token_output,
        "co2_kg_per_token_total": co2_per_token_total,
        "energy_kwh_per_token_total": energy_kwh_per_token_total,
        "avg_ram_mb": float(avg_ram_mb) if avg_ram_mb is not None else None,
        "max_ram_mb": float(max_ram_mb),
        "query_history": query_history,
    }

    return results


def main():
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)

    mode_str = "rag" if USE_RAG else "baseline"
    print("=== BENCHMARK ENERGÉTICO - DeepSeek R1 Qwen 1.5B ===")
    print(f"Device       : {DEVICE}")
    print(f"Mode         : {mode_str}")
    print(f"Queries file : {QUERIES_FILE}")
    print(f"Results file : {RESULTS_FILE}")
    print("====================================\n")

    queries = load_queries(QUERIES_FILE)
    print(f"Total de queries: {len(queries)}")

    model, tokenizer = load_model_and_tokenizer()

    results = run_model_on_queries(model, tokenizer, queries)

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nResultados salvos em: {RESULTS_FILE}")
    print(f"Emissões totais (kg CO2): {results['total_emissions_kg']}")
    print(f"Energia total (kWh)     : {results['total_energy_kwh']}")
    print(f"Potência média (W)      : {results['avg_power_w']}")
    print(f"RAM média (MB)          : {results['avg_ram_mb']}")
    print(f"RAM máxima (MB)         : {results['max_ram_mb']}")


if __name__ == "__main__":
    main()
