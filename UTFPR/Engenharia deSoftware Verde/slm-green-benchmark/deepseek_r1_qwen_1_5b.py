import os
import json
import argparse
import time
from dataclasses import dataclass, asdict
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from codecarbon import EmissionsTracker
from tqdm import tqdm

import gpt2
import phi2

# Identificação do modelo
MODEL_KEY = "deepseek_r1_qwen_1_5b"
MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
TRUST_REMOTE_CODE = True  # DeepSeek costuma exigir trust_remote_code=True

# Parâmetros de geração padronizados para todos os modelos
GEN_KWARGS = {
    "max_new_tokens": 128,
    "temperature": 0.0,   # determinístico para comparação
    "do_sample": False,
    "top_p": 1.0,
}


@dataclass
class QueryRecord:
    """
    Registro de uma única query no benchmark.
    """
    query: str           # texto da query
    response: str        # resposta completa do modelo
    emissions: float     # emissões em kg CO2eq (CodeCarbon)
    duration_s: float    # tempo de execução da query (segundos)
    tokens_input: int    # número de tokens na entrada
    tokens_output: int   # número de tokens gerados
    timestamp: int       # ordem de execução (1..N)


@dataclass
class ModelRunResult:
    """
    Resultado agregado do benchmark de um modelo.
    """
    model_hf_id: str
    model_key: str
    device: str
    total_queries: int
    total_emissions: float      # soma das emissões de todas as queries
    total_time_s: float         # soma dos tempos de todas as queries
    avg_time_s_per_query: float # média de tempo por query
    query_history: List[QueryRecord]


def load_queries(path: str) -> List[str]:
    """
    Carrega as queries de um arquivo texto, uma por linha.
    Limita a 100 queries para padronizar o experimento.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Arquivo de queries não encontrado: {path}")
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines()]
    queries = [l for l in lines if l]
    if len(queries) == 0:
        raise ValueError("Arquivo de queries está vazio.")
    if len(queries) > 100:
        queries = queries[:100]
    return queries


def load_model_and_tokenizer(device: str):
    """
    Carrega o modelo e o tokenizer no dispositivo especificado (cpu ou cuda).
    """
    print(f"\n🔄 Carregando modelo: {MODEL_ID} (device={device})")

    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA não está disponível, mas device='cuda' foi solicitado.")
        torch_dtype = torch.float16
        device_arg = "cuda"
        device_map = {"": 0}
    else:
        torch_dtype = torch.float32
        device_arg = "cpu"
        device_map = None

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        trust_remote_code=TRUST_REMOTE_CODE
    )
    # Garantir pad_token (alguns modelos Qwen não definem)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=TRUST_REMOTE_CODE,
        torch_dtype=torch_dtype,
        device_map=device_map
    )

    if device_map is None:
        model.to(device_arg)

    model.eval()
    print(f"✅ Modelo {MODEL_ID} carregado em {device_arg}")
    return model, tokenizer, device_arg


def run_model_on_queries(
    queries: List[str],
    device: str,
    country_iso_code: str | None = None
) -> ModelRunResult:
    """
    Executa o modelo sobre a lista de queries, medindo emissões e tempo por query.
    """
    model, tokenizer, device_arg = load_model_and_tokenizer(device=device)

    # Warm-up (não é medido energeticamente)
    n_warmup = min(5, len(queries))
    if n_warmup > 0:
        print(f"🔥 Warm-up com {n_warmup} queries (sem medir emissões)...")
        for q in queries[:n_warmup]:
            inputs = tokenizer(q, return_tensors="pt").to(device_arg)
            with torch.no_grad():
                _ = model.generate(**inputs, **GEN_KWARGS)

    print(f"\n⚡ Iniciando medição energética para {len(queries)} queries "
          f"({MODEL_KEY}, device={device_arg})")

    query_history: List[QueryRecord] = []
    total_emissions = 0.0
    total_time_s = 0.0

    for idx, q in enumerate(tqdm(queries, desc=f"Queries {MODEL_KEY}"), start=1):
        tracker_kwargs = {
            "project_name": f"{MODEL_KEY}_query",
            "measure_power_secs": 1,
            "log_level": "error",
            # Se quiser CSV do CodeCarbon por query, descomente:
            # "save_to_file": True,
            # "output_dir": "results",
            # "output_file": f"{MODEL_KEY}_per_query.csv",
        }
        if country_iso_code:
            tracker_kwargs["country_iso_code"] = country_iso_code

        query_tracker = EmissionsTracker(**tracker_kwargs)

        start_time = time.time()
        query_tracker.start()

        # Preparação da entrada
        inputs = tokenizer(q, return_tensors="pt").to(device_arg)
        n_tokens_input = inputs["input_ids"].shape[1]

        # Geração
        with torch.no_grad():
            generated_ids = model.generate(**inputs, **GEN_KWARGS)

        emissions = query_tracker.stop()
        end_time = time.time()

        duration_s = end_time - start_time
        total_time_s += duration_s
        total_emissions += emissions

        # Decodifica texto completo gerado (entrada + saída)
        full_text = tokenizer.decode(
            generated_ids[0],
            skip_special_tokens=True
        )
        # Tokens de saída = total gerado - tokens de entrada
        n_tokens_output = generated_ids[0].shape[0] - n_tokens_input

        query_history.append(
            QueryRecord(
                query=q,
                response=full_text,
                emissions=emissions,
                duration_s=duration_s,
                tokens_input=n_tokens_input,
                tokens_output=n_tokens_output,
                timestamp=idx
            )
        )

    avg_time_s_per_query = total_time_s / len(queries)

    return ModelRunResult(
        model_hf_id=MODEL_ID,
        model_key=MODEL_KEY,
        device=device_arg,
        total_queries=len(queries),
        total_emissions=total_emissions,
        total_time_s=total_time_s,
        avg_time_s_per_query=avg_time_s_per_query,
        query_history=query_history
    )


def save_model_run_to_json(run: ModelRunResult, output_path: str):
    """
    Salva o resultado completo do benchmark em um arquivo JSON.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    data = {
        "model": run.model_hf_id,
        "model_key": run.model_key,
        "device": run.device,
        "total_queries": run.total_queries,
        "total_emissions": run.total_emissions,
        "total_time_s": run.total_time_s,
        "avg_time_s_per_query": run.avg_time_s_per_query,
        "query_history": [
            asdict(q) for q in run.query_history
        ]
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"💾 JSON salvo: {output_path}")
    print(f"   • total_emissions = {run.total_emissions:.6f} kg CO₂")
    print(f"   • total_time_s    = {run.total_time_s:.2f} s")
    print(f"   • total_queries   = {run.total_queries}")
    print(f"   • avg_time/query  = {run.avg_time_s_per_query:.2f} s")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark energético do modelo DeepSeek-R1-Distill-Qwen-1.5B."
    )
    parser.add_argument(
        "--queries_path",
        type=str,
        default=os.path.join("data", "queries.txt"),
        help="Caminho para o arquivo com as queries (uma por linha).",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=os.path.join("results"),
        help="Diretório de saída para o JSON.",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default="cpu",
        help="Dispositivo para executar o modelo (cpu ou cuda).",
    )
    parser.add_argument(
        "--country_iso_code",
        type=str,
        default=None,
        help="Código ISO do país (ex: 'BRA') para o CodeCarbon (opcional)."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=== BENCHMARK ENERGÉTICO - DeepSeek-R1-Distill-Qwen-1.5B ===")
    print(f"Device       : {args.device}")
    print(f"Queries file : {args.queries_path}")
    print(f"Results dir  : {args.results_dir}")
    print("============================================================\n")

    # Usa todos os núcleos da CPU para acelerar o pré-processamento
    torch.set_num_threads(os.cpu_count() or 4)

    queries = load_queries(args.queries_path)
    print(f"Total de queries carregadas: {len(queries)}\n")

    run_result = run_model_on_queries(
        queries=queries,
        device=args.device,
        country_iso_code=args.country_iso_code,
    )

    json_name = f"environmental_data_{MODEL_KEY}.json"
    output_path = os.path.join(args.results_dir, json_name)
    save_model_run_to_json(run_result, output_path)

    print(
        f"\nResumo {MODEL_KEY}: "
        f"total_emissions={run_result.total_emissions:.6f} kg CO₂ | "
        f"total_time={run_result.total_time_s:.2f} s | "
        f"avg_time/query={run_result.avg_time_s_per_query:.2f} s\n"
    )


if __name__ == "__main__":
    main()

