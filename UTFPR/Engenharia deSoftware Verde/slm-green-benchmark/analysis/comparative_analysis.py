import os
import json
import pandas as pd

RESULTS_DIR = "results"


def load_json_files(results_dir: str):
    """Carrega todos os arquivos JSON da pasta results/."""
    data = []
    for file in os.listdir(results_dir):
        if file.endswith(".json"):
            path = os.path.join(results_dir, file)
            with open(path, "r", encoding="utf-8") as f:
                entry = json.load(f)
                data.append(entry)
    return data


def safe_get(entry: dict, key: str, default=None):
    """Pega uma chave com valor padrão caso não exista."""
    return entry.get(key, default)


def infer_total_time_from_history(entry: dict):
    """Se total_time_s não existir, calcula a partir dos duration_s da query_history (se houver)."""
    if "total_time_s" in entry:
        return entry["total_time_s"]

    durations = [
        q.get("duration_s", 0.0)
        for q in entry.get("query_history", [])
        if "duration_s" in q
    ]
    return float(sum(durations)) if durations else None


def build_comparative_table(data):
    """Constrói uma tabela com métricas comparativas entre os modelos."""
    rows = []

    for entry in data:
        model = entry.get("model", "desconhecido")
        model_key = entry.get("model_key", "desconhecido")
        total_emissions = float(entry.get("total_emissions", 0.0))
        total_queries = int(entry.get("total_queries", len(entry.get("query_history", [])) or 1))

        # Emissões por query
        emissions_list = [q.get("emissions", 0.0) for q in entry.get("query_history", [])]
        series_emissions = pd.Series(emissions_list) if emissions_list else pd.Series([0.0])

        # Tempo total
        total_time_s = infer_total_time_from_history(entry)
        if total_time_s is None:
            total_time_s = 0.0

        # Tempo por query
        durations_list = [q.get("duration_s", 0.0) for q in entry.get("query_history", [])]
        series_durations = pd.Series(durations_list) if durations_list else pd.Series([0.0])

        # Tokens (se existirem)
        tokens_in_list = [q.get("tokens_input", q.get("tokens", 0)) for q in entry.get("query_history", [])]
        tokens_out_list = [q.get("tokens_output", 0) for q in entry.get("query_history", [])]

        series_tokens_in = pd.Series(tokens_in_list) if tokens_in_list else pd.Series([0])
        series_tokens_out = pd.Series(tokens_out_list) if tokens_out_list else pd.Series([0])

        row = {
            "Modelo": model_key,
            "HuggingFace ID": model,
            "Total Queries": total_queries,

            # Emissões
            "Emissão Total (kg CO₂eq)": total_emissions,
            "Média Emissão por Query (kg CO₂eq)": total_emissions / total_queries if total_queries else 0.0,
            "Desvio Padrão Emissão": series_emissions.std(),
            "Emissão Mínima": series_emissions.min(),
            "Emissão Máxima": series_emissions.max(),

            # Tempo
            "Tempo Total (s)": total_time_s,
            "Média Tempo por Query (s)": total_time_s / total_queries if total_queries else 0.0,
            "Desvio Padrão Tempo (s)": series_durations.std(),
            "Tempo Mínimo (s)": series_durations.min(),
            "Tempo Máximo (s)": series_durations.max(),

            # Tokens
            "Média Tokens Entrada": series_tokens_in.mean(),
            "Média Tokens Saída": series_tokens_out.mean(),
        }

        rows.append(row)

    df = pd.DataFrame(rows)
    # Ordena por emissão total, depois por tempo total
    df = df.sort_values(["Emissão Total (kg CO₂eq)", "Tempo Total (s)"], ascending=True)
    return df


def main():
    print("=== Comparative Analysis (Emissão e Tempo) ===")

    data = load_json_files(RESULTS_DIR)

    if not data:
        print("Nenhum arquivo JSON encontrado em results/. Execute os benchmarks antes.")
        return

    df = build_comparative_table(data)

    output_path = os.path.join(RESULTS_DIR, "comparative_results.csv")
    df.to_csv(output_path, sep=";", index=False)

    print("\nTabela comparativa gerada em:")
    print(output_path)
    print("\nConteúdo:\n")
    print(df)


if __name__ == "__main__":
    main()
