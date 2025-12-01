import os
import json
import matplotlib.pyplot as plt

RESULTS_DIR = "results"
plt.rcParams.update({"figure.figsize": (10, 6)})


def load_json_files(results_dir: str):
    data = []
    for file in os.listdir(results_dir):
        if file.endswith(".json"):
            with open(os.path.join(results_dir, file), "r", encoding="utf-8") as f:
                data.append(json.load(f))
    return data


def infer_total_time(entry: dict):
    if "total_time_s" in entry:
        return float(entry["total_time_s"])
    durations = [q.get("duration_s", 0.0) for q in entry.get("query_history", [])]
    return float(sum(durations)) if durations else 0.0


def main():
    print("=== Visualizations (Emissão e Tempo) ===")

    data = load_json_files(RESULTS_DIR)
    if not data:
        print("Nenhum JSON encontrado em results/")
        return

    models = [d.get("model_key", "desconhecido") for d in data]

    # -----------------------------
    # EMISSÕES
    # -----------------------------
    totals_emissions = [float(d.get("total_emissions", 0.0)) for d in data]
    means_emissions = [
        float(d.get("total_emissions", 0.0)) / int(d.get("total_queries", len(d.get("query_history", [])) or 1))
        for d in data
    ]
    emissions_lists = [
        [q.get("emissions", 0.0) for q in d.get("query_history", [])]
        for d in data
    ]

    # 1) Emissões totais
    plt.bar(models, totals_emissions)
    plt.title("Emissões Totais por Modelo (kg CO₂eq)")
    plt.ylabel("kg CO₂eq")
    plt.xlabel("Modelo")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "emissions_total.png"))
    plt.clf()

    # 2) Emissões médias por query
    plt.bar(models, means_emissions)
    plt.title("Emissões Médias por Query (kg CO₂eq)")
    plt.ylabel("kg CO₂eq")
    plt.xlabel("Modelo")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "emissions_mean.png"))
    plt.clf()

    # 3) Boxplot de emissões por query
    plt.boxplot(emissions_lists, labels=models)
    plt.title("Distribuição das Emissões por Query")
    plt.ylabel("kg CO₂eq")
    plt.xlabel("Modelo")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "emissions_boxplot.png"))
    plt.clf()

    # 4) Evolução temporal das emissões (por query)
    for d in data:
        model_key = d.get("model_key", "desconhecido")
        timestamps = [q.get("timestamp", i + 1) for i, q in enumerate(d.get("query_history", []))]
        emissions = [q.get("emissions", 0.0) for q in d.get("query_history", [])]
        plt.plot(timestamps, emissions, label=model_key)

    plt.title("Evolução Temporal das Emissões por Query")
    plt.xlabel("Query")
    plt.ylabel("kg CO₂eq")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "emissions_temporal_evolution.png"))
    plt.clf()

    # -----------------------------
    # TEMPO
    # -----------------------------
    totals_time = [infer_total_time(d) for d in data]
    means_time = [
        infer_total_time(d) / int(d.get("total_queries", len(d.get("query_history", [])) or 1))
        for d in data
    ]
    durations_lists = [
        [q.get("duration_s", 0.0) for q in d.get("query_history", [])]
        for d in data
    ]

    # 5) Tempo total por modelo
    plt.bar(models, totals_time)
    plt.title("Tempo Total por Modelo (s)")
    plt.ylabel("Segundos")
    plt.xlabel("Modelo")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "time_total.png"))
    plt.clf()

    # 6) Tempo médio por query
    plt.bar(models, means_time)
    plt.title("Tempo Médio por Query (s)")
    plt.ylabel("Segundos")
    plt.xlabel("Modelo")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "time_mean.png"))
    plt.clf()

    # 7) Boxplot de tempo por query
    plt.boxplot(durations_lists, labels=models)
    plt.title("Distribuição do Tempo por Query")
    plt.ylabel("Segundos")
    plt.xlabel("Modelo")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "time_boxplot.png"))
    plt.clf()

    # 8) Evolução temporal do tempo por query
    for d in data:
        model_key = d.get("model_key", "desconhecido")
        timestamps = [q.get("timestamp", i + 1) for i, q in enumerate(d.get("query_history", []))]
        durations = [q.get("duration_s", 0.0) for q in d.get("query_history", [])]
        plt.plot(timestamps, durations, label=model_key)

    plt.title("Evolução Temporal do Tempo por Query")
    plt.xlabel("Query")
    plt.ylabel("Segundos")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "time_temporal_evolution.png"))
    plt.clf()

    print("Gráficos de emissão e tempo gerados em results/.")


if __name__ == "__main__":
    main()
