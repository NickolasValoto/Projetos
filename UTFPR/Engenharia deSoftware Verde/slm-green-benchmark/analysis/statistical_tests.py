import os
import json
from scipy.stats import shapiro, kruskal, f_oneway, mannwhitneyu, ttest_ind

RESULTS_DIR = "results"


def load_json_files(results_dir: str):
    data = {}
    for file in os.listdir(results_dir):
        if file.endswith(".json"):
            name = file.replace("environmental_data_", "").replace(".json", "")
            with open(os.path.join(results_dir, file), "r", encoding="utf-8") as f:
                data[name] = json.load(f)
    return data


def extract_emissions_and_time(data):
    emissions = {}
    durations = {}

    for model_key, info in data.items():
        emissions_list = [q.get("emissions", 0.0) for q in info.get("query_history", [])]
        durations_list = [q.get("duration_s", 0.0) for q in info.get("query_history", [])]

        # Garante que não fiquem vazias (para evitar erro nos testes)
        if not emissions_list:
            emissions_list = [0.0]
        if not durations_list:
            durations_list = [0.0]

        emissions[model_key] = emissions_list
        durations[model_key] = durations_list

    return emissions, durations


def main():
    print("=== Statistical Tests (Emissão e Tempo) ===")

    raw = load_json_files(RESULTS_DIR)
    if not raw:
        print("Nenhum JSON encontrado.")
        return

    emissions, durations = extract_emissions_and_time(raw)
    models = list(emissions.keys())

    # ----------------------------------------------------
    # Teste de normalidade - Emissões
    # ----------------------------------------------------
    print("\n### Teste de Normalidade (Shapiro–Wilk) - Emissões ###")
    emissions_normality = {}
    for m in models:
        stat, p = shapiro(emissions[m])
        emissions_normality[m] = (p > 0.05)
        print(f"{m}: p-value={p:.6f} {'(normal)' if p > 0.05 else '(não normal)'}")

    # ----------------------------------------------------
    # Teste de normalidade - Tempo
    # ----------------------------------------------------
    print("\n### Teste de Normalidade (Shapiro–Wilk) - Tempo ###")
    time_normality = {}
    for m in models:
        stat, p = shapiro(durations[m])
        time_normality[m] = (p > 0.05)
        print(f"{m}: p-value={p:.6f} {'(normal)' if p > 0.05 else '(não normal)'}")

    # ----------------------------------------------------
    # Teste Global - Emissões (ANOVA ou Kruskal–Wallis)
    # ----------------------------------------------------
    print("\n### Teste Global - Emissões (ANOVA ou Kruskal–Wallis) ###")
    all_emissions_normal = all(emissions_normality[m] for m in models)

    if all_emissions_normal:
        stat, p = f_oneway(*[emissions[m] for m in models])
        print(f"ANOVA (emissões): p-value={p:.6f}")
    else:
        stat, p = kruskal(*[emissions[m] for m in models])
        print(f"Kruskal–Wallis (emissões): p-value={p:.6f}")

    # ----------------------------------------------------
    # Teste Global - Tempo (ANOVA ou Kruskal–Wallis)
    # ----------------------------------------------------
    print("\n### Teste Global - Tempo (ANOVA ou Kruskal–Wallis) ###")
    all_time_normal = all(time_normality[m] for m in models)

    if all_time_normal:
        stat, p = f_oneway(*[durations[m] for m in models])
        print(f"ANOVA (tempo): p-value={p:.6f}")
    else:
        stat, p = kruskal(*[durations[m] for m in models])
        print(f"Kruskal–Wallis (tempo): p-value={p:.6f}")

    # ----------------------------------------------------
    # Testes pareados - Emissões e Tempo
    # ----------------------------------------------------
    print("\n### Testes Pareados entre Modelos ###")

    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            m1, m2 = models[i], models[j]

            # Emissões
            norm_pair_emissions = emissions_normality[m1] and emissions_normality[m2]
            if norm_pair_emissions:
                stat_e, p_e = ttest_ind(emissions[m1], emissions[m2])
                test_name_e = "t-test"
            else:
                stat_e, p_e = mannwhitneyu(emissions[m1], emissions[m2])
                test_name_e = "Mann–Whitney"

            # Tempo
            norm_pair_time = time_normality[m1] and time_normality[m2]
            if norm_pair_time:
                stat_t, p_t = ttest_ind(durations[m1], durations[m2])
                test_name_t = "t-test"
            else:
                stat_t, p_t = mannwhitneyu(durations[m1], durations[m2])
                test_name_t = "Mann–Whitney"

            print(f"\nComparação {m1} vs {m2}:")
            print(f"  Emissões - {test_name_e}: p-value={p_e:.6f}")
            print(f"  Tempo    - {test_name_t}: p-value={p_t:.6f}")

    print("\nTestes estatísticos concluídos.")


if __name__ == "__main__":
    main()
