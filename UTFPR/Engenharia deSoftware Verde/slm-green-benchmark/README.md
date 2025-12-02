# SLM Green Benchmark  
Benchmark de Eficiência Energética de Modelos de Linguagem Pequenos

---

## 1. Visão Geral

Este projeto implementa um benchmark de eficiência energética para três Small Language Models (SLMs) executados localmente:

- microsoft/phi-2  
- openai-community/gpt2  
- deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B

O objetivo principal é medir o custo energético, consumo de recursos e desempenho desses modelos ao processarem um conjunto padronizado de queries matemáticas.

Os benchmarks registram:

- Consumo energético (kWh) via CodeCarbon  
- Emissões estimadas de CO₂  
- Potência média (W)  
- Tempo total e por query  
- Uso de RAM  
- Tokens de entrada e saída  
- Métricas normalizadas de eficiência energética  
- Estrutura para modo **baseline** e **RAG** (desativado nos experimentos oficiais)

---

## 2. Objetivo do Projeto

O benchmark avalia:

- Consumo energético por execução e por query  
- Emissões de CO₂ associadas  
- Latência (tempo por query)  
- RAM utilizada em cada inferência  
- Tokens de entrada e saída  
- Eficiência energética relativa entre modelos  

Nos resultados apresentados, **não foi utilizado RAG (Retrieval-Augmented Generation)**.  
As respostas foram geradas apenas por inferência direta.

---

## 3. Estrutura do Repositório

```
slm-green-benchmark/
│
├── data/
│   └── queries.txt
│
├── results/
│   ├── environmental_data_gpt2.json
│   ├── environmental_data_phi2.json
│   ├── environmental_data_deepseek_r1_qwen_1_5b.json
│
├── analysis/
│   ├── comparative_analysis.py
│   ├── visualizations.py
│   └── statistical_tests.py
│
├── gpt2_benchmark.py
├── phi2_benchmark.py
├── deepseek_benchmark.py
│
└── README.md
```

---

## 4. Instalação

### 4.1 Criar ambiente virtual
```
python -m venv .venv
```

### 4.2 Ativar ambiente

**Windows:**
```
.venv\Scripts\activate
```

**Linux / macOS:**
```
source .venv/bin/activate
```

### 4.3 Instalar dependências

```
pip install -r requirements.txt
```

Dependências principais:

- transformers  
- accelerate  
- torch  
- codecarbon  
- psutil  
- tqdm  
- openpyxl  

---

## 5. Execução dos Benchmarks

### Ambiente de Execução

Todos os experimentos foram realizados localmente, em uma única máquina, garantindo condições controladas e reprodutíveis.

### Hardware
- **CPU:** Intel Core i3-4160 3.60GHz  
- **GPU:** NVIDIA GeForce GTX 1050 Ti (4 GB) *(não utilizada)*  
- **RAM:** 16 GB DDR3  
- **Armazenamento:** SSD SATA  

### Sistema Operacional
- **Windows 10** (64 bits)

### Ambiente Python
- **Python 3.11**
- Bibliotecas principais:
  - `transformers`  
  - `torch`  
  - `accelerate`  
  - `codecarbon`  
  - `psutil`  
  - `tqdm`  

### Configuração de Execução
- Todos os modelos foram executados **exclusivamente em CPU**  
- **RAG desativado** (`USE_RAG=False`) nos experimentos principais  
- Queries independentes, sem memória entre chamadas  
- Modelos carregados localmente via HuggingFace Transformers  

---

## 6. Executando Cada Benchmark

### 6.1 Phi-2
```
python phi2_benchmark.py
```

### 6.2 GPT-2
```
python gpt2_benchmark.py
```

### 6.3 DeepSeek R1 Qwen 1.5B
```
python deepseek_benchmark.py
```

Cada execução:

1. Carrega o modelo  
2. Executa warmup  
3. Processa todas as queries de `data/queries.txt`  
4. Mede energia, CO₂, RAM, tempo e tokens  
5. Salva os resultados em `results/*.json`  

---

## 7. Estrutura dos Arquivos JSON

Exemplo simplificado:

```json
{
  "model": "phi2",
  "device": "cpu",
  "mode": "baseline",
  "total_emissions_kg": 0.00123,
  "total_energy_kwh": 0.00045,
  "total_queries": 100,
  "avg_emissions_kg_per_query": 0.0000123,
  "total_time_s_codecarbon": 512.45,
  "avg_time_s_per_query": 5.12,
  "query_history": [
    {
      "timestamp": 1,
      "query": "27 + 58",
      "response": "85",
      "duration_s": 4.91,
      "tokens_input": 7,
      "tokens_output": 5,
      "emissions": 0.0000121,
      "energy_kwh": 0.0000045,
      "power_w": 33.0,
      "ram_mb": 612.2
    }
  ]
}
```

---

## 8. Explicação das Métricas

### emissions  
Quantidade estimada de CO₂ emitida (kg) pela query.

### energy_kwh  
Energia consumida estimada (kWh) atribuída à query.

### power_w  
Potência média estimada durante a query.

### duration_s  
Tempo total de execução da query.

### tokens_input  
Quantidade de tokens do prompt final enviado ao modelo.

### tokens_output  
Quantidade de tokens gerados pelo modelo.

### ram_mb  
Uso aproximado de RAM durante a execução da query.

### timestamp  
A ordem em que a query foi processada.

---

## 9. Scripts de Análise

Localizados em `/analysis`:

### comparative_analysis.py  
Compara métricas entre modelos (tempo, energia, tokens etc.).

### visualizations.py  
Gera gráficos como:
- boxplot de latência  
- boxplot de energia  
- distribuição de tokens  
- linhas de potência  

### statistical_tests.py  
Executa testes como:
- Mann–Whitney  
- ANOVA  
- Correlações entre variáveis  

---

## 10. Reprodutibilidade

- Utilize o mesmo arquivo `queries.txt`  
- Execute sempre no mesmo hardware  
- Não altere parâmetros de geração  
- Registre CPU, RAM, SO e versão do Python  
- Rode baseline e RAG separadamente  
- Use ambiente virtual isolado  

---

## 11. Licença

Projeto acadêmico para fins de pesquisa.  
Os modelos seguem as licenças de seus respectivos desenvolvedores.

---

## 12. Contato

**Nickolas Acelino Valoto Santos**
