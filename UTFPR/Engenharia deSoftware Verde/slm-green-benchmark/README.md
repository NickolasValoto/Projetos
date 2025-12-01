# SLM Green Benchmark  
Benchmark de Eficiência Energética de Modelos de Linguagem Pequenos

---

## 1. Visão Geral

Este projeto implementa um benchmark de eficiência energética para três Small Language Models (SLMs) executados localmente:

- microsoft/phi-2  
- openai-community/gpt2  
- deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B

O objetivo principal é medir o custo energético e o comportamento desses modelos ao processarem um conjunto padronizado de queries matemáticas.

---

## 2. Objetivo do Projeto

O benchmark avalia:

- Consumo energético (via CodeCarbon)  
- Emissões estimadas de CO₂  
- Tempo por query  
- Quantidade de tokens de entrada e saída  
- Estabilidade e coerência das respostas  
- Comparação direta entre modelos executando as mesmas queries

Não há uso de RAG (Retrieval-Augmented Generation).  
As respostas são geradas apenas por inferência direta.

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
│   └── environmental_data_deepseek.json
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
Windows:
```
.venv\Scripts\activate
```

Linux / macOS:
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
- tqdm  
- openpyxl  

---

## 5. Execução dos Benchmarks

## Ambiente de Execução

Todos os experimentos foram realizados localmente, em uma única máquina, para garantir condições controladas e reprodutíveis. A seguir estão as especificações completas do ambiente utilizado.

### Hardware
- **CPU:** Intel Core i3-4160 3.60GHz  
- **GPU:** NVIDIA GeForce GTX 1050 Ti (4 GB)  
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
  - `tqdm`  
  - `openpyxl`  

### Configuração de Execução
- Todos os modelos foram executados **exclusivamente em CPU**.  
- Nenhuma técnica de *Retrieval-Augmented Generation (RAG)* foi utilizada.  
- Todas as queries foram processadas de forma independente, sem memória de contexto.  
- Os modelos foram carregados localmente via HuggingFace Transformers.  

Essa especificação permite que qualquer pesquisador replique integralmente os experimentos realizados neste benchmark.


Cada modelo possui um script independente.

### 5.1 Phi-2
```
python phi2_benchmark.py
```

### 5.2 GPT-2
```
python gpt2_benchmark.py
```

### 5.3 DeepSeek R1 Qwen
```
python deepseek_benchmark.py
```

Cada execução:

1. Carrega o modelo  
2. Executa warmup  
3. Processa todas as queries do arquivo `data/queries.txt`  
4. Mede energia e tempo  
5. Gera um arquivo JSON em `results/`

---

## 6. Estrutura dos Arquivos JSON

Exemplo simplificado:

```json
{
  "model": "phi2",
  "device": "cpu",
  "total_emissions": 0.00123,
  "total_queries": 100,
  "avg_emissions_per_query": 0.0000123,
  "total_time_s": 512.45,
  "avg_time_s_per_query": 5.12,
  "query_history": [
      {
        "query": "27 + 58",
        "response": "85",
        "emissions": 0.0000121,
        "duration_s": 4.91,
        "tokens_input": 7,
        "tokens_output": 5,
        "timestamp": 1
      }
  ]
}
```

---

## 7. Explicação das Métricas

### emissions  
Valor estimado de CO₂ emitido na execução da query.

### duration_s  
Tempo total para processar a query.

### tokens_input  
Quantidade de tokens que o modelo recebeu como entrada.

### tokens_output  
Quantidade de tokens gerados pelo modelo na resposta.

### timestamp  
Ordem sequencial da query na execução.

---

## 8. Scripts de Análise

Localizados em `/analysis`:

### comparative_analysis.py  
Compara métricas entre os modelos.

### visualizations.py  
Gera gráficos utilizando arquivos JSON.

### statistical_tests.py  
Executa testes estatísticos sobre tempo, emissões e tokens.

---

## 9. Reprodutibilidade

Para garantir reprodutibilidade:

- Utilize o mesmo arquivo `queries.txt`  
- Execute sempre no mesmo hardware quando comparar modelos  
- Não altere parâmetros de geração entre execuções  
- Registre CPU, GPU, RAM e sistema operacional

---

## 10. Licença

Projeto acadêmico para fins de pesquisa.  
Os modelos seguem as licenças de seus respectivos desenvolvedores.

---

## 11. Contato

Para dúvidas, contribuições ou análise adicional, entre em contato com o autor do projeto.
