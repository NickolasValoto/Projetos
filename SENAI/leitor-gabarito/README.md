
# 🧠 Leitor Automático de Gabarito

Sistema desenvolvido para **leitura automatizada de gabaritos de provas** a partir de imagens ou arquivos PDF, utilizando técnicas de **visão computacional** e **OCR**.

---

## 🚀 Funcionalidades

✔ Leitura de provas em imagem (JPG, PNG, etc.)  
✔ Conversão automática de PDF para imagens  
✔ Detecção automática da tabela do gabarito  
✔ Leitura das respostas por análise de pixels  
✔ Identificação de múltiplas respostas (`?`)  
✔ Identificação de questões em branco (`-`)  
✔ Extração do nome do aluno via OCR  
✔ Correção automática de provas invertidas (180°)  
✔ Estimativa do ângulo da prova  
✔ Interface gráfica (Tkinter)  
✔ Barra de progresso em tempo real  
✔ Cancelamento do processamento  
✔ Exportação dos resultados para CSV  
✔ Modo debug com visualização das regiões detectadas  

---

## 🖥️ Interface

O sistema possui uma interface gráfica simples e intuitiva:

- Seleção de pasta com imagens ou arquivo PDF
- Configuração do número de questões
- Configuração das alternativas (ex: A,B,C,D,E)
- Ativação do modo debug
- Visualização de logs em tempo real
- Barra de progresso
- Botão de cancelamento

---

## 📊 Saída (CSV)

Após o processamento, é gerado um arquivo `.csv` com os resultados:

```csv
arquivo;aluno;angulo_detectado;rotacionada;Q01;Q02;Q03;Q04
prova1.jpg;João Silva;1.25;SIM;A;B;?;D

```
## Campos:
- arquivo → nome da imagem analisada
- aluno → nome identificado via OCR
- angulo_detectado → inclinação da prova
- rotacionada → se houve correção de rotação
- Q01...Qn → respostas identificadas

---

## 🧠 Como funciona

O sistema utiliza uma combinação de técnicas:

📌 Visão Computacional (OpenCV)
Detecção de linhas para localizar o gabarito
Segmentação da tabela em células
Contagem de pixels para identificar marcações
📌 OCR (EasyOCR)
Identificação da palavra "Aluno"
Extração da região do nome
Leitura do nome do estudante
📌 Regras de decisão
Alternativa mais marcada → resposta válida
Duas alternativas fortes → ?
Nenhuma marcação → -

---

## ⚙️ Tecnologias utilizadas

- Python 3
- OpenCV
- EasyOCR
- NumPy
- Pandas
- Tkinter
- PDFium (pypdfium2)

---

## 📦 Instalação

Clone o repositório:
```bash
git clone https://github.com/seu-usuario/leitor-gabarito.git
cd leitor-gabarito
```

Instale as dependências:
```bash
pip install opencv-python numpy pandas easyocr pypdfium2 pillow
```
Instale no local padrão da biblioteca.

---

## ▶️ Como usar

Execute o programa:
```bash
python main.py
```

### Na interface:

1. Selecione uma pasta com imagens ou um arquivo PDF
2. Informe o número de questões
3. Informe as alternativas (ex: A,B,C,D,E)
4. Clique em Processar

## ⚠️ Observações importantes
- A qualidade da imagem influencia diretamente no resultado
- Pequenas variações no layout do gabarito podem exigir ajustes
- O OCR pode ter variações dependendo da caligrafia
- O sistema assume um padrão de estrutura de gabarito

## 🔧 Modo Debug

Ao ativar o modo debug, o sistema gera imagens com:

- Região da tabela detectada
- Área de leitura das respostas
- Região do nome do aluno
- Palavra "Aluno"

Essas imagens são salvas na pasta #DEBUG

---

## 🚀 Melhorias futuras
Suporte a diferentes modelos de gabarito
Treinamento de modelo para OCR personalizado
Exportação para Excel
Interface mais avançada
Versão executável (.exe)


## 👨‍💻 Autor
### Desenvolvido por Nickolas Valoto


## 📄 Licença

Este projeto é de uso educacional e pode ser adaptado conforme necessidade.




