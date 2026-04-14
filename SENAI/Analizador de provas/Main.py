# ============================================================
# LEITOR AUTOMÁTICO DE GABARITO
# ============================================================
# Sistema de leitura automatizada de provas em imagem ou PDF.
#
# Funcionalidades:
# - Conversão de PDF para imagens
# - Detecção automática da tabela do gabarito
# - Leitura das respostas via análise de pixels
# - Identificação de múltiplas marcações ("?")
# - Identificação de questões em branco ("-")
# - Extração do nome do aluno via OCR (EasyOCR)
# - Correção automática de imagem invertida (180°)
# - Estimativa do ângulo da prova
# - Interface gráfica com barra de progresso e cancelamento
#
# Tecnologias:
# - OpenCV (visão computacional)
# - EasyOCR (OCR)
# - Tkinter (interface)
# - Pandas (exportação CSV)
#
# Autor: Nickolas Valoto
# ============================================================

import os
import re
import cv2
import numpy as np
import pandas as pd
import pytesseract
import easyocr
import threading
import queue
import pypdfium2 as pdfium
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# ============================================================
# CONFIGURAÇÕES GLOBAIS
# ============================================================
# Variáveis que controlam o comportamento do sistema.
# São alteradas dinamicamente pela interface gráfica.
# Possibilita o cancelamento da execução principal do programa

CANCELAR_PROCESSAMENTO = False

# Ativa/desativa a exportação da imagem de debug geral
SALVAR_DEBUG = False

# Guarda a posição da palavra "Aluno" encontrada na primeira imagem,
# para reutilizar nas próximas e acelerar o OCR
BOX_ALUNO_REFERENCIA = None

# OCR do EasyOCR é carregado uma única vez
reader = easyocr.Reader(['pt'], gpu=False)

# Caminho do executável do Tesseract (caso você ainda use em algum ponto futuro)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
os.environ["TESSDATA_PREFIX"] = r"C:\Program Files\Tesseract-OCR\tessdata"

# Essas variáveis passam a ser definidas pela interface
PASTA_IMAGENS = ""
ARQUIVO_SAIDA = ""
NUM_QUESTOES = 27
ALTERNATIVAS = ["A", "B", "C", "D", "E"]

# Parâmetros da leitura da tabela
PAD_X = 0.18
PAD_Y = 0.22
LIMIAR_MIN_PIXELS = 35
LIMIAR_RAZAO = 1.25

# Extensões aceitas para imagens
EXTENSOES = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")

# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================
# Funções de apoio para:
# - manipulação de nomes de arquivos
# - conversão de PDF
# - limpeza de texto OCR
# - pré-processamento de imagem

def nome_seguro(texto):
    texto = re.sub(r'[<>:"/\\|?*]', '_', texto)
    texto = texto.replace('º', 'o')
    texto = texto.replace('ª', 'a')
    return texto

def converter_pdf_para_imagens(caminho_pdf, pasta_saida=None, dpi=200, callback_log=None):
    """
    Converte todas as páginas de um PDF em imagens JPG.
    Retorna a lista de caminhos das imagens geradas.
    """
    def escrever(msg):
        if callback_log:
            callback_log(msg)
        else:
            print(msg)

    escrever(f"Convertendo PDF: {os.path.basename(caminho_pdf)}")

    if pasta_saida is None:
        nome_base_seguro = nome_seguro(os.path.splitext(os.path.basename(caminho_pdf))[0])

        pasta_saida = os.path.join(
            os.path.dirname(caminho_pdf),
            nome_base_seguro
        )

    os.makedirs(pasta_saida, exist_ok=True)

    pdf = pdfium.PdfDocument(caminho_pdf)
    arquivos_gerados = []

    nome_base = nome_seguro(os.path.splitext(os.path.basename(caminho_pdf))[0])

    for i in range(len(pdf)):
        page = pdf[i]
        pil_image = page.render(scale=dpi / 72).to_pil()

        nome_arquivo = f"{nome_base}_page-{i+1:04d}.jpg"
        caminho_saida = os.path.join(pasta_saida, nome_arquivo)

        pil_image.save(caminho_saida, "JPEG", quality=95)
        arquivos_gerados.append(caminho_saida)

    return arquivos_gerados

def listar_arquivos_para_processar(pasta, callback_log=None):
    def escrever(msg):
        if callback_log:
            callback_log(msg)
        else:
            print(msg)

    """
    Lista imagens diretamente na pasta e também converte PDFs encontrados.
    """
    arquivos_processar = []

    for arquivo in os.listdir(pasta):
        caminho = os.path.join(pasta, arquivo)

        if os.path.isdir(caminho):
            continue

        nome_lower = arquivo.lower()

        if nome_lower.endswith(EXTENSOES):
            arquivos_processar.append(caminho)

        elif nome_lower.endswith(".pdf"):
            escrever(f"Convertendo PDF: {os.path.basename(caminho)}")
            imagens_pdf = converter_pdf_para_imagens(caminho, callback_log=callback_log)
            arquivos_processar.extend(imagens_pdf)

    return arquivos_processar

def melhorar_contraste(gray):
    """
    Aplica CLAHE para melhorar contraste local da imagem em tons de cinza.
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)

def limpar_nome_texto(texto):
    """
    Limpa ruídos comuns do OCR e mantém apenas letras e espaços.
    """
    texto = texto.replace("\n", " ").replace("\r", " ")
    texto = texto.replace("|", "I")
    texto = texto.replace("_", " ")
    texto = texto.replace(":", "")
    texto = texto.replace(";", "")
    texto = texto.replace(",", " ")
    texto = texto.replace(".", " ")
    texto = re.sub(r"\s+", " ", texto).strip()
    texto = re.sub(r"[^A-Za-zÀ-ÿ ]", "", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto

def preprocessar_binaria(gray):
    """
    Gera uma imagem binária invertida a partir da imagem em cinza.
    Essa versão é reaproveitada para acelerar a leitura das respostas.
    """
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, bin_img = cv2.threshold(blur, 170, 255, cv2.THRESH_BINARY_INV)
    return bin_img

# ============================================================
# LOCALIZAÇÃO DA TABELA DO GABARITO
# ============================================================
# Detecta automaticamente a região do gabarito usando:
# - detecção de linhas horizontais e verticais
# - operações morfológicas
# - filtros geométricos (largura, altura e proporção)

def localizar_grade(gray, img, caminho_imagem=None):
    """
    Localiza a região da tabela do gabarito com base em linhas horizontais e verticais.
    """
    bin_img = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        21, 10
    )

    h, w = bin_img.shape

    # Faixa vertical onde o gabarito costuma aparecer
    y1_busca = int(h * 0.18)
    y2_busca = int(h * 0.58)
    faixa = bin_img[y1_busca:y2_busca, :]

    # Kernels mais fortes, principalmente úteis para páginas vindas de PDF
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (60, 1))
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 24))

    horiz = cv2.morphologyEx(faixa, cv2.MORPH_OPEN, kernel_h)
    vert = cv2.morphologyEx(faixa, cv2.MORPH_OPEN, kernel_v)
    grade = cv2.bitwise_or(horiz, vert)

    contornos, _ = cv2.findContours(grade, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    melhor = None
    melhor_area = 0

    for c in contornos:
        x, y, wc, hc = cv2.boundingRect(c)
        area = wc * hc
        proporcao = wc / max(hc, 1)

        # Filtro geométrico básico da tabela
        if wc > w * 0.45 and hc > 40 and proporcao > 4:
            if area > melhor_area:
                melhor_area = area
                melhor = (x, y + y1_busca, wc, hc)

    return melhor

# ============================================================
# LEITURA DAS RESPOSTAS
# ============================================================
# A leitura é baseada na contagem de pixels ativos em cada célula.
#
# Regras:
# - "-"  -> questão em branco
# - "?"  -> múltiplas marcações ou ambiguidade
# - "A-E" -> alternativa dominante
#
# A decisão considera:
# - maior marcação
# - segunda maior marcação
# - média das demais alternativas

def ler_respostas(bin_img, box_grade):
    """
    Lê as respostas marcadas usando a imagem binária já pré-processada.

    Regras:
    - "-"  -> questão em branco / marcação fraca
    - "?"  -> marcação ambígua ou duas alternativas marcadas
    - "A"..."E" -> alternativa claramente dominante
    """
    x, y, w, h = box_grade

    # Remove cabeçalho das questões e coluna das alternativas
    x1 = x + int(w * 0.035)
    x2 = x + w - int(w * 0.003)
    y1 = y + int(h * 0.23)
    y2 = y + h - int(h * 0.02)

    roi = bin_img[y1:y2, x1:x2]

    h_roi, w_roi = roi.shape
    cell_w = w_roi / NUM_QUESTOES
    cell_h = h_roi / len(ALTERNATIVAS)

    respostas = []

    # Ajuste fino para detectar dupla marcação
    LIMIAR_SEGUNDA_MARCACAO = 0.75 # se a 2ª tiver >= 75% da 1ª, considera ambígua

    for q in range(NUM_QUESTOES):
        scores = []

        for i, alt in enumerate(ALTERNATIVAS):
            cx1 = int(q * cell_w)
            cx2 = int((q + 1) * cell_w)
            cy1 = int(i * cell_h)
            cy2 = int((i + 1) * cell_h)

            cell = roi[cy1:cy2, cx1:cx2]

            hh, ww = cell.shape
            px = int(ww * PAD_X)
            py = int(hh * PAD_Y)

            if ww - 2 * px <= 1 or hh - 2 * py <= 1:
                miolo = cell
            else:
                miolo = cell[py:hh - py, px:ww - px]

            score = cv2.countNonZero(miolo)
            scores.append(score)

        # Ordena do maior para o menor
        ordem = np.argsort(scores)[::-1]
        idx_maior = int(ordem[0])
        maior = scores[idx_maior]
        segunda = scores[int(ordem[1])] if len(scores) > 1 else 0

        # Questão em branco / marcação muito fraca
        if maior < LIMIAR_MIN_PIXELS:
            resp = "-"

        # Duas alternativas marcadas ou muito próximas
        elif segunda >= LIMIAR_MIN_PIXELS and segunda >= maior * LIMIAR_SEGUNDA_MARCACAO:
            resp = "?"

        else:
            # Mantém sua checagem de dominância
            outras = [s for j, s in enumerate(scores) if j != idx_maior]
            media_outras = max(1, np.mean(outras))
            razao = maior / media_outras

            if razao < LIMIAR_RAZAO:
                resp = "?"
            else:
                resp = ALTERNATIVAS[idx_maior]

        respostas.append(resp)

    return respostas, (x1, y1, x2, y2)

# ============================================================
# LEITURA DO NOME DO ALUNO (OCR)
# ============================================================
# Estratégia:
# 1. Localizar a palavra "Aluno"
# 2. Usar essa posição como referência
# 3. Extrair a região do nome
#
# Otimização:
# - A posição encontrada é reutilizada nas próximas imagens,
#   reduzindo o custo do OCR

def extrair_nome_aluno_easyocr(gray, img, caminho_imagem):
    """
    Procura a palavra 'Aluno' apenas quando necessário.
    Depois reutiliza essa posição nas próximas imagens.
    """
    global BOX_ALUNO_REFERENCIA

    h, w = gray.shape
    box_aluno = None

    # Reaproveita a posição encontrada anteriormente
    if BOX_ALUNO_REFERENCIA is not None:
        box_aluno = BOX_ALUNO_REFERENCIA
    else:
        topo_y1 = 0
        topo_y2 = int(h * 0.25)
        topo = gray[topo_y1:topo_y2, :]

        # Ampliação moderada para reduzir custo do OCR
        topo_ampliado = cv2.resize(topo, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
        resultados = reader.readtext(topo_ampliado, detail=1, paragraph=False)

        for r in resultados:
            box, texto, conf = r
            texto_limpo = texto.strip().lower()

            if "aluno" in texto_limpo:
                pts = np.array(box, dtype=np.float32) / 1.5

                xs = pts[:, 0]
                ys = pts[:, 1]

                x1 = int(xs.min())
                x2 = int(xs.max())
                y1 = int(ys.min())
                y2 = int(ys.max())

                box_aluno = (x1, y1, x2, y2)
                BOX_ALUNO_REFERENCIA = box_aluno
                break

    # Define a região do nome a partir da palavra "Aluno"
    if box_aluno is not None:
        ax1, ay1, ax2, ay2 = box_aluno

        nome_x1 = ax2 + int(w * 0.01)
        nome_x2 = min(w, nome_x1 + int(w * 0.48))

        altura_palavra = ay2 - ay1
        nome_y1 = max(0, ay1 - int(altura_palavra * 0.35))
        nome_y2 = min(h, ay2 + int(altura_palavra * 0.55))
    else:
        # Fallback por coordenadas fixas
        nome_x1 = int(w * 0.178)
        nome_x2 = int(w * 0.68)
        nome_y1 = int(h * 0.148)
        nome_y2 = int(h * 0.170)

    nome_roi = gray[nome_y1:nome_y2, nome_x1:nome_x2]
    nome_roi = cv2.resize(nome_roi, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

    resultado_nome = reader.readtext(nome_roi, detail=0, paragraph=True)
    texto_nome = " ".join(resultado_nome).strip()
    texto_nome = limpar_nome_texto(texto_nome)

    return texto_nome, nome_roi, (nome_x1, nome_y1, nome_x2, nome_y2), box_aluno

# ============================================================
# DEBUG VISUAL
# ============================================================
# Gera uma imagem com as regiões detectadas:
# - tabela
# - respostas
# - nome
# - palavra "Aluno"
#
# Utilizado para validação e ajuste fino do algoritmo

def salvar_debug(img, caminho_imagem, box_grade=None, box_respostas=None, box_nome=None, box_aluno=None):
    """
    Salva uma única imagem de debug com as caixas principais desenhadas.
    """
    if not SALVAR_DEBUG:
        return

    pasta = os.path.dirname(caminho_imagem)
    nome_arquivo = os.path.basename(caminho_imagem)

    pasta_debug = os.path.join(pasta, "#DEBUG")
    os.makedirs(pasta_debug, exist_ok=True)

    dbg = img.copy()

    if box_grade is not None:
        x, y, w, h = box_grade
        cv2.rectangle(dbg, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(dbg, "TABELA", (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    if box_respostas is not None:
        x1, y1, x2, y2 = box_respostas
        cv2.rectangle(dbg, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(dbg, "RESPOSTAS", (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    if box_aluno is not None:
        x1, y1, x2, y2 = box_aluno
        cv2.rectangle(dbg, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(dbg, "ALUNO", (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    if box_nome is not None:
        x1, y1, x2, y2 = box_nome
        cv2.rectangle(dbg, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(dbg, "NOME", (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imwrite(os.path.join(pasta_debug, f"DEBUG_GERAL_{nome_arquivo}"), dbg)

# ============================================================
# ESTIMATIVA DO ÂNGULO
# ============================================================
# Estima a inclinação da prova sem rotacionar a imagem.
# Utiliza:
# - Canny (detecção de bordas)
# - HoughLinesP (detecção de linhas)
# - Mediana dos ângulos encontrados

def estimar_angulo_pela_tabela(gray, img, box_grade, caminho_imagem=None):
    """
    Estima a inclinação da tabela sem rotacionar a imagem.
    Retorna o ângulo em graus ou None quando não houver linhas válidas.
    """
    x, y, w, h = box_grade

    margem_x = int(w * 0.05)
    margem_y = int(h * 0.20)

    rx1 = max(0, x - margem_x)
    ry1 = max(0, y - margem_y)
    rx2 = min(gray.shape[1], x + w + margem_x)
    ry2 = min(gray.shape[0], y + h + margem_y)

    roi = gray[ry1:ry2, rx1:rx2]

    blur = cv2.GaussianBlur(roi, (3, 3), 0)
    edges = cv2.Canny(blur, 50, 150, apertureSize=3)

    linhas = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=35,
        minLineLength=max(40, roi.shape[1] // 8),
        maxLineGap=40
    )

    angulos = []

    if linhas is not None:
        for linha in linhas:
            x1l, y1l, x2l, y2l = linha[0]

            dx = x2l - x1l
            dy = y2l - y1l

            if dx == 0:
                continue

            comprimento = np.hypot(dx, dy)
            ang = np.degrees(np.arctan2(dy, dx))

            if comprimento > roi.shape[1] * 0.10 and -20 <= ang <= 20:
                angulos.append(ang)

    if not angulos:
        return None

    return float(np.median(angulos))

# ============================================================
# PROCESSAMENTO DE UMA IMAGEM
# ============================================================
# Fluxo:
# 1. Leitura da imagem
# 2. Pré-processamento
# 3. Localização da tabela
# 4. Tentativa de correção (rotação 180°)
# 5. OCR do nome
# 6. Leitura das respostas
# 7. Estimativa de ângulo
# 8. Geração de debug (opcional)

def ler_imagem_segura(caminho_imagem):
    """
    Lê imagem de forma compatível com caminhos Unicode no Windows.
    """
    try:
        arquivo_bytes = np.fromfile(caminho_imagem, dtype=np.uint8)
        img = cv2.imdecode(arquivo_bytes, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None

def processar_imagem(caminho_imagem, callback_log=None):
    """
    Fluxo principal de processamento de uma única imagem.
    Se a tabela não for encontrada, tenta novamente com a imagem girada 180°.
    """

    def girar_180(img):
        """
        Gira a imagem 180 graus.
        """
        return cv2.rotate(img, cv2.ROTATE_180)

    def escrever(msg):
        if callback_log:
            callback_log(msg)
        else:
            print(msg)

    img = ler_imagem_segura(caminho_imagem)
    if img is None:
        escrever(f"Erro ao abrir imagem: {caminho_imagem}")
        return None, None, None

    # ============================================================
    # 1ª tentativa: imagem original
    # ============================================================
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = melhorar_contraste(gray)
    bin_img = preprocessar_binaria(gray)

    box_grade = localizar_grade(gray, img, caminho_imagem)

    imagem_foi_girada = False

    # ============================================================
    # 2ª tentativa: imagem girada 180°
    # ============================================================
    if box_grade is None:
        escrever("⛔ Tabela do gabarito não encontrada. Tentando girar a imagem 180°...")

        img = girar_180(img)
        imagem_foi_girada = True

        global BOX_ALUNO_REFERENCIA
        BOX_ALUNO_REFERENCIA = None

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = melhorar_contraste(gray)
        bin_img = preprocessar_binaria(gray)

        box_grade = localizar_grade(gray, img, caminho_imagem)

    # ============================================================
    # Falha definitiva
    # ============================================================
    if box_grade is None:
        escrever("⛔ Tabela do gabarito não encontrada mesmo após girar 180°.")
        return None, None, None

    # OCR do nome só depois de confirmar a orientação correta
    nome_aluno, nome_roi, box_nome, box_aluno = extrair_nome_aluno_easyocr(gray, img, caminho_imagem)

    # Estimativa do ângulo
    angulo = estimar_angulo_pela_tabela(gray, img, box_grade, caminho_imagem)

    if angulo is None:
        escrever(f"[DEBUG] Ângulo detectado para {os.path.basename(caminho_imagem)}: N/A")
    else:
        escrever(f"[DEBUG] Ângulo detectado para {os.path.basename(caminho_imagem)}: {angulo:.2f}")

    if imagem_foi_girada:
        escrever("↻ Imagem processada após rotação de 180°.")

    # Leitura das respostas
    respostas, box_respostas = ler_respostas(bin_img, box_grade)

    # Debug opcional
    salvar_debug(
        img,
        caminho_imagem,
        box_grade=box_grade,
        box_respostas=box_respostas,
        box_nome=box_nome,
        box_aluno=box_aluno
    )

    return nome_aluno, respostas, angulo

# ============================================================
# PROCESSAMENTO DE ARQUIVO ÚNICO
# ============================================================
# Processa:
# - uma única imagem
# - ou um PDF convertido em múltiplas imagens
#
# Inclui:
# - controle de progresso
# - tratamento de erro
# - cancelamento

def processar_entrada_unica(caminho, callback_log=None, callback_progresso=None):
    """
    Processa um único arquivo: imagem ou PDF.
    """
    def escrever(msg):
        if callback_log:
            callback_log(msg)
        else:
            print(msg)

    resultados = []

    if caminho.lower().endswith(".pdf"):
        arquivos = converter_pdf_para_imagens(caminho, callback_log=callback_log)
    else:
        arquivos = [caminho]

    total = len(arquivos)

    for indice, caminho_arquivo in enumerate(arquivos, start=1):
        if CANCELAR_PROCESSAMENTO:
            escrever("⛔ Processamento cancelado pelo usuário.")
            break

        arquivo = os.path.basename(caminho_arquivo)

        if callback_progresso:
            callback_progresso(indice - 1, total)

        escrever(f"Processando: {arquivo}")

        try:
            nome_aluno, respostas, angulo = processar_imagem(caminho_arquivo, callback_log)

            if nome_aluno is None:
                escrever("⛔ Processamento interrompido pelo sistema.")
                break

            linha = {
                "arquivo": arquivo,
                "aluno": nome_aluno,
                "angulo_detectado": "" if angulo is None else round(angulo, 2),
                "rotacionada": "" if angulo is None else ("SIM" if abs(angulo) >= 1.0 else "NAO")
            }

            for i, resp in enumerate(respostas, start=1):
                linha[f"Q{i:02d}"] = resp

            resultados.append(linha)

        except Exception as e:
            escrever(f"Erro em {arquivo}: {e}")
            linha = {
                "arquivo": arquivo,
                "aluno": "ERRO",
                "angulo_detectado": "",
                "rotacionada": "",
                "erro": str(e)
            }
            for i in range(1, NUM_QUESTOES + 1):
                linha[f"Q{i:02d}"] = ""
            resultados.append(linha)

        if callback_progresso:
            callback_progresso(indice, total)

    if callback_progresso:
        callback_progresso(total, total)

    df = pd.DataFrame(resultados)
    df.to_csv(ARQUIVO_SAIDA, index=False, encoding="utf-8-sig", sep=";")

    escrever(f"Concluído. Resultado salvo em: {ARQUIVO_SAIDA}")

# ============================================================
# PROCESSAMENTO EM LOTE
# ============================================================
# Processa todos os arquivos válidos da pasta:
# - imagens
# - PDFs (convertidos automaticamente)
#
# Gera um único arquivo CSV consolidado

def processar_lote(callback_log=None, callback_progresso=None):
    """
    Processa todos os arquivos válidos encontrados na pasta selecionada.
    """
    def escrever(msg):
        if callback_log:
            callback_log(msg)
        else:
            print(msg)

    arquivos = listar_arquivos_para_processar(PASTA_IMAGENS, callback_log)

    if not arquivos:
        escrever("Nenhuma imagem encontrada na pasta.")
        return

    resultados = []
    total = len(arquivos)

    for indice, caminho in enumerate(arquivos, start=1):
        if CANCELAR_PROCESSAMENTO:
            escrever("⛔ Processamento cancelado pelo usuário.")
            break

        arquivo = os.path.basename(caminho)

        if callback_progresso:
            callback_progresso(indice - 1, total)

        escrever(f"Processando: {arquivo}")

        try:
            nome_aluno, respostas, angulo = processar_imagem(caminho, callback_log)

            if nome_aluno is None:
                escrever("⛔ Processamento interrompido pelo sistema.")
                break

            linha = {
                "arquivo": arquivo,
                "aluno": nome_aluno,
                "angulo_detectado": "" if angulo is None else round(angulo, 2),
                "rotacionada": "" if angulo is None else ("SIM" if abs(angulo) >= 1.0 else "NAO")
            }

            for i, resp in enumerate(respostas, start=1):
                linha[f"Q{i:02d}"] = resp

            resultados.append(linha)

        except Exception as e:
            escrever(f"Erro em {arquivo}: {e}")

            linha = {
                "arquivo": arquivo,
                "aluno": "ERRO",
                "angulo_detectado": "",
                "rotacionada": "",
                "erro": str(e)
            }
            for i in range(1, NUM_QUESTOES + 1):
                linha[f"Q{i:02d}"] = ""
            resultados.append(linha)

        if callback_progresso:
            callback_progresso(indice, total)

    if callback_progresso:
        callback_progresso(total, total)

    df = pd.DataFrame(resultados)
    df.to_csv(ARQUIVO_SAIDA, index=False, encoding="utf-8-sig", sep=";")

    escrever(f"Concluído. Resultado salvo em: {ARQUIVO_SAIDA}")

# ============================================================
# INTERFACE GRÁFICA
# ============================================================
# Interface construída com Tkinter.
#
# Recursos:
# - Seleção de pasta/arquivo
# - Configuração de questões e alternativas
# - Barra de progresso
# - Log em tempo real
# - Cancelamento de execução

def abrir_interface():
    """
    Abre a interface gráfica principal do sistema.
    """
    fila_ui = queue.Queue()

    def processar_fila_ui():
        try:
            while True:
                tipo, payload = fila_ui.get_nowait()

                if tipo == "log":
                    txt_log.insert(tk.END, payload + "\n")
                    txt_log.see(tk.END)

                elif tipo == "progresso":
                    atual, total = payload
                    barra_progresso["maximum"] = total
                    barra_progresso["value"] = atual

                elif tipo == "fim_ok":
                    btn_processar.config(state="normal")
                    btn_cancelar.config(state="disabled")
                    messagebox.showinfo("Concluído", f"Processamento finalizado.\nCSV salvo em:\n{ARQUIVO_SAIDA}")

                elif tipo == "fim_cancelado":
                    btn_processar.config(state="normal")
                    btn_cancelar.config(state="disabled")
                    messagebox.showinfo("Cancelado", "Processamento cancelado pelo usuário.")

                elif tipo == "erro":
                    btn_processar.config(state="normal")
                    btn_cancelar.config(state="disabled")
                    messagebox.showerror("Erro", payload)

        except queue.Empty:
            pass

        janela.after(100, processar_fila_ui)

    def executar_processamento(caminho):
        try:
            if os.path.isfile(caminho):
                processar_entrada_unica(
                    caminho,
                    callback_log=log,
                    callback_progresso=atualizar_progresso
                )
            else:
                processar_lote(
                    callback_log=log,
                    callback_progresso=atualizar_progresso
                )

            if CANCELAR_PROCESSAMENTO:
                fila_ui.put(("fim_cancelado", None))
            else:
                fila_ui.put(("fim_ok", None))

        except Exception as e:
            fila_ui.put(("erro", str(e)))

    def cancelar():
        global CANCELAR_PROCESSAMENTO
        CANCELAR_PROCESSAMENTO = True
        log("⛔ Cancelamento solicitado pelo usuário...")
        barra_progresso.stop()  # opcional se usar modo indeterminado
    def escolher_pasta():
        pasta = filedialog.askdirectory(title="Selecione a pasta com as provas")
        if pasta:
            entry_pasta.delete(0, tk.END)
            entry_pasta.insert(0, pasta)

    def escolher_arquivo():
        arquivo = filedialog.askopenfilename(
            title="Selecione um PDF ou imagem",
            filetypes=[
                ("PDF", "*.pdf"),
                ("Imagens", "*.png *.jpg *.jpeg *.bmp *.webp *.tif *.tiff")
            ]
        )
        if arquivo:
            entry_pasta.delete(0, tk.END)
            entry_pasta.insert(0, arquivo)

    def atualizar_progresso(atual, total):
        fila_ui.put(("progresso", (atual, total)))

    def log(mensagem):
        fila_ui.put(("log", mensagem))

    def iniciar():
        global PASTA_IMAGENS, ARQUIVO_SAIDA, NUM_QUESTOES, ALTERNATIVAS
        global SALVAR_DEBUG, BOX_ALUNO_REFERENCIA, CANCELAR_PROCESSAMENTO

        caminho = entry_pasta.get().strip()
        questoes = entry_questoes.get().strip()
        alternativas_txt = entry_alternativas.get().strip()

        SALVAR_DEBUG = debug_var.get()
        BOX_ALUNO_REFERENCIA = None
        CANCELAR_PROCESSAMENTO = False

        if not caminho:
            messagebox.showerror("Erro", "Informe a pasta ou arquivo.")
            return

        if not questoes.isdigit():
            messagebox.showerror("Erro", "O número de questões deve ser numérico.")
            return

        lista_alternativas = [a.strip().upper() for a in alternativas_txt.split(",") if a.strip()]
        if not lista_alternativas:
            messagebox.showerror("Erro", "Informe as alternativas separadas por vírgula. Ex: A,B,C,D,E")
            return

        NUM_QUESTOES = int(questoes)
        ALTERNATIVAS = lista_alternativas

        if os.path.isdir(caminho):
            PASTA_IMAGENS = caminho
            ARQUIVO_SAIDA = os.path.join(PASTA_IMAGENS, "resultados.csv")
        else:
            pasta_base = os.path.dirname(caminho)
            PASTA_IMAGENS = pasta_base
            ARQUIVO_SAIDA = os.path.join(PASTA_IMAGENS, "resultados.csv")

        txt_log.delete("1.0", tk.END)
        barra_progresso["value"] = 0
        log("Iniciando processamento...")

        btn_processar.config(state="disabled")
        btn_cancelar.config(state="normal")

        # AQUI fica só o start da thread
        thread = threading.Thread(
            target=executar_processamento,
            args=(caminho,),
            daemon=True
        )
        thread.start()

    # Janela principal
    janela = tk.Tk()
    janela.title("Leitor de Gabarito")
    janela.geometry("760x520")
    janela.resizable(False, False)

    # Barra de progresso
    barra_progresso = ttk.Progressbar(janela, orient="horizontal", length=650, mode="determinate")
    barra_progresso.pack(padx=10, pady=(10, 5), fill="x")

    # Opção de debug
    debug_var = tk.BooleanVar(value=False)
    tk.Checkbutton(
        janela,
        text="Gerar imagens de debug",
        variable=debug_var
    ).pack(anchor="w", padx=10, pady=(5, 0))

    # Campo de caminho
    tk.Label(janela, text="Pasta ou arquivo (imagem/PDF):").pack(anchor="w", padx=10, pady=(10, 2))

    frame_caminho = tk.Frame(janela)
    frame_caminho.pack(fill="x", padx=10)

    entry_pasta = tk.Entry(frame_caminho, width=70)
    entry_pasta.pack(side="left", fill="x", expand=True)

    tk.Button(frame_caminho, text="Pasta de imagens", command=escolher_pasta).pack(side="left", padx=5)
    tk.Button(frame_caminho, text="Arquivo PDF", command=escolher_arquivo).pack(side="left")

    # Número de questões
    tk.Label(janela, text="Número de questões:").pack(anchor="w", padx=10, pady=(15, 2))
    entry_questoes = tk.Entry(janela, width=20)
    entry_questoes.pack(anchor="w", padx=10)
    entry_questoes.insert(0, "27")

    # Alternativas
    tk.Label(janela, text="Alternativas (separadas por vírgula):").pack(anchor="w", padx=10, pady=(15, 2))
    entry_alternativas = tk.Entry(janela, width=30)
    entry_alternativas.pack(anchor="w", padx=10)
    entry_alternativas.insert(0, "A,B,C,D,E")

    # Botão de processamento
    frame_botoes = tk.Frame(janela)
    frame_botoes.pack(pady=15)

    btn_processar = tk.Button(
        frame_botoes,
        text="Processar",
        command=iniciar,
        height=2,
        width=15
    )
    btn_processar.pack(side="left", padx=5)

    # Botão de cancelamento
    btn_cancelar = tk.Button(
        frame_botoes,
        text="Cancelar",
        command=cancelar,
        height=2,
        width=15,
        state="disabled"
    )
    btn_cancelar.pack(side="left", padx=5)
    # Área de log
    tk.Label(janela, text="Saída do processamento:").pack(anchor="w", padx=10, pady=(10, 2))

    frame_log = tk.Frame(janela)
    frame_log.pack(fill="both", expand=True, padx=10, pady=(0, 10))

    scroll_log = tk.Scrollbar(frame_log)
    scroll_log.pack(side="right", fill="y")

    txt_log = tk.Text(frame_log, height=12, wrap="word", yscrollcommand=scroll_log.set)
    txt_log.pack(side="left", fill="both", expand=True)

    scroll_log.config(command=txt_log.yview)
    janela.after(100, processar_fila_ui)
    janela.mainloop()

# ============================================================
# EXECUÇÃO EM THREAD
# ============================================================
# O processamento roda em thread separada para evitar travamento
# da interface gráfica.
#
# Comunicação feita via fila (queue).

if __name__ == "__main__":
    abrir_interface()
