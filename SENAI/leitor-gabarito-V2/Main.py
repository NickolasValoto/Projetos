"""
Leitor de Gabarito - Processamento automático de folhas de respostas.

Desenvolvido por Nickolas Valoto.

Este script lê arquivos PDF ou imagens contendo gabaritos preenchidos, corrige
alinhamento da página, permite selecionar manualmente a tabela de respostas na
primeira página, reutiliza essa referência nas páginas seguintes, identifica o
nome do aluno por OCR e exporta um CSV com as respostas por questão.

Fluxo geral:
1. O usuário seleciona um PDF ou imagem pela interface Tkinter.
2. Se for PDF, as páginas são convertidas para imagens JPG.
3. A primeira página é alinhada e exibida para seleção manual da tabela.
4. A caixa da tabela é reutilizada nas demais páginas.
5. A cada página, o sistema:
   - corrige inclinação/perspectiva quando possível;
   - recorta a tabela;
   - identifica marcações nas alternativas;
   - tenta ler o nome do aluno via Tesseract OCR;
   - adiciona o resultado à lista final.
6. Ao final, gera um arquivo resultado_geral.csv.

Dependências principais:
- OpenCV (cv2): processamento de imagem.
- PyMuPDF (fitz): conversão de PDF em imagem.
- NumPy: operações matriciais.
- pytesseract: OCR para localizar rótulo e ler nome do aluno.
- Tkinter: interface gráfica.

Observação:
Este código foi mantido em arquivo único para facilitar execução e manutenção
inicial. Em uma evolução futura, pode ser separado em módulos como config.py,
pdf_utils.py, image_alignment.py, ocr_name.py, table_reader.py e gui.py.
"""

from __future__ import annotations

import os
import re
import queue
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import cv2
import fitz  # PyMuPDF
import numpy as np
import pytesseract
import tkinter as tk
from tkinter import ttk, filedialog, messagebox


# ============================================================
# CONFIGURAÇÕES
# ============================================================

@dataclass
class Configuracao:
    """
    Centraliza todos os parâmetros configuráveis do sistema.

    Os campos desta classe controlam desde a quantidade de questões e
    alternativas até parâmetros finos de OCR, marcação, debug e interface.
    A interface gráfica preenche parte desses valores antes do processamento.
    """
    num_questoes: int = 27
    alternativas: tuple[str, ...] = ("A", "B", "C", "D", "E")
    salvar_debug: bool = True
    tesseract_cmd: str = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    zoom_pdf: float = 3.0

    # leitura das marcações
    limiar_marcacao: int = 35
    fator_dominancia: float = 1.25
    fator_roi_x: float = 0.88
    fator_roi_y: float = 0.95
    fator_deslocamento_variacao: float = 0.10

    # OCR do nome
    palavra_nome: str = "Aluno"
    distancia_esquerda_nome: int = 90
    distancia_superior_nome: int = 0
    largura_retangulo_nome: int = 600
    altura_retangulo_nome: int = 80

    # janela de seleção manual
    largura_maxima_selecao: int = 900
    altura_barra_selecao: int = 95


@dataclass
class ContextoProcessamento:
    """
    Guarda referências calculadas na primeira página e reaproveitadas nas demais.

    A ideia é evitar que o usuário precise selecionar a tabela em cada página.
    Depois que a primeira página define box da tabela, box do nome, tamanho
    padrão e rotação 180°, esses dados são mantidos neste contexto.
    """
    box_relativa: Optional[tuple[float, float, float, float]] = None
    box_nome_relativa: Optional[tuple[float, float, float, float]] = None
    tamanho_padrao: Optional[tuple[int, int]] = None
    rotacionar_180: Optional[bool] = None


@dataclass
class SelecaoInicial:
    """
    Representa a seleção manual feita na primeira página.

    Contém a caixa da tabela selecionada, se houve rotação manual de 180°
    e o tamanho padrão usado para padronizar as páginas seguintes.
    """
    box: tuple[int, int, int, int]
    rotacionado_180: bool
    tamanho_padrao: tuple[int, int]


LogCallback = Optional[Callable[[str], None]]
ProgressCallback = Optional[Callable[[int, int], None]]


# ============================================================
# UTILITÁRIOS DE LOG E ARQUIVOS
# ============================================================

class DebugWriter:
    """
    Controla a gravação de imagens de debug.

    Quando salvar_debug=False, as chamadas de salvamento são ignoradas,
    evitando escrita desnecessária em disco.
    """
    def __init__(self, salvar_debug: bool):
        self.salvar_debug = salvar_debug

    def salvar(self, caminho: str | Path, img) -> bool:
        if not self.salvar_debug:
            return False
        return salvar_imagem_segura(caminho, img)


def escrever_log(msg: str, callback_log: LogCallback = None) -> None:
    """Envia uma mensagem para a interface gráfica ou para o terminal."""
    if callback_log:
        callback_log(msg)
    else:
        print(msg)


def salvar_imagem_segura(caminho: str | Path, img) -> bool:
    """
    Salva uma imagem OpenCV em disco de forma compatível com caminhos do Windows.

    Usa cv2.imencode + escrita binária para evitar problemas com caracteres
    especiais no caminho, como acentos, espaços e símbolos.
    """
    if img is None:
        return False

    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)

    ext = caminho.suffix or ".png"
    if not caminho.suffix:
        caminho = caminho.with_suffix(ext)

    try:
        ok, buffer = cv2.imencode(ext, img)
        if not ok:
            return False

        with open(caminho, "wb") as f:
            f.write(buffer.tobytes())

        return True
    except Exception:
        return False


def ler_imagem_segura(caminho: str | Path):
    """
    Lê uma imagem de forma segura, inclusive em caminhos com caracteres especiais.

    Retorna uma imagem BGR do OpenCV ou None caso a leitura falhe.
    """
    caminho = Path(caminho)
    if not caminho.exists():
        return None

    try:
        data = np.frombuffer(caminho.read_bytes(), np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def caminho_pasta_paginas(caminho_pdf: str | Path) -> Path:
    """Retorna a pasta onde as páginas convertidas do PDF serão armazenadas."""
    caminho_pdf = Path(caminho_pdf)
    return caminho_pdf.parent / caminho_pdf.stem


def obter_pasta_debug_pdf(caminho_pdf: str | Path) -> Path:
    """Cria e retorna a pasta de debug associada ao PDF processado."""
    pasta_debug = caminho_pasta_paginas(caminho_pdf) / "debug"
    pasta_debug.mkdir(parents=True, exist_ok=True)
    return pasta_debug


# ============================================================
# PDF
# ============================================================

def converter_pdf_para_jpgs(caminho_pdf: str | Path, pasta_saida: str | Path, zoom: float = 3.0) -> list[str]:
    """
    Converte cada página de um PDF em imagem JPG.

    Se a imagem da página já existir na pasta de saída, ela é reaproveitada.
    Isso evita converter o mesmo PDF duas vezes, acelerando o fluxo após a
    seleção inicial da primeira página.
    """
    pasta_saida = Path(pasta_saida)
    pasta_saida.mkdir(parents=True, exist_ok=True)

    caminhos_imagens: list[str] = []
    doc = fitz.open(str(caminho_pdf))

    try:
        for i, pagina in enumerate(doc):
            caminho_img = pasta_saida / f"pagina_{i + 1:03d}.jpg"

            # reaproveita imagem já convertida
            if caminho_img.exists():
                caminhos_imagens.append(str(caminho_img))
                continue

            mat = fitz.Matrix(zoom, zoom)
            pix = pagina.get_pixmap(matrix=mat, alpha=False)
            pix.save(str(caminho_img))
            caminhos_imagens.append(str(caminho_img))
    finally:
        doc.close()

    return caminhos_imagens


# ============================================================
# COORDENADAS E CANVAS
# ============================================================

def box_absoluta_para_relativa(box: tuple[int, int, int, int], shape_img) -> tuple[float, float, float, float]:
    """Converte uma caixa em pixels para coordenadas relativas à imagem."""
    x, y, w, h = box
    altura, largura = shape_img[:2]
    return x / largura, y / altura, w / largura, h / altura


def box_relativa_para_absoluta(box_rel: tuple[float, float, float, float], shape_img) -> tuple[int, int, int, int]:
    """Converte uma caixa relativa para coordenadas absolutas em pixels."""
    x_rel, y_rel, w_rel, h_rel = box_rel
    altura, largura = shape_img[:2]

    x = int(round(x_rel * largura))
    y = int(round(y_rel * altura))
    w = int(round(w_rel * largura))
    h = int(round(h_rel * altura))

    x = max(0, min(x, largura - 1))
    y = max(0, min(y, altura - 1))
    w = max(1, min(w, largura - x))
    h = max(1, min(h, altura - y))

    return x, y, w, h


def encaixar_em_canvas_padrao_topo_esquerda(img, tamanho_padrao: tuple[int, int]):
    """
    Redimensiona uma imagem mantendo proporção e a posiciona no canto superior esquerdo.

    O restante do canvas é preenchido em branco. Essa padronização permite
    reutilizar coordenadas relativas entre páginas diferentes.
    """
    largura_padrao, altura_padrao = tamanho_padrao
    h, w = img.shape[:2]

    escala = min(largura_padrao / w, altura_padrao / h)
    novo_w = int(round(w * escala))
    novo_h = int(round(h * escala))

    img_redim = cv2.resize(img, (novo_w, novo_h), interpolation=cv2.INTER_CUBIC)
    canvas = np.full((altura_padrao, largura_padrao, 3), 255, dtype=np.uint8)
    canvas[0:novo_h, 0:novo_w] = img_redim
    return canvas


def encaixar_em_canvas_padrao(img, tamanho_padrao: tuple[int, int]):
    """Redimensiona uma imagem mantendo proporção e a centraliza em canvas branco."""
    largura_padrao, altura_padrao = tamanho_padrao
    h, w = img.shape[:2]

    escala = min(largura_padrao / w, altura_padrao / h)
    novo_w = int(round(w * escala))
    novo_h = int(round(h * escala))

    img_redim = cv2.resize(img, (novo_w, novo_h), interpolation=cv2.INTER_CUBIC)
    canvas = np.full((altura_padrao, largura_padrao, 3), 255, dtype=np.uint8)

    x_ini = (largura_padrao - novo_w) // 2
    y_ini = (altura_padrao - novo_h) // 2
    canvas[y_ini:y_ini + novo_h, x_ini:x_ini + novo_w] = img_redim
    return canvas


def redimensionar_para_tamanho_padrao(img, tamanho_padrao: tuple[int, int]):
    """Redimensiona diretamente a imagem para o tamanho padrão informado."""
    largura_padrao, altura_padrao = tamanho_padrao
    return cv2.resize(img, (largura_padrao, altura_padrao), interpolation=cv2.INTER_CUBIC)


# ============================================================
# ALINHAMENTO DA PÁGINA
# ============================================================

def ordenar_pontos(pts):
    """Ordena quatro pontos no padrão: superior-esquerdo, superior-direito, inferior-direito, inferior-esquerdo."""
    pts = np.array(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]

    return np.array([tl, tr, br, bl], dtype=np.float32)


def detectar_folha_por_contorno(img, pasta_debug: str | Path, debug: DebugWriter, nome_base_debug: str = "debug"):
    """
    Tenta detectar o contorno da folha na imagem.

    Usa Canny, dilatação/erosão e busca por contornos quadriláteros grandes.
    Se encontrar uma folha plausível, retorna seus quatro pontos. Caso contrário,
    retorna None e o processamento segue usando apenas correção de rotação.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blur, 50, 150)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.dilate(edges, kernel, iterations=2)
    edges = cv2.erode(edges, kernel, iterations=1)

    contornos, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h_img, w_img = img.shape[:2]
    area_img = h_img * w_img
    melhor = None
    melhor_area = 0

    for cnt in contornos:
        area = cv2.contourArea(cnt)
        if area < area_img * 0.20:
            continue

        peri = cv2.arcLength(cnt, True)
        aprox = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        if len(aprox) == 4 and area > melhor_area:
            melhor = aprox.reshape(4, 2)
            melhor_area = area

    if melhor is not None:
        debug_contornos = img.copy()
        cv2.polylines(debug_contornos, [melhor.astype(np.int32)], True, (0, 0, 255), 4)
        debug.salvar(Path(pasta_debug) / f"{nome_base_debug}_01b_contorno_folha_detectado.jpg", debug_contornos)

    return melhor


def corrigir_perspectiva_folha(img, pts, pasta_debug: str | Path, debug: DebugWriter, nome_base_debug: str = "debug"):
    """Aplica transformação de perspectiva para deixar a folha frontalizada."""
    pts = ordenar_pontos(pts)
    tl, tr, br, bl = pts

    largura_a = np.linalg.norm(br - bl)
    largura_b = np.linalg.norm(tr - tl)
    altura_a = np.linalg.norm(tr - br)
    altura_b = np.linalg.norm(tl - bl)

    max_w = int(max(largura_a, largura_b))
    max_h = int(max(altura_a, altura_b))

    destino = np.array([
        [0, 0],
        [max_w - 1, 0],
        [max_w - 1, max_h - 1],
        [0, max_h - 1]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(pts, destino)
    warp = cv2.warpPerspective(
        img,
        M,
        (max_w, max_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255)
    )

    debug.salvar(Path(pasta_debug) / f"{nome_base_debug}_02_folha_warp.jpg", warp)
    return warp


def rotacionar_imagem_sem_cortar(img, angulo: float):
    """Rotaciona a imagem expandindo o canvas para evitar cortes nas bordas."""
    h, w = img.shape[:2]
    centro = (w / 2, h / 2)

    M = cv2.getRotationMatrix2D(centro, angulo, 1.0)
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])

    novo_w = int((h * sin) + (w * cos))
    novo_h = int((h * cos) + (w * sin))

    M[0, 2] += (novo_w / 2) - centro[0]
    M[1, 2] += (novo_h / 2) - centro[1]

    return cv2.warpAffine(
        img,
        M,
        (novo_w, novo_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255)
    )


def estimar_angulo_rotacao(img, pasta_debug: str | Path, debug: DebugWriter, nome_base_debug: str = "debug") -> float:
    """
    Estima a inclinação da página com base em linhas horizontais longas.

    Primeiro tenta usar HoughLinesP para encontrar linhas próximas da horizontal.
    Se não houver linhas suficientes, usa minAreaRect como fallback.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    bin_img = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        10
    )

    linhas = cv2.HoughLinesP(
        bin_img,
        1,
        np.pi / 180,
        threshold=120,
        minLineLength=max(img.shape[1], img.shape[0]) // 3,
        maxLineGap=40
    )

    debug_linhas = None
    if debug.salvar_debug:
        debug_linhas = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    angulos = []

    if linhas is not None:
        for linha in linhas:
            x1, y1, x2, y2 = linha[0]
            dx = x2 - x1
            dy = y2 - y1

            if dx == 0 and dy == 0:
                continue

            ang = np.degrees(np.arctan2(dy, dx))

            while ang <= -90:
                ang += 180
            while ang > 90:
                ang -= 180

            if -20 <= ang <= 20:
                comp = np.hypot(dx, dy)
                angulos.append((ang, comp))
                if debug_linhas is not None:
                    cv2.line(debug_linhas, (x1, y1), (x2, y2), (0, 0, 255), 2)

    if debug_linhas is not None:
        debug.salvar(Path(pasta_debug) / f"{nome_base_debug}_04_linhas_usadas_rotacao.jpg", debug_linhas)

    if angulos:
        valores = np.array([a for a, _ in angulos], dtype=np.float32)
        pesos = np.array([p for _, p in angulos], dtype=np.float32)

        med = np.median(valores)
        mask = np.abs(valores - med) <= 3.0
        if np.any(mask):
            valores = valores[mask]
            pesos = pesos[mask]

        return float(np.average(valores, weights=pesos))

    coords = cv2.findNonZero(bin_img)
    if coords is None:
        return 0.0

    rect = cv2.minAreaRect(coords)
    ang = rect[-1]

    if ang < -45:
        ang = 90 + ang
    elif ang > 45:
        ang = ang - 90

    return float(ang)


def alinhar_imagem(img, pasta_debug: str | Path, debug: DebugWriter, nome_base_debug: str = "debug", callback_log: LogCallback = None):
    """
    Executa o alinhamento da página.

    Tenta corrigir perspectiva pela folha detectada e, em seguida, corrige o
    ângulo fino. Se a folha não for detectada, aplica apenas a correção angular.
    """
    pts_folha = detectar_folha_por_contorno(img, pasta_debug, debug, nome_base_debug)

    if pts_folha is not None:
        img_corrigida = corrigir_perspectiva_folha(img, pts_folha, pasta_debug, debug, nome_base_debug)
        angulo_detectado = estimar_angulo_rotacao(img_corrigida, pasta_debug, debug, nome_base_debug)
        origem = "folha detectada com sucesso"
    else:
        img_corrigida = img
        angulo_detectado = estimar_angulo_rotacao(img_corrigida, pasta_debug, debug, nome_base_debug)
        origem = "folha não detectada; usando fallback por rotação"

    angulo_aplicado = 0.0 if abs(angulo_detectado) < 0.03 else angulo_detectado

    if angulo_aplicado == 0.0:
        img_alinhada = img_corrigida.copy()
    else:
        img_alinhada = rotacionar_imagem_sem_cortar(img_corrigida, angulo_aplicado)

    debug.salvar(Path(pasta_debug) / f"{nome_base_debug}_05_imagem_alinhada.jpg", img_alinhada)

    escrever_log(f"[INFO] {origem}", callback_log)
    escrever_log(f"[INFO] ângulo detectado: {angulo_detectado:.4f} graus", callback_log)
    escrever_log(f"[INFO] ângulo aplicado: {angulo_aplicado:.4f} graus", callback_log)

    return img_alinhada, angulo_aplicado


# ============================================================
# SELEÇÃO MANUAL DA TABELA
# ============================================================

def selecionar_tabela_manual_opencv(img, config: Configuracao):
    """
    Abre uma janela OpenCV para seleção manual da tabela do gabarito.

    Controles:
    - Arrastar mouse: define a área da tabela.
    - ENTER: confirma a seleção.
    - R: gira a visualização em 180°.
    - C: limpa a seleção.
    - ESC: cancela.
    """
    img_vis = img.copy()
    h_original, w_original = img_vis.shape[:2]
    escala = 1.0

    if w_original > config.largura_maxima_selecao:
        escala = config.largura_maxima_selecao / float(w_original)
        img_vis = cv2.resize(img_vis, None, fx=escala, fy=escala, interpolation=cv2.INTER_AREA)

    barra_h = config.altura_barra_selecao
    desenhando = False
    x_ini = y_ini = x_fim = y_fim = -1
    roi_confirmada = None
    rotacionado_180 = False

    def montar_tela(img_exibicao, roi_temp=None):
        painel = np.full((barra_h, img_exibicao.shape[1], 3), 245, dtype=np.uint8)

        cv2.putText(painel, "Selecao da tabela", (20, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (40, 40, 40), 2)

        cv2.putText(painel,
                    "Selecione a area do GABARITO | ENTER confirmar | R girar 180 graus | C limpar",
                    (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (70, 70, 70), 1)

        tela = np.vstack([painel, img_exibicao.copy()])

        if roi_temp:
            x1, y1, x2, y2 = roi_temp
            cv2.rectangle(tela, (x1, y1 + barra_h), (x2, y2 + barra_h), (0, 255, 0), 2)

        return tela

    def mouse(event, x, y, flags, param):
        nonlocal desenhando, x_ini, y_ini, x_fim, y_fim, roi_confirmada

        if y < barra_h:
            return

        y -= barra_h

        if event == cv2.EVENT_LBUTTONDOWN:
            desenhando = True
            x_ini, y_ini = x, y

        elif event == cv2.EVENT_MOUSEMOVE and desenhando:
            x_fim, y_fim = x, y

        elif event == cv2.EVENT_LBUTTONUP:
            desenhando = False
            x_fim, y_fim = x, y
            roi_confirmada = (
                min(x_ini, x_fim),
                min(y_ini, y_fim),
                max(x_ini, x_fim),
                max(y_ini, y_fim)
            )

    cv2.namedWindow("Selecao")
    cv2.setMouseCallback("Selecao", mouse)

    while True:
        if desenhando:
            roi_temp = (min(x_ini, x_fim), min(y_ini, y_fim), max(x_ini, x_fim), max(y_ini, y_fim))
        else:
            roi_temp = roi_confirmada

        cv2.imshow("Selecao", montar_tela(img_vis, roi_temp))
        tecla = cv2.waitKey(20) & 0xFF

        if tecla == 13:  # ENTER
            if roi_confirmada:
                x1, y1, x2, y2 = roi_confirmada
                x = int(x1 / escala)
                y = int(y1 / escala)
                w = int((x2 - x1) / escala)
                h = int((y2 - y1) / escala)
                cv2.destroyAllWindows()
                return x, y, w, h, rotacionado_180

        elif tecla in (ord("r"), ord("R")):
            img_vis = cv2.rotate(img_vis, cv2.ROTATE_180)
            rotacionado_180 = not rotacionado_180
            roi_confirmada = None

        elif tecla in (ord("c"), ord("C")):
            roi_confirmada = None

        elif tecla in (27, 15):  # ESC ou CTRL+O em alguns teclados
            cv2.destroyAllWindows()
            raise RuntimeError("Seleção cancelada.")


# ============================================================
# OCR DO NOME
# ============================================================

def normalizar_texto_ocr(txt: str) -> str:
    """Normaliza textos vindos do OCR para facilitar comparação com o rótulo procurado."""
    txt = (txt or "").strip().lower()

    trocas = {
        "|": "l", "!": "l", "1": "l", "0": "o", "5": "s", "$": "s",
        "(": "", ")": "", "[": "", "]": "", "{": "", "}": "",
        ":": "", ";": "", ",": "", ".": "", "_": "", "-": "",
        "/": "", "\\": "", "'": "", '"': "", " ": ""
    }

    for a, b in trocas.items():
        txt = txt.replace(a, b)

    return txt


def score_rotulo_cabecalho(txt_norm: str, palavra_norm: str = "aluno") -> int:
    """Pontua a chance de um texto OCR corresponder ao rótulo do campo de nome."""
    if not txt_norm:
        return 0

    if txt_norm == palavra_norm:
        return 100

    if txt_norm.startswith(palavra_norm[:4]):
        return 90

    if palavra_norm[:4] in txt_norm:
        return 80

    # mantém compatibilidade com os padrões antigos
    fortes = {"aluno": 100, "aluna": 100, "alunoa": 100, "nome": 95}

    if txt_norm in fortes:
        return fortes[txt_norm]

    if txt_norm.startswith("alun"):
        return 85
    if "alun" in txt_norm:
        return 75
    if txt_norm.startswith("nom"):
        return 80
    if "nome" in txt_norm:
        return 70

    return 0

def detectar_box_nome_aluno(img, config: Configuracao, pasta_debug: str | Path | None = None, nome_base_debug: str = "debug"):
    """
    Detecta a região onde está escrito o nome do aluno.

    Primeiro localiza no topo da folha uma palavra configurável, por padrão
    "Aluno". A partir da posição desse rótulo, monta uma ROI usando os valores
    configuráveis de distância, largura e altura. Essa ROI é usada para OCR do
    nome e pode ser visualizada no debug.
    """
    h, w = img.shape[:2]
    topo = img[0:int(h * 0.28), 0:w].copy()

    gray = cv2.cvtColor(topo, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    variantes = []

    bin1 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 12)
    variantes.append(("bin1", bin1, 1.0, 1.0))

    bin2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 8)
    variantes.append(("bin2", bin2, 1.0, 1.0))

    gray_big = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    bin3 = cv2.adaptiveThreshold(gray_big, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10)
    variantes.append(("bin3_big", bin3, 0.5, 0.5))

    melhor = None

    for nome_variante, img_ocr, escala_x, escala_y in variantes:
        for psm in (6, 11, 12):
            try:
                dados = pytesseract.image_to_data(
                    img_ocr,
                    lang="eng",
                    config=f"--oem 3 --psm {psm}",
                    output_type=pytesseract.Output.DICT
                )
            except Exception:
                continue

            for i, bruto in enumerate(dados["text"]):
                txt_norm = normalizar_texto_ocr(bruto)
                palavra_norm = normalizar_texto_ocr(config.palavra_nome)
                score = score_rotulo_cabecalho(txt_norm, palavra_norm)

                if score <= 0:
                    continue

                candidato = {
                    "score": score,
                    "texto": bruto,
                    "x": int(round(dados["left"][i] * escala_x)),
                    "y": int(round(dados["top"][i] * escala_y)),
                    "w": int(round(dados["width"][i] * escala_x)),
                    "h": int(round(dados["height"][i] * escala_y)),
                    "variante": nome_variante,
                    "psm": psm,
                }

                if melhor is None or candidato["score"] > melhor["score"]:
                    melhor = candidato

    if melhor is None or melhor["score"] < 75:
        if pasta_debug:
            salvar_imagem_segura(Path(pasta_debug) / f"{nome_base_debug}_debug_nome_detectado.jpg", topo)
        return None

    x = melhor["x"]
    y = melhor["y"]
    w_box = melhor["w"]
    h_box = melhor["h"]

    x_ini = min(
        topo.shape[1] - 1,
        x + w_box + config.distancia_esquerda_nome
    )

    y_ini = max(
        0,
        y + config.distancia_superior_nome
    )

    x_fim = min(
        topo.shape[1],
        x_ini + config.largura_retangulo_nome
    )

    y_fim = min(
        topo.shape[0],
        y_ini + config.altura_retangulo_nome
    )

    if x_fim <= x_ini or y_fim <= y_ini:
        return None

    if pasta_debug:
        debug_final = topo.copy()
        cv2.rectangle(debug_final, (x, y), (x + w_box, y + h_box), (0, 0, 255), 2)
        cv2.putText(debug_final, f"{melhor['texto']} ({melhor['score']})",
                    (x, max(18, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.rectangle(debug_final, (x_ini, y_ini), (x_fim, y_fim), (0, 255, 0), 2)
        salvar_imagem_segura(Path(pasta_debug) / f"{nome_base_debug}_debug_nome_detectado.jpg", debug_final)

    return x_ini, y_ini, x_fim - x_ini, y_fim - y_ini


def limpar_nome_ocr(txt: str) -> str:
    """Remove ruídos comuns do OCR e mantém apenas caracteres compatíveis com nomes."""
    txt = txt.replace("\n", " ").replace("\r", " ")
    txt = re.sub(r"\s+", " ", txt).strip()

    txt = re.sub(r"^[\s:;,\-_.|]+", "", txt)
    txt = re.sub(r"[\s:;,\-_.|]+$", "", txt)

    txt = re.split(
        r"\b(Serie|Série|Turma|Data|Disciplina|Professor|Prof|Nota|Curso|Ano|N[oº]|Matricula|Matr[ií]cula|RM|RGM|RA)\b",
        txt,
        maxsplit=1,
        flags=re.IGNORECASE
    )[0].strip()

    # Remove números e qualquer sujeira que não seja letra, acento, espaço, apóstrofo ou hífen.
    txt = re.sub(r"[^A-Za-zÀ-ÿ '\-]", " ", txt)
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt


def pontuar_nome(txt: str) -> int:
    """Atribui uma pontuação simples para escolher a melhor leitura OCR do nome."""
    if not txt:
        return -999

    score = 0
    palavras = txt.split()
    tamanho = len(txt)

    if tamanho >= 6:
        score += 20
    if 2 <= len(palavras) <= 6:
        score += 30
    elif len(palavras) == 1:
        score += 10
    elif len(palavras) > 6:
        score -= 10

    letras = sum(ch.isalpha() for ch in txt)
    score += int((letras / max(1, len(txt))) * 40)

    if re.search(r"(aluno|aluna|nome|serie|série|turma|data|disciplina|professor|prof|nota|curso|ano|matricula|matrícula|rm|rgm|ra)",
                 txt, flags=re.IGNORECASE):
        score -= 40

    return score


def salvar_debug_box_nome(img, box_nome, pasta_debug: str | Path, nome_base_debug: str, debug: DebugWriter) -> None:
    """Salva imagens de debug mostrando a ROI usada para leitura do nome."""
    if box_nome is None:
        return

    x, y, w, h = box_nome
    debug_img = img.copy()
    cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(debug_img, "ROI NOME", (x, max(20, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)

    debug.salvar(Path(pasta_debug) / f"{nome_base_debug}_debug_nome_detectado.jpg", debug_img)

    roi_nome = img[y:y + h, x:x + w].copy()
    if roi_nome.size > 0:
        debug.salvar(Path(pasta_debug) / f"{nome_base_debug}_debug_roi_nome.jpg", roi_nome)


def ler_nome_aluno_da_box(img, box_nome, pasta_debug: str | Path | None = None, nome_base_debug: str = "debug") -> str:
    """
    Executa OCR na ROI do nome e escolhe a melhor leitura.

    Testa diferentes pré-processamentos, modos PSM e idiomas do Tesseract.
    A melhor leitura é escolhida por pontuação, limpa e devolvida como string.
    """
    x, y, w, h = box_nome
    roi_nome = img[y:y + h, x:x + w].copy()

    if roi_nome.size == 0:
        return ""

    margem_x = max(1, int(w * 0.02))
    margem_y = max(1, int(h * 0.08))

    roi_nome = roi_nome[margem_y:max(margem_y + 1, h - margem_y), margem_x:max(margem_x + 1, w - margem_x)].copy()

    if roi_nome.size == 0:
        return ""

    gray = cv2.cvtColor(roi_nome, cv2.COLOR_BGR2GRAY)
    variantes = []

    v1 = cv2.resize(gray, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
    variantes.append(("gray_big", v1))

    v2 = cv2.GaussianBlur(gray, (3, 3), 0)
    v2 = cv2.resize(v2, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
    v2 = cv2.adaptiveThreshold(v2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10)
    variantes.append(("adapt", v2))

    v3 = cv2.GaussianBlur(gray, (3, 3), 0)
    v3 = cv2.resize(v3, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
    _, v3 = cv2.threshold(v3, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variantes.append(("otsu", v3))

    melhores = []

    for nome_variante, img_ocr in variantes[:2]:
        for psm in (7,):
            for lang in ("por",):
                try:
                    texto = pytesseract.image_to_string(
                        img_ocr,
                        lang=lang,
                        config=f"--oem 3 --psm {psm}"
                    )
                except Exception:
                    continue

                texto_limpo = limpar_nome_ocr(texto)
                melhores.append({
                    "variante": nome_variante,
                    "psm": psm,
                    "lang": lang,
                    "bruto": texto,
                    "limpo": texto_limpo,
                    "score": pontuar_nome(texto_limpo)
                })

    if not melhores:
        return ""

    melhores.sort(key=lambda item: item["score"], reverse=True)
    melhor = melhores[0]
    nome_final = melhor["limpo"]

    if pasta_debug:
        try:
            with open(Path(pasta_debug) / f"{nome_base_debug}_debug_ocr_nome.txt", "w", encoding="utf-8") as f:
                f.write(f"Melhor nome: {nome_final}\n")
                f.write(f"Score: {melhor['score']}\n")
                f.write(f"Variante: {melhor['variante']}\n")
                f.write(f"PSM: {melhor['psm']}\n")
                f.write(f"Linguagem: {melhor['lang']}\n\n")

                for item in melhores:
                    f.write(
                        f"Variante={item['variante']} | PSM={item['psm']} | "
                        f"LANG={item['lang']} | SCORE={item['score']}\n"
                    )
                    f.write(f"BRUTO : {repr(item['bruto'])}\n")
                    f.write(f"LIMPO : {repr(item['limpo'])}\n")
                    f.write("-" * 80 + "\n")
        except Exception:
            pass

    return nome_final


# ============================================================
# LEITURA DA TABELA
# ============================================================

def suavizar_vetor(vetor, k: int = 9):
    """Suaviza uma projeção unidimensional usando média móvel."""
    vetor = vetor.astype(np.float32)
    if k <= 1:
        return vetor
    kernel = np.ones(k, dtype=np.float32) / k
    return np.convolve(vetor, kernel, mode="same")


def segmentos_ativos(v):
    """Identifica intervalos contínuos onde um vetor booleano está ativo."""
    segmentos = []
    ini = None

    for i, val in enumerate(v):
        if val and ini is None:
            ini = i
        elif not val and ini is not None:
            segmentos.append((ini, i - 1))
            ini = None

    if ini is not None:
        segmentos.append((ini, len(v) - 1))

    return segmentos


def centros_de_segmentos(segmentos):
    """Calcula o centro de cada segmento ativo."""
    return [int((a + b) / 2) for a, b in segmentos]


def detectar_linhas_grade(tabela):
    """
    Detecta linhas horizontais e verticais da tabela por morfologia matemática.

    O resultado também é usado para separar a grade da marcação feita pelo aluno.
    """
    gray = cv2.cvtColor(tabela, cv2.COLOR_BGR2GRAY)

    bin_img = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        10
    )

    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (max(25, tabela.shape[1] // 25), 3))
    horizontais = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel_h)

    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (3, max(20, tabela.shape[0] // 6)))
    verticais = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel_v)

    proj_x = np.sum(verticais > 0, axis=0).astype(np.float32)
    proj_x_suave = suavizar_vetor(proj_x, k=9)
    limiar_x = max(5.0, 0.35 * np.max(proj_x_suave))
    xs = centros_de_segmentos(segmentos_ativos(proj_x_suave > limiar_x))

    proj_y = np.sum(horizontais > 0, axis=1).astype(np.float32)
    proj_y_suave = suavizar_vetor(proj_y, k=9)
    limiar_y = max(5.0, 0.35 * np.max(proj_y_suave))
    ys = centros_de_segmentos(segmentos_ativos(proj_y_suave > limiar_y))

    return {
        "gray": gray,
        "bin_img": bin_img,
        "horizontais": horizontais,
        "verticais": verticais,
        "xs": xs,
        "ys": ys
    }


def gerar_grade_fixa(tabela_shape, num_questoes: int, alternativas: int):
    """Divide a tabela em uma grade fixa com base na quantidade de questões e alternativas."""
    h, w = tabela_shape[:2]
    total_colunas = 1 + num_questoes
    total_linhas = 1 + alternativas

    xs = np.linspace(0, w, total_colunas + 1).astype(int).tolist()
    ys = np.linspace(0, h, total_linhas + 1).astype(int).tolist()

    return xs, ys


def calcular_retangulo_interno_celula(x1, y1, x2, y2, fator_x=2.01, fator_y=2.01):
    """Calcula a região interna analisada dentro de uma célula da grade."""
    largura = x2 - x1
    altura = y2 - y1

    if largura <= 1 or altura <= 1:
        return x1, y1, max(x1 + 1, x2), max(y1 + 1, y2)

    nova_largura = largura * fator_x
    nova_altura = altura * fator_y

    offset_x = (largura - nova_largura) / 5.0
    offset_y = (altura - nova_altura) / 5.0

    rx1 = int(round(x1 + offset_x))
    ry1 = int(round(y1 + offset_y))
    rx2 = int(round(x1 + offset_x + nova_largura))
    ry2 = int(round(y1 + offset_y + nova_altura))

    rx1 = max(x1, min(rx1, x2 - 1))
    ry1 = max(y1, min(ry1, y2 - 1))
    rx2 = max(rx1 + 1, min(rx2, x2))
    ry2 = max(ry1 + 1, min(ry2, y2))

    return rx1, ry1, rx2, ry2


def deslocar_retangulo(rx1, ry1, rx2, ry2, dx, dy, shape_img):
    """Desloca uma ROI sem permitir que ela saia dos limites da imagem."""
    h, w = shape_img[:2]
    largura = rx2 - rx1
    altura = ry2 - ry1

    rx1_n = rx1 + dx
    ry1_n = ry1 + dy
    rx2_n = rx2 + dx
    ry2_n = ry2 + dy

    if rx1_n < 0:
        rx1_n = 0
        rx2_n = min(w, largura)
    if ry1_n < 0:
        ry1_n = 0
        ry2_n = min(h, altura)

    if rx2_n > w:
        rx2_n = w
        rx1_n = max(0, w - largura)
    if ry2_n > h:
        ry2_n = h
        ry1_n = max(0, h - altura)

    return int(rx1_n), int(ry1_n), int(rx2_n), int(ry2_n)


def decidir_marcacao_por_scores(scores, alternativas, limiar_marcacao: int, fator_dominancia: float):
    """Decide qual alternativa foi marcada com base nos scores de pixels preenchidos."""
    maior_idx = int(np.argmax(scores))
    score_max = scores[maior_idx]
    scores_ordenados = sorted(scores, reverse=True)
    segundo = scores_ordenados[1] if len(scores_ordenados) > 1 else 0

    if score_max > limiar_marcacao and score_max > segundo * fator_dominancia:
        return alternativas[maior_idx]

    return "-"


def ler_respostas_variacao(
    tabela,
    marcas,
    xs,
    ys,
    pasta_debug: str | Path,
    nome_base_debug: str,
    sufixo_debug: str,
    config: Configuracao,
    debug: DebugWriter,
    deslocamento_x: int = 0,
    deslocamento_y: int = 0,
):
    """
    Lê as respostas aplicando uma variação de deslocamento na ROI das células.

    Essa estratégia aumenta a robustez quando a grade está levemente deslocada.
    Cada variação gera uma lista de respostas e detalhes por questão.
    """
    debug_celulas = tabela.copy()
    respostas = []
    detalhes = []

    for q in range(config.num_questoes):
        col_idx = q + 1
        scores = []

        for alt_idx, alt in enumerate(config.alternativas):
            row_idx = alt_idx + 1

            x1 = xs[col_idx]
            x2 = xs[col_idx + 1]
            y1 = ys[row_idx]
            y2 = ys[row_idx + 1]

            rx1, ry1, rx2, ry2 = calcular_retangulo_interno_celula(
                x1, y1, x2, y2,
                fator_x=config.fator_roi_x,
                fator_y=config.fator_roi_y
            )

            rx1, ry1, rx2, ry2 = deslocar_retangulo(
                rx1, ry1, rx2, ry2,
                dx=deslocamento_x,
                dy=deslocamento_y,
                shape_img=marcas.shape
            )

            roi = marcas[ry1:ry2, rx1:rx2]
            score = int(np.sum(roi > 0))
            scores.append(score)
            cv2.rectangle(debug_celulas, (rx1, ry1), (rx2, ry2), (255, 0, 0), 1)

        marcado = decidir_marcacao_por_scores(
            scores,
            config.alternativas,
            config.limiar_marcacao,
            config.fator_dominancia
        )

        respostas.append(marcado)
        detalhes.append({
            "questao": q + 1,
            "scores": dict(zip(config.alternativas, scores)),
            "escolha": marcado,
            "deslocamento_x": deslocamento_x,
            "deslocamento_y": deslocamento_y
        })

        maior_idx = int(np.argmax(scores))
        row_idx = maior_idx + 1
        x1 = xs[col_idx]
        x2 = xs[col_idx + 1]
        y1 = ys[row_idx]
        y2 = ys[row_idx + 1]

        rx1, ry1, rx2, ry2 = calcular_retangulo_interno_celula(
            x1, y1, x2, y2,
            fator_x=config.fator_roi_x,
            fator_y=config.fator_roi_y
        )

        rx1, ry1, rx2, ry2 = deslocar_retangulo(
            rx1, ry1, rx2, ry2,
            dx=deslocamento_x,
            dy=deslocamento_y,
            shape_img=marcas.shape
        )

        cor = (0, 255, 0) if marcado != "-" else (0, 0, 255)
        cv2.rectangle(debug_celulas, (rx1, ry1), (rx2, ry2), cor, 2)
        cv2.putText(debug_celulas, f"{q + 1}:{marcado}", (x1 + 2, max(12, y1 + 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, cor, 1, cv2.LINE_AA)

    debug.salvar(Path(pasta_debug) / f"{nome_base_debug}_{config.num_questoes}_celulas_{sufixo_debug}.jpg", debug_celulas)

    return {"respostas": respostas, "detalhes": detalhes}


def consolidar_respostas_multiplas(*listas_respostas):
    """
    Consolida múltiplas leituras da mesma tabela.

    Se todas concordam, mantém a resposta. Se só uma resposta válida aparece,
    usa essa resposta. Se houver conflito entre alternativas válidas, retorna "-".
    """
    respostas_finais = []
    detalhes_finais = []
    total = len(listas_respostas[0])
    nomes = ["original", "direita", "baixo", "cima"]

    for i in range(total):
        leituras = [lista[i] for lista in listas_respostas]
        validas = [r for r in leituras if r != "-"]
        unicas_validas = sorted(set(validas))

        if len(set(leituras)) == 1:
            final = leituras[0]
        elif len(unicas_validas) == 1:
            final = unicas_validas[0]
        else:
            final = "-"

        respostas_finais.append(final)

        detalhe = {"questao": i + 1, "final": final}
        detalhe.update({nome: valor for nome, valor in zip(nomes, leituras)})
        detalhes_finais.append(detalhe)

    return respostas_finais, detalhes_finais


def ler_respostas_da_tabela(tabela, pasta_debug: str | Path, nome_base_debug: str, config: Configuracao, debug: DebugWriter):
    """Lê todas as respostas da tabela recortada e retorna respostas, detalhes e grade usada."""
    dados_grade = detectar_linhas_grade(tabela)

    bin_img = dados_grade["bin_img"]
    horizontais = dados_grade["horizontais"]
    verticais = dados_grade["verticais"]

    xs, ys = gerar_grade_fixa(
        tabela.shape,
        num_questoes=config.num_questoes,
        alternativas=len(config.alternativas)
    )

    grade_mask = cv2.bitwise_or(horizontais, verticais)
    marcas = cv2.subtract(bin_img, grade_mask)

    deslocamento_px_x = max(1, int(round((xs[2] - xs[1]) * config.fator_deslocamento_variacao)))
    deslocamento_px_y = max(1, int(round((ys[2] - ys[1]) * config.fator_deslocamento_variacao)))

    leituras = {
        "original": ler_respostas_variacao(tabela, marcas, xs, ys, pasta_debug, nome_base_debug, "original", config, debug, 0, 0),
        "direita": ler_respostas_variacao(tabela, marcas, xs, ys, pasta_debug, nome_base_debug, "deslocada_direita", config, debug, deslocamento_px_x, 0),
        "baixo": ler_respostas_variacao(tabela, marcas, xs, ys, pasta_debug, nome_base_debug, "deslocada_baixo", config, debug, 0, deslocamento_px_y),
        "cima": ler_respostas_variacao(tabela, marcas, xs, ys, pasta_debug, nome_base_debug, "deslocada_cima", config, debug, 0, -deslocamento_px_y),
    }

    respostas_finais, detalhes_consolidados = consolidar_respostas_multiplas(
        leituras["original"]["respostas"],
        leituras["direita"]["respostas"],
        leituras["baixo"]["respostas"],
        leituras["cima"]["respostas"]
    )

    return {
        "respostas": respostas_finais,
        "detalhes": {
            "original": leituras["original"]["detalhes"],
            "direita": leituras["direita"]["detalhes"],
            "baixo": leituras["baixo"]["detalhes"],
            "cima": leituras["cima"]["detalhes"],
            "consolidado": detalhes_consolidados,
        },
        "xs": xs,
        "ys": ys
    }


def analisar_tabela_recortada(tabela, pasta_debug: str | Path, nome_base_debug: str, config: Configuracao, debug: DebugWriter):
    """Função de alto nível para analisar a tabela recortada do gabarito."""
    resultado = ler_respostas_da_tabela(tabela, pasta_debug, nome_base_debug, config, debug)

    print("\nRespostas lidas:")
    for i, r in enumerate(resultado["respostas"], start=1):
        print(f"Q{i:02d}: {r}")

    return resultado


# ============================================================
# PROCESSADOR PRINCIPAL
# ============================================================

class LeitorGabarito:
    """
    Orquestra todo o processamento do gabarito.

    Esta classe conecta conversão de PDF, alinhamento de imagens, seleção da
    tabela, OCR do nome, leitura das marcações e exportação final para CSV.
    """
    def __init__(
        self,
        config: Configuracao,
        cancel_event: Optional[threading.Event] = None,
        callback_log: LogCallback = None,
        callback_progresso: ProgressCallback = None,
    ):
        self.config = config
        self.cancel_event = cancel_event or threading.Event()
        self.callback_log = callback_log
        self.callback_progresso = callback_progresso
        self.debug = DebugWriter(config.salvar_debug)

        pytesseract.pytesseract.tesseract_cmd = config.tesseract_cmd

    def log(self, msg: str) -> None:
        """Registra mensagem no callback da interface ou no console."""
        escrever_log(msg, self.callback_log)

    def progresso(self, atual: int, total: int) -> None:
        """Atualiza a barra de progresso quando há callback configurado."""
        if self.callback_progresso:
            self.callback_progresso(atual, total)

    def preparar_selecao_primeira_pagina(self, caminho_arquivo: str | Path) -> SelecaoInicial:
        """Converte/carrega a primeira página, alinha e solicita seleção manual da tabela."""
        caminho_arquivo = Path(caminho_arquivo)
        ext = caminho_arquivo.suffix.lower()

        if ext == ".pdf":
            pasta_paginas = caminho_pasta_paginas(caminho_arquivo)
            caminhos_imagens = converter_pdf_para_jpgs(caminho_arquivo, pasta_paginas, zoom=self.config.zoom_pdf)
            if not caminhos_imagens:
                raise RuntimeError("Nenhuma página foi convertida do PDF.")
            caminho_primeira = Path(caminhos_imagens[0])
        else:
            caminho_primeira = caminho_arquivo

        img = ler_imagem_segura(caminho_primeira)
        if img is None:
            raise FileNotFoundError(str(caminho_primeira))

        pasta_debug = caminho_primeira.parent / "debug_temp"
        pasta_debug.mkdir(parents=True, exist_ok=True)

        img_alinhada, _ = alinhar_imagem(img, pasta_debug, self.debug, "primeira_pagina", self.callback_log)
        h0, w0 = img_alinhada.shape[:2]
        tamanho_padrao = (w0, h0)

        img_padronizada = encaixar_em_canvas_padrao_topo_esquerda(img_alinhada, tamanho_padrao)
        x, y, w, h, rotacionado_180 = selecionar_tabela_manual_opencv(img_padronizada, self.config)

        return SelecaoInicial(
            box=(x, y, w, h),
            rotacionado_180=rotacionado_180,
            tamanho_padrao=tamanho_padrao
        )

    def processar_arquivo(self, caminho_arquivo: str | Path, selecao_primeira_pagina: Optional[SelecaoInicial] = None):
        """Decide se o arquivo de entrada é PDF ou imagem e chama o fluxo adequado."""
        caminho_arquivo = Path(caminho_arquivo)
        if not caminho_arquivo.exists():
            raise FileNotFoundError(str(caminho_arquivo))

        ext = caminho_arquivo.suffix.lower()

        if ext == ".pdf":
            return self.processar_pdf(caminho_arquivo, selecao_primeira_pagina)

        if ext in [".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"]:
            return self.processar_imagem_unica(caminho_arquivo)

        raise ValueError("Formato não suportado. Selecione PDF ou imagem.")

    def processar_pdf(self, caminho_pdf: str | Path, selecao_primeira_pagina: Optional[SelecaoInicial] = None):
        """Processa todas as páginas de um PDF e gera o CSV final."""
        caminho_pdf = Path(caminho_pdf)
        pasta_paginas = caminho_pasta_paginas(caminho_pdf)
        pasta_paginas.mkdir(parents=True, exist_ok=True)
        pasta_debug_geral = obter_pasta_debug_pdf(caminho_pdf)

        self.log("[INFO] Convertendo PDF para imagens...")
        caminhos_imagens = converter_pdf_para_jpgs(caminho_pdf, pasta_paginas, zoom=self.config.zoom_pdf)
        self.log(f"[INFO] {len(caminhos_imagens)} páginas convertidas.")

        contexto = ContextoProcessamento()
        if selecao_primeira_pagina:
            contexto.tamanho_padrao = selecao_primeira_pagina.tamanho_padrao

        resultados = []
        total = len(caminhos_imagens)

        for idx, caminho_img in enumerate(caminhos_imagens, start=1):
            if self.cancel_event.is_set():
                self.log("⛔ Processamento cancelado pelo usuário.")
                break

            self.progresso(idx - 1, total)
            self.log(f"[INFO] Página {idx}")

            resultado = self.processar_pagina(
                caminho_imagem=caminho_img,
                pasta_debug_geral=pasta_debug_geral,
                indice_pagina=idx,
                contexto=contexto,
                selecao_primeira_pagina=selecao_primeira_pagina if idx == 1 else None
            )

            if contexto.box_relativa is None:
                contexto.box_relativa = resultado["box_relativa"]
            if contexto.box_nome_relativa is None:
                contexto.box_nome_relativa = resultado["box_nome_relativa"]
            if contexto.tamanho_padrao is None:
                contexto.tamanho_padrao = resultado["tamanho_padrao"]
            if contexto.rotacionar_180 is None:
                contexto.rotacionar_180 = resultado["rotacionar_180"]
                self.log(f"[INFO] Rotação global: {contexto.rotacionar_180}")

            resultados.append({
                "pagina": idx,
                "imagem": caminho_img,
                "box": resultado["box"],
                "aluno": resultado["nome_aluno"],
                "respostas": resultado["resultado_tabela"]["respostas"]
            })

            self.progresso(idx, total)

        caminho_csv_geral = pasta_paginas / "resultado_geral.csv"
        self.salvar_csv_resultado(caminho_csv_geral, resultados)

        self.log("[INFO] Finalizado")
        self.log(f"[INFO] CSV geral salvo em: {caminho_csv_geral}")

        return {
            "pdf": str(caminho_pdf),
            "pasta_paginas": str(pasta_paginas),
            "pasta_debug_geral": str(pasta_debug_geral),
            "box_relativa": contexto.box_relativa,
            "box_nome_relativa": contexto.box_nome_relativa,
            "tamanho_padrao": contexto.tamanho_padrao,
            "rotacionar_180": contexto.rotacionar_180,
            "resultados": resultados,
            "csv_geral": str(caminho_csv_geral)
        }

    def processar_pagina(
        self,
        caminho_imagem: str | Path,
        pasta_debug_geral: str | Path,
        indice_pagina: int,
        contexto: ContextoProcessamento,
        selecao_primeira_pagina: Optional[SelecaoInicial] = None,
    ):
        nome_base_debug = f"pagina_{indice_pagina:03d}"
        img = ler_imagem_segura(caminho_imagem)
        if img is None:
            raise FileNotFoundError(str(caminho_imagem))

        img_alinhada, angulo = alinhar_imagem(img, pasta_debug_geral, self.debug, nome_base_debug, self.callback_log)

        if contexto.rotacionar_180:
            img_alinhada = cv2.rotate(img_alinhada, cv2.ROTATE_180)

        tamanho_padrao = contexto.tamanho_padrao
        if tamanho_padrao is None:
            h0, w0 = img_alinhada.shape[:2]
            tamanho_padrao = (w0, h0)

        img_padronizada = encaixar_em_canvas_padrao_topo_esquerda(img_alinhada, tamanho_padrao)

        box, box_relativa, rotacionado_180 = self.definir_box_tabela(
            img_padronizada,
            contexto.box_relativa,
            contexto.rotacionar_180,
            selecao_primeira_pagina
        )

        x, y, w, h = box

        box_nome_relativa = self.definir_box_nome(
            img_padronizada,
            contexto.box_nome_relativa,
            pasta_debug_geral,
            nome_base_debug
        )

        nome_aluno = ""
        if box_nome_relativa is not None:
            box_nome = box_relativa_para_absoluta(box_nome_relativa, img_padronizada.shape)
            salvar_debug_box_nome(img_padronizada, box_nome, pasta_debug_geral, nome_base_debug, self.debug)
            nome_aluno = ler_nome_aluno_da_box(img_padronizada, box_nome, pasta_debug_geral, nome_base_debug)

        tabela = img_padronizada[y:y + h, x:x + w].copy()
        self.debug.salvar(Path(pasta_debug_geral) / f"{nome_base_debug}_08_recorte_tabela.jpg", tabela)

        resultado_tabela = analisar_tabela_recortada(
            tabela,
            pasta_debug_geral,
            nome_base_debug,
            self.config,
            self.debug
        )

        return {
            "box": box,
            "box_relativa": box_relativa,
            "box_nome_relativa": box_nome_relativa,
            "rotacionar_180": rotacionado_180,
            "tamanho_padrao": tamanho_padrao,
            "nome_aluno": nome_aluno,
            "resultado_tabela": resultado_tabela,
            "angulo_aplicado": angulo,
        }

    def definir_box_tabela(
        self,
        img_padronizada,
        box_relativa_atual,
        rotacionar_180_atual,
        selecao_primeira_pagina: Optional[SelecaoInicial]
    ):
        if box_relativa_atual is None:
            if selecao_primeira_pagina is not None:
                x, y, w, h = selecao_primeira_pagina.box
                rotacionado_180 = selecao_primeira_pagina.rotacionado_180

                if rotacionado_180:
                    img_padronizada[:] = cv2.rotate(img_padronizada, cv2.ROTATE_180)

                box = (x, y, w, h)
                box_relativa = box_absoluta_para_relativa(box, img_padronizada.shape)
            else:
                x, y, w, h, rotacionado_180 = selecionar_tabela_manual_opencv(img_padronizada, self.config)

                if rotacionado_180:
                    img_padronizada[:] = cv2.rotate(img_padronizada, cv2.ROTATE_180)

                box = (x, y, w, h)
                box_relativa = box_absoluta_para_relativa(box, img_padronizada.shape)
        else:
            box = box_relativa_para_absoluta(box_relativa_atual, img_padronizada.shape)
            box_relativa = box_relativa_atual
            rotacionado_180 = rotacionar_180_atual

        return box, box_relativa, rotacionado_180

    def definir_box_nome(self, img_padronizada, box_nome_relativa_atual, pasta_debug, nome_base_debug):
        """Obtém ou reaproveita a ROI relativa usada para OCR do nome do aluno."""
        if box_nome_relativa_atual is not None:
            return box_nome_relativa_atual

        box_nome = detectar_box_nome_aluno(
            img_padronizada,
            self.config,
            pasta_debug=pasta_debug,
            nome_base_debug=nome_base_debug
        )

        if box_nome is None:
            return None

        # Mantém exatamente a região detectada.
        # Não desloca para a direita, pois isso pode cortar o começo do nome.
        return box_absoluta_para_relativa(box_nome, img_padronizada.shape)

    def processar_imagem_unica(self, caminho_imagem: str | Path):
        """Processa uma única imagem, sem fluxo de múltiplas páginas."""
        if self.cancel_event.is_set():
            self.log("⛔ Processamento cancelado antes de iniciar.")
            return None

        self.progresso(0, 1)
        self.log(f"[INFO] Processando imagem: {Path(caminho_imagem).name}")

        img = ler_imagem_segura(caminho_imagem)
        if img is None:
            raise FileNotFoundError(f"Não foi possível abrir a imagem: {caminho_imagem}")

        pasta_debug = Path(caminho_imagem).parent / "debug_temp"
        pasta_debug.mkdir(parents=True, exist_ok=True)

        self.debug.salvar(pasta_debug / "00_original.jpg", img)
        img_alinhada, angulo = alinhar_imagem(img, pasta_debug, self.debug, "imagem_unica", self.callback_log)

        x, y, w, h, rotacionado_180 = selecionar_tabela_manual_opencv(img_alinhada, self.config)
        self.log(f"Tabela selecionada manualmente em: {(x, y, w, h)}")

        if rotacionado_180:
            img_alinhada = cv2.rotate(img_alinhada, cv2.ROTATE_180)

        debug_img = img_alinhada.copy()
        cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 180, 0), 2)
        self.debug.salvar(pasta_debug / "07_tabela_selecionada_manual.jpg", debug_img)

        tabela = img_alinhada[y:y + h, x:x + w].copy()
        self.debug.salvar(pasta_debug / "08_recorte_tabela_manual.jpg", tabela)

        resultado_tabela = analisar_tabela_recortada(tabela, pasta_debug, "imagem_unica", self.config, self.debug)

        caminho_csv_simples = pasta_debug / "respostas_compacto.csv"
        with open(caminho_csv_simples, "w", encoding="utf-8") as f:
            f.write(";".join(resultado_tabela["respostas"]))

        self.log(f"[INFO] CSV compacto salvo em: {caminho_csv_simples}")
        self.progresso(1, 1)

        return {
            "box": (x, y, w, h),
            "angulo_aplicado": angulo,
            "pasta_debug": str(pasta_debug),
            "resultado_tabela": resultado_tabela
        }

    def salvar_csv_resultado(self, caminho_csv: str | Path, resultados: list[dict]) -> None:
        """Exporta página, nome do aluno e respostas por questão em CSV separado por ponto e vírgula."""
        caminho_csv = Path(caminho_csv)
        caminho_csv.parent.mkdir(parents=True, exist_ok=True)

        with open(caminho_csv, "w", encoding="utf-8-sig", newline="") as f:
            separador = ";"
            cabecalho = ["pagina", "aluno"] + [f"Q{i}" for i in range(1, self.config.num_questoes + 1)]
            f.write(separador.join(cabecalho) + "\n")

            for item in resultados:
                aluno = (item.get("aluno") or "").replace(";", " ").replace("\n", " ").replace("\r", " ").strip()
                respostas = list(item.get("respostas", [])[:self.config.num_questoes])

                if len(respostas) < self.config.num_questoes:
                    respostas += [""] * (self.config.num_questoes - len(respostas))

                linha = [str(item["pagina"]), aluno] + respostas
                f.write(separador.join(linha) + "\n")


# ============================================================
# INTERFACE TKINTER
# ============================================================

def abrir_interface():
    """
    Inicializa a interface gráfica Tkinter.

    A interface permite selecionar o arquivo, definir quantidade de questões,
    alternativas, parâmetros do OCR do nome, modo debug e acompanhar progresso.
    """
    cancel_event = threading.Event()
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
                    barra_progresso["maximum"] = max(total, 1)
                    barra_progresso["value"] = atual

                elif tipo == "fim_ok":
                    btn_processar.config(state="normal")
                    btn_cancelar.config(state="disabled")
                    messagebox.showinfo("Concluído", "Processamento finalizado com sucesso.")

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

    def log(msg: str):
        fila_ui.put(("log", msg))

    def atualizar_progresso(atual: int, total: int):
        fila_ui.put(("progresso", (atual, total)))

    def criar_configuracao_da_tela() -> Configuracao:
        txt_questoes = entry_questoes.get().strip()
        txt_alternativas = entry_alternativas.get().strip()

        if not txt_questoes.isdigit():
            raise ValueError("A quantidade de questões deve ser numérica.")

        lista_alternativas = [a.strip().upper() for a in txt_alternativas.split(",") if a.strip()]
        if not lista_alternativas:
            raise ValueError("Informe as alternativas separadas por vírgula. Ex: A,B,C,D,E")

        return Configuracao(
            num_questoes=int(txt_questoes),
            alternativas=tuple(lista_alternativas),
            salvar_debug=var_debug.get(),
            palavra_nome=entry_palavra_nome.get().strip() or "Aluno",
            distancia_esquerda_nome=int(entry_dist_esq_nome.get().strip() or 90),
            distancia_superior_nome=int(entry_dist_sup_nome.get().strip() or 0),
            largura_retangulo_nome=int(entry_largura_nome.get().strip() or 600),
            altura_retangulo_nome=int(entry_altura_nome.get().strip() or 60)
        )

    def executar_processamento(caminho, config: Configuracao, selecao_primeira_pagina: Optional[SelecaoInicial]):
        try:
            leitor = LeitorGabarito(
                config=config,
                cancel_event=cancel_event,
                callback_log=log,
                callback_progresso=atualizar_progresso
            )

            resultado = leitor.processar_arquivo(caminho, selecao_primeira_pagina=selecao_primeira_pagina)

            if cancel_event.is_set():
                fila_ui.put(("fim_cancelado", None))
            else:
                fila_ui.put(("fim_ok", resultado))

        except Exception as e:
            fila_ui.put(("erro", str(e)))

    def cancelar():
        cancel_event.set()
        log("⛔ Cancelamento solicitado pelo usuário...")

    def escolher_arquivo():
        arquivo = filedialog.askopenfilename(
            title="Selecione um PDF ou imagem",
            filetypes=[
                ("PDF", "*.pdf"),
                ("Imagens", "*.png *.jpg *.jpeg *.bmp *.webp *.tif *.tiff")
            ]
        )
        if arquivo:
            entry_caminho.delete(0, tk.END)
            entry_caminho.insert(0, arquivo)

    def iniciar():
        caminho = entry_caminho.get().strip()

        if not caminho:
            messagebox.showerror("Erro", "Selecione um arquivo PDF ou imagem.")
            return

        if not os.path.exists(caminho):
            messagebox.showerror("Erro", "O caminho informado não existe.")
            return

        try:
            config = criar_configuracao_da_tela()
        except Exception as e:
            messagebox.showerror("Erro", str(e))
            return

        cancel_event.clear()
        txt_log.delete("1.0", tk.END)
        barra_progresso["value"] = 0

        log("Iniciando processamento...")
        log(f"Quantidade de questões: {config.num_questoes}")
        log(f"Alternativas: {', '.join(config.alternativas)}")
        log(f"Debug: {'ativado' if config.salvar_debug else 'desativado'}")

        btn_processar.config(state="disabled")
        btn_cancelar.config(state="normal")

        log("Preparando seleção manual da primeira página...")
        processar_fila_ui()
        janela.update_idletasks()
        janela.update()

        try:
            leitor_preparacao = LeitorGabarito(
                config=config,
                cancel_event=cancel_event,
                callback_log=log,
                callback_progresso=atualizar_progresso
            )
            selecao_primeira_pagina = leitor_preparacao.preparar_selecao_primeira_pagina(caminho)
        except Exception as e:
            btn_processar.config(state="normal")
            btn_cancelar.config(state="disabled")
            messagebox.showerror("Erro", f"Falha na seleção inicial:\n{e}")
            return

        thread = threading.Thread(
            target=executar_processamento,
            args=(caminho, config, selecao_primeira_pagina),
            daemon=True
        )
        thread.start()

    janela = tk.Tk()
    janela.title("Leitor de Gabarito")
    janela.geometry("820x620")
    janela.resizable(False, False)

    barra_progresso = ttk.Progressbar(janela, orient="horizontal", length=760, mode="determinate")
    barra_progresso.pack(padx=10, pady=(10, 5), fill="x")

    tk.Label(janela, text="Arquivo PDF ou imagem:").pack(anchor="w", padx=10, pady=(10, 2))

    frame_caminho = tk.Frame(janela)
    frame_caminho.pack(fill="x", padx=10)

    entry_caminho = tk.Entry(frame_caminho, width=85)
    entry_caminho.pack(side="left", fill="x", expand=True)

    tk.Button(frame_caminho, text="Selecionar arquivo", command=escolher_arquivo).pack(side="left", padx=5)

    frame_config = tk.Frame(janela)
    frame_config.pack(fill="x", padx=10, pady=(15, 5))

    tk.Label(frame_config, text="Quantidade de questões:").grid(row=0, column=0, sticky="w")
    entry_questoes = tk.Entry(frame_config, width=10)
    entry_questoes.grid(row=1, column=0, sticky="w", padx=(0, 20))
    entry_questoes.insert(0, "27")

    tk.Label(frame_config, text="Alternativas da prova:").grid(row=0, column=1, sticky="w")
    entry_alternativas = tk.Entry(frame_config, width=25)
    entry_alternativas.grid(row=1, column=1, sticky="w", padx=(0, 20))
    entry_alternativas.insert(0, "A,B,C,D,E")

    var_debug = tk.BooleanVar(value=False)
    chk_debug = tk.Checkbutton(frame_config, text="Gerar debug", variable=var_debug)
    chk_debug.grid(row=1, column=2, sticky="w")

    frame_nome = tk.LabelFrame(janela, text="Configuração da leitura do nome")
    frame_nome.pack(fill="x", padx=10, pady=(5, 5))

    tk.Label(frame_nome, text="Palavra a encontrar:").grid(row=0, column=0, sticky="w")
    entry_palavra_nome = tk.Entry(frame_nome, width=18)
    entry_palavra_nome.grid(row=1, column=0, sticky="w", padx=(0, 15))
    entry_palavra_nome.insert(0, "Aluno")

    tk.Label(frame_nome, text="Distância esquerda (px):").grid(row=0, column=1, sticky="w")
    entry_dist_esq_nome = tk.Entry(frame_nome, width=10)
    entry_dist_esq_nome.grid(row=1, column=1, sticky="w", padx=(0, 15))
    entry_dist_esq_nome.insert(0, "90")

    tk.Label(frame_nome, text="Distância superior (px):").grid(row=0, column=2, sticky="w")
    entry_dist_sup_nome = tk.Entry(frame_nome, width=10)
    entry_dist_sup_nome.grid(row=1, column=2, sticky="w", padx=(0, 15))
    entry_dist_sup_nome.insert(0, "0")

    tk.Label(frame_nome, text="Largura retângulo (px):").grid(row=0, column=3, sticky="w")
    entry_largura_nome = tk.Entry(frame_nome, width=10)
    entry_largura_nome.grid(row=1, column=3, sticky="w", padx=(0, 15))
    entry_largura_nome.insert(0, "600")

    tk.Label(frame_nome, text="Altura retângulo (px):").grid(row=0, column=4, sticky="w")
    entry_altura_nome = tk.Entry(frame_nome, width=10)
    entry_altura_nome.grid(row=1, column=4, sticky="w", padx=(0, 15))
    entry_altura_nome.insert(0, "60")

    frame_botoes = tk.Frame(janela)
    frame_botoes.pack(pady=15)

    btn_processar = tk.Button(frame_botoes, text="Processar", command=iniciar, height=2, width=15)
    btn_processar.pack(side="left", padx=5)

    btn_cancelar = tk.Button(frame_botoes, text="Cancelar", command=cancelar, height=2, width=15, state="disabled")
    btn_cancelar.pack(side="left", padx=5)

    tk.Label(janela, text="Saída do processamento:").pack(anchor="w", padx=10, pady=(10, 2))

    frame_log = tk.Frame(janela)
    frame_log.pack(fill="both", expand=True, padx=10, pady=(0, 10))

    scroll_log = tk.Scrollbar(frame_log)
    scroll_log.pack(side="right", fill="y")

    txt_log = tk.Text(frame_log, height=20, wrap="word", yscrollcommand=scroll_log.set)
    txt_log.pack(side="left", fill="both", expand=True)

    scroll_log.config(command=txt_log.yview)

    janela.after(100, processar_fila_ui)
    janela.mainloop()


if __name__ == "__main__":
    abrir_interface()
