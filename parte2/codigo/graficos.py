"""Funções auxiliares de visualização (mesmo padrão em tons de cinza da Atividade 1)."""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy import stats
from statsmodels.nonparametric.smoothers_lowess import lowess

SAIDA = Path(__file__).resolve().parent.parent / "figuras"
SAIDA.mkdir(exist_ok=True)

PRETO = "#000000"
CINZA = "#555555"
GRADE = "#E5E5E5"
PREENCHIMENTO = "#B8B8B8"
PREENCHIMENTO_CLARO = "#E6E6E6"
REFERENCIA = "#666666"

plt.rcParams.update({
    "font.family": ["Arial", "DejaVu Sans"],
    "font.size": 10,
    "axes.edgecolor": CINZA,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": GRADE,
    "grid.linewidth": 0.8,
    "axes.axisbelow": True,
    "xtick.color": CINZA,
    "ytick.color": CINZA,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "ytick.major.size": 0,
    "xtick.major.size": 3,
    "legend.frameon": False,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})


def painel(ax, titulo: str, rotulo_y: str = "", rotulo_x: str = "", letra: str | None = None):
    """Título em negrito, rótulo do eixo y na horizontal e letra do painel, como na Atividade 1."""
    ax.annotate(titulo, (0, 1), xycoords="axes fraction", xytext=(0, 24), textcoords="offset points",
                fontsize=11, fontweight="bold", ha="left", va="bottom")
    if rotulo_y:
        ax.annotate(rotulo_y, (0, 1), xycoords="axes fraction", xytext=(0, 6), textcoords="offset points",
                    fontsize=10, ha="left", va="bottom")
    if letra:
        ax.annotate(letra, (0, 1), xycoords="axes fraction", xytext=(-46, 24), textcoords="offset points",
                    fontsize=11, fontweight="bold", ha="left", va="bottom")
    ax.set_xlabel(rotulo_x, fontsize=10)


def salvar(fig, nome: str):
    caminho = SAIDA / f"{nome}.png"
    fig.savefig(caminho, dpi=180, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    Image.open(caminho).convert("L").save(caminho, optimize=True)


def letras(n: int) -> list[str]:
    return [f"({chr(ord('a') + i)})" for i in range(n)]


def grade(n: int, colunas: int, largura: float = 7.5, altura_painel: float = 2.6):
    linhas = math.ceil(n / colunas)
    fig, eixos = plt.subplots(linhas, colunas, figsize=(largura, altura_painel * linhas), squeeze=False)
    for ax in eixos.flat[n:]:
        ax.set_visible(False)
    return fig, eixos.flat[:n]


def histogramas(df: pd.DataFrame, colunas: dict[str, str], nome: str, colunas_grade: int = 3):
    fig, eixos = grade(len(colunas), colunas_grade)
    for ax, (coluna, rotulo), letra in zip(eixos, colunas.items(), letras(len(colunas))):
        valores = df[coluna].to_numpy(dtype=float)
        if np.unique(valores).size <= 20:
            contagens = pd.Series(valores).value_counts().sort_index()
            ax.bar(contagens.index, contagens.values, width=0.8, color=PREENCHIMENTO, edgecolor=CINZA, linewidth=0.6)
        else:
            ax.hist(valores, bins=40, color=PREENCHIMENTO, edgecolor=CINZA, linewidth=0.5)
        assimetria = f"{stats.skew(valores):.2f}".replace(".", ",")
        ax.text(0.98, 0.95, f"assimetria = {assimetria}", transform=ax.transAxes,
                ha="right", va="top", fontsize=8.5, color=CINZA)
        painel(ax, rotulo, "Frequência", "", letra)
    fig.tight_layout(h_pad=3.2, w_pad=2.2)
    salvar(fig, nome)


def barras(df: pd.DataFrame, colunas: dict[str, str], nome: str):
    fig, eixos = grade(len(colunas), 2, altura_painel=2.5)
    for ax, (coluna, rotulo), letra in zip(eixos, colunas.items(), letras(len(colunas))):
        contagens = df[coluna].value_counts().sort_values()
        ax.barh(contagens.index.astype(str), contagens.values, color=PREENCHIMENTO, edgecolor=CINZA, linewidth=0.6)
        for i, v in enumerate(contagens.values):
            ax.text(v, i, f"  {v} ({100 * v / len(df):.1f}%)".replace(".", ","), va="center", fontsize=8.5, color=CINZA)
        ax.set_xlim(0, contagens.max() * 1.45)
        ax.grid(axis="x", color=GRADE)
        ax.grid(axis="y", visible=False)
        painel(ax, rotulo, "", "Número de imóveis", letra)
    fig.tight_layout(h_pad=3.0, w_pad=2.5)
    salvar(fig, nome)


def matriz_dispersao(df: pd.DataFrame, colunas: dict[str, str], nome: str, amostra: int = 3000, semente: int = 1):
    """Matriz adaptada: dispersões abaixo da diagonal, histogramas na diagonal e r de Pearson acima."""
    nomes = list(colunas)
    k = len(nomes)
    pontos = df.sample(min(amostra, len(df)), random_state=semente)
    correlacoes = df[nomes].corr()
    fig, eixos = plt.subplots(k, k, figsize=(7.6, 7.6))
    for i, linha in enumerate(nomes):
        for j, coluna in enumerate(nomes):
            ax = eixos[i, j]
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])
            if i == j:
                ax.hist(df[linha], bins=25, color=PREENCHIMENTO, edgecolor=CINZA, linewidth=0.4)
                ax.spines["left"].set_visible(False)
            elif i > j:
                jitter = np.random.default_rng(semente).uniform(-0.15, 0.15, size=(len(pontos), 2))
                x = pontos[coluna] + (jitter[:, 0] if df[coluna].nunique() <= 20 else 0)
                y = pontos[linha] + (jitter[:, 1] if df[linha].nunique() <= 20 else 0)
                ax.scatter(x, y, s=1.2, color=PRETO, alpha=0.18, linewidths=0, rasterized=True)
            else:
                r = correlacoes.loc[linha, coluna]
                ax.text(0.5, 0.5, f"{r:.2f}".replace(".", ","), transform=ax.transAxes, ha="center", va="center",
                        fontsize=8 + 7 * abs(r), fontweight="bold" if abs(r) >= 0.5 else "normal",
                        color=PRETO if abs(r) >= 0.3 else CINZA)
                for lado in ("left", "bottom"):
                    ax.spines[lado].set_visible(False)
            if j == 0:
                ax.set_ylabel(colunas[linha], fontsize=8.5, rotation=0, ha="right", va="center", labelpad=4)
            if i == k - 1:
                ax.set_xlabel(colunas[coluna], fontsize=8.5, rotation=30, ha="right")
    fig.subplots_adjust(wspace=0.08, hspace=0.08, left=0.15, bottom=0.13, right=0.99, top=0.99)
    salvar(fig, nome)


def boxplots(df: pd.DataFrame, resposta: str, rotulo_resposta: str, categorias: dict[str, str], nome: str):
    fig, eixos = grade(len(categorias), 2, altura_painel=2.9)
    for ax, (coluna, rotulo), letra in zip(eixos, categorias.items(), letras(len(categorias))):
        niveis = df.groupby(coluna, observed=True)[resposta].median().sort_values().index
        dados = [df.loc[df[coluna] == n, resposta] for n in niveis]
        ax.boxplot(dados, vert=False, widths=0.6, patch_artist=True, showfliers=True,
                   boxprops=dict(facecolor=PREENCHIMENTO, edgecolor=CINZA),
                   medianprops=dict(color=PRETO, linewidth=1.6), whiskerprops=dict(color=CINZA),
                   capprops=dict(color=CINZA),
                   flierprops=dict(marker="o", markersize=1.8, markerfacecolor=PRETO, markeredgecolor="none", alpha=0.35))
        ax.set_yticks(range(1, len(niveis) + 1), [str(n) for n in niveis])
        ax.grid(axis="x", color=GRADE)
        ax.grid(axis="y", visible=False)
        painel(ax, rotulo, "", rotulo_resposta, letra)
    fig.tight_layout(h_pad=3.0, w_pad=2.5)
    salvar(fig, nome)


def diagnostico(modelo, nome: str, maximo_pontos: int | None = None):
    """Histograma com densidade normal, resíduos versus preditos e Q-Q (três painéis)."""
    residuos = np.asarray(modelo.resid)
    preditos = np.asarray(modelo.fittedvalues)
    sigma = np.sqrt(modelo.scale)
    fig = plt.figure(figsize=(7.5, 6.4))
    ax1 = fig.add_axes([0.08, 0.57, 0.39, 0.33])
    ax2 = fig.add_axes([0.58, 0.57, 0.39, 0.33])
    ax3 = fig.add_axes([0.33, 0.07, 0.39, 0.33])

    contagens, bordas, _ = ax1.hist(residuos, bins=45, color=PREENCHIMENTO, edgecolor=CINZA, linewidth=0.4)
    xs = np.linspace(bordas[0], bordas[-1], 300)
    ax1.plot(xs, stats.norm.pdf(xs, 0, sigma) * len(residuos) * (bordas[1] - bordas[0]), color=PRETO, lw=1.8,
             label="Densidade normal")
    ax1.legend(loc="upper right", fontsize=8.5)
    painel(ax1, "Histograma e densidade normal", "Frequência", "Resíduos", "(a)")

    ax2.scatter(preditos, residuos, s=2, color=PRETO, alpha=0.25, linewidths=0, rasterized=True)
    ax2.axhline(0, color=REFERENCIA, lw=1.2, ls="--")
    suave = lowess(residuos, preditos, frac=0.2, it=0, return_sorted=True)
    ax2.plot(suave[:, 0], suave[:, 1], color=PRETO, lw=1.6, label="LOWESS")
    ax2.legend(loc="upper right", fontsize=8.5)
    painel(ax2, "Resíduos versus valores preditos", "Resíduos", "Valores preditos", "(b)")

    ordenados = np.sort(residuos)
    n = len(ordenados)
    teoricos = stats.norm.ppf((np.arange(1, n + 1) - 0.375) / (n + 0.25))
    q_x, q_y = np.quantile(teoricos, [0.25, 0.75]), np.quantile(ordenados, [0.25, 0.75])
    inclinacao = (q_y[1] - q_y[0]) / (q_x[1] - q_x[0])
    intercepto = q_y[0] - inclinacao * q_x[0]
    ax3.scatter(teoricos, ordenados, s=2.5, color=PRETO, linewidths=0, rasterized=True)
    limites = np.array([teoricos.min(), teoricos.max()])
    ax3.plot(limites, intercepto + inclinacao * limites, color=REFERENCIA, lw=1.1)
    painel(ax3, "Gráfico Q-Q dos resíduos", "Quantis dos resíduos", "Quantis teóricos N(0,1)", "(c)")
    salvar(fig, nome)


ABREVIACOES = {"São Paulo": "SP", "Rio de Janeiro": "RJ", "Belo Horizonte": "BH", "Porto Alegre": "POA",
               "Campinas": "CPS", "Não mobiliado": "Não mob.", "Apartamento": "Apto."}


def residuos_preditores(modelo, df: pd.DataFrame, numericas: dict[str, str], categoricas: dict[str, str], nome: str):
    """Resíduos contra cada preditora, com LOWESS para as numéricas e boxplots para as categóricas."""
    residuos = np.asarray(modelo.resid)
    total = len(numericas) + len(categoricas)
    fig, eixos = grade(total, 3, altura_painel=2.45)
    rotulos = letras(total)
    for ax, (coluna, rotulo), letra in zip(eixos, numericas.items(), rotulos):
        x = df[coluna].to_numpy(dtype=float)
        desloc = np.random.default_rng(3).uniform(-0.18, 0.18, len(x)) if np.unique(x).size <= 30 else 0
        ax.scatter(x + desloc, residuos, s=1.2, color=PRETO, alpha=0.15, linewidths=0, rasterized=True)
        ax.axhline(0, color=REFERENCIA, lw=1.0, ls="--")
        suave = lowess(residuos, x, frac=0.3, it=0, return_sorted=True)
        ax.plot(suave[:, 0], suave[:, 1], color=PRETO, lw=1.6)
        painel(ax, rotulo, "Resíduos", "", letra)
    for ax, (coluna, rotulo), letra in zip(eixos[len(numericas):], categoricas.items(), rotulos[len(numericas):]):
        niveis = list(df[coluna].cat.categories) if hasattr(df[coluna], "cat") else sorted(df[coluna].unique())
        dados = [residuos[(df[coluna] == n).to_numpy()] for n in niveis]
        ax.boxplot(dados, widths=0.6, patch_artist=True,
                   boxprops=dict(facecolor=PREENCHIMENTO, edgecolor=CINZA),
                   medianprops=dict(color=PRETO, linewidth=1.4), whiskerprops=dict(color=CINZA),
                   capprops=dict(color=CINZA),
                   flierprops=dict(marker="o", markersize=1.5, markerfacecolor=PRETO, markeredgecolor="none", alpha=0.3))
        ax.set_xticks(range(1, len(niveis) + 1), [ABREVIACOES.get(str(n), str(n)) for n in niveis], fontsize=8.5)
        ax.axhline(0, color=REFERENCIA, lw=1.0, ls="--")
        painel(ax, rotulo, "Resíduos", "", letra)
    fig.tight_layout(h_pad=3.0, w_pad=1.8)
    salvar(fig, nome)


def influencia(modelo, nome: str):
    infl = modelo.get_influence()
    cook = infl.cooks_distance[0]
    alavancagem = infl.hat_matrix_diag
    estudentizados = infl.resid_studentized_external
    n, p = len(cook), int(modelo.df_model) + 1
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 3.0))
    ax1.vlines(np.arange(n), 0, cook, color=PRETO, lw=0.5)
    ax1.axhline(4 / n, color=REFERENCIA, lw=1.0, ls="--", label="4/n")
    ax1.legend(loc="upper right", fontsize=8.5)
    painel(ax1, "Distância de Cook", "D de Cook", "Índice do registro", "(a)")
    ax2.scatter(alavancagem, estudentizados, s=2, color=PRETO, alpha=0.3, linewidths=0, rasterized=True)
    ax2.axvline(2 * p / n, color=REFERENCIA, lw=1.0, ls="--", label="2p/n")
    for limite in (-3, 3):
        ax2.axhline(limite, color=REFERENCIA, lw=1.0, ls=":")
    ax2.legend(loc="upper right", fontsize=8.5)
    painel(ax2, "Alavancagem e resíduos", "Resíduo estudentizado", "Alavancagem h", "(b)")
    fig.tight_layout(w_pad=3.0)
    salvar(fig, nome)


def comparacao_escalas(modelo_original, modelo_final, nome: str):
    fig, eixos = plt.subplots(1, 2, figsize=(7.5, 3.1))
    for ax, modelo, titulo, letra in zip(eixos, (modelo_original, modelo_final),
                                         ("Aluguel em R$ (sem transformações)", "log10(Aluguel) (modelo final)"),
                                         ("(a)", "(b)")):
        ax.scatter(modelo.fittedvalues, modelo.resid, s=2, color=PRETO, alpha=0.25, linewidths=0, rasterized=True)
        ax.axhline(0, color=REFERENCIA, lw=1.0, ls="--")
        painel(ax, titulo, "Resíduos", "Valores preditos", letra)
    eixos[0].ticklabel_format(style="plain")
    fig.tight_layout(w_pad=3.0)
    salvar(fig, nome)
