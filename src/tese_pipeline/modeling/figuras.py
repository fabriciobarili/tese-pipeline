"""Figuras de diagnóstico do treinamento (para embasar a pesquisa).

Exporta, a partir de um classificador treinado e de suas métricas/SHAP:
  1. matriz de confusão (normalizada por linha, com contagens);
  2. peso de cada variável — importância SHAP global e por classe;
  3. gráficos de apoio — métricas por classe, distribuição das classes, curvas ROC e PR (OvR).

Paleta categórica Okabe–Ito (à prova de daltonismo); matriz de confusão em matiz sequencial
único (azul claro→escuro). Figuras estáticas (matplotlib) prontas para o artigo/.docx.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import auc, precision_recall_curve, roc_curve  # noqa: E402
from sklearn.preprocessing import label_binarize  # noqa: E402

LOG = logging.getLogger(__name__)

# Okabe–Ito — categórica segura para CVD (ordem fixa, nunca ciclada).
OKABE_ITO = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#F0E442"]
INK = "#222222"
GRID = "#CCCCCC"


def _cor_classe(i: int) -> str:
    return OKABE_ITO[i % len(OKABE_ITO)]


def matriz_confusao(cm, classes, out_png: Path) -> None:
    """Heatmap da matriz de confusão: cor = recall por linha (normalizado), texto = contagem + %."""
    cm = np.asarray(cm, dtype=float)
    linhas = cm.sum(axis=1, keepdims=True)
    cmn = np.divide(cm, linhas, out=np.zeros_like(cm), where=linhas > 0)
    n = len(classes)
    fig, ax = plt.subplots(figsize=(1.6 + 1.1 * n, 1.4 + 1.0 * n))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n), classes, rotation=20, ha="right")
    ax.set_yticks(range(n), classes)
    ax.set_xlabel("Classe predita", color=INK)
    ax.set_ylabel("Classe real", color=INK)
    ax.set_title("Matriz de confusão (normalizada por linha)", color=INK)
    for i in range(n):
        for j in range(n):
            cor = "white" if cmn[i, j] > 0.55 else INK
            ax.text(j, i, f"{int(cm[i, j]):,}\n{cmn[i, j]:.1%}".replace(",", "."),
                    ha="center", va="center", color=cor, fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="proporção da linha (recall)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def pesos_variaveis(ranking_global: dict, out_png: Path, top: int = 15) -> None:
    """Barras horizontais do peso (|SHAP| médio) de cada variável — ranking global."""
    itens = sorted(ranking_global.items(), key=lambda kv: kv[1], reverse=True)[:top]
    feats = [k for k, _ in itens][::-1]
    vals = [v for _, v in itens][::-1]
    fig, ax = plt.subplots(figsize=(7.2, 0.42 * len(feats) + 1.0))
    ax.barh(range(len(feats)), vals, color=OKABE_ITO[0], height=0.68)
    ax.set_yticks(range(len(feats)), feats)
    ax.set_xlabel("peso da variável — |SHAP| médio", color=INK)
    ax.set_title("Peso de cada variável (importância global)", color=INK)
    for i, v in enumerate(vals):
        ax.text(v, i, f" {v:.3f}", va="center", ha="left", fontsize=8, color=INK)
    ax.margins(x=0.12)
    _limpa(ax)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def pesos_por_classe(per_class: dict, classes, out_png: Path, top: int = 8) -> None:
    """Pequenos múltiplos: peso (|SHAP|) das variáveis por classe (uma faceta por classe)."""
    presentes = [c for c in classes if c in per_class]
    if not presentes:
        return
    fig, axes = plt.subplots(1, len(presentes), figsize=(4.4 * len(presentes), 0.4 * top + 1.2))
    if len(presentes) == 1:
        axes = [axes]
    for ax, c in zip(axes, presentes, strict=False):
        itens = sorted(per_class[c].items(), key=lambda kv: kv[1], reverse=True)[:top]
        feats = [k for k, _ in itens][::-1]
        vals = [v for _, v in itens][::-1]
        ax.barh(range(len(feats)), vals, color=_cor_classe(list(classes).index(c)), height=0.68)
        ax.set_yticks(range(len(feats)), feats, fontsize=8)
        ax.set_title(c, color=INK)
        ax.set_xlabel("|SHAP| médio", color=INK, fontsize=9)
        _limpa(ax)
    fig.suptitle("Peso das variáveis por classe de evento", color=INK)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def metricas_por_classe(per_class: dict, classes, out_png: Path) -> None:
    """Barras agrupadas: precisão, recall e F1 por classe."""
    metricas = [("precision", "precisão"), ("recall", "recall"), ("f1-score", "F1")]
    presentes = [c for c in classes if c in per_class]
    x = np.arange(len(presentes))
    w = 0.26
    fig, ax = plt.subplots(figsize=(1.6 + 1.6 * len(presentes), 4))
    for j, (chave, rotulo) in enumerate(metricas):
        vals = [per_class[c].get(chave, 0.0) for c in presentes]
        ax.bar(x + (j - 1) * w, vals, w, label=rotulo, color=OKABE_ITO[j])
        for xi, v in zip(x + (j - 1) * w, vals, strict=False):
            ax.text(xi, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=7, color=INK)
    ax.set_xticks(x, presentes, rotation=15, ha="right")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("valor", color=INK)
    ax.set_title("Desempenho por classe", color=INK)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    _limpa(ax)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def distribuicao_classes(dist_pct: dict, out_png: Path) -> None:
    """Barras da distribuição natural das classes (destaca o desbalanceamento)."""
    itens = sorted(dist_pct.items(), key=lambda kv: kv[1], reverse=True)
    labels = [k for k, _ in itens]
    vals = [v for _, v in itens]
    fig, ax = plt.subplots(figsize=(1.6 + 1.4 * len(labels), 3.8))
    cores = [_cor_classe(i) for i in range(len(labels))]
    ax.bar(labels, vals, color=cores, width=0.6)
    for i, v in enumerate(vals):
        ax.text(i, v + max(vals) * 0.01, f"{v:.1f}%", ha="center", va="bottom", fontsize=9,
                color=INK)
    ax.set_ylabel("% das corridas", color=INK)
    ax.set_title("Distribuição natural das classes (desbalanceamento)", color=INK)
    _limpa(ax)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def curvas_roc_pr(y_true, proba, classes, out_png: Path) -> None:
    """ROC e Precision-Recall One-vs-Rest, uma curva por classe."""
    yb = label_binarize(y_true, classes=list(classes))
    if yb.shape[1] == 1:  # binário degenerado
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
    for k, c in enumerate(classes):
        cor = _cor_classe(k)
        fpr, tpr, _ = roc_curve(yb[:, k], proba[:, k])
        ax1.plot(fpr, tpr, color=cor, lw=2, label=f"{c} (AUC {auc(fpr, tpr):.3f})")
        prec, rec, _ = precision_recall_curve(yb[:, k], proba[:, k])
        ax2.plot(rec, prec, color=cor, lw=2,
                 label=f"{c} (AP {auc(rec, prec):.3f})")
    ax1.plot([0, 1], [0, 1], ls="--", color=GRID, lw=1)
    ax1.set_xlabel("falso-positivo", color=INK)
    ax1.set_ylabel("verdadeiro-positivo", color=INK)
    ax1.set_title("ROC (OvR)", color=INK)
    ax1.legend(frameon=False, fontsize=8, loc="lower right")
    ax2.set_xlabel("recall", color=INK)
    ax2.set_ylabel("precisão", color=INK)
    ax2.set_title("Precision–Recall (OvR)", color=INK)
    ax2.legend(frameon=False, fontsize=8, loc="lower left")
    for ax in (ax1, ax2):
        _limpa(ax)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _limpa(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color=GRID, lw=0.6, alpha=0.5)
    ax.set_axisbelow(True)


def exportar_classificacao(y_true, proba, classes, metrics: dict, shap_imp: dict,
                           out_dir: Path) -> list[str]:
    """Gera todas as figuras de diagnóstico. Cada peça falha isolada (não derruba o run)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gerados: list[str] = []
    tarefas = [
        ("matriz_confusao.png",
         lambda p: matriz_confusao(metrics["confusion_matrix"]["matrix"], classes, p)),
        ("pesos_variaveis_global.png",
         lambda p: pesos_variaveis(shap_imp["global"], p)),
        ("pesos_variaveis_por_classe.png",
         lambda p: pesos_por_classe(shap_imp.get("per_class", {}), classes, p)),
        ("metricas_por_classe.png",
         lambda p: metricas_por_classe(metrics["per_class"], classes, p)),
        ("distribuicao_classes.png",
         lambda p: distribuicao_classes(metrics["target_distribution_pct"], p)),
        ("curvas_roc_pr.png",
         lambda p: curvas_roc_pr(y_true, proba, classes, p)),
    ]
    for nome, fn in tarefas:
        try:
            fn(out_dir / nome)
            gerados.append(nome)
        except Exception as e:  # noqa: BLE001
            LOG.warning("figura %s falhou: %s", nome, e)
    LOG.info("figuras exportadas em %s: %s", out_dir, gerados)
    return gerados
