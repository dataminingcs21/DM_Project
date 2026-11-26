"""Step 18: build the full written report as a PDF (French), from the results the other scripts wrote.

Usage (from the project root, with the virtual environment active):

    python scripts/18_build_report_pdf.py

Every table is read from the reports of steps 01-17 (JSON / CSV), never retyped, so the PDF follows the data.
A few extra figures (French, matplotlib) are drawn into reports/report/figures/; the other figures are the ones
the pipeline already produced.  The demo capture is analysed live with the saved models for chapter 12.

Output: reports/report/rapport_data_mining.pdf

Exit codes: 0 done; 1 a needed report is missing (run the step it names).
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle  # noqa: E402
from reportlab.lib.units import cm  # noqa: E402
from reportlab.pdfbase import pdfmetrics  # noqa: E402
from reportlab.pdfbase.ttfonts import TTFont  # noqa: E402
from reportlab.platypus import (  # noqa: E402
    BaseDocTemplate,
    CondPageBreak,
    Frame,
    Image,
    KeepTogether,
    ListFlowable,
    ListItem,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.tableofcontents import TableOfContents  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REP = ROOT / "reports"
OUT_DIR = REP / "report"
FIG = OUT_DIR / "figures"
PDF = OUT_DIR / "rapport_data_mining.pdf"

ACCENT = "#1f6fd1"
GREY = "#b8b8ae"
INK = "#1d2230"
MUTED = "#5b6474"


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def need(path: Path, step: str) -> Path:
    if not path.is_file():
        print(f"ERROR: {path} is missing; run {step}")
        sys.exit(1)
    return path


def rj(rel: str, step: str):
    return json.loads(need(ROOT / rel, step).read_text(encoding="utf-8"))


def rc(rel: str, step: str) -> pd.DataFrame:
    return pd.read_csv(need(ROOT / rel, step))


def fr(x: float, d: int = 3) -> str:
    """French decimal comma."""
    return f"{x:.{d}f}".replace(".", ",")


def pct(x: float, d: int = 1) -> str:
    return f"{x * 100:.{d}f} %".replace(".", ",")


def n(x: int) -> str:
    return f"{int(x):,}".replace(",", "\u202f")


def esc(text: str) -> str:
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# ---------------------------------------------------------------------------
# Fonts and styles
# ---------------------------------------------------------------------------
def register_fonts() -> str:
    fonts = Path("C:/Windows/Fonts")
    files = {"Arial": "arial.ttf", "Arial-Bold": "arialbd.ttf", "Arial-Italic": "ariali.ttf", "Arial-BoldItalic": "arialbi.ttf"}
    if all((fonts / f).is_file() for f in files.values()):
        for name, f in files.items():
            pdfmetrics.registerFont(TTFont(name, str(fonts / f)))
        pdfmetrics.registerFontFamily("Arial", normal="Arial", bold="Arial-Bold", italic="Arial-Italic", boldItalic="Arial-BoldItalic")
        return "Arial"
    return "Helvetica"  # fallback: accents fine, a few symbols may be missing


FONT = register_fonts()
BOLD = f"{FONT}-Bold" if FONT == "Arial" else "Helvetica-Bold"
S = {
    "body": ParagraphStyle("body", fontName=FONT, fontSize=10, leading=14.2, alignment=TA_JUSTIFY, spaceAfter=6, textColor=colors.HexColor(INK), allowWidows=0, allowOrphans=0),
    "h1": ParagraphStyle("h1", fontName=BOLD, fontSize=17, leading=21, spaceBefore=4, spaceAfter=10, textColor=colors.HexColor(ACCENT), keepWithNext=1),
    "h2": ParagraphStyle("h2", fontName=BOLD, fontSize=12.5, leading=16, spaceBefore=10, spaceAfter=5, textColor=colors.HexColor(INK), keepWithNext=1),
    "h3": ParagraphStyle("h3", fontName=BOLD, fontSize=10.5, leading=14, spaceBefore=6, spaceAfter=3, textColor=colors.HexColor(INK)),
    "cap": ParagraphStyle("cap", fontName=FONT, fontSize=8.5, leading=11, alignment=TA_CENTER, textColor=colors.HexColor(MUTED), spaceBefore=3, spaceAfter=10),
    "cell": ParagraphStyle("cell", fontName=FONT, fontSize=8.6, leading=10.8, textColor=colors.HexColor(INK)),
    "cellb": ParagraphStyle("cellb", fontName=BOLD, fontSize=8.6, leading=10.8, textColor=colors.white),
    "key": ParagraphStyle("key", fontName=FONT, fontSize=10, leading=14, leftIndent=10, borderPadding=8, backColor=colors.HexColor("#eef4fc"),
                          borderColor=colors.HexColor("#c9dcf5"), borderWidth=0.6, spaceBefore=6, spaceAfter=10, textColor=colors.HexColor(INK)),
    "code": ParagraphStyle("code", fontName="Courier", fontSize=8.4, leading=11, leftIndent=8, backColor=colors.HexColor("#f4f5f7"), borderPadding=5, spaceAfter=8),
    "toc1": ParagraphStyle("toc1", fontName=FONT, fontSize=10.5, leading=16, leftIndent=10),
    "toc2": ParagraphStyle("toc2", fontName=FONT, fontSize=9, leading=12.5, leftIndent=26, textColor=colors.HexColor(MUTED)),
}


def P(text: str, style: str = "body") -> Paragraph:
    return Paragraph(text, S[style])


def H1(text: str) -> Paragraph:
    p = Paragraph(text, S["h1"])
    p._toc = (0, text)
    return p


def H2(text: str) -> Paragraph:
    p = Paragraph(text, S["h2"])
    p._toc = (1, text)
    return p


def bullets(items: list[str], numbered: bool = False) -> ListFlowable:
    return ListFlowable(
        [ListItem(P(t), leftIndent=14) for t in items],
        bulletType="1" if numbered else "bullet", start="1" if numbered else None,
        bulletFontName=FONT, bulletFontSize=9, leftIndent=14, bulletColor=colors.HexColor(ACCENT),
    )


def key(text: str) -> KeepTogether:
    return KeepTogether([P(text, "key")])


def figure(path: Path, caption: str, width_cm: float = 16.0) -> KeepTogether:
    path = Path(path)
    if not path.is_file():
        return KeepTogether([P(f"<i>Figure manquante : {esc(path.name)}</i>", "cap")])
    from PIL import Image as PILImage

    with PILImage.open(path) as im:
        w, h = im.size
    width = width_cm * cm
    height = width * h / w
    if height > 21 * cm:  # keep it on one page
        height = 21 * cm
        width = height * w / h
    return KeepTogether([Image(str(path), width=width, height=height), P(caption, "cap")])


def table(rows: list[list], widths: list[float], num_cols: tuple[int, ...] = (), bold_rows: tuple[int, ...] = ()) -> Table:
    """First row = header. Cells are wrapped in Paragraphs; numeric columns right-aligned."""
    data = []
    for i, row in enumerate(rows):
        cells = []
        for j, v in enumerate(row):
            style = "cellb" if i == 0 else "cell"
            text = esc(v) if not isinstance(v, Paragraph) else v
            if isinstance(text, str):
                if i in bold_rows and i:
                    text = f"<b>{text}</b>"
                ps = ParagraphStyle(f"c{i}{j}", parent=S[style], alignment=2 if (j in num_cols and i) else 0)
                cells.append(Paragraph(text, ps))
            else:
                cells.append(text)
        data.append(cells)
    t = Table(data, colWidths=[w * cm for w in widths], repeatRows=1)
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(ACCENT)),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f3f6fa")]),
                ("LINEBELOW", (0, 0), (-1, -1), 0.3, colors.HexColor("#d6dde8")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 3.5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return t


def tblock(t: Table, caption: str) -> KeepTogether:
    return KeepTogether([t, P(caption, "cap")])


# ---------------------------------------------------------------------------
# Figures drawn for the report (French)
# ---------------------------------------------------------------------------
def style_axes(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color("#999")
    ax.spines["bottom"].set_color("#999")
    ax.tick_params(colors="#333", labelsize=9)


def fig_pipeline() -> Path:
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.set_xlim(0, 110)
    ax.set_ylim(0, 46)
    ax.axis("off")

    def box(x, y, w, h, title, sub, main=False):
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor="#e3eefc" if main else "white", edgecolor=ACCENT if main else "#999", lw=2 if main else 1.2, zorder=2))
        ax.text(x + w / 2, y + h * 0.62, title, ha="center", va="center", fontsize=10.5, weight="bold", color=INK, zorder=3)
        ax.text(x + w / 2, y + h * 0.28, sub, ha="center", va="center", fontsize=8.5, color=MUTED, zorder=3)

    top = [("Données brutes", "8 CSV · 2,83 M flux", 2), ("Exploration", "statistiques, figures", 24), ("Nettoyage", "2,52 M flux · 66 var.", 46), ("Prétraitement", "80/20 · 40 variables", 68), ("Application", "Streamlit · .pcap", 90)]
    for i, (t, s, x) in enumerate(top):
        box(x, 32, 18, 10, t, s, main=(i == 3))
        if i:
            ax.annotate("", xy=(x, 37), xytext=(x - 4, 37), arrowprops=dict(arrowstyle="->", color="#777"))
    bottom = [("Classification", "11 modèles · vote", 2), ("Clustering", "k-means · Ward · DBSCAN", 24), ("Règles", "FP-Growth", 46), ("Anomalies", "IF · LOF · autoencodeur", 68), ("Évaluation critique", "5 expériences", 90)]
    for t, s, x in bottom:
        box(x, 6, 18, 10, t, s)
    ax.plot([77, 77], [32, 24], color="#777", lw=1.1)
    ax.plot([11, 99], [24, 24], color="#777", lw=1.1)
    for _, _, x in bottom:
        ax.annotate("", xy=(x + 9, 16.2), xytext=(x + 9, 24), arrowprops=dict(arrowstyle="->", color="#777"))
    ax.text(79, 26.2, "appris sur les lignes d'entraînement seulement", fontsize=8.5, color=MUTED)
    ax.text(2, 1.5, "Le jeu de test (504 473 flux) ne sert qu'à mesurer.", fontsize=8.5, color=MUTED)
    path = FIG / "pipeline.png"
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_f1_models(models: list[dict], final_vote: float) -> Path:
    rows = []
    for m in models:
        use = m["final_set_uses"]
        f1 = m["tuned_macro_f1"] if use == "tuned" else m["baseline_macro_f1"]
        rows.append((f"{m['model']} ({'réglé' if use == 'tuned' else 'base'})", f1, m["key"] in ("random_forest", "xgboost", "decision_tree")))
    rows.sort(key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(9, 5.2))
    y = np.arange(len(rows))
    ax.barh(y, [r[1] for r in rows], color=[ACCENT if r[2] else GREY for r in rows], height=0.62)
    ax.set_yticks(y, [r[0] for r in rows])
    for i, r in enumerate(rows):
        ax.text(r[1] + 0.008, i, fr(r[1]), va="center", fontsize=8.5, color=INK, weight="bold" if r[2] else "normal")
    ax.axvline(final_vote, color="#555", ls="--", lw=1)
    ax.text(final_vote - 0.01, -1.15, f"vote final {fr(final_vote)}", ha="right", fontsize=8.5, color="#333")
    ax.set_ylim(-1.5, len(rows) - 0.4)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("F1 macro sur le jeu de test")
    ax.set_title("Les modèles à base d'arbres dominent ; le vote reste derrière les deux meilleurs", fontsize=11, loc="left")
    style_axes(ax)
    path = FIG / "f1_modeles.png"
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_anomaly(per_class: pd.DataFrame) -> Path:
    d = per_class[(per_class["class"] != "BENIGN") & (per_class["test_rows"] >= 100)].sort_values("Autoencoder")
    fig, ax = plt.subplots(figsize=(9, 5.2))
    y = np.arange(len(d))
    ax.hlines(y, 0, 1, color="#eee", lw=0.8, zorder=0)
    ax.scatter(d["Isolation Forest"], y, marker="s", s=38, color=GREY, label="Isolation Forest", zorder=2)
    ax.scatter(d["Local Outlier Factor"], y, s=46, facecolors="none", edgecolors="#888", linewidths=1.6, label="LOF", zorder=3)
    ax.scatter(d["Autoencoder"], y, s=60, color=ACCENT, label="Autoencodeur", zorder=4)
    ax.set_yticks(y, d["class"])
    for i, v in enumerate(d["Autoencoder"]):
        ax.text(1.03, i, pct(v, 0), va="center", fontsize=8.5, weight="bold")
    benign = per_class.loc[per_class["class"] == "BENIGN", "Autoencoder"].iloc[0]
    ax.axvline(benign, color="#555", ls="--", lw=1)
    ax.set_xlim(-0.02, 1.0)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel(f"Part des flux signalés (seuil : 1 % de fausses alertes ; trafic normal signalé : {pct(benign)})")
    ax.set_title("Appris sur le seul trafic normal, l'autoencodeur repère surtout les DoS", fontsize=11, loc="left")
    ax.legend(loc="lower right", fontsize=8.5, frameon=False)
    style_axes(ax)
    path = FIG / "anomalies_par_classe.png"
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_importance() -> Path:
    rf = pd.read_csv(REP / "mining/explain/importance_permutation_random_forest.csv").set_index("feature")["f1_drop_mean"]
    xg = pd.read_csv(REP / "mining/explain/importance_permutation_xgboost.csv").set_index("feature")["f1_drop_mean"]
    top = (rf + xg).sort_values(ascending=False).head(10).index[::-1]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    y = np.arange(len(top))
    col = [ACCENT if f == "Destination Port" else GREY for f in top]
    ax.barh(y + 0.2, rf[top], height=0.38, color=col, alpha=0.55, label="Random Forest")
    ax.barh(y - 0.2, xg[top], height=0.38, color=col, label="XGBoost")
    ax.set_yticks(y, top)
    for i, f in enumerate(top):
        ax.text(rf[f] + 0.002, i + 0.2, fr(rf[f]), va="center", fontsize=7.5, color="#444")
        ax.text(xg[f] + 0.002, i - 0.2, fr(xg[f]), va="center", fontsize=7.5, color="#222")
    ax.set_xlabel("Baisse du F1 macro quand la variable est mélangée (50 000 flux de test, 5 répétitions)")
    ax.set_title("Le port de destination est la variable la plus importante des deux modèles", fontsize=11, loc="left")
    ax.legend(loc="lower right", fontsize=8.5, frameon=False, title="barre claire : RF · foncée : XGB", title_fontsize=8)
    style_axes(ax)
    path = FIG / "importance.png"
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_day_split(per_class: pd.DataFrame, pooled: float) -> Path:
    d = per_class[per_class["class"] != "Heartbleed"].sort_values("detected_unseen")
    fig, ax = plt.subplots(figsize=(9, 5.4))
    y = np.arange(len(d))
    ax.hlines(y, d["detected_unseen"], d["detected_random_split"], color=GREY, lw=2, zorder=1)
    ax.scatter(d["detected_random_split"], y, s=40, color=GREY, label="découpage aléatoire (attaque vue)", zorder=2)
    ax.scatter(d["detected_unseen"], y, s=60, color=ACCENT, label="jour retenu (attaque jamais vue)", zorder=3)
    ax.set_yticks(y, d["class"])
    for i, v in enumerate(d["detected_unseen"]):
        ax.text(1.04, i, pct(v, 0), va="center", fontsize=8.5, weight="bold")
    ax.axvline(pooled, color="#555", ls="--", lw=1)
    ax.text(pooled + 0.01, -0.9, f"toutes attaques jamais vues : {pct(pooled, 0)}", fontsize=8.5, color="#333")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-1.3, len(d) - 0.4)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel("Part des flux d'attaque détectés (XGBoost binaire)")
    ax.set_title("Jamais vue, une attaque est détectée une fois sur trois ; vue, presque toujours", fontsize=11, loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=8.5, frameon=False)
    style_axes(ax)
    path = FIG / "separation_par_jour.png"
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return path


def demo_capture() -> tuple[Path | None, dict]:
    """Analyse the demo capture with the saved models (both flag modes) and draw the result."""
    try:
        from nids.capture.demo import SCANNER, build_demo_pcap
        from nids.capture.pcap_flows import extract_flows
        from nids.prediction.predictor import Predictor
    except Exception as exc:  # pragma: no cover
        return None, {"error": str(exc)}
    predictor = Predictor.load("final")
    data = build_demo_pcap()
    out = {}
    frames = {}
    for mode in ("cic", "real"):
        flows = extract_flows(data, flag_mode=mode)
        pred = predictor.predict_frame(flows.features, missing="median").frame["prediction"].reset_index(drop=True)
        frame = flows.meta.assign(prediction=pred)
        scan = frame["src_ip"] == SCANNER
        out[mode] = {
            "packets": flows.stats["packets_read"], "flows": len(frame), "scan_flows": int(scan.sum()),
            "scan_detected": int((scan & (frame["prediction"] == "PortScan")).sum()),
            "normal_flows": int((~scan).sum()), "normal_alarms": int((~scan & (frame["prediction"] != "BENIGN")).sum()),
        }
        frames[mode] = frame
    f = frames["cic"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1, 2]})
    counts = f["prediction"].value_counts()
    axes[0].barh(counts.index, counts.values, color=[GREY if c == "BENIGN" else "#d64545" for c in counts.index])
    for i, v in enumerate(counts.values):
        axes[0].text(v + 3, i, str(v), va="center", fontsize=9)
    axes[0].set_title("Classes prédites", fontsize=10, loc="left")
    style_axes(axes[0])
    t = (f["start_time"] - f["start_time"].min()).dt.total_seconds()
    attack = f["prediction"] != "BENIGN"
    axes[1].scatter(t[~attack], f.loc[~attack, "dst_port"], s=14, color=GREY, label="classé BENIGN")
    axes[1].scatter(t[attack], f.loc[attack, "dst_port"], s=14, color="#d64545", label="classé attaque")
    axes[1].set_yscale("symlog")
    axes[1].set_xlabel("secondes depuis le début de la capture")
    axes[1].set_ylabel("port de destination")
    axes[1].set_title("Flux dans le temps : le scan balaie les ports 1 à 300", fontsize=10, loc="left")
    axes[1].legend(fontsize=8.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    style_axes(axes[1])
    path = FIG / "demo_capture.png"
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return path, out


def attach_headings(story: list) -> list:
    """Glue every heading to the block that follows it, so a heading never ends a page alone.

    ``keepWithNext`` does not hold when the next flowable is itself a KeepTogether (table + caption, figure + caption).
    The heading's ``_toc`` mark is kept on the heading, inside the group, so the table of contents still finds it.
    """
    out: list = []
    i = 0
    while i < len(story):
        item = story[i]
        next_is_toc = i + 1 < len(story) and isinstance(story[i + 1], TableOfContents)  # the TOC must stay a top-level flowable
        if isinstance(item, Paragraph) and item.style.name in ("h1", "h2", "h3") and i + 1 < len(story) and not next_is_toc:
            group = [item]
            j = i + 1
            while j < len(story) and isinstance(story[j], Paragraph) and story[j].style.name in ("h1", "h2", "h3"):
                group.append(story[j])
                j += 1
            if j < len(story):
                group.append(story[j])
                j += 1
            out.append(KeepTogether(group))
            i = j
        else:
            out.append(item)
            i += 1
    return out


# ---------------------------------------------------------------------------
# Document template (header, footer, table of contents)
# ---------------------------------------------------------------------------
class ReportDoc(BaseDocTemplate):
    def __init__(self, filename: str, **kw):
        super().__init__(filename, pagesize=A4, leftMargin=2.2 * cm, rightMargin=2.2 * cm, topMargin=2.2 * cm, bottomMargin=2 * cm,
                         title="Détection d'intrusions réseau par data mining sur CIC-IDS2017", author="Projet DM", **kw)
        frame = Frame(self.leftMargin, self.bottomMargin, self.width, self.height, id="f")
        self.addPageTemplates([PageTemplate(id="cover", frames=[frame], onPage=self._cover), PageTemplate(id="body", frames=[frame], onPage=self._page)])

    def afterFlowable(self, flowable):
        if hasattr(flowable, "_toc"):
            level, text = flowable._toc
            key_name = f"h{id(flowable)}"
            self.canv.bookmarkPage(key_name)
            self.canv.addOutlineEntry(text, key_name, level=level, closed=level > 0)
            self.notify("TOCEntry", (level, text, self.page, key_name))

    def _cover(self, canv, doc):
        canv.saveState()
        canv.setFillColor(colors.HexColor(ACCENT))
        canv.rect(0, A4[1] - 9 * cm, A4[0], 9 * cm, stroke=0, fill=1)
        canv.restoreState()

    def _page(self, canv, doc):
        canv.saveState()
        canv.setFont(FONT, 8)
        canv.setFillColor(colors.HexColor(MUTED))
        canv.drawString(doc.leftMargin, A4[1] - 1.3 * cm, "Détection d'intrusions réseau par data mining · CIC-IDS2017")
        canv.drawRightString(A4[0] - doc.rightMargin, 1.2 * cm, f"{doc.page}")
        canv.setStrokeColor(colors.HexColor("#d6dde8"))
        canv.line(doc.leftMargin, A4[1] - 1.45 * cm, A4[0] - doc.rightMargin, A4[1] - 1.45 * cm)
        canv.restoreState()


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------
def build() -> int:
    FIG.mkdir(parents=True, exist_ok=True)
    # ---- data from the reports
    meta = rj("data/preprocessed/preprocessing_metadata.json", "scripts/04_preprocess.py")
    cleaning = rj("data/cleaned/cleaning_report.json", "scripts/03_clean_data.py")
    clean_counts = rc("data/cleaned/cleaning_class_counts.csv", "scripts/03_clean_data.py")
    split_counts = rc("data/preprocessed/preprocessing_class_counts.csv", "scripts/04_preprocess.py")
    comparison = rj("reports/evaluation/model_comparison_tuned.json", "scripts/06_evaluate.py --stage tuned")
    clustering = rj("reports/mining/clustering/clustering_report.json", "scripts/09_cluster.py")
    composition = rc("reports/mining/clustering/composition_k15.csv", "scripts/09_cluster.py")
    anomaly = rj("reports/mining/anomaly/anomaly_report.json", "scripts/10_anomaly.py")
    anomaly_pc = rc("reports/mining/anomaly/per_class_detection.csv", "scripts/10_anomaly.py")
    assoc = rj("reports/mining/association/association_report.json", "scripts/11_association_rules.py")
    class_rules = rc("reports/mining/association/class_rules.csv", "scripts/11_association_rules.py")
    rule_pc = rc("reports/mining/association/rule_classifier_per_class.csv", "scripts/11_association_rules.py")
    explain = rj("reports/mining/explain/explain_report.json", "scripts/12_explain.py")
    ens = rc("reports/experiments/ensembles/ensembles_comparison.csv", "scripts/13_ensembles.py")
    port = rj("reports/experiments/port_ablation/port_ablation_report.json", "scripts/14_port_ablation.py")
    day = rj("reports/experiments/day_split/day_split_report.json", "scripts/15_day_split.py")
    day_pc = rc("reports/experiments/day_split/per_class.csv", "scripts/15_day_split.py")
    imb = rc("reports/experiments/imbalance/imbalance_comparison.csv", "scripts/16_imbalance.py")
    gran = rc("reports/experiments/granularity/granularity_comparison.csv", "scripts/17_granularity.py")

    models = comparison["models"]
    votes = {v["key"]: v for v in comparison["votes"]}
    final_vote = votes["majority_vote_final"]

    print("Drawing the report figures ...")
    f_pipe = fig_pipeline()
    f_f1 = fig_f1_models(models, final_vote["macro_f1"])
    f_anom = fig_anomaly(anomaly_pc)
    f_imp = fig_importance()
    f_day = fig_day_split(day_pc, day["pooled_detection_unseen"])
    print("Analysing the demo capture with the saved models ...")
    f_demo, demo = demo_capture()

    story: list = []
    # ---------------- cover
    story += [
        Spacer(1, 2.2 * cm),
        Paragraph("<font color='white' size='24'><b>Détection d'intrusions réseau<br/>par data mining</b></font>", ParagraphStyle("t", fontName=BOLD, fontSize=24, leading=30)),
        Spacer(1, 0.4 * cm),
        Paragraph("<font color='white' size='13'>Classification, clustering, règles d'association et détection d'anomalies sur CIC-IDS2017</font>", ParagraphStyle("s", fontName=FONT, fontSize=13, leading=17)),
        Spacer(1, 4.2 * cm),
        P("<b>Rapport de projet</b> · rapport complet de la démarche, de la donnée brute à l'application", "body"),
        P(f"Version du {date.today().strftime('%d/%m/%Y')}", "body"),
        Spacer(1, 0.8 * cm),
    ]
    vote_line = (f"vote majoritaire : F1 macro {fr(final_vote['macro_f1'])}, détection {pct(final_vote['detection_rate'], 2)}, "
                 f"fausses alertes {pct(final_vote['false_alarm_rate'], 2)}")
    story += [
        table(
            [
                ["Élément", "Valeur"],
                ["Jeu de données", f"CIC-IDS2017 · {n(cleaning['rows_in'])} flux bruts · {n(cleaning['rows_out'])} après nettoyage · 15 classes"],
                ["Méthodes", "11 classifieurs + vote · k-means, Ward, DBSCAN · FP-Growth · Isolation Forest, LOF, autoencodeur"],
                ["Résultat principal", vote_line],
                ["Attaques jamais vues", f"détection {pct(day['pooled_detection_unseen'], 0)} (contre {pct(day['random_split']['detection_rate'], 2)} si l'attaque est connue)"],
                ["Livrables", "18 scripts reproductibles · application Streamlit (CSV et .pcap) · 719 tests automatisés"],
            ],
            [3.6, 13.0],
        ),
        NextPageTemplate("body"),
        CondPageBreak(9 * cm),
    ]
    toc = TableOfContents()
    toc.levelStyles = [S["toc1"], S["toc2"]]
    story += [P("<b>Sommaire</b>", "h1"), toc, PageBreak()]

    # ---------------- 0. Résumé
    story += [
        H1("Résumé"),
        P(f"Sur le jeu de données CIC-IDS2017 ({n(cleaning['rows_out'])} flux réseau après nettoyage, 15 classes), un vote majoritaire de 11 "
          f"classifieurs atteint un F1 macro de {fr(final_vote['macro_f1'])} et détecte {pct(final_vote['detection_rate'], 1)} des flux d'attaque avec "
          f"{pct(final_vote['false_alarm_rate'], 2)} de fausses alertes. Ce score ne vaut que pour des types d'attaques déjà vus : une attaque d'un type "
          f"absent de l'entraînement n'est détectée qu'une fois sur trois ({pct(day['pooled_detection_unseen'], 0)})."),
        P("Le projet couvre les quatre grandes tâches du data mining. La <b>classification supervisée</b> compare 11 modèles réglés par validation croisée. "
          "Le <b>clustering</b> montre que les attaques forment des groupes naturels sans étiquettes. Les <b>règles d'association</b> (FP-Growth) donnent des "
          "signatures lisibles. La <b>détection d'anomalies</b>, apprise sur le seul trafic normal, repère les attaques volumineuses sans aucune étiquette."),
        P("Cinq expériences mesurent ensuite la portée réelle des scores : séparation par jour de capture, suppression du port de destination, ensembles pondérés "
          "et stacking, rééquilibrage des classes, regroupement des étiquettes. Une application Streamlit analyse enfin des fichiers de capture (.pcap). Son "
          "développement a révélé que les modèles ont en partie appris un défaut de l'outil qui a fabriqué le jeu de données."),
        key("<b>Message principal.</b> Les modèles à base d'arbres reconnaissent très bien les attaques connues de ce laboratoire. La généralisation à des attaques "
            "nouvelles ou à un autre réseau est un problème différent et bien plus difficile : c'est ce que montrent les chapitres 10 à 12."),
        PageBreak(),
    ]

    # ---------------- 1. Introduction
    story += [
        H1("1. Introduction et démarche"),
        P("Un système de détection d'intrusions (IDS) surveille le trafic réseau pour repérer les attaques. S'il manque une attaque, l'intrus agit ; s'il déclenche "
          "trop de fausses alertes, ses opérateurs finissent par l'ignorer. Le data mining permet d'apprendre automatiquement à distinguer le trafic normal du "
          "trafic malveillant à partir d'exemples."),
        H2("1.1 Questions de recherche"),
        bullets([
            "Quelle performance atteignent des méthodes supervisées sur le jeu de référence CIC-IDS2017 ?",
            "Que révèlent les méthodes non supervisées (clustering, règles d'association, anomalies) sur la structure du trafic ?",
            "Ces scores tiennent-ils face à des attaques nouvelles et à du trafic réel ?",
        ], numbered=True),
        H2("1.2 Démarche CRISP-DM"),
        P("Le projet suit les six phases de CRISP-DM : compréhension du problème, compréhension des données (inspection et exploration), préparation (nettoyage et "
          "prétraitement), modélisation, évaluation, déploiement. Chaque étape est un script Python reproductible qui écrit son propre rapport ; les chiffres de ce "
          "document sont lus directement dans ces rapports."),
        figure(f_pipe, "Figure 1 – Chaîne de traitement. Tout ce qui est appris (modèles, clusters, seuils, poids) l'est sur l'entraînement ; le test ne sert qu'à mesurer."),
        table(
            [
                ["Phase CRISP-DM", "Scripts", "Ce qui est produit"],
                ["Compréhension des données", "01_inspect_dataset, 02_run_eda", "inventaire des fichiers, statistiques, 11 figures d'exploration"],
                ["Préparation", "03_clean_data, 04_preprocess", "données nettoyées, découpage, pipeline, 40 variables"],
                ["Modélisation", "05_train, 07_optimize, 09 à 12", "11 classifieurs (base et réglés), clusters, règles, détecteurs, projections"],
                ["Évaluation", "06_evaluate, 08, 13 à 17", "scores sur le test, matrices de confusion, 5 expériences critiques"],
                ["Déploiement", "app/streamlit_app.py, 18", "application web (flux, CSV, .pcap) et ce rapport"],
            ],
            [4.0, 5.0, 7.6],
        ),
        Spacer(1, 6),
        P("Le code est organisé en paquet Python (<font name='Courier'>src/nids</font>) : données, nettoyage, exploration, prétraitement, modèles, entraînement, "
          "évaluation, optimisation, vote, persistance, prédiction, data mining (<font name='Courier'>mining</font>), expériences et captures réseau "
          "(<font name='Courier'>capture</font>). Tous les paramètres (chemins, graine aléatoire, seuils) sont centralisés dans "
          "<font name='Courier'>config/settings.py</font>."),
        CondPageBreak(9 * cm),
    ]

    # ---------------- 2. Données et exploration
    raw_counts = clean_counts[["label", "rows_before"]].sort_values("rows_before", ascending=False)
    story += [
        H1("2. Compréhension et exploration des données"),
        H2("2.1 Le jeu CIC-IDS2017"),
        P(f"CIC-IDS2017 a été publié par le Canadian Institute for Cybersecurity (Université du Nouveau-Brunswick). Le trafic d'un réseau de laboratoire a été "
          f"capturé pendant cinq jours ouvrés, avec des attaques jouées à des moments connus, puis converti en <b>flux</b> par l'outil CICFlowMeter. Un flux résume "
          f"un échange bidirectionnel entre deux machines : durée, nombre et taille des paquets dans chaque sens, délais entre paquets, flags TCP, port de "
          f"destination. Le jeu brut compte {n(cleaning['rows_in'])} flux, {cleaning['n_feature_columns_in']} variables et 15 classes."),
        figure(REP / "eda/figures/class_distribution.png", "Figure 2 – Nombre de flux par classe (échelle logarithmique), données brutes. BENIGN représente 80,3 % des flux, Heartbleed 11 flux.", 15),
        P("Le jeu est <b>très déséquilibré</b> : un rapport de plus de 200 000 entre la classe la plus grande et la plus petite. L'exactitude (accuracy) n'a donc pas "
          "de sens : prédire toujours BENIGN donnerait 80 %. Toutes les comparaisons utilisent le <b>F1 macro</b>, moyenne des F1 de chaque classe."),
        figure(REP / "eda/figures/label_by_file.png", "Figure 3 – Classes par fichier de capture. Chaque type d'attaque n'apparaît que dans un seul fichier.", 15),
        key("Chaque type d'attaque n'apparaît que dans <b>un seul</b> des 8 fichiers. Cette particularité est exploitée au chapitre 10 pour tester la détection d'attaques jamais vues."),
        H2("2.2 Exploration (script 02)"),
        P("L'exploration a été faite sur les données <b>non nettoyées</b>, pour décider des règles de nettoyage. Ses principaux constats :"),
        bullets([
            "<b>8 colonnes constantes</b> (flags PSH et URG arrière, six moyennes « bulk ») qui n'apportent aucune information.",
            "<b>41 paires de variables corrélées</b> à plus de 0,95, dont 10 parfaitement identiques (par exemple Total Fwd Packets et Subflow Fwd Packets). "
            "Un filtre au seuil 0,95 retirerait environ 24 variables.",
            "<b>Des valeurs infinies et négatives impossibles</b> : débits infinis quand la durée du flux est nulle, durées et délais négatifs.",
            "<b>Des distributions extrêmement asymétriques</b> : la plupart des variables ont une asymétrie supérieure à 10 et beaucoup de valeurs hors des moustaches. "
            "D'où le choix d'une transformation logarithmique avant la standardisation.",
            "Les variables qui séparent le mieux les classes (test ANOVA F) sont les longueurs de paquets retour, le flag PSH et les tailles de fenêtre TCP.",
        ]),
        figure(REP / "eda/figures/missing_and_infinite.png", "Figure 4 – Valeurs manquantes et infinies par variable (données brutes).", 14),
        figure(REP / "eda/figures/skewness.png", "Figure 5 – Asymétrie des variables : la plupart sont très fortement asymétriques.", 14),
        figure(REP / "eda/figures/correlation_heatmap.png", "Figure 6 – Matrice de corrélation (échantillon de 500 000 lignes) : de larges blocs de variables redondantes.", 14),
        CondPageBreak(9 * cm),
    ]

    # ---------------- 3. Nettoyage
    steps = cleaning["steps"] if isinstance(cleaning["steps"], list) else []
    cc = clean_counts.sort_values("rows_before", ascending=False)
    rows = [["Classe", "Avant", "Après", "Retirés", "% retirés"]]
    rows += [[r.label, n(r.rows_before), n(r.rows_after), n(r.rows_removed), fr(r.removed_percent, 1)] for r in cc.itertuples()]
    story += [
        H1("3. Nettoyage des données"),
        P(f"Le nettoyage (script 03) applique des règles décidées à partir de l'inspection et de l'exploration ; chaque règle est écrite dans "
          f"<font name='Courier'>config/settings.py</font> avec sa justification. Résultat : <b>{n(cleaning['rows_in'])} → {n(cleaning['rows_out'])} flux</b> "
          f"et <b>{cleaning['n_feature_columns_in']} → {cleaning['n_feature_columns_out']} variables</b>."),
        bullets([
            "<b>Doublons exacts</b> : 308 381 lignes identiques supprimées (10,9 %). Ils gonfleraient artificiellement les scores, car une même ligne pourrait être à la fois en entraînement et en test.",
            "<b>Libellés endommagés</b> : les trois attaques web contenaient un caractère illisible à la place du tiret (« Web Attack ? XSS »), remplacé par un tiret (2 180 lignes).",
            "<b>Colonnes identiques</b> : 12 colonnes strictement égales à une autre sont retirées (Subflow Fwd Packets, SYN Flag Count, colonnes « bulk »...).",
            "<b>Valeurs impossibles</b> : les valeurs négatives dans les durées, débits et longueurs d'en-tête (2 926 lignes, 93 % BENIGN) deviennent manquantes ; "
            "les valeurs infinies (2 867 lignes, toutes de durée nulle) aussi. Elles seront imputées au prétraitement.",
            "<b>Vecteurs contradictoires</b> : 698 vecteurs identiques portent deux étiquettes différentes (0,055 %). Ils sont conservés et fixent un plancher d'erreur inévitable.",
        ]),
        tblock(table(rows, [5.8, 2.6, 2.6, 2.6, 2.2], num_cols=(1, 2, 3, 4)), "Tableau 1 – Flux par classe avant et après suppression des doublons. PortScan (43 %) et SSH-Patator (45 %) contenaient beaucoup de répétitions."),
        CondPageBreak(9 * cm),
    ]

    # ---------------- 4. Prétraitement
    sc = split_counts.copy()
    rows = [["Classe", "Entraînement", "Test"]] + [[r[2], n(r[3]), n(r[4])] for r in sc.itertuples()]
    dropped = [f for f in meta.get("feature_selection", {}).get("dropped", [])] if isinstance(meta.get("feature_selection"), dict) else []
    story += [
        H1("4. Prétraitement"),
        P(f"Le prétraitement (script 04) découpe les données puis construit un pipeline scikit-learn <b>ajusté sur l'entraînement seulement</b>, pour qu'aucune "
          f"information du test ne fuie dans les modèles."),
        H2("4.1 Découpage"),
        P(f"Découpage stratifié 80/20, graine 42 : <b>{n(meta['X_train_shape'][0])} flux d'entraînement</b> et <b>{n(meta['X_test_shape'][0])} flux de test</b>. "
          f"La stratification garantit que chaque classe garde la même proportion des deux côtés."),
        tblock(table(rows, [7.0, 3.5, 3.5], num_cols=(1, 2)), "Tableau 2 – Flux par classe dans l'entraînement et le test. Heartbleed (2), Sql Injection (4) et Infiltration (7) n'ont que quelques flux de test."),
        key("Les scores des trois classes minuscules sont <b>indicatifs</b> : un flux de plus ou de moins les fait varier de plusieurs dizaines de points. "
            "Le rapport donne aussi un « F1 macro fiable » calculé sur les classes ayant au moins 30 flux de test."),
        H2("4.2 Pipeline"),
        table(
            [
                ["Étape", "Ce qu'elle fait", "Pourquoi"],
                ["1. Infinis → manquants", "remplace ±∞ par une valeur manquante", "les débits sont infinis quand la durée est nulle"],
                ["2. Imputation", "médiane de l'entraînement", "robuste aux valeurs extrêmes, contrairement à la moyenne"],
                ["3. log1p signé", "sign(x) · log(1 + |x|)", "réduit l'asymétrie extrême (chapitre 2) en gardant le signe"],
                ["4. Filtre", "retire les variables constantes et corrélées à plus de 0,95", "supprime la redondance : 66 → 40 variables"],
                ["5. Standardisation", "moyenne 0, écart-type 1", "nécessaire pour KNN, SVC, la régression logistique, k-means"],
            ],
            [3.6, 6.0, 7.0],
        ),
        Spacer(1, 6),
        P(f"Le pipeline conserve <b>{meta['n_model_features']} des {meta['n_raw_features']} variables</b>. Il est enregistré avec un « hash » de ses réglages : "
          "si un réglage change, les données prétraitées et les modèles sont déclarés périmés au lieu d'être réutilisés par erreur."),
        CondPageBreak(9 * cm),
    ]

    # ---------------- 5. Classification
    mrows = [["Modèle", "F1 base", "F1 réglé", "Écart", "Version gardée", "Paramètres réglés"]]
    for m in sorted(models, key=lambda m: -max(m["baseline_macro_f1"], m["tuned_macro_f1"])):
        mrows.append([m["model"], fr(m["baseline_macro_f1"]), fr(m["tuned_macro_f1"]), fr(m["delta_macro_f1"]), "réglée" if m["final_set_uses"] == "tuned" else "base",
                      (m.get("tuned_parameters") or "")[:60]])
    vrows = [["Vote", "F1 macro", "F1 fiable", "Détection", "Fausses alertes"]]
    names = {"majority_vote_baseline": "11 modèles de base", "majority_vote_tuned": "11 modèles réglés", "majority_vote_final": "Ensemble final (règle fixée)"}
    for k, label in names.items():
        v = votes[k]
        vrows.append([label, fr(v["macro_f1"], 4), fr(v["macro_f1_reliable"], 4), pct(v["detection_rate"], 2), pct(v["false_alarm_rate"], 3)])
    story += [
        H1("5. Classification supervisée"),
        P("Onze algorithmes de familles différentes sont comparés, puis combinés par vote majoritaire. Chacun existe en version « de base » (réglages par défaut) "
          "et en version « réglée »."),
        H2("5.1 Les modèles"),
        bullets([
            "<b>Plus proches voisins</b> : KNN.",
            "<b>Arbres et ensembles d'arbres</b> : Decision Tree, Random Forest, AdaBoost, Gradient Boosting, XGBoost (sur carte graphique).",
            "<b>Modèles probabilistes</b> : Naive Bayes gaussien et Bernoulli.",
            "<b>Modèles linéaires et à noyau</b> : régression logistique, SGD, SVC à noyau RBF.",
        ]),
        P("KNN, SVC et Gradient Boosting sont trop lents sur 2 millions de lignes. Ils sont entraînés sur un sous-échantillon <b>plafonné par classe</b> "
          "(50 000 à 100 000 lignes) : toutes les lignes des petites classes sont gardées et les grandes classes sont plafonnées."),
        H2("5.2 Réglage des hyperparamètres"),
        P("Chaque modèle est réglé par GridSearchCV (validation croisée en 5 plis, score = F1 macro), y compris le choix d'une pondération des classes. "
          "Le choix entre version de base et version réglée a été <b>fixé avant tout score sur le test</b> : la version réglée n'est gardée que si son gain "
          "moyen dépasse l'écart-type des différences entre plis. Cette règle évite de choisir sur le test, ce qui gonflerait les résultats."),
        tblock(table(mrows, [3.6, 1.6, 1.6, 1.5, 2.2, 6.1], num_cols=(1, 2, 3)), "Tableau 3 – F1 macro sur le test, versions de base et réglées. L'écart est réglé moins base."),
        figure(f_f1, "Figure 7 – F1 macro de la version utilisée par l'ensemble final. Les modèles à base d'arbres dominent nettement.", 15.5),
        H2("5.3 Vote majoritaire"),
        P("Chaque modèle vote pour une classe, avec une voix chacun et sans pondération. En cas d'égalité, une classe d'attaque l'emporte sur BENIGN : "
          "rater une attaque coûte plus cher qu'une fausse alerte."),
        tblock(table(vrows, [5.8, 2.4, 2.4, 2.6, 3.2], num_cols=(1, 2, 3, 4)), "Tableau 4 – Les trois votes sur le test (504 473 flux)."),
        P("Les trois votes ne se distinguent pas nettement : les écarts sont de quelques millièmes et dépendent surtout des trois classes minuscules. Surtout, "
          "<b>Random Forest seul (0,915) et XGBoost seul (0,912) font mieux que le vote</b> : les modèles faibles diluent les bons."),
        figure(REP / "evaluation/figures/confusion_majority_vote.png", "Figure 8 – Matrice de confusion du vote (en % de chaque classe réelle). La diagonale est correcte.", 14),
        H2("5.4 Erreurs principales"),
        bullets([
            "<b>XSS classé Brute Force</b> : 69 % des XSS. Les deux attaques web se ressemblent presque parfaitement dans l'espace des variables.",
            "<b>Bot classé BENIGN</b> : 12 à 14 % des Bot, donc des attaques manquées.",
            "<b>Infiltration</b> : 3 des 7 flux de test classés BENIGN.",
            "Les fausses alertes restent rares : 183 flux normaux classés PortScan, 89 DoS Hulk, 58 Bot, sur 419 297 flux normaux.",
        ]),
        CondPageBreak(9 * cm),
    ]

    # ---------------- 6. Clustering
    sweep = pd.DataFrame(clustering["kmeans_sweep"])
    srows = [["k", "Silhouette", "ARI", "NMI", "Pureté", "F1 classifieur par cluster"]]
    for r in sweep.itertuples():
        srows.append([str(r.k), fr(r.silhouette), fr(r.ari), fr(r.nmi), fr(r.purity), fr(r.holdout_macro_f1)])
    cmp_rows = [["Algorithme", "Clusters", "ARI", "NMI", "Pureté", "Silhouette"]]
    for r in clustering["kmeans_vs_ward"]:
        cmp_rows.append([r["algorithm"].replace("agglomerative", "hiérarchique"), str(r["n_clusters"]), fr(r["ari"]), fr(r["nmi"]), fr(r["purity"]), fr(r["silhouette"])])
    db = pd.DataFrame(clustering["dbscan"])
    db_rows = [["Quantile eps", "eps", "Clusters", "Bruit", "ARI", "NMI", "Pureté"]]
    for r in db.itertuples():
        db_rows.append([fr(r.eps_quantile, 2), fr(r.eps, 3), str(r.n_clusters), pct(r.noise_fraction), fr(r.ari), fr(r.nmi), fr(r.purity)])
    comp = composition.head(15)
    comp_rows = [["Cluster", "Taille", "Classe majoritaire", "Part", "Deuxième classe", "Part"]]
    for r in comp.itertuples():
        comp_rows.append([str(r.cluster), n(r.size), r.majority_class, pct(r.majority_share, 0), r.second_class if isinstance(r.second_class, str) else "", pct(r.second_share, 0)])
    story += [
        H1("6. Clustering"),
        P("Question : sans aucune étiquette, les flux se regroupent-ils naturellement par type d'attaque ? Les étiquettes ne servent jamais à construire les "
          "clusters, seulement à les juger ensuite."),
        H2("6.1 Méthode"),
        bullets([
            f"<b>Données</b> : {n(clustering['fit_rows'])} flux d'entraînement plafonnés par classe pour ajuster les clusters, {n(clustering['eval_rows'])} autres pour l'évaluation.",
            "<b>k-means</b> pour k de 2 à 50 ; <b>Ward</b> (classification hiérarchique ascendante) sur 10 000 flux, car sa mémoire croît en n² ; <b>DBSCAN</b> avec un eps choisi sur la courbe des k-distances.",
            "<b>Mesures internes</b> (sans étiquettes) : silhouette et Davies-Bouldin. <b>Mesures externes</b> : indice de Rand ajusté (ARI), information mutuelle normalisée (NMI), pureté.",
            "<b>Classifieur par cluster</b> : chaque cluster prend le nom de sa classe majoritaire, puis la règle est évaluée sur d'autres flux. Son F1 macro mesure la part de structure capturée.",
        ]),
        H2("6.2 Résultats k-means"),
        tblock(table(srows, [1.2, 2.4, 2.0, 2.0, 2.0, 4.2], num_cols=(0, 1, 2, 3, 4, 5)), "Tableau 5 – Balayage de k pour k-means."),
        figure(REP / "mining/clustering/figures/kmeans_sweep.png", "Figure 9 – Gauche : coude (inertie) et silhouette. Droite : accord des clusters avec les vraies classes.", 16),
        P(f"La silhouette, qui n'utilise aucune étiquette, est maximale à <b>k = {clustering['best_k_silhouette']}</b>. La pureté continue de monter avec k "
          f"(0,92 à k = 50) alors que l'ARI baisse après k = 15 : les classes se découpent en sous-groupes homogènes. Il faut bien plus de clusters que de "
          f"classes, car une même attaque produit plusieurs formes de flux."),
        H2("6.3 Comparaison des algorithmes"),
        tblock(table(cmp_rows, [5.2, 1.9, 1.8, 1.8, 1.8, 2.0], num_cols=(1, 2, 3, 4, 5)), "Tableau 6 – k-means et Ward sur les mêmes 10 000 flux, k = 15."),
        tblock(table(db_rows, [2.4, 1.8, 1.8, 1.8, 1.8, 1.8, 1.8], num_cols=(0, 1, 2, 3, 4, 5, 6)), "Tableau 7 – DBSCAN (20 000 flux, min_samples = 10) pour plusieurs valeurs d'eps."),
        P("Ward fait mieux que k-means sur toutes les mesures. DBSCAN atteint la meilleure pureté (0,94) en laissant 4 à 8 % des flux comme « bruit », mais "
          "avec 80 à 140 clusters."),
        H2("6.4 Contenu des clusters"),
        tblock(table(comp_rows, [1.6, 1.8, 3.8, 1.4, 3.8, 1.4], num_cols=(1, 3, 5)), "Tableau 8 – Composition des 15 clusters k-means."),
        figure(REP / "mining/clustering/figures/composition_k15.png", "Figure 10 – Répartition de chaque classe entre les 15 clusters (ligne = classe).", 15),
        P("Plusieurs clusters sont presque purs (PortScan, DoS Hulk, BENIGN). Les mélanges révèlent des parentés : DoS Hulk avec DDoS, FTP-Patator avec SSH-Patator, "
          "les DoS lents entre eux. Les attaques web et Bot restent noyées dans des clusters dominés par BENIGN."),
        CondPageBreak(9 * cm),
    ]

    # ---------------- 7. Règles d'association
    top_rules = class_rules.sort_values(["confidence", "class_coverage"], ascending=False).groupby("class").head(1)
    rrows = [["Règle", "Couverture", "Confiance", "Lift"]]
    for r in top_rules.sort_values("lift", ascending=False).itertuples():
        rrows.append([r.rule.replace("class=", "").replace("->", "→"), fr(r.class_coverage, 2), fr(r.confidence, 2), fr(r.lift, 1)])
    pcr = [["Classe", "Flux", "Précision", "Rappel", "F1"]] + [[r[1], n(r[2]), fr(r[3], 2), fr(r[4], 2), fr(r[5], 2)] for r in rule_pc.itertuples()]
    story += [
        H1("7. Règles d'association"),
        P("Question : quelles propriétés des flux vont ensemble, et lesquelles signent une attaque ? L'algorithme <b>FP-Growth</b> extrait les ensembles d'items "
          "fréquents, puis les règles « si A alors B » de forte confiance."),
        H2("7.1 Des flux aux transactions"),
        bullets([
            f"<b>Transactions</b> : {n(assoc['transactions'])} flux d'entraînement plafonnés par classe, pris dans les données nettoyées en unités d'origine "
            "(octets, microsecondes) pour que les règles restent lisibles.",
            f"<b>Items</b> ({assoc['items']}) : 13 variables découpées en trois tranches de quantiles (low, mid, high) avec 0 comme item à part ; présence des "
            "flags TCP ; port nommé par service (HTTP, SSH, FTP...) ; classe. Les bornes sont apprises sur l'entraînement.",
            "<b>Mesures</b> : support (fréquence de la règle), confiance (P(B | A)), lift (confiance / P(B) ; > 1 = association positive).",
        ]),
        H2("7.2 Règles générales"),
        P(f"Avec un support minimal de 0,002 et une confiance minimale de 0,8, FP-Growth trouve <b>{n(assoc['frequent_itemsets'])} itemsets fréquents</b> et "
          f"<b>{n(assoc['general_rules'])} règles</b>. Les règles de plus fort lift sont surtout des redondances (une longueur totale nulle implique une "
          f"longueur moyenne nulle) : elles confirment la redondance que le filtre de corrélation retire."),
        H2("7.3 Règles de classe"),
        P("Un support global cacherait les petites classes (Heartbleed n'a que 11 flux). Les itemsets sont donc extraits <b>à l'intérieur de chaque classe</b> "
          "(au moins 30 % de la classe), puis leur confiance et leur lift sont mesurés sur tous les flux : c'est l'idée de la classification par associations (CBA)."),
        tblock(table(rrows, [10.4, 2.0, 2.0, 2.0], num_cols=(1, 2, 3)), "Tableau 9 – Meilleure règle de chaque classe (couverture = part de la classe couverte)."),
        P("Les règles se lisent comme des signatures : Slowhttptest ouvre de longues connexions sans réponse (Total Backward Packets = 0), Bot passe par le port "
          f"8080, SSH-Patator par le port SSH. <b>Aucune règle</b> n'atteint 0,8 de confiance pour {', '.join(assoc['classes_without_rule'])} : "
          "leurs flux ressemblent à du trafic HTTP ordinaire."),
        H2("7.4 Les règles comme classifieur"),
        P(f"Les {assoc['class_rules']} règles, rangées par confiance (la première qui s'applique décide, sinon BENIGN), donnent un F1 macro de "
          f"<b>{fr(assoc['rule_classifier_macro_f1'])}</b> sur {n(assoc['rule_classifier_test_rows'])} flux de test : précision presque parfaite, rappel faible, "
          "et chaque décision s'explique en une ligne."),
        tblock(table(pcr, [6.0, 2.0, 2.4, 2.4, 2.4], num_cols=(1, 2, 3, 4)), "Tableau 10 – Classifieur à règles, par classe."),
        CondPageBreak(9 * cm),
    ]

    # ---------------- 8. Anomalies
    arows = [["Détecteur", "Lignes d'apprentissage", "AUC ROC", "Précision moy.", "Détection à 1 %", "Détection à 5 %", "Fausses alertes mesurées à 1 %"]]
    for d in anomaly["detectors"]:
        ops = {op["target_false_alarm_rate"]: op for op in d["operating_points"]}
        arows.append([d["name"], n(d["fit_rows"]), fr(d["roc_auc"]), fr(d["average_precision"]), pct(ops[0.01]["detection_rate"]), pct(ops[0.05]["detection_rate"]), pct(ops[0.01]["false_alarm_rate"], 2)])
    sup = anomaly["supervised_reference"]
    story += [
        H1("8. Détection d'anomalies non supervisée"),
        P("Question : peut-on détecter des attaques en n'apprenant que le trafic normal ? C'est le cadre réaliste face à une attaque nouvelle, qui par définition "
          "n'a pas d'étiquette."),
        H2("8.1 Méthode"),
        bullets([
            f"<b>Apprentissage</b> : {n(anomaly['fit_rows'])} flux BENIGN d'entraînement uniquement (50 000 pour LOF, plus coûteux).",
            "<b>Isolation Forest</b> : une anomalie s'isole en peu de coupes aléatoires.",
            "<b>Local Outlier Factor</b> : compare la densité d'un point à celle de ses voisins.",
            "<b>Autoencodeur</b> : réseau de neurones 40-32-16-8-16-32-40 entraîné à reproduire le trafic normal ; l'erreur de reconstruction sert de score.",
            f"<b>Seuils</b> : fixés sur {n(anomaly['calibration_rows'])} <i>autres</i> flux BENIGN d'entraînement, pour 0,1 %, 1 % et 5 % de fausses alertes. "
            "Aucune étiquette d'attaque ne sert ni à l'apprentissage ni au seuil.",
        ]),
        H2("8.2 Résultats"),
        tblock(table(arows, [2.9, 2.3, 1.6, 1.9, 2.1, 2.1, 3.0], num_cols=(1, 2, 3, 4, 5, 6)), f"Tableau 11 – Détecteurs sur les {n(anomaly['test_rows'])} flux de test."),
        P(f"L'autoencodeur sépare attaques et trafic normal avec une AUC ROC de <b>0,964</b>, sans avoir vu une seule attaque. À 1 % de fausses alertes, il ne "
          f"détecte pourtant que 28 % des flux d'attaque, contre {pct(sup['detection_rate'], 1)} pour XGBoost supervisé (qui a vu toutes les attaques). Les seuils "
          f"tiennent leur promesse : le taux de fausses alertes mesuré sur le test reste proche de la cible."),
        figure(REP / "mining/anomaly/figures/roc_curves.png", "Figure 11 – Courbes ROC des trois détecteurs (axe des fausses alertes en log) ; l'étoile = XGBoost supervisé.", 11),
        figure(f_anom, "Figure 12 – Part de chaque attaque signalée au seuil de 1 % de fausses alertes.", 15.5),
        P("Les attaques volumineuses (DoS, DDoS) modifient fortement les statistiques des flux et ressortent. Les attaques applicatives (brute force FTP/SSH, "
          "attaques web, Bot) ressemblent à des sessions ordinaires : un détecteur qui ne connaît que le normal ne peut pas les distinguer avec ces variables."),
        CondPageBreak(9 * cm),
    ]

    # ---------------- 9. Projections et importance
    pca = explain["pca"]
    story += [
        H1("9. Projections et importance des variables"),
        H2("9.1 Analyse en composantes principales"),
        P(f"Sur 15 000 flux d'entraînement plafonnés par classe, les deux premières composantes n'expliquent que <b>{pct(pca['pc1_pc2_variance'], 0)}</b> de la "
          f"variance ; il en faut {pca['components_for_90pct']} pour 90 % et {pca['components_for_95pct']} pour 95 %. Les classes ne se séparent pas dans une vue "
          "linéaire à deux dimensions."),
        figure(REP / "mining/explain/figures/pca_scree.png", "Figure 13 – Variance expliquée par composante (éboulis).", 11),
        figure(REP / "mining/explain/figures/pca_map.png", "Figure 14 – Projection des flux sur les deux premières composantes.", 14),
        H2("9.2 t-SNE"),
        P("Le t-SNE préserve les voisinages plutôt que les distances. La plupart des attaques forment leurs propres îlots, souvent plusieurs par classe, ce qui "
          "rejoint le clustering. Web Attack Brute Force et XSS partagent le même îlot : c'est la confusion que font tous les modèles."),
        figure(REP / "mining/explain/figures/tsne_map.png", "Figure 15 – Carte t-SNE. Les distances entre îlots ne sont pas interprétables ; seuls les voisinages le sont.", 15),
        H2("9.3 Importance des variables"),
        P("L'importance par permutation mesure la perte de F1 macro quand on mélange une variable, sur des flux de test : elle ne dépend pas du modèle et se "
          "mesure sur des données non vues, contrairement à l'importance intégrée des arbres."),
        figure(f_imp, "Figure 16 – Importance par permutation des 10 variables principales.", 15.5),
        P(f"Le <b>port de destination</b> est premier pour les deux modèles, suivi des tailles de fenêtre TCP initiales, qui dépendent du système d'exploitation de "
          f"chaque machine du laboratoire. Les deux modèles partagent {explain['rf_vs_xgb']['n_shared']} de leurs 10 variables principales (Spearman "
          f"{fr(explain['rf_vs_xgb']['spearman'], 2)}). Des variables corrélées se partagent l'importance : mélanger l'une laisse sa jumelle intacte."),
        CondPageBreak(9 * cm),
    ]

    # ---------------- 10. Évaluation critique
    folds = pd.DataFrame(day["folds"])
    frows = [["Jour retenu", "Attaques jamais vues", "Flux d'attaque", "Détection", "Fausses alertes", "AUC"]]
    days = {"Friday-Afternoon-DDos": "Vendredi après-midi (DDoS)", "Friday-Afternoon-PortScan": "Vendredi après-midi (PortScan)",
            "Friday-Morning": "Vendredi matin", "Thursday-Afternoon-Infilteration": "Jeudi après-midi", "Thursday-Morning-WebAttacks": "Jeudi matin",
            "Tuesday": "Mardi", "Wednesday": "Mercredi"}
    for r in folds.itertuples():
        frows.append([days.get(r.held_out, r.held_out), r.unseen_attacks, n(r.attack_rows), pct(r.detection_rate), pct(r.false_alarm_rate, 2), fr(r.roc_auc)])
    prow = [["Modèle", "Avec le port", "Sans le port", "Écart", "F1 fiable sans"]] + [[m["model"], fr(m["macro_f1_with"], 4), fr(m["macro_f1_without"], 4), fr(m["delta"], 4), fr(m["reliable_without"], 4)] for m in port["models"]]
    erows = [["Combinaison (modèles de base)", "F1 OOF (entraînement)", "F1 test", "F1 test fiable"]]
    for r in ens.itertuples():
        erows.append([r.combiner.replace("Single model", "Modèle seul").replace("Equal vote", "Vote égal").replace("Weighted vote", "Vote pondéré").replace("Stacking, logistic regression", "Stacking, régression logistique").replace("per-class OOF F1", "F1 OOF par classe").replace("by OOF F1", "par F1 OOF").replace("11 models (baseline set)", "11 modèles"),
                      "" if pd.isna(r.oof_macro_f1) else fr(r.oof_macro_f1, 4), fr(r.test_macro_f1, 4), fr(r.test_macro_f1_reliable, 4)])
    irows = [["Modèle", "Variante", "Lignes", "F1 macro", "F1 fiable", "Fausses alertes"]]
    vmap = {"none": "aucune", "balanced weights": "poids équilibrés", "SMOTE": "SMOTE", "undersample + SMOTE": "sous-éch. + SMOTE"}
    for r in imb.itertuples():
        irows.append([r.model, vmap.get(r.variant, r.variant), n(r.train_rows), fr(r.macro_f1, 4), fr(r.macro_f1_reliable, 4), pct(r.false_alarm_rate, 3)])
    grows = [["Modèle", "Tâche", "Variante", "F1 macro", "F1 fiable", "Détection"]]
    gmap = {"trained on 15 classes": "entraîné 15 classes", "trained on families": "entraîné familles", "15-class model, collapsed": "15 classes regroupé", "trained binary": "entraîné binaire"}
    tmap = {"15 classes": "15 classes", "8 families": "8 familles", "binary": "binaire"}
    for r in gran.itertuples():
        grows.append([r.model, tmap.get(r.task, r.task), gmap.get(r.variant, r.variant), fr(r.macro_f1, 4), fr(r.macro_f1_reliable, 4), pct(r.detection_rate, 2)])
    story += [
        H1("10. Évaluation critique : jusqu'où tiennent les scores ?"),
        P("Les scores du chapitre 5 sont mesurés avec un découpage aléatoire. Cinq expériences testent leur portée ; chacune n'apprend que sur l'entraînement."),
        H2("10.1 Séparation par jour : des attaques jamais vues"),
        P("Le découpage aléatoire montre au modèle des exemples de chaque attaque, souvent issus de la même session que les flux de test. Ici, un détecteur binaire "
          "(BENIGN contre attaque, XGBoost) est entraîné sur sept fichiers et testé sur le huitième ; le pipeline est réajusté à chaque tour. Comme chaque attaque "
          "n'apparaît que dans un fichier, les attaques testées sont toujours d'un type absent de l'entraînement."),
        tblock(table(frows, [3.6, 5.4, 2.1, 1.8, 2.1, 1.4], num_cols=(2, 3, 4, 5)), "Tableau 12 – Un tour par fichier contenant des attaques."),
        figure(f_day, "Figure 17 – Détection par classe, attaque vue (gris) contre jamais vue (bleu).", 15.5),
        key(f"Avec le découpage aléatoire, le même détecteur repère {pct(day['random_split']['detection_rate'], 2)} des attaques. Sur des types jamais vus : "
            f"<b>{pct(day['pooled_detection_unseen'], 0)}</b>. Une attaque nouvelle n'est repérée que si elle ressemble à une attaque connue."),
        H2("10.2 Sans le port de destination"),
        tblock(table(prow, [3.6, 2.8, 2.8, 2.4, 3.0], num_cols=(1, 2, 3, 4)), "Tableau 13 – Modèles réentraînés sans la variable Destination Port."),
        P("Les grandes classes ne bougent presque pas ; les pertes portent sur Sql Injection, Infiltration, XSS et Slowhttptest. XGBoost retrouve l'information dans "
          "les autres variables : le port est un raccourci, pas une nécessité."),
        H2("10.3 Ensembles pondérés et stacking"),
        P("Les poids et méta-modèles sont appris sur les prédictions <i>hors pli</i> (out-of-fold) de la validation croisée : pour chaque ligne d'entraînement, la "
          "prédiction d'un modèle qui ne l'a pas vue. Le test n'est utilisé qu'une fois, à la fin."),
        tblock(table(erows, [8.6, 2.8, 2.2, 2.6], num_cols=(1, 2, 3)), "Tableau 14 – Combinaisons des 11 modèles de base."),
        P("Pondérer aide, mais <b>aucune combinaison ne bat Random Forest seul</b>. Le vote pondéré par classe s'effondre : chaque modèle a un F1 proche de 1 sur "
          "BENIGN, donc ses votes BENIGN pèsent plus que ses votes pour les petites classes. Le vote égal reproduit exactement le score de l'évaluation (0,9095), "
          "ce qui valide toute la chaîne."),
        H2("10.4 Déséquilibre des classes"),
        tblock(table(irows, [3.2, 3.6, 2.4, 2.2, 2.2, 2.6], num_cols=(2, 3, 4, 5)), "Tableau 15 – Poids équilibrés, SMOTE (classes < 20 000 portées à 20 000) et sous-échantillonnage de BENIGN à 500 000 + SMOTE."),
        P("<b>Aucune variante n'améliore le F1 macro global.</b> Ces méthodes échangent de la précision contre du rappel sur les petites classes et ajoutent des "
          "fausses alertes. Les arbres apprennent déjà bien les petites classes ; les cas difficiles (XSS contre Brute Force) sont des chevauchements, pas un manque d'exemples."),
        H2("10.5 Granularité des étiquettes"),
        tblock(table(grows, [3.0, 2.4, 4.0, 2.2, 2.2, 2.4], num_cols=(3, 4, 5)), "Tableau 16 – 15 classes, 8 familles (DoS, DDoS, PortScan, Brute Force, Web, Bot, Infiltration, BENIGN) et binaire."),
        P("Regrouper en familles supprime les distinctions les plus dures et absorbe les classes minuscules (F1 macro 0,94-0,95). Un modèle à 15 classes regroupé "
          "après coup fait aussi bien qu'un modèle entraîné sur les familles : garder les étiquettes fines ne coûte rien."),
        CondPageBreak(9 * cm),
    ]

    # ---------------- 11. Application
    app_items = [
        "<b>Analyser une capture</b> : dépôt d'un fichier .pcap / .pcapng (ou bouton de démonstration) ; verdict, indicateurs, classes prédites, chronologie, hôtes impliqués, détail de chaque flux et export CSV.",
        "<b>Flux unique</b> : saisie des 40 variables ou chargement d'un vrai exemple ; vote de chaque modèle.",
        "<b>Fichier CSV</b> : jusqu'à 50 000 flux ; les lignes invalides sont refusées avec leur motif.",
        "<b>Les modèles</b> : version utilisée, scores, erreurs connues et limites.",
    ]
    story += [
        H1("11. Application : de la capture réseau à la décision"),
        P("Le déploiement prend la forme d'une application web Streamlit (<font name='Courier'>streamlit run app/streamlit_app.py</font>), au thème sombre de "
          "console de sécurité, organisée en quatre onglets :"),
        bullets(app_items, numbered=True),
        H2("11.1 Des paquets aux flux"),
        P("Les modèles lisent les 40 statistiques de flux de CICFlowMeter, pas des paquets. Le module <font name='Courier'>capture/pcap_flows.py</font> "
          "(bibliothèque dpkt) reproduit ses règles : flux bidirectionnels, sens « aller » donné par le premier paquet, fin sur FIN ou RST, durée maximale 120 s, "
          "périodes d'activité séparées par 5 s d'inactivité, temps en microsecondes entières. Chaque valeur a été vérifiée à la main sur une session TCP synthétique."),
        H2("11.2 Deux particularités découvertes dans CIC-IDS2017"),
        bullets([
            "<b>Le bourrage Ethernet est compté comme des données.</b> Une simple réponse RST affiche 6 octets de charge utile, soit exactement le bourrage d'une trame minimale de 60 octets. Le module l'imite.",
            "<b>Les colonnes de flags TCP sont fausses</b> (défaut documenté par Engelen et al., 2021). Dans les données, « PSH Flag Count » et « ACK Flag Count » "
            "sont presque exclusifs (les deux à 1 dans seulement 1 330 flux TCP sur 1,6 million) ; « PSH » vaut 1 pour tous les PortScan, qui commencent par un SYN.",
        ]),
    ]
    if f_demo is not None:
        c, r = demo["cic"], demo["real"]
        story += [
            figure(f_demo, "Figure 18 – Analyse de la capture de démonstration (mode compatible CIC-IDS2017), calculée par les modèles enregistrés.", 16),
            tblock(table(
                [
                    ["Mode des flags", "Flux du scan détectés", "Fausses alertes sur le trafic normal"],
                    ["Compatible CIC-IDS2017 (par défaut)", f"{c['scan_detected']} sur {c['scan_flows']}", f"{c['normal_alarms']} sur {c['normal_flows']}"],
                    ["Flags TCP réels", f"{r['scan_detected']} sur {r['scan_flows']}", f"{r['normal_alarms']} sur {r['normal_flows']}"],
                ],
                [6.0, 4.5, 6.0], num_cols=(1, 2),
            ), f"Tableau 17 – Capture de démonstration : {n(c['packets'])} paquets, {c['flows']} flux, dont un scan SYN de 300 ports."),
            key("Avec les vrais flags, le scan passe <b>entièrement inaperçu</b> ; avec le motif du dataset, il est reconnu. Les modèles ont donc en partie appris "
                "un défaut de l'outil qui a fabriqué leurs données. Le mode compatible est une approximation empirique de ce défaut, et l'application le signale."),
        ]
    story += [CondPageBreak(9 * cm)]

    # ---------------- 12. Qualité logicielle
    story += [
        H1("12. Qualité logicielle et reproductibilité"),
        bullets([
            "<b>719 tests automatisés</b> (pytest) : 665 pour la chaîne de classification et l'application, 36 pour les modules de data mining et d'expériences, "
            "18 pour la lecture des captures et l'onglet correspondant. Ils n'utilisent que des données synthétiques ; l'application est testée sans navigateur.",
            "<b>Écritures atomiques</b> : chaque fichier est écrit dans un fichier temporaire puis renommé, pour qu'une interruption ne laisse jamais un fichier à moitié écrit.",
            "<b>Contrôle de fraîcheur</b> : données prétraitées et modèles portent le hash de leurs réglages ; un réglage modifié les déclare périmés.",
            "<b>Pas de fuite de données</b> : le test n'est lu que par l'évaluation ; toutes les autres étapes n'apprennent que sur l'entraînement.",
            "<b>Dépendances figées</b> dans requirements.txt (Python 3.13, scikit-learn 1.8, XGBoost 3.1, mlxtend, imbalanced-learn, dpkt, Streamlit 1.52).",
        ]),
        table(
            [
                ["Commande", "Rôle", "Durée"],
                ["python scripts/01_inspect_dataset.py … 04_preprocess.py", "inspection, exploration, nettoyage, prétraitement", "quelques minutes"],
                ["python scripts/05_train.py, 06_evaluate.py", "11 modèles de base et leur évaluation", "≈ 18 min + validation croisée"],
                ["python scripts/07_optimize.py", "réglage par GridSearchCV", "plusieurs heures"],
                ["python scripts/09_cluster.py … 12_explain.py", "clustering, anomalies, règles, projections", "2 à 10 min chacun"],
                ["python scripts/13_ensembles.py … 17_granularity.py", "5 expériences critiques", "10 à 30 min chacune"],
                ["python scripts/18_build_report_pdf.py", "ce rapport", "≈ 1 min"],
                ["streamlit run app/streamlit_app.py", "application", "immédiat"],
                ["python -m pytest -q", "719 tests", "≈ 2 min"],
            ],
            [7.6, 6.0, 3.0],
        ),
        CondPageBreak(9 * cm),
    ]

    # ---------------- 13. Discussion
    story += [
        H1("13. Discussion et limites"),
        P("Les résultats décrivent un très bon détecteur des attaques connues de ce laboratoire, pas un IDS prêt pour un autre réseau. Trois raisons, chacune mesurée :"),
        bullets([
            f"<b>Le découpage aléatoire est optimiste</b> : séparer par jour fait chuter la détection de {pct(day['random_split']['detection_rate'], 2)} à {pct(day['pooled_detection_unseen'], 0)}.",
            "<b>Les modèles s'appuient sur le montage du laboratoire</b> : port de destination et fenêtres TCP initiales sont les variables les plus importantes.",
            "<b>Les modèles ont appris des défauts de l'outil de mesure</b> : avec les vrais flags TCP, un scan de ports passe inaperçu.",
        ], numbered=True),
        P("Les méthodes non supervisées apportent un contrepoint : le clustering confirme des groupes naturels, les règles donnent des signatures lisibles (sauf pour les "
          "attaques web), et l'autoencodeur est la seule méthode qui ne demande aucune attaque étiquetée."),
        H2("Autres limites"),
        bullets([
            "Classes minuscules (2, 4 et 7 flux de test) : scores très instables.",
            "Clustering, règles, projections et importance travaillent sur des échantillons plafonnés, pas sur les proportions réelles.",
            "Plusieurs hyperparamètres retenus sont en bord de grille ; une grille plus large pourrait mieux faire.",
            "Aucune capture étiquetée d'un autre réseau n'était disponible pour mesurer l'application en conditions réelles.",
            "Le module de captures réimplémente CICFlowMeter et n'en reproduit le défaut des flags que de façon empirique.",
        ]),
        H1("14. Conclusion et perspectives"),
        P(f"Sur CIC-IDS2017, les modèles à base d'arbres reconnaissent les attaques connues avec un F1 macro de 0,91 à 0,92 ; un Random Forest seul suffit. Mais une "
          f"attaque d'un type nouveau n'est détectée qu'une fois sur trois, et une part de la performance repose sur le port et sur un défaut de l'outil d'extraction. "
          f"Les quatre tâches de data mining se complètent : la classification pour le connu, le clustering pour la structure, les règles pour l'explication, "
          f"l'anomalie pour l'inconnu."),
        H2("Perspectives"),
        bullets([
            "Combiner un classifieur (attaques connues) et un autoencodeur (tout ce qui ne ressemble à rien de connu).",
            "Retirer les variables liées au laboratoire (port, fenêtres TCP, flags) et mesurer ce qui reste.",
            "Régénérer les flux avec la version corrigée de CICFlowMeter et réentraîner.",
            "Tester sur un autre jeu (CSE-CIC-IDS2018, UNSW-NB15) ou une capture étiquetée d'un autre réseau.",
            "Élargir les grilles de réglage autour des hyperparamètres en bord de grille.",
        ], numbered=True),
        H1("Références"),
        bullets([
            "I. Sharafaldin, A. H. Lashkari, A. A. Ghorbani, « Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization », ICISSP 2018.",
            "G. Engelen, V. Rimmer, W. Joosen, « Troubleshooting an Intrusion Detection Dataset: the CICIDS2017 Case Study », IEEE Security and Privacy Workshops 2021 (lirias.kuleuven.be/retrieve/713070).",
            "CICFlowMeter corrigé, dépôt github.com/lisa-lthorrold/CICFlowMeter.",
            "J. Han, J. Pei, Y. Yin, « Mining Frequent Patterns without Candidate Generation » (FP-Growth), SIGMOD 2000.",
            "F. T. Liu, K. M. Ting, Z.-H. Zhou, « Isolation Forest », ICDM 2008.",
            "N. V. Chawla et al., « SMOTE: Synthetic Minority Over-sampling Technique », JAIR 2002.",
            "Bibliothèques : scikit-learn, XGBoost, mlxtend, imbalanced-learn, dpkt, Streamlit, ReportLab.",
        ], numbered=True),
    ]

    doc = ReportDoc(str(PDF))
    doc.multiBuild(attach_headings(story))
    print(f"Report: {PDF} ({PDF.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(build())
