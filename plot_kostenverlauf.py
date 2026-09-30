#!/usr/bin/env python3
"""
Kostenverlauf über der Gruppenanzahl K für die Präsentation.

Liest die Hauptauswertung (haupt_gesamt.csv), bildet je Szenario und je K den
Mittelwert der Gesamtkosten über alle Läufe, die dieses K untersucht haben, und
zeichnet das 95-%-Konfidenzintervall des Mittelwerts als Band
(Mittelwert ± t · s / Wurzel(n), t-Verteilung mit n − 1 Freiheitsgraden). Unter jedem Punkt steht die Anzahl
der Läufe, die zu diesem K beitragen.

Aufruf:
    python3 plot_kostenverlauf.py                       # Szenario 1 bis 4
    python3 plot_kostenverlauf.py --szenario 3
    python3 plot_kostenverlauf.py --csv ~/map_ws/ergebnisse/haupt_gesamt.csv --out ~/map_ws/abbildungen
"""
import argparse
import ast
import csv
import math
import re
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator

BLAU = "#00509B"        # LUH-Blau, wie in der Präsentation
GRUEN = "#AFCA00"       # match-Grün
GRAU = "#6E6E6E"
NAMEN = {1: "Quadrat", 2: "Quadrat mit Wand", 3: "U-Profil", 4: "U-Profil, verbaut"}
UNZULAESSIG = 1e7       # Kosten ab hier stammen aus Strafen, nicht aus der Schätzgüte

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size": 16, "axes.labelsize": 16, "axes.titlesize": 18,
    "xtick.labelsize": 14, "ytick.labelsize": 14,
    "axes.spines.top": False, "axes.spines.right": False,
})


# 97,5-%-Quantile der t-Verteilung für 1 bis 30 Freiheitsgrade
T975 = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228,
        2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086,
        2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042]


def t_quantil(df):
    try:
        from scipy.stats import t
        return float(t.ppf(0.975, df))
    except ImportError:
        if df <= 30:
            return T975[df - 1]
        return 1.96 + 2.4 / df          # Näherung, Abweichung unter 0,5 %


def ki_halbbreite(werte):
    n = len(werte)
    if n < 2:
        return 0.0
    return t_quantil(n - 1) * st.stdev(werte) / math.sqrt(n)


def komma(x, _pos=None):
    return f"{x:g}".replace(".", ",")


def lade(csv_pfad):
    """Liefert {Szenario: {K: [Kosten je Lauf]}} und {Szenario: [K* je Lauf]}."""
    kosten, kstern = {}, {}
    with open(csv_pfad, newline="") as fh:
        for row in csv.DictReader(fh):
            m = re.search(r"_S(\d+)_", row["Task_ID"])
            if not m or row.get("Status") == "keine_loesung":
                continue
            sc = int(m.group(1))
            try:
                ks = ast.literal_eval(row["k_Verlauf"])
                cs = ast.literal_eval(row["Kosten_Verlauf"])
            except (ValueError, SyntaxError):
                continue
            for k, c in zip(ks, cs):
                if c is None or not math.isfinite(float(c)) or float(c) >= UNZULAESSIG:
                    continue            # unzulässiges K: nicht in den Mittelwert
                kosten.setdefault(sc, {}).setdefault(int(k), []).append(float(c))
            kstern.setdefault(sc, []).append(int(float(row["Gewaehltes_k"])))
    return kosten, kstern


def zeichne(sc, daten, kstern, out):
    ks = sorted(daten)
    mw = [st.mean(daten[k]) for k in ks]
    hb = [ki_halbbreite(daten[k]) for k in ks]
    n = [len(daten[k]) for k in ks]

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    unten = [m - h for m, h in zip(mw, hb)]
    oben = [m + h for m, h in zip(mw, hb)]
    ax.fill_between(ks, unten, oben, color=BLAU, alpha=0.30, linewidth=0,
                    label="95-%-Konfidenzintervall")
    ax.plot(ks, mw, color=BLAU, linewidth=2.5, marker="o", markersize=8,
            markeredgecolor="white", label="Mittelwert")

    # häufigste gewählte Gruppenanzahl hervorheben
    if kstern:
        k_haeufig = max(set(kstern), key=kstern.count)
        if k_haeufig in ks:
            i = ks.index(k_haeufig)
            ax.plot([k_haeufig], [mw[i]], marker="o", markersize=14, color=GRUEN,
                    markeredgecolor=BLAU, zorder=5,
                    label=f"häufigste Wahl K* = {k_haeufig}")

    # Anzahl der Läufe je K unter den Punkten
    y0, y1 = min(unten), max(oben)
    spanne = y1 - y0 if y1 > y0 else 1.0
    ax.set_ylim(y0 - 0.16 * spanne, y1 + 0.08 * spanne)
    ax.set_xlim(ks[0] - 0.4, ks[-1] + 0.4)
    for k, cnt in zip(ks, n):
        ax.text(k, y0 - 0.13 * spanne, f"n = {cnt}", ha="center", va="bottom",
                fontsize=12, color=GRAU)

    ax.set_xlabel("Gruppenanzahl K")
    ax.set_ylabel("Gesamtkosten C$_{\\mathrm{ges}}$")
    ax.set_title(f"Szenario {sc}: {NAMEN.get(sc, '')}")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_formatter(FuncFormatter(komma))
    ax.grid(axis="y", color="#D8D8D8", linewidth=0.8)
    ax.legend(frameon=False, fontsize=13, loc="upper right")
    fig.tight_layout()

    out.mkdir(parents=True, exist_ok=True)
    for endung in ("png", "pdf"):
        fig.savefig(out / f"kostenverlauf_S{sc}.{endung}", dpi=300)
    plt.close(fig)

    print(f"Szenario {sc}:")
    for k, m, h, cnt in zip(ks, mw, hb, n):
        print(f"  K = {k}: {komma(round(m, 1))}, 95-%-KI ± {komma(round(h, 1))}  (n = {cnt})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(Path.home() / "map_ws" / "ergebnisse" / "haupt_gesamt.csv"))
    ap.add_argument("--szenario", type=int, nargs="*", default=[1, 2, 3, 4])
    ap.add_argument("--out", default=str(Path.home() / "map_ws" / "ergebnisse" / "abbildungen"))
    a = ap.parse_args()
    kosten, kstern = lade(Path(a.csv).expanduser())
    for sc in a.szenario:
        if sc in kosten:
            zeichne(sc, kosten[sc], kstern.get(sc, []), Path(a.out).expanduser())
        else:
            print(f"Szenario {sc}: keine Daten gefunden")
    print(f"Abbildungen in {Path(a.out).expanduser()}")


if __name__ == "__main__":
    main()