#!/usr/bin/env python3
"""
Auswertung der Voruntersuchung V1 (Abbruchkriterien der PSO).

Liest die zusammengeführten Ergebnisse der Reihen tpat, tpat_eps und tpat_s3
aus ~/map_ws/ergebnisse sowie die Protokolle der einzelnen Läufe. Die Werte von
t_pat und epsilon stehen in der Task_ID, die Anzahl der Iterationen und der
Abbruchgrund je PSO-Aufruf in den Protokollzeilen "PSO_STOP ...".

Aufruf:
    python3 auswertung_tpat.py                  # alle vorhandenen Reihen
    python3 auswertung_tpat.py --dir ~/map_ws/ergebnisse tpat

Ausgabe: Tabellen im Terminal, v1_zusammenfassung.csv und v1_latex.txt im
Ergebnisordner.
"""
import argparse
import csv
import math
import re
import statistics as st
from pathlib import Path

try:
    from scipy import stats
except ImportError:              # Tests sind optional
    stats = None

TID_RE = re.compile(r"_S(?P<sc>\d+)_.*_tp(?P<tp>\d+)_eps(?P<eps>[0-9.e+-]+)_r(?P<run>\d+)$")
STOP_RE = re.compile(r"PSO_STOP iterationen=(?P<it>\d+) grund=(?P<grund>\w+)")


def lade_protokoll(pfad):
    """Iterationen und Abbruchgründe aller PSO-Aufrufe eines Laufs."""
    if not pfad.exists():
        return []
    with open(pfad, errors='replace') as fh:
        return [(int(m['it']), m['grund']) for m in STOP_RE.finditer(fh.read())]


def lade_reihe(ordner, reihe):
    datei = ordner / f'{reihe}_gesamt.csv'
    if not datei.exists():
        return []
    laeufe = []
    with open(datei, newline='') as fh:
        for row in csv.DictReader(fh):
            m = TID_RE.search(row['Task_ID'])
            if not m:
                continue
            stops = lade_protokoll(ordner / reihe / 'logs' / f"{row['Task_ID']}.log")
            laeufe.append(dict(
                reihe=reihe, sc=int(m['sc']), tp=int(m['tp']), eps=float(m['eps']),
                status=row['Status'],
                K=float(row['Gewaehltes_k']),
                J=float(row['Finale_Kosten_J']),
                crlb=float(row['cost_crlb']),
                zeit=float(row['Rechenzeit_gesamt_s']),
                aufrufe=len(stops),
                it_mittel=st.mean([s[0] for s in stops]) if stops else math.nan,
                anteil_stag=(sum(s[1] == 'stagnation' for s in stops) / len(stops)) if stops else math.nan,
                anteil_max=(sum(s[1] == 'max_iter' for s in stops) / len(stops)) if stops else math.nan,
            ))
    return laeufe


def mw(werte):
    werte = [w for w in werte if not math.isnan(w)]
    return st.mean(werte) if werte else math.nan


def sd(werte):
    werte = [w for w in werte if not math.isnan(w)]
    return st.stdev(werte) if len(werte) > 1 else math.nan


def zusammenfassen(laeufe):
    gruppen = {}
    for l in laeufe:
        gruppen.setdefault((l['sc'], l['tp'], l['eps']), []).append(l)
    zeilen = []
    for (sc, tp, eps), g in sorted(gruppen.items()):
        ok = [l for l in g if not math.isnan(l['J'])]
        zeilen.append(dict(
            Szenario=sc, t_pat=tp, epsilon=eps, n=len(g),
            zulaessig=sum(l['status'] == 'abbruchkriterium' for l in g),
            K_mittel=mw([l['K'] for l in ok]), K_sd=sd([l['K'] for l in ok]),
            CRLB_mittel=mw([l['crlb'] for l in ok]),
            Cges_mittel=mw([l['J'] for l in ok]), Cges_sd=sd([l['J'] for l in ok]),
            Zeit_mittel_s=mw([l['zeit'] for l in g]),
            Iterationen_mittel=mw([l['it_mittel'] for l in g]),
            Anteil_Stagnation=mw([l['anteil_stag'] for l in g]),
            Anteil_tmax=mw([l['anteil_max'] for l in g]),
        ))
    return zeilen


def komma(x, n=2):
    return '–' if math.isnan(x) else f"{x:.{n}f}".replace('.', ',')


def ausgeben(zeilen, laeufe, ordner):
    kopf = (f"{'Sz':>2} {'t_pat':>5} {'eps':>7} {'n':>3} {'zul':>3} {'K':>5} {'sK':>5} "
            f"{'CRLB':>8} {'C_ges':>8} {'s_C':>6} {'Zeit/s':>7} {'Iter':>5} {'Stag%':>6} {'tmax%':>6}")
    print(kopf)
    print('-' * len(kopf))
    for z in zeilen:
        print(f"{z['Szenario']:>2} {z['t_pat']:>5} {z['epsilon']:>7.0e} {z['n']:>3} {z['zulaessig']:>3} "
              f"{z['K_mittel']:5.2f} {z['K_sd']:5.2f} {z['CRLB_mittel']:8.2f} {z['Cges_mittel']:8.2f} "
              f"{z['Cges_sd']:6.2f} {z['Zeit_mittel_s']:7.1f} {z['Iterationen_mittel']:5.1f} "
              f"{100 * z['Anteil_Stagnation']:6.1f} {100 * z['Anteil_tmax']:6.1f}")

    # Statistische Tests je Szenario über t_pat (nur epsilon = 1e-4)
    if stats:
        print("\nTests über t_pat (epsilon = 1e-4):")
        for sc in sorted({l['sc'] for l in laeufe}):
            gr = {}
            for l in laeufe:
                if l['sc'] == sc and abs(l['eps'] - 1e-4) < 1e-12 and not math.isnan(l['J']):
                    gr.setdefault(l['tp'], []).append(l['J'])
            gr = {k: v for k, v in gr.items() if len(v) > 2}
            if len(gr) > 1:
                kw = stats.kruskal(*gr.values())
                lv = stats.levene(*gr.values())
                print(f"  Szenario {sc}: Kruskal-Wallis p = {kw.pvalue:.3f} (Mittelwerte), "
                      f"Levene p = {lv.pvalue:.3f} (Streuung)")
        eps_gr = {}
        for l in laeufe:
            if l['sc'] == 1 and l['tp'] == 15 and not math.isnan(l['J']):
                eps_gr.setdefault(l['eps'], []).append(l['J'])
        eps_gr = {k: v for k, v in eps_gr.items() if len(v) > 2}
        if len(eps_gr) > 1:
            kw = stats.kruskal(*eps_gr.values())
            print(f"  Szenario 1, t_pat = 15: Kruskal-Wallis über epsilon p = {kw.pvalue:.3f}")
    else:
        print("\n(scipy nicht installiert: keine Tests. Installation: pip install scipy)")

    print("\nHinweis: Die Rechenzeiten entstehen unter paralleler Last und eignen sich nur "
          "für den Vergleich der Stufen untereinander.")

    # Dateien schreiben
    with open(ordner / 'v1_zusammenfassung.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(zeilen[0].keys()))
        w.writeheader()
        w.writerows(zeilen)
    with open(ordner / 'v1_latex.txt', 'w') as fh:
        fh.write("% Szenario & t_pat & epsilon & K & s_K & C_CRLB & C_ges & s_C & Zeit/s & Iterationen & Stagnation %\n")
        for z in zeilen:
            eps_tex = f"$10^{{{int(round(math.log10(z['epsilon'])))}}}$"
            fh.write(f"{z['Szenario']} & {z['t_pat']} & {eps_tex} & {komma(z['K_mittel'])} & {komma(z['K_sd'])} & "
                     f"{komma(z['CRLB_mittel'])} & {komma(z['Cges_mittel'])} & {komma(z['Cges_sd'])} & "
                     f"{komma(z['Zeit_mittel_s'], 1)} & {komma(z['Iterationen_mittel'], 1)} & "
                     f"{komma(100 * z['Anteil_Stagnation'], 0)} \\\\\n")
    print(f"\nGeschrieben: {ordner / 'v1_zusammenfassung.csv'}, {ordner / 'v1_latex.txt'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('reihen', nargs='*', default=['tpat', 'tpat_eps', 'tpat_s3'])
    ap.add_argument('--dir', default=str(Path.home() / 'map_ws' / 'ergebnisse'))
    a = ap.parse_args()
    ordner = Path(a.dir).expanduser()
    laeufe = []
    for r in a.reihen:
        l = lade_reihe(ordner, r)
        print(f"Reihe {r}: {len(l)} Läufe")
        laeufe += l
    if not laeufe:
        print("Keine Ergebnisse gefunden. Wurde run_batch.py bis zum Zusammenführen ausgeführt?")
        return
    ausgeben(zusammenfassen(laeufe), laeufe, ordner)


if __name__ == '__main__':
    main()