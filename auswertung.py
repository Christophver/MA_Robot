#!/usr/bin/env python3
"""
Auswertung der Versuchsreihen für Kapitel 8.

Aufruf (im Ordner ~/map_ws/ergebnisse):
    python3 ~/map_ws/auswertung.py haupt_gesamt.csv
    python3 ~/map_ws/auswertung.py typen_gesamt.csv
    python3 ~/map_ws/auswertung.py flotte_gesamt.csv
    python3 ~/map_ws/auswertung.py zeit_gesamt.csv

Gruppiert automatisch nach Szenario und Flotte (Teil der Task_ID) und gibt je
Gruppe aus: Anzahl, Erfolgsquote, K (Mittel, Std., Verteilung), Kostenanteile,
C_inter, kleinster Roboterabstand und Rechenzeit.
"""
import csv, json, math, sys, statistics as st
from collections import Counter, defaultdict

def mstd(v):
    if not v:
        return float('nan'), float('nan')
    return st.mean(v), (st.stdev(v) if len(v) > 1 else 0.0)

def min_robot_distance(formations):
    """Kleinster Abstand zweier Roboter innerhalb derselben Formation (m)."""
    dmin = float('inf')
    for f in formations:
        pts = [(f[i], f[i + 1]) for i in range(0, len(f), 2)]
        for a in range(len(pts)):
            for b in range(a + 1, len(pts)):
                dmin = min(dmin, math.dist(pts[a], pts[b]))
    return dmin

def main(path):
    rows = list(csv.DictReader(open(path, newline='')))
    groups = defaultdict(list)
    for r in rows:
        teile = r['Task_ID'].split('_')          # reihe_S1_N6_wm1.0_wo1.0_r01
        groups[(teile[1], teile[2], teile[3], teile[4])].append(r)

    print(f"Datei: {path}  ({len(rows)} Zeilen)\n")
    for key in sorted(groups):
        g = groups[key]
        ok = [r for r in g if r['Status'] == 'abbruchkriterium']
        if not ok:
            print(f"== {' '.join(key)}  | Läufe {len(g)}, kein Lauf mit Abbruchkriterium: "
                  f"{dict(Counter(r['Status'] for r in g))}\n")
            continue
        k = [int(r['Gewaehltes_k']) for r in ok]
        f = lambda s: [float(r[s]) for r in ok]
        crlb, move, obs, inter, ges, zeit = map(f, ['cost_crlb', 'cost_move', 'cost_obs',
                                                    'cost_inter', 'Finale_Kosten_J',
                                                    'Rechenzeit_gesamt_s'])
        dmins = [min_robot_distance(json.loads(r['Formationen'])) for r in ok]
        km, ks = mstd(k)
        print(f"== {' '.join(key)}  | Läufe {len(g)}, davon Abbruchkriterium {len(ok)} "
              f"({100 * len(ok) / len(g):.0f} %), Status: {dict(Counter(r['Status'] for r in g))}")
        print(f"   K*: Mittel {km:.2f}, Std {ks:.2f}, Verteilung {dict(sorted(Counter(k).items()))}")
        for name, v in [('C_ges', ges), ('C_CRLB [µm²]', crlb), ('w_move*C_move', move),
                        ('w_obs*C_obs', obs), ('C_inter', inter), ('Rechenzeit [s]', zeit)]:
            m, s = mstd(v)
            print(f"   {name:<15} Mittel {m:10.2f}  Std {s:9.2f}  Min {min(v):10.2f}  Max {max(v):10.2f}")
        anteil = [c / j * 100 for c, j in zip(crlb, ges) if j]
        print(f"   Anteil CRLB an C_ges: {st.mean(anteil):.1f} %")
        print(f"   C_inter = 0 in {sum(1 for v in inter if v == 0)} von {len(inter)} Läufen; "
              f"kleinster Roboterabstand {min(dmins):.2f} m (Mittel der Minima {st.mean(dmins):.2f} m)")
        print()

if __name__ == '__main__':
    main(sys.argv[1])