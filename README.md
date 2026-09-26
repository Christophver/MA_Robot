# MA_Robot# Formationsplanung mobiler Tracker-Roboter für drohnengestützte Messungen

Dieses Repository enthält die Implementierung zur Masterarbeit. Das Verfahren plant, wo eine Flotte mobiler Bodenroboter mit Trackern stehen muss, damit sie eine Drohne an allen vorgegebenen Messposen um ein Messobjekt erfassen kann. Es wägt dabei die Schätzgüte der Drohnenposen, Sicherheitsabstände zu Hindernissen und zwischen den Robotern sowie den Aufwand für Umpositionierungen gegeneinander ab.

Das Verfahren arbeitet in zwei Ebenen. Die äußere Ebene teilt die Drohnenposen mit K-Means in Gruppen ein und erhöht die Gruppenanzahl schrittweise, bis sich die Gesamtkosten nicht mehr verbessern. Die innere Ebene bestimmt für jede Gruppe mit einer Partikelschwarmoptimierung eine Formation der Roboter. Als Gütemaß dient das A-Kriterium, also die Spur der inversen Fisher-Informationsmatrix (Cramér-Rao-Schranke).

Die Umsetzung erfolgt in ROS 2 Jazzy. Die Ergebnisse lassen sich in RViz darstellen. Eine Physiksimulation (Gazebo) wird nicht verwendet, da das Planungsergebnis nicht von der Fahrdynamik der Roboter abhängt.


## Inhalt

1. [Voraussetzungen](#voraussetzungen)
2. [Aufbau des Repositorys](#aufbau-des-repositorys)
3. [Installation](#installation)
4. [Schnellstart: ein einzelner Planungslauf](#schnellstart-ein-einzelner-planungslauf)
5. [Knoten und Topics](#knoten-und-topics)
6. [Parameter](#parameter)
7. [Eingabe: Drohnenposen](#eingabe-drohnenposen)
8. [Ausgabe: Ergebnisse und finale Lösung](#ausgabe-ergebnisse-und-finale-lösung)
9. [Versuchsreihen mit run_batch.py](#versuchsreihen-mit-run_batchpy)
10. [Auswertung mit auswertung.py](#auswertung-mit-auswertungpy)
11. [Feste Modellwerte im Quelltext](#feste-modellwerte-im-quelltext)
12. [Reproduzierbarkeit](#reproduzierbarkeit)
13. [Bekannte Einschränkungen](#bekannte-einschränkungen)
14. [Fehlersuche](#fehlersuche)


## Voraussetzungen

| Komponente | Version |
|---|---|
| Betriebssystem | Ubuntu 24.04 |
| ROS 2 | Jazzy |
| Python | 3.12 |
| Python-Bibliotheken | NumPy, SciPy, scikit-learn, scikit-image, OpenCV, Matplotlib |

Die Bibliotheken lassen sich über die Paketverwaltung installieren:

```bash
sudo apt install python3-numpy python3-scipy python3-sklearn python3-skimage python3-opencv python3-matplotlib
```


## Aufbau des Repositorys

| Pfad | Inhalt |
|---|---|
| `src/multi_robot_optimizer/multi_robot_optimizer/image_to_ros.py` | Kartenknoten: erzeugt die Belegungskarte (aus einem Luftbild oder einem synthetischen Szenario) und die Drohnenposen |
| `src/multi_robot_optimizer/multi_robot_optimizer/optimizer_node.py` | Optimierungsknoten: Distanzfeld, äußere Ebene (K-Means, Wahl der Gruppenanzahl), Visualisierung, Protokollierung, Export |
| `src/multi_robot_optimizer/multi_robot_optimizer/pso_algorithm.py` | Innere Ebene: Partikelschwarmoptimierung, Messmodell, Straffunktionen; ohne Abhängigkeit von ROS 2 |
| `src/multi_robot_optimizer/config/pso_params.yaml` | Parameter des Optimierungsknotens für den Start über die Launch-Datei |
| `src/multi_robot_optimizer/launch/optimizer.launch.py` | Startet den Optimierungsknoten mit den Parametern aus `pso_params.yaml` |
| `run_batch.py` | Automatisierte Versuchsreihen (parallel, wiederaufnehmbar) |
| `auswertung.py` | Statistische Auswertung der Versuchsreihen |
| `ergebnisse/*_gesamt.csv` | Ergebnisse der Versuchsreihen der Arbeit |


## Installation

Das Repository wird als ROS-2-Arbeitsbereich verwendet. Nach dem Klonen wird das Paket gebaut und die Umgebung geladen:

```bash
git clone <URL-des-Repositorys> ~/map_ws
cd ~/map_ws
colcon build --packages-select multi_robot_optimizer
source install/setup.bash
```

Der Befehl `source install/setup.bash` ist in jedem neuen Terminal nötig. Nach jeder Änderung an Dateien in `src/` muss das Paket neu gebaut werden, da `ros2 run` die installierte Kopie unter `install/` ausführt.


## Schnellstart: ein einzelner Planungslauf

Ein Planungslauf benötigt zwei Knoten. Der Optimierungsknoten wartet auf Karte und Drohnenposen, der Kartenknoten sendet beides. Der Optimierungsknoten muss zuerst laufen.

**Terminal 1 – Optimierungsknoten:**

```bash
cd ~/map_ws && source install/setup.bash
ros2 run multi_robot_optimizer optimizer_node --ros-args \
    -p scenario_name:=1 -p timer_period:=0.5 \
    -p formation_export:=~/map_ws/loesung/finale.csv
```

**Terminal 2 – Kartenknoten (innerhalb von 60 Sekunden starten):**

```bash
cd ~/map_ws && source install/setup.bash
ros2 run multi_robot_optimizer image_to_ros --ros-args -p scenario:=1
```

Der Kartenknoten sendet Karte und Drohnenposen etwa zehn Sekunden lang und beendet sich. Der Optimierungsknoten untersucht danach nacheinander die Gruppenanzahlen K = 1, 2, 3 und beendet sich nach Erfüllung des Abbruchkriteriums. Die finale Lösung steht anschließend in `~/map_ws/loesung/finale.csv`.

**Darstellung in RViz (optional):** RViz mit `rviz2` starten, als *Fixed Frame* `map` eintragen und die Anzeigen `MarkerArray` für die Topics `formation_markers` und `drone_targets_rviz` sowie `Marker` für `debug_grid` hinzufügen.


## Knoten und Topics

| Knoten | Ausführbare Datei | Aufgabe |
|---|---|---|
| `map_publisher_node` | `image_to_ros` | Karte und Drohnenposen erzeugen oder laden und senden |
| `formation_optimizer_node` | `optimizer_node` | Formationen planen, darstellen, protokollieren und exportieren |

| Topic | Nachrichtentyp | Inhalt |
|---|---|---|
| `/map` | `nav_msgs/OccupancyGrid` | Belegungsraster (0,1 m je Zelle) |
| `/drone_targets` | `geometry_msgs/PoseArray` | Drohnenposen (Position, Gierwinkel), Rahmen `map` |
| `formation_markers` | `visualization_msgs/MarkerArray` | Formationen und Gruppen |
| `drone_targets_rviz` | `visualization_msgs/MarkerArray` | Drohnenposen, eingefärbt nach Gruppe |
| `debug_grid` | `visualization_msgs/Marker` | Hindernisse und Distanzfeld zur Kontrolle |

Der Optimierungsknoten übernimmt jeweils die erste empfangene Karte und die ersten empfangenen Drohnenposen. Da beide Knoten Standardnachrichten verwenden, lassen sich Karte oder Drohnenposen auch aus anderen Quellen einspeisen.


## Parameter

Parameter werden beim Start mit `--ros-args -p name:=wert` übergeben. Gleitkommaparameter benötigen einen Dezimalpunkt (`1.0` statt `1`), Listen werden in eckigen Klammern angegeben. Unbekannte oder falsch geschriebene Parameternamen ignoriert ROS 2 ohne Meldung.

### Optimierungsknoten (`optimizer_node`)

| Parameter | Standard | Bedeutung |
|---|---|---|
| `w_move` | `1.0` | Gewicht der Bewegungskosten (µm²/m) |
| `w_obs` | `1.0` | Gewicht der Hinderniskosten |
| `robot_types` | `['A','A','A','A','B','B']` | Typ je Roboter: A = ein Tracker, B = zwei Tracker im Abstand von 0,2 m |
| `start_robot_poses` | 6 Roboter im Raster | Ausgangsaufstellung `[x1, y1, x2, y2, …]` in m; legt die Roboteranzahl fest |
| `d_safe` | `1.0` | Wirkungsbereich der weichen Hindernisstrafe in m |
| `d_crash` | `0.15` | Kollisionsradius gegenüber Hindernissen in m |
| `max_iterations` | `50` | Höchstzahl der PSO-Iterationen je Gruppe |
| `t_patience` | `15` | Iterationen ohne Verbesserung bis zum Abbruch |
| `epsilon` | `0.0001` | Mindestverbesserung der besten Position |
| `inertia_weight` | `0.5` | Trägheitsgewicht der PSO |
| `c1`, `c2` | `1.5` | Beschleunigungskoeffizienten der PSO |
| `timer_period` | `4.0` | Sekunden je untersuchter Gruppenanzahl (nur Wartezeit, ohne Einfluss auf das Ergebnis) |
| `startup_timeout` | `60.0` | Sekunden bis zum Selbstabbruch, falls keine Karte oder Drohnenposen eintreffen |
| `csv_path` | `~/map_ws/pso_evaluation_results.csv` | Ergebnisdatei mit den Kennzahlen je Lauf (wird fortgeschrieben) |
| `formation_export` | leer | Pfad für die finale Lösung; leer = kein Export |
| `task_id` | leer | Kennung des Laufs in der Ergebnisdatei (muss mit einem Buchstaben beginnen) |
| `scenario_name` | `1` | Szenarionummer für die Bezeichnung in der Ergebnisdatei |

Fehlen Einträge in `robot_types`, ergänzt der Knoten Roboter vom Typ A. Die Flotte muss mindestens sechs Tracker umfassen.

### Kartenknoten (`image_to_ros`)

| Parameter | Standard | Bedeutung |
|---|---|---|
| `scenario` | `0` | `0` = Luftbild aus `custom_image_path`; `1` bis `4` = synthetische Szenarien |
| `custom_image_path` | `/home/vboxuser/map_ws/Mein_Luftbild.png` | Luftbild für `scenario:=0` |
| `drohnen_quelle` | `generiert` | Herkunft der Drohnenposen: `generiert`, `datei` oder `extern` |
| `drohnen_datei` | leer | CSV-Datei mit Drohnenposen für `drohnen_quelle:=datei` |
| `drohnen_export` | leer | Speichert die verwendeten Drohnenposen als CSV (Vorlage oder Kontrolle) |

Die synthetischen Szenarien bilden jeweils eine Fläche von 20 m × 20 m ab:

| Szenario | Messobjekt | Weitere Hindernisse | Drohnenposen |
|---|---|---|---|
| 1 | Quadrat 4 m × 4 m | keine | 24 |
| 2 | Quadrat 4 m × 4 m | Wand | 24 |
| 3 | U-Profil 10 m × 8 m | keine | 76 |
| 4 | U-Profil 10 m × 8 m | Wand, zwei diagonale Riegel | 72 |


## Eingabe: Drohnenposen

Standardmäßig erzeugt der Kartenknoten die Drohnenposen selbst: Messsäulen im Abstand von etwa 2,5 m zum Messobjekt mit je vier Höhen (1,25 / 3,75 / 6,25 / 8,75 m). Stammen die Posen aus einer anderen Anwendung, etwa einer Flugplanung, gibt es zwei Wege.

**Aus einer Datei (`drohnen_quelle:=datei`):**

```bash
ros2 run multi_robot_optimizer image_to_ros --ros-args \
    -p scenario:=1 -p drohnen_quelle:=datei -p drohnen_datei:=~/map_ws/drohnenposen.csv
```

Die Datei enthält eine Pose je Zeile im Format `x, y, z[, yaw]`:

```
x,y,z,yaw
3.3850,-3.5550,1.2500,2.3317
3.3850,-3.5550,3.7500,2.3317
6.0,6.0,4.0
```

Die Koordinaten liegen im Kartenrahmen `map` in Metern: Ursprung in der Kartenmitte, x nach rechts, y nach oben. Der Gierwinkel in Radiant ist optional. Fehlt er, richtet sich die Drohne zum Messobjekt aus. Trennzeichen ist das Komma (alternativ Semikolon), Dezimalzeichen der Punkt. Eine Kopfzeile und Kommentare mit `#` sind erlaubt. Die Datei sollte mit einem Texteditor bearbeitet werden. Tabellenprogramme mit deutscher Spracheinstellung speichern Dezimalkommas, die nicht gelesen werden können.

Beim Laden verwirft der Knoten Posen außerhalb der Karte, in Hindernissen oder mit z ≤ 0 und meldet jede einzeln. Die Zeile `--> X von Y Drohnenposen … übernommen` zeigt das Ergebnis. Eine Vorlage erzeugt der Parameter `drohnen_export`:

```bash
ros2 run multi_robot_optimizer image_to_ros --ros-args -p scenario:=1 -p drohnen_export:=~/map_ws/drohnenposen.csv
```

**Von einem anderen ROS-2-Knoten (`drohnen_quelle:=extern`):** Der Kartenknoten sendet dann nur die Karte. Ein anderer Knoten sendet die Posen als `geometry_msgs/PoseArray` im Rahmen `map` auf `/drone_targets`. Er sollte die Nachricht mehrfach senden, da der Optimierungsknoten nur Nachrichten empfängt, die nach seinem Start eintreffen.


## Ausgabe: Ergebnisse und finale Lösung

**Ergebnisdatei (`csv_path`):** Jeder Lauf hängt eine Zeile an. Die wichtigsten Spalten:

| Spalte | Inhalt |
|---|---|
| `Task_ID`, `Messobjekt`, `Status` | Kennung, Szenario, Ende des Laufs (`abbruchkriterium`, `max_k`, `keine_loesung`) |
| `Gewaehltes_k`, `Finale_Kosten_J` | gewählte Gruppenanzahl K* und Gesamtkosten |
| `cost_crlb`, `cost_move`, `cost_obs`, `cost_inter` | Kostenanteile (Schätzgüte in µm², gewichtete Bewegungs- und Hinderniskosten, Roboterabstand) |
| `k_Verlauf`, `Kosten_Verlauf` | untersuchte zulässige Gruppenanzahlen und ihre Gesamtkosten |
| `Rechenzeit_Verlauf_s`, `Rechenzeit_gesamt_s` | Rechenzeit je untersuchter Gruppenanzahl und gesamt |
| `w_move`, `w_obs`, `d_safe`, `d_crash`, `robot_types`, `N_Roboter`, `N_Drohnenposen` | Konfiguration des Laufs |
| `Gruppengroessen`, `Formationen` | Posen je Gruppe und Roboterpositionen je Formation (JSON) |

**Finale Lösung (`formation_export`):** Ist ein Pfad gesetzt, schreibt der Knoten am Ende zwei Dateien. Eine Kommentarzeile nennt jeweils Szenario, K, Gesamtkosten und Gewichte.

`finale.csv` – Roboterpositionen je Formation:

```
gruppe,roboter,typ,x,y,start_x,start_y
1,1,A,-0.6952,-5.8010,0.0000,0.0000
```

`finale_drohnen.csv` – welche Drohnenposen jede Formation erfasst:

```
gruppe,x,y,z,yaw
1,3.3850,-3.5550,1.2500,2.3317
```

Endet ein Lauf ohne zulässige Lösung, entsteht keine Exportdatei.


## Versuchsreihen mit run_batch.py

`run_batch.py` führt ganze Versuchsreihen ohne Eingriff aus. Es startet für jeden Lauf beide Knoten, wartet auf das Ergebnis und fasst am Ende alle Läufe einer Reihe zusammen.

```bash
cd ~/map_ws && source install/setup.bash
python3 run_batch.py test                               # je Szenario ein Lauf
python3 run_batch.py haupt --dry-run                    # nur anzeigen, was laufen würde
nohup python3 run_batch.py haupt typen flotte zeit --workers 3 > lauf.log 2>&1 &
```

Mit `nohup` laufen die Reihen weiter, wenn das Terminal geschlossen wird. Den Fortschritt zeigt `tail -f lauf.log`, abgebrochen wird mit `pkill -INT -f run_batch.py`.

**Vordefinierte Reihen** (in `EXPERIMENTS` im Skript anpassbar):

| Reihe | Zweck | Szenario | Variation | Läufe |
|---|---|---|---|---|
| `test` | Funktionstest | 1 bis 4 | – | 1 |
| `wmove` | Kontrolle von `w_move` | 1 | 0,63 / 1,0 / 1,6 / 2,5 / 4,0 | 30 |
| `wobs` | Kontrolle von `w_obs` | 3 | 0,1 / 1,0 / 10 | 30 |
| `haupt` | Hauptauswertung | 1 bis 4 | – | 30 |
| `flotte` | Roboteranzahl | 1 | 4 / 6 / 8 Roboter | 30 |
| `typen` | Zusammensetzung bei 8 Trackern | 1 | 8A bis 4B | 30 |
| `zeit` | Rechenzeit ohne parallele Last | 1 | 4 / 6 / 8 Roboter | 5 |

Eine Reihe mit eigenen Drohnenposen erhält den Eintrag `drohnen_datei='~/map_ws/drohnenposen.csv'`. Ein auskommentiertes Beispiel steht im Skript.

**Arbeitsweise:**

Die Option `--workers` legt fest, wie viele Läufe gleichzeitig laufen. Jeder Strang erhält eine eigene `ROS_DOMAIN_ID` (ab 40) und eine eigene CSV-Datei. Als Faustregel gilt: Anzahl der Prozessorkerne minus eins. Die Reihe `zeit` läuft immer mit einem Strang, damit parallele Läufe die Rechenzeitmessung nicht verfälschen.

Bereits abgeschlossene Läufe erkennt das Skript an ihrer `Task_ID` und überspringt sie. Nach einem Abbruch setzt ein erneuter Start daher an der Stelle fort. Fehlgeschlagene Läufe wiederholt es einmal. Eine Sperre verhindert, dass zwei Instanzen gleichzeitig laufen.

Werden Code oder Gewichte geändert, muss der Ordner der betroffenen Reihe vorher verschoben oder gelöscht werden. Sonst übernimmt das Skript alte Läufe als erledigt.

**Ablage:**

| Pfad | Inhalt |
|---|---|
| `ergebnisse/<reihe>_gesamt.csv` | alle Läufe einer Reihe |
| `ergebnisse/<reihe>/worker_N.csv` | Rohdaten je Strang |
| `ergebnisse/<reihe>/logs/<Task_ID>.log` | Ausgabe beider Knoten je Lauf |
| `ergebnisse/<reihe>/fehler.txt` | fehlgeschlagene Läufe (Timeout, kein CSV-Eintrag) |


## Auswertung mit auswertung.py

Das Skript fasst eine Gesamtdatei nach Szenario und Flotte zusammen. Es berechnet Erfolgsquote, Mittelwert und Streuung der Gruppenanzahl, die Kostenanteile, die Roboterabstände, die Abstände zu Hindernissen und die Rechenzeit je untersuchter Gruppenanzahl.

```bash
cd ~/map_ws && source install/setup.bash && cd ergebnisse
python3 ~/map_ws/auswertung.py haupt_gesamt.csv
for r in haupt typen flotte zeit; do python3 ~/map_ws/auswertung.py ${r}_gesamt.csv | tee auswertung_${r}.txt; done
```

Für die Abstände zu Hindernissen erzeugt das Skript die Karten mit dem Code des installierten Pakets neu, ohne ROS 2 zu starten. Deshalb muss vorher `source install/setup.bash` ausgeführt werden.


## Feste Modellwerte im Quelltext

Einige Werte sind keine ROS-Parameter, sondern stehen als Konstanten im Quelltext:

| Wert | Datei | Bedeutung |
|---|---|---|
| σ_pos = 0,2 µm + 0,3 µm/m · d | `pso_algorithm.py` | Abstandsunsicherheit der Tracker |
| d_max = 30 m | `pso_algorithm.py` | Messreichweite |
| σ_ang = 0,05 rad + 0,01 rad/m · d | `pso_algorithm.py` | angenommene Winkelunsicherheit |
| d_min = 1,5 m, Kollision bei 0,6 m | `pso_algorithm.py` | Abstände zwischen Robotern |
| Strafwerte 10⁸ / 10⁷ / 10⁹ / 10⁶ | `pso_algorithm.py` | nicht beobachtbare Pose / je ungültigem Tracker / numerischer Fehler / Kollision |
| v_max = 2 m, Startbereich ± 15 m | `pso_algorithm.py` | Geschwindigkeitsgrenze und Initialisierung der PSO |
| Populationsgröße max(30, 10 · 2N) | `pso_algorithm.py` | Anzahl der Partikel |
| 0,1 m je Zelle, Arbeitsabstand 2,5 m | `image_to_ros.py` | Kartenauflösung und Abstand der Drohnen zum Messobjekt |
| K-Means mit festem Startwert, 10 Wiederholungen | `optimizer_node.py` | reproduzierbare Gruppierung |

Die Positionsvarianzen und das A-Kriterium haben die Einheit µm².


## Reproduzierbarkeit

Der Codestand, mit dem die Ergebnisse der Arbeit entstanden, ist mit dem Tag `versuche-final` markiert:

```bash
git checkout versuche-final
```

Die Gruppierung ist durch den festen Startwert von K-Means in jedem Lauf gleich. Die Partikelschwarmoptimierung ist stochastisch. Jeder Lauf startet einen neuen Prozess, die Läufe sind daher unabhängige Stichproben. Die Arbeit wertet deshalb 30 Läufe je Konfiguration aus.


## Bekannte Einschränkungen

Das Messmodell bewertet im Wesentlichen die Position eines Referenzpunkts der Drohne. Die Orientierung geht nur über angenommene Winkelunsicherheiten ein. Die Sichtlinienprüfung behandelt Hindernisse als unendlich hoch. Die Bewegungskosten messen den geradlinigen Abstand von der Ausgangsaufstellung und summieren die Wege aller Roboter.

Die weichen Sicherheitsabstände (`d_safe`, `d_min`) werden in verbauten Umgebungen häufig unterschritten. Garantiert sind nur die Kollisionsgrenzen. Die beiden Kollisionsgrenzen gehen von unterschiedlichen Roboterradien aus (0,15 m gegenüber Hindernissen, 0,3 m zwischen Robotern) und sollten für einen realen Einsatz an die tatsächlichen Abmessungen angepasst werden.

Die Erprobung erfolgte kinematisch mit synthetischen Karten. Die Rechenzeit wächst etwa quadratisch mit dem Umfang des Messobjekts.


## Fehlersuche

**Der Optimierungsknoten beendet sich nach einer Minute mit „Keine Daten nach Startfrist“.** Der Kartenknoten wurde zu spät oder gar nicht gestartet. Beide Knoten müssen im selben Netzwerk und mit derselben `ROS_DOMAIN_ID` laufen.

**Eine Änderung im Code wirkt nicht.** Nach Änderungen in `src/` muss neu gebaut werden: `colcon build --packages-select multi_robot_optimizer && source install/setup.bash`. Ob eine bestimmte Änderung installiert ist, zeigt:

```bash
python3 -c "import inspect, multi_robot_optimizer.image_to_ros as m; print('drohnen_quelle' in inspect.getsource(m))"
```

**Ein Parameter wirkt nicht.** Der Name ist vermutlich falsch geschrieben. ROS 2 ignoriert unbekannte Parameter ohne Meldung. Bei Gleitkommaparametern muss der Wert einen Dezimalpunkt enthalten.

**Die Drohnenposen aus der Datei werden nicht übernommen.** Die Ausgabe des Kartenknotens enthält in diesem Fall „Generiere … Dummy-Drohnen“ statt „… Drohnenposen aus … übernommen“. Dann läuft der Knoten im Modus `generiert`. Außerdem prüfen, ob die Datei gespeichert ist (`tail ~/map_ws/drohnenposen.csv`).

**`run_batch.py` meldet „ABBRUCH: Es läuft bereits eine Instanz“.** Eine andere Instanz läuft noch. Anzeigen mit `pgrep -af run_batch.py`, beenden mit `pkill -INT -f run_batch.py`.

**Eine Versuchsreihe rechnet nichts und meldet alle Läufe als vorhanden.** Im Ordner der Reihe liegen noch Ergebnisse eines früheren Durchgangs. Den Ordner verschieben, dann neu starten.

**Die Rechenzeiten unterscheiden sich stark zwischen Reihen.** Parallele Läufe teilen sich die Prozessorkerne. Vergleichbare Rechenzeiten liefert nur die Reihe `zeit`.