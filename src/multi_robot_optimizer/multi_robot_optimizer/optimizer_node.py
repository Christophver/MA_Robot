import rclpy
from rclpy.node import Node
import numpy as np
import math
from visualization_msgs.msg import Marker, MarkerArray
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import time

from multi_robot_optimizer.pso_algorithm import SwarmOptimizer
from visualization_msgs.msg import Marker, MarkerArray

class OptimizerNode(Node):
    def __init__(self):
        super().__init__('formation_optimizer_node')

        # --- HIER: DIE DEFINITION DER HINDERNISSE ---
        self.target_obstacles = [[0.0, 0.0, 5.0, 20.0, 15.0, 10.0]]
        self.avoidance_obstacles = [[15.0, 12.0, 5.0, 5.0, 5.0, 10.0]]
        
        # Falls du alte Funktionen hast, die zwingend 'obstacle_coords' brauchen,
        # kombiniere sie hier einmalig:
        self.obstacle_coords = self.target_obstacles + self.avoidance_obstacles
        
        # ==========================================
        # Schritt 1: Dateneingabe
        # ==========================================
        # Vorhandene Parameter
        self.declare_parameter('w_crlb', 1.0)
        self.declare_parameter('w_move', 0.1)
        self.declare_parameter('w_obs', 0.5)
        self.declare_parameter('max_drone_dist', 30.0)
        self.declare_parameter('robot_types', ['A', 'A', 'B', 'B'])
        
        # NEU: Die Hyperparameter und Constraints aus der YAML anmelden
        self.declare_parameter('max_iterations', 50)
        self.declare_parameter('c1', 1.5)
        self.declare_parameter('c2', 1.5)
        self.declare_parameter('inertia_weight', 0.5)
        self.declare_parameter('d_safe', 2.0)
        self.declare_parameter('d_crash', 0.15) # Auf 30cm Roboter (Radius 15cm) angepasst
        
        params = {
            'w_crlb': self.get_parameter('w_crlb').value,
            'w_move': self.get_parameter('w_move').value,
            'w_obs': self.get_parameter('w_obs').value,
            'max_drone_dist': self.get_parameter('max_drone_dist').value,
            'robot_types': self.get_parameter('robot_types').value,
            
            # NEU: Werte ins Dictionary packen
            'max_iterations': self.get_parameter('max_iterations').value,
            'c1': self.get_parameter('c1').value,
            'c2': self.get_parameter('c2').value,
            'inertia_weight': self.get_parameter('inertia_weight').value,
            'd_safe': self.get_parameter('d_safe').value,
            'd_crash': self.get_parameter('d_crash').value,
        }
        
        self.optimizer = SwarmOptimizer(params)
        self.marker_pub = self.create_publisher(MarkerArray, 'formation_markers', 10)
        self.obstacle_pub = self.create_publisher(MarkerArray, 'obstacle_markers', 10)

        # Feste Farbpalette für bis zu 20 Cluster
        # Basierend auf Matplotlibs 'tab20' (Wissenschaftlicher Standard)
        # (R, G, B) im Bereich 0.0 bis 1.0
        # ==========================================
        self.cluster_colors = [
            (0.122, 0.467, 0.706),  # Cluster 0: Dunkelblau
            (0.682, 0.780, 0.910),  # Cluster 1: Hellblau
            (1.000, 0.498, 0.055),  # Cluster 2: Orange
            (1.000, 0.733, 0.471),  # Cluster 3: Hellorange
            (0.173, 0.627, 0.173),  # Cluster 4: Grün
            (0.596, 0.875, 0.541),  # Cluster 5: Hellgrün
            (0.839, 0.153, 0.157),  # Cluster 6: Rot
            (1.000, 0.596, 0.588),  # Cluster 7: Hellrot
            (0.580, 0.404, 0.741),  # Cluster 8: Lila
            (0.773, 0.690, 0.835),  # Cluster 9: Helllila
            (0.549, 0.337, 0.294),  # Cluster 10: Braun
            (0.769, 0.612, 0.580),  # Cluster 11: Hellbraun
            (0.890, 0.467, 0.761),  # Cluster 12: Pink
            (0.969, 0.714, 0.824),  # Cluster 13: Hellpink
            (0.498, 0.498, 0.498),  # Cluster 14: Grau
            (0.780, 0.780, 0.780),  # Cluster 15: Hellgrau
            (0.737, 0.741, 0.133),  # Cluster 16: Olivgrün
            (0.859, 0.859, 0.553),  # Cluster 17: Hellolivgrün
            (0.090, 0.745, 0.812),  # Cluster 18: Cyan
            (0.620, 0.855, 0.898),  # Cluster 19: Hellcyan
        ]

        # Startpositionen der Roboter (x1, y1, x2, y2, x3, y3)
        # Startpositionen für 6 Roboter (x1, y1, x2, y2, ...)
        self.start_robot_poses = [
            0.0, 0.0,  # Roboter 1
            1.0, 0.0,  # Roboter 2
            2.0, 0.0,  # Roboter 3
            0.0, 1.0,  # Roboter 4
            1.0, 1.0,  # Roboter 5
            2.0, 1.0   # Roboter 6
        ]
        
        
        # NEU: Automatisierte 3D-Grid-Generierung passend zum Gebäude
        # 1 Drohne pro 1 qm Wandfläche, genau 1m Abstand zur Wand
        generated_drones_3d = []
        
        
        # Wand-Dimensionen des Gebäudes: Länge (X)=20m, Breite (Y)=15m, Höhe (Z)=10m
        # Das Gebäude zentriert sich von X=[-10, 10], Y=[-7.5, 7.5], Z=[0, 10]
        
        # A) Kurze Wände (Vorne & Hinten bei X = -11.0 und +11.0 wegen 1m Abstand)
        # Y-Spanne: 15m Breite -> 15 Punkte im 1m-Abstand
        # Z-Spanne: 10m Höhe -> 10 Punkte im 1m-Abstand (0.5m bis 9.5m, keine Drohnen über 10m)
        y_coords = np.linspace(-7.0, 7.0, 5)
        z_coords = np.linspace(0.5, 9.5, 2)
        
        for x in [-11.0, 11.0]:
            for y in y_coords:
                for z in z_coords:
                    generated_drones_3d.append([float(x), float(y), float(z)])
                    
        # B) Lange Wände (Links & Rechts bei Y = -8.5 und +8.5 wegen 1m Abstand)
        # X-Spanne: 20m Länge -> 20 Punkte im 1m-Abstand
        x_coords = np.linspace(-9.5, 9.5, 7)
        
        for y in [-8.5, 8.5]:
            for x in x_coords:
                for z in z_coords:
                    generated_drones_3d.append([float(x), float(y), float(z)])

        # Logge die exakte Anzahl im ROS 2 Terminal (erzeugt mathematisch exakt 700 Punkte)
        self.get_logger().info(f"Test-Szenario generiert: {len(generated_drones_3d)} Drohnen-Messpunkte erzeugt.")
        
        # Umwandlung in 6D-Posen über deine bestehende Methode
        drones_6d = []
        main_obstacle = self.obstacle_coords[0] 
        
        for pos in generated_drones_3d:
            pose = self.calculate_drone_pose_facing_wall(pos, main_obstacle)
            drones_6d.append(pose)
            
        # self.all_drones enthält jetzt die vollständige 6D-Matrix [x, y, z, roll, pitch, yaw]
        self.all_drones = np.array(drones_6d)
        
        # ==========================================
        # Schritt 2: Initialisierung
        # ==========================================
        self.current_k = 1

        self.optimization_done = False # Neuer Kontroll-Zustand

        # --- Parameter Fallback ---
        # Falls ROS 2 die pso_params.yaml nicht findet, nutzen wir diese Standardwerte.
        # Hier kannst du die Gewichtung deiner Multi-Kriterien-Optimierung direkt steuern!
        self.params = {
            'w_crlb': 0.4,   # 40% Fokus auf Sensordaten-Qualität
            'w_obs': 0.3,    # 30% Fokus auf Hindernisvermeidung
            'w_inter': 0.2,  # 20% Fokus auf Formationsabstand (keine Kollisionen)
            'w_move': 0.1    # 10% Fokus auf kurze Fahrtwege
        }

        self.k_history = []
        self.total_cost_history = []
        self.cluster_costs_history = {} # Speichert die Einzelkosten pro k

        self.previous_total_cost = float('inf')
        self.best_marker_array = None # Speicher für die beste RViz-Ausgabe (k-1)
        
        
        # Starte die hochpräzise Stoppuhr
        start_time = time.perf_counter()
        
        # Timer startet den Loop (alle 4 Sekunden ein neuer Schritt k)
        self.timer = self.create_timer(4.0, self.evaluate_next_k)
        self.get_logger().info("Schritt 1 & 2: Dateneingabe abgeschlossen, k=1 initialisiert.")

    def evaluate_next_k(self):

        # 1. NEU: Hindernisse IMMER und SOFORT publizieren!
        # Egal ob der Algorithmus rechnet, scheitert oder fertig ist:
        self.publish_obstacles()

        # 1. UHR STARTEN: Genau hier, bevor die schwere Mathematik beginnt!
        start_time = time.perf_counter()

        # Wenn wir fertig sind, optimieren wir nicht mehr, sondern halten nur RViz am Leben!
        if self.optimization_done:
            if self.best_marker_array is not None:
                self.marker_pub.publish(self.best_marker_array)
            return
        
        if self.current_k > len(self.all_drones):
            self.get_logger().info("Maximale Cluster-Anzahl erreicht. Abbruch.")
            self.timer.cancel()
            return

        self.get_logger().info(f"\n------------------------------------------------")
        self.get_logger().info(f"Starte Evaluierung für k = {self.current_k}")
        
        # ==========================================
        # Schritt 3: Clusterbildung
        # ==========================================
        self.get_logger().info("Schritt 3: Aufteilung der Drohnenpunkte in k Gruppen...")
        clusters = []
        if self.current_k == 1:
            clusters.append(self.all_drones.tolist())
        else:
            kmeans = KMeans(n_clusters=self.current_k, random_state=42, n_init=10).fit(self.all_drones)
            for i in range(self.current_k):
                clusters.append(self.all_drones[kmeans.labels_ == i].tolist())
                
        # ==========================================
        # Schritt 4 & 5: Positions-Optimierung & Gesamtkosten
        # ==========================================
        self.get_logger().info("Schritt 4 & 5: PSO und Berechnung der Fahrtkosten (C_move)...")
        current_total_cost = 0.0
        formations = []
        current_cluster_costs = [] 
        valid_solution = True
        
        w_move = self.params.get('w_move', 0.1)
        
        # 1. NEU HIER EINBAUEN: Beide Hindernis-Listen kombinieren
        all_physical_obstacles = self.target_obstacles + self.avoidance_obstacles
        
        for idx, cluster_drones in enumerate(clusters):
            # 1. INNERE SCHLEIFE (PSO)
            # 2. HIER ANPASSEN: all_physical_obstacles statt self.obstacle_coords übergeben
            best_form, pso_cost = self.optimizer.run_pso(cluster_drones, self.start_robot_poses, all_physical_obstacles)
            
            # NEUE LOGIK: Wenn die Kosten explodieren, ist diese Cluster-Aufteilung physikalisch unmöglich
            if best_form is None or pso_cost >= 1e8:
                self.get_logger().warn(f"PSO für Cluster {idx+1} gescheitert (Sichtlinie blockiert oder Singularität).")
                valid_solution = False
                break # Bricht nur die Cluster-Berechnung für dieses k ab
                
            # 2. ÄUSSERE SCHLEIFE: Bewegungskomponente (C_move) berechnen
            c_move_raw = 0.0
            num_robots_in_form = len(best_form) // 2
            
            for i in range(num_robots_in_form):
                sx = self.start_robot_poses[i*2]
                sy = self.start_robot_poses[i*2+1]
                tx = best_form[i*2]
                ty = best_form[i*2+1]
                c_move_raw += math.hypot(tx - sx, ty - sy)
                
            # 3. NORMIERUNG VON C_move
            max_travel_dist = num_robots_in_form * 50.0 
            norm_move = min(c_move_raw / max_travel_dist, 1.0) if max_travel_dist > 0 else 0.0
            
            # 4. GESAMTKOSTEN ADDIEREN
            cluster_total_cost = pso_cost + (w_move * norm_move)
            current_total_cost += cluster_total_cost
            formations.append(best_form)
            current_cluster_costs.append(cluster_total_cost)
            
        # ⏱️ Timer Stoppen (egal ob gültig oder nicht)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        self.get_logger().info(f"⏱️ Evaluierung für k={self.current_k} abgeschlossen in {elapsed_time:.3f} Sekunden.")

        # --- DAS SICHERHEITSNETZ ---
        if not valid_solution:
            self.get_logger().warn(f"--> k={self.current_k} ist physikalisch ungültig. Erhöhe k erzwungenermaßen.")
            self.previous_total_cost = float('inf') # Setze Historie auf Unendlich, damit das nächste k als "besser" gilt
            self.current_k += 1
            return # Raus hier! Der ROS-Timer ruft die Funktion gleich automatisch für k+1 auf
        
        # ==========================================
        # Schritt 5: Berechnung der Gesamtkosten
        # ==========================================
        self.get_logger().info(f"Schritt 5: Gesamtkosten J_total = {current_total_cost:.4f}")

        self.k_history.append(self.current_k)
        self.total_cost_history.append(current_total_cost)
        self.cluster_costs_history[self.current_k] = current_cluster_costs
        
        # ==========================================
        # Schritt 6: Entscheidung
        # ==========================================
        self.get_logger().info("Schritt 6: Entscheidung - C_total höher als bei k-1?")
        
        if self.current_k > 1 and current_total_cost > self.previous_total_cost:
            self.get_logger().info("--> JA (Stopp)")
            self.get_logger().info(f"    (Aktuell: {current_total_cost:.4f} > Vorher: {self.previous_total_cost:.4f})")
            
            self.get_logger().info(f"\n+++ ERGEBNIS +++")
            self.get_logger().info(f"Beste Formation ermittelt für k = {self.current_k - 1} Cluster.")
            
            self.optimization_done = True 
            self.plot_results() 
            return

        self.get_logger().info("--> NEIN")
        self.get_logger().info(f"    Erhöhe k = k + 1 (Gehe in der nächsten Runde auf {self.current_k + 1})")
        
        # --- RViz-Visualisierung für den aktuellen, gültigen Schritt aufbauen ---
        marker_array = MarkerArray()

        if self.best_marker_array is not None:
            self.marker_pub.publish(self.best_marker_array)
                
        self.publish_obstacles()
        
        # Cluster räumlich nach ihrem Winkel sortieren (stabile Farben)
        sorted_clusters = []
        for c_drones, form in zip(clusters, formations):
            center = np.mean(c_drones, axis=0)
            angle = np.arctan2(center[1] - 5.0, center[0] - 5.0)
            sorted_clusters.append((angle, c_drones, form))
            
        sorted_clusters.sort(key=lambda item: item[0])

        for idx, (_, cluster_drones, form) in enumerate(sorted_clusters):
            # 1. Hole die einzelne (R, G, B) Farbe für dieses Cluster
            c_color = self.cluster_colors[idx % len(self.cluster_colors)]
            
            # 2. Übergebe diese Farbe sowohl an drone_color als auch an robot_color
            self.add_cluster_markers(marker_array, cluster_drones, form, 
                                     drone_color=c_color, robot_color=c_color, base_id=(idx+1)*100)
        
        self.best_marker_array = marker_array 
        self.marker_pub.publish(marker_array)

        # Werte für den nächsten Iterationsschritt überschreiben
        self.previous_total_cost = current_total_cost
        self.current_k += 1

        
    # --- Hilfsfunktionen für RViz ---
    def publish_obstacles(self):
        marker_array = MarkerArray()
        
        # Hilfsfunktion für die Marker-Generierung
        def create_markers(obs_list, r, g, b):
            for obs in obs_list:
                marker = Marker()
                marker.header.frame_id = "map"
                marker.header.stamp = self.get_clock().now().to_msg()
                marker.ns = "obstacles"
                
                # Vergibt automatisch immer die nächste freie ID (0, 1, 2, ...)
                marker.id = len(marker_array.markers) 
                marker.type = Marker.CUBE
                marker.action = Marker.ADD
                
                # Format auslesen: [x, y, z, size_x, size_y, size_z]
                marker.pose.position.x = float(obs[0])
                marker.pose.position.y = float(obs[1])
                marker.pose.position.z = float(obs[2])
                
                marker.pose.orientation.w = 1.0 
                marker.pose.orientation.x = 0.0
                marker.pose.orientation.y = 0.0
                marker.pose.orientation.z = 0.0

                marker.scale.x = float(obs[3])
                marker.scale.y = float(obs[4])
                marker.scale.z = float(obs[5])

                # Die übergebene Farbe anwenden
                marker.color.r = r
                marker.color.g = g
                marker.color.b = b
                marker.color.a = 0.7 

                marker_array.markers.append(marker)

        # 1. Mess-Objekte / Zielgebäude zeichnen (Grün)
        create_markers(self.target_obstacles, 0.2, 0.8, 0.2)
        
        # 2. Ausweich-Hindernisse / Störobjekte zeichnen (Rot)
        create_markers(self.avoidance_obstacles, 0.8, 0.2, 0.2)

        # Alles gebündelt an RViz senden
        self.obstacle_pub.publish(marker_array)

    def add_cluster_markers(self, marker_array, drones, formation, drone_color, robot_color, base_id):
        # 1. Drohnen zeichnen (bleibt unverändert)
        for idx, d in enumerate(drones):
            # d[2] sorgt dafür, dass die Kugel in RViz in der Luft schwebt!
            drone_marker = self.create_base_marker(base_id + idx, Marker.SPHERE, d[0], d[1], d[2], *drone_color)
            drone_marker.scale.x, drone_marker.scale.y, drone_marker.scale.z = 0.5, 0.5, 0.5
            marker_array.markers.append(drone_marker)
            
        # 2. Roboter-Typen laden (Fallback-Liste)
        robot_types = getattr(self, 'params', {}).get('robot_types', ['A', 'A', 'A', 'A', 'B', 'B'])

        # 3. Dynamische Anzahl der Roboter aus der Formations-Länge ableiten
        num_robots = len(formation) // 2
        for i in range(num_robots):
            rx = float(formation[i*2])
            ry = float(formation[i*2+1])
            
            # Welcher Typ ist dieser Roboter?
            r_type = robot_types[i] if i < len(robot_types) else 'A'

            # 4. Marker mit typspezifischer Höhe, aber CLUSTER-FARBE erstellen
            if r_type == 'A':
                # Typ A: Flach (0.2m hoch)
                scale_z = 0.2
                robot_marker = self.create_base_marker(base_id + 50 + i, Marker.CYLINDER, rx, ry, scale_z / 2.0, *robot_color)
                robot_marker.scale.x, robot_marker.scale.y, robot_marker.scale.z = 0.5, 0.5, scale_z
            else:
                # Typ B: Hoch (1.2m hoch)
                scale_z = 1.2
                robot_marker = self.create_base_marker(base_id + 50 + i, Marker.CYLINDER, rx, ry, scale_z / 2.0, *robot_color)
                robot_marker.scale.x, robot_marker.scale.y, robot_marker.scale.z = 0.6, 0.6, scale_z

            marker_array.markers.append(robot_marker)

    def create_base_marker(self, m_id, m_type, x, y, z, r, g, b):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "pso_formation"
        marker.id = m_id
        marker.type = m_type
        marker.action = Marker.ADD
        marker.pose.position.x = float(x)
        marker.pose.position.y = float(y)
        marker.pose.position.z = float(z)
        marker.pose.orientation.w = 1.0
        marker.color.r = float(r)
        marker.color.g = float(g)
        marker.color.b = float(b)
        marker.color.a = 1.0
        return marker
    
    def plot_results(self):
        plt.figure(figsize=(10, 6))
        
        # Gesamtkosten mit LaTeX-Notation für J_total
        plt.plot(self.k_history, self.total_cost_history, 'k-o', linewidth=2, label=r'Gesamtkosten ($J_{total}$)')
        
        # Einzelne Clusterkosten als Punkte (Scatter) eintragen
        for k, costs in self.cluster_costs_history.items():
            for idx, c in enumerate(costs):
                # 1. Die exakt gleiche Farbe wie in RViz aus der Liste laden!
                c_color = self.cluster_colors[idx % len(self.cluster_colors)]
                
                # Punkte einzeichnen (leicht versetzt auf der x-Achse für bessere Lesbarkeit)
                offset = (idx - len(costs)/2) * 0.05 
                # 2. Das color=c_color Argument hinzufügen
                plt.scatter(k + offset, c, s=100, zorder=5, color=c_color)
                # Cluster-Nummer daneben schreiben
                plt.text(k + offset + 0.05, c, f'C{idx+1}', fontsize=9, verticalalignment='center')

        # Vertikale Linie beim besten k markieren
        best_k = self.current_k - 1
        plt.axvline(x=best_k, color='green', linestyle='--', alpha=0.5, label=f'Optimales k = {best_k}')

        # Diagramm hübsch machen und exakt an die Thesis anpassen!
        plt.title(r'Entwicklung der Gesamtkosten ($J_{total}$) über die Iterationen')
        plt.xlabel('Anzahl der Cluster (k)')
        plt.ylabel(r'Kosten ($J_{total}$)')
        plt.xticks(self.k_history) # Nur ganze Zahlen auf der x-Achse
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        
        # Speichername anpassen, um alte Plots nicht zu überschreiben
        save_path = '/home/vboxuser/map_ws/j_total_plot.png'
        plt.savefig(save_path, dpi=300)
        self.get_logger().info(f"Plot wurde erfolgreich gespeichert unter: {save_path}")
        
        # Fenster öffnen
        plt.show()

    def calculate_drone_pose_facing_wall(self, drone_pos, obs):
        """
        Berechnet den 6D-Zustandsvektor der Drohne. Die Drohne richtet ihre Sensorik
        stets senkrecht auf die nächstgelegene Oberfläche des Hindernis-Quaders (AABB) aus.
        """
        import math
        
        # 1. Daten robust entpacken (unterstützt das neue 6D-Listenformat)
        if isinstance(obs, dict):
            cx = obs.get('x', 0.0)
            cy = obs.get('y', 0.0)
            sx = obs.get('sx', obs.get('s', 2.0))
            sy = obs.get('sy', obs.get('s', 2.0))
            sz = obs.get('h', 5.0)
            cz = obs.get('z', sz / 2.0)
        else:
            if len(obs) == 6:
                cx, cy, cz, sx, sy, sz = obs
            else:
                cx, cy = obs[0], obs[1]
                sx, sy, sz = 2.0, 2.0, 5.0
                cz = sz / 2.0

        dx, dy, dz = drone_pos[0], drone_pos[1], drone_pos[2]

        # 2. Nächstgelegenen Oberflächenpunkt auf dem Quader finden (AABB Projection)
        # Formeln exakt wie in Kapitel 4 der Masterarbeit
        x_surf = max(cx - sx/2.0, min(dx, cx + sx/2.0))
        y_surf = max(cy - sy/2.0, min(dy, cy + sy/2.0))
        z_surf = max(cz - sz/2.0, min(dz, cz + sz/2.0))

        # 3. Blickvektor von der Drohne zur Wand
        vx = x_surf - dx
        vy = y_surf - dy
        vz = z_surf - dz

        # Fallback: Falls die Drohne exakt IM Gebäude wäre, schaut sie zur Mitte
        if vx == 0 and vy == 0 and vz == 0:
            vx = cx - dx
            vy = cy - dy
            vz = cz - dz

        # 4. Winkel berechnen
        yaw = math.atan2(vy, vx)
        pitch = math.atan2(vz, math.hypot(vx, vy))
        roll = 0.0 # Wankwinkel ist konstant 0 (stabiler Schwebeflug)

        # 6D Vektor zurückgeben: [x, y, z, roll, pitch, yaw]
        return [dx, dy, dz, roll, pitch, yaw]

def main(args=None):
    rclpy.init(args=args)
    node = OptimizerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()

if __name__ == '__main__':
    main()