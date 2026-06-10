import rclpy
from rclpy.node import Node
import numpy as np
import math
from visualization_msgs.msg import Marker, MarkerArray
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

from multi_robot_optimizer.pso_algorithm import SwarmOptimizer
from visualization_msgs.msg import Marker, MarkerArray

class OptimizerNode(Node):
    def __init__(self):
        super().__init__('formation_optimizer_node')
        
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
        self.obstacle_coords = [[0.0, 0.0, 5.0, 20.0, 15.0, 10.0]]
        
        # NEU: Automatisierte 3D-Grid-Generierung passend zum Gebäude
        # 1 Drohne pro 1 qm Wandfläche, genau 1m Abstand zur Wand
        generated_drones_3d = []
        import numpy as np
        
        # Wand-Dimensionen des Gebäudes: Länge (X)=20m, Breite (Y)=15m, Höhe (Z)=10m
        # Das Gebäude zentriert sich von X=[-10, 10], Y=[-7.5, 7.5], Z=[0, 10]
        
        # A) Kurze Wände (Vorne & Hinten bei X = -11.0 und +11.0 wegen 1m Abstand)
        # Y-Spanne: 15m Breite -> 15 Punkte im 1m-Abstand
        # Z-Spanne: 10m Höhe -> 10 Punkte im 1m-Abstand (0.5m bis 9.5m, keine Drohnen über 10m)
        y_coords = np.linspace(-7.0, 7.0, 15)
        z_coords = np.linspace(0.5, 9.5, 10)
        
        for x in [-11.0, 11.0]:
            for y in y_coords:
                for z in z_coords:
                    generated_drones_3d.append([float(x), float(y), float(z)])
                    
        # B) Lange Wände (Links & Rechts bei Y = -8.5 und +8.5 wegen 1m Abstand)
        # X-Spanne: 20m Länge -> 20 Punkte im 1m-Abstand
        x_coords = np.linspace(-9.5, 9.5, 20)
        
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
        
        ## Einheitliche Farben für Drohnen und Roboter desselben Clusters
        self.cluster_colors = [
            ((0.0, 1.0, 0.0), (0.0, 1.0, 0.0)), # Hellgrün
            ((1.0, 1.0, 0.0), (1.0, 1.0, 0.0)), # Gelb
            ((0.0, 1.0, 1.0), (0.0, 1.0, 1.0)), # Cyan
            ((1.0, 0.0, 1.0), (1.0, 0.0, 1.0)), # Magenta
            ((1.0, 0.5, 0.0), (1.0, 0.5, 0.0)), # Orange
            ((1.0, 0.2, 0.2), (1.0, 0.2, 0.2)), # Hellrot
            ((0.5, 0.5, 1.0), (0.5, 0.5, 1.0)), # Hellblau
            ((1.0, 1.0, 1.0), (1.0, 1.0, 1.0))  # Weiß
        ]
        
        # Timer startet den Loop (alle 4 Sekunden ein neuer Schritt k)
        self.timer = self.create_timer(4.0, self.evaluate_next_k)
        self.get_logger().info("Schritt 1 & 2: Dateneingabe abgeschlossen, k=1 initialisiert.")

    def evaluate_next_k(self):

        # 1. NEU: Hindernisse IMMER und SOFORT publizieren!
        # Egal ob der Algorithmus rechnet, scheitert oder fertig ist:
        self.publish_obstacles()

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
        
        # Gewicht für die Bewegung (aus Parametern)
        w_move = self.params.get('w_move', 0.1)
        
        for idx, cluster_drones in enumerate(clusters):
            # 1. INNERE SCHLEIFE (PSO): Berechnet Formationsgüte (CRLB + C_obs + C_inter)
            best_form, pso_cost = self.optimizer.run_pso(cluster_drones, self.start_robot_poses, self.obstacle_coords)
            
            if pso_cost >= 1e9:
                self.get_logger().warn("PSO-Algorithmus konnte keine valide Formation finden (Sichtlinie blockiert oder Singularität).")
            # Hier kannst du das k-Erhöhen ggf. überspringen

            if best_form is None or pso_cost == float('inf'):
                self.get_logger().error(f"PSO fehlgeschlagen für Cluster {idx+1} (Kosten unendlich).")
                valid_solution = False
                break
                
            # 2. ÄUSSERE SCHLEIFE: Bewegungskomponente (C_move) berechnen
            c_move_raw = 0.0
            num_robots_in_form = len(best_form) // 2
            
            for i in range(num_robots_in_form):
                # Startposition (x, y) des Roboters i
                sx = self.start_robot_poses[i*2]
                sy = self.start_robot_poses[i*2+1]
                
                # Zielposition (x, y) des Roboters i aus der PSO
                tx = best_form[i*2]
                ty = best_form[i*2+1]
                
                # Euklidische Distanz für diesen Roboter addieren
                c_move_raw += math.hypot(tx - sx, ty - sy)
                
            # 3. NORMIERUNG VON C_move
            # Wir definieren eine maximale zumutbare Fahrstrecke (z.B. 50 Meter pro Roboter)
            max_travel_dist = num_robots_in_form * 50.0 
            norm_move = min(c_move_raw / max_travel_dist, 1.0) if max_travel_dist > 0 else 0.0
            
            # 4. GESAMTKOSTEN FÜR DIESES CLUSTER ADDIEREN
            # pso_cost enthält bereits w_crlb, w_obs, w_inter
            cluster_total_cost = pso_cost + (w_move * norm_move)
            
            current_total_cost += cluster_total_cost
            formations.append(best_form)
            current_cluster_costs.append(cluster_total_cost)
            
        if not valid_solution:
            self.timer.cancel()
            return
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
            
            # Ergebnis ausgeben
            self.get_logger().info(f"\n+++ ERGEBNIS +++")
            self.get_logger().info(f"Beste Formation ermittelt für k = {self.current_k - 1} Cluster.")
            
            # Zustand umschalten, Plot generieren und raus!
            # (Der Timer läuft im Hintergrund weiter und hält RViz am Leben)
            self.optimization_done = True 
            self.plot_results() 
            return

        self.get_logger().info("--> NEIN")
        self.get_logger().info(f"    Erhöhe k = k + 1 (Gehe in der nächsten Runde auf {self.current_k + 1})")
        
        # --- RViz-Visualisierung für den aktuellen, gültigen Schritt aufbauen ---
        marker_array = MarkerArray()

        if self.best_marker_array is not None:
                self.marker_pub.publish(self.best_marker_array)
                
        # NEU: Rufe hier stattdessen die neue Hindernis-Funktion auf
        self.publish_obstacles()
        
        # NEU: Cluster räumlich nach ihrem Winkel zum Zentrum (5.0, 5.0) sortieren.
        # Das verhindert, dass die Farben beim Erhöhen von k wild durchtauschen.
        sorted_clusters = []
        for c_drones, form in zip(clusters, formations):
            center = np.mean(c_drones, axis=0)
            # Berechne den Winkel im Kreis (arctan2)
            angle = np.arctan2(center[1] - 5.0, center[0] - 5.0)
            sorted_clusters.append((angle, c_drones, form))
            
        # Nach dem Winkel sortieren (gegen den Uhrzeigersinn)
        sorted_clusters.sort(key=lambda item: item[0])

        # Jetzt die Marker mit stabilen Farben publishen
        for idx, (_, cluster_drones, form) in enumerate(sorted_clusters):
            d_color, r_color = self.cluster_colors[idx % len(self.cluster_colors)]
            self.add_cluster_markers(marker_array, cluster_drones, form, 
                                     drone_color=d_color, robot_color=r_color, base_id=(idx+1)*100)
        
        self.best_marker_array = marker_array 
        self.marker_pub.publish(marker_array)
        
        # Werte für den nächsten Iterationsschritt überschreiben
        self.previous_total_cost = current_total_cost
        self.current_k += 1

    # --- Hilfsfunktionen für RViz ---
    def publish_obstacles(self):
        # Erstelle ein neues MarkerArray für die Hindernisse
        marker_array = MarkerArray()
        
        # Standardwerte (exakt die gleichen wie in der Mathematik!)
        DEFAULT_SIZE = 2.0
        DEFAULT_HEIGHT = 5.0
        
        for i, obs in enumerate(self.obstacle_coords):
            marker = Marker()
            marker.header.frame_id = "map" # Wichtig für RViz
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "obstacles"
            marker.id = i
            marker.type = Marker.CUBE  # <--- HIER: Wir nutzen jetzt Quader!
            marker.action = Marker.ADD
            
            # 1. Daten extrahieren (Genau wie in pso_algorithm.py)
            if isinstance(obs, dict):
                x = obs.get('x', 0.0)
                y = obs.get('y', 0.0)
                s = obs.get('s', DEFAULT_SIZE)
                h = obs.get('h', DEFAULT_HEIGHT)
                z = obs.get('z', h / 2.0)
            else:
                x, y = obs[0], obs[1]
                s, h = DEFAULT_SIZE, DEFAULT_HEIGHT
                z = h / 2.0

            # 2. Position an RViz übergeben
            marker.pose.position.x = float(x)
            marker.pose.position.y = float(y)
            marker.pose.position.z = float(z)
            
            # Ausrichtung (Quader steht gerade)
            marker.pose.orientation.w = 1.0 
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0

            # 3. Skalierung direkt an die Mathematik koppeln
            marker.scale.x = float(s) # Breite (X-Achse)
            marker.scale.y = float(s) # Tiefe (Y-Achse)
            marker.scale.z = float(h) # Höhe (Z-Achse)

            # 4. Optik (Ziegelrot und leicht transparent)
            marker.color.r = 0.8
            marker.color.g = 0.2
            marker.color.b = 0.2
            marker.color.a = 0.7 # Transparenz, damit man Drohnen dahinter noch erahnen kann

            marker_array.markers.append(marker)

        # Publisher aufrufen (stelle sicher, dass self.obstacle_pub in __init__ definiert ist)
        self.obstacle_pub.publish(marker_array)

    def add_cluster_markers(self, marker_array, drones, formation, drone_color, robot_color, base_id):
        for idx, d in enumerate(drones):
            # d[2] sorgt dafür, dass die Kugel in RViz in der Luft schwebt!
            drone_marker = self.create_base_marker(base_id + idx, Marker.SPHERE, d[0], d[1], d[2], *drone_color)
            drone_marker.scale.x, drone_marker.scale.y, drone_marker.scale.z = 0.5, 0.5, 0.5
            marker_array.markers.append(drone_marker)
            
        # Dynamische Anzahl der Roboter aus der Formations-Länge ableiten
        num_robots = len(formation) // 2
        for i in range(num_robots):
            rx = float(formation[i*2])
            ry = float(formation[i*2+1])
            robot_marker = self.create_base_marker(base_id + 50 + i, Marker.CYLINDER, rx, ry, 0.2, *robot_color)
            robot_marker.scale.x, robot_marker.scale.y, robot_marker.scale.z = 0.6, 0.6, 0.4
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
                # Punkte einzeichnen (leicht versetzt auf der x-Achse für bessere Lesbarkeit)
                offset = (idx - len(costs)/2) * 0.05 
                plt.scatter(k + offset, c, s=100, zorder=5)
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
        dx, dy, dz = drone_pos
        
        # Hindernis-Grenzen (Bounding Box)
        s_half = obs['s'] / 2.0
        h_half = obs.get('h', 5.0) / 2.0 
        z_center = obs.get('z', h_half)
        
        x_min, x_max = obs['x'] - s_half, obs['x'] + s_half
        y_min, y_max = obs['y'] - s_half, obs['y'] + s_half
        z_min, z_max = z_center - h_half, z_center + h_half
        
        # Nächstgelegenen Punkt auf der Quader-Oberfläche finden
        nx = max(x_min, min(dx, x_max))
        ny = max(y_min, min(dy, y_max))
        nz = max(z_min, min(dz, z_max))
        
        # Blickvektor
        look_x = nx - dx
        look_y = ny - dy
        look_z = nz - dz
        
        # Winkel berechnen
        yaw = math.atan2(look_y, look_x)
        dist_xy = math.hypot(look_x, look_y)
        pitch = math.atan2(look_z, dist_xy) 
        roll = 0.0 
        
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