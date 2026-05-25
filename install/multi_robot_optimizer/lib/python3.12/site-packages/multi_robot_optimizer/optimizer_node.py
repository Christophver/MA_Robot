import rclpy
from rclpy.node import Node
import numpy as np
from visualization_msgs.msg import Marker, MarkerArray
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

from multi_robot_optimizer.pso_algorithm import SwarmOptimizer

class OptimizerNode(Node):
    def __init__(self):
        super().__init__('formation_optimizer_node')
        
        # ==========================================
        # Schritt 1: Dateneingabe
        # ==========================================
        self.declare_parameter('w_crlb', 1.0)
        self.declare_parameter('w_move', 0.1)
        self.declare_parameter('w_obs', 0.5)
        self.declare_parameter('max_drone_dist', 30.0)
        
        params = {
            'w_crlb': self.get_parameter('w_crlb').value,
            'w_move': self.get_parameter('w_move').value,
            'w_obs': self.get_parameter('w_obs').value,
            'max_drone_dist': self.get_parameter('max_drone_dist').value,
        }
        
        self.optimizer = SwarmOptimizer(params)
        self.marker_pub = self.create_publisher(MarkerArray, '/formation_markers', 10)
        
        self.start_robot_poses = [0.0, 0.0, 2.0, 0.0, 0.0, 2.0]
        self.obstacle_coords = [[5.0, 5.0]] 
        
        # Zentrales Messobjekt steht bei (x=5.0, y=5.0)
        # 16 Drohnen in 4 verschiedenen Höhenschichten um das Zentrum verteilt
        self.all_drones = np.array([
            # --- Ebene 1: Niedrige Höhe (z = 4.0 m), enger Radius (2 m) ---
            [5.0, 3.0, 4.0], 
            [3.0, 5.0, 4.0], 
            [7.0, 5.0, 4.0], 
            [5.0, 7.0, 4.0],

            # --- Ebene 2: Mittlere Höhe (z = 7.0 m), mittlerer Radius (3 m), diagonal versetzt ---
            [2.8, 2.8, 7.0], 
            [7.1, 2.8, 7.0], 
            [2.8, 7.1, 7.0], 
            [7.1, 7.1, 7.0],

            # --- Ebene 3: Hohe Höhe (z = 10.0 m), weiter Radius (4 m) ---
            [5.0, 1.0, 10.0], 
            [1.0, 5.0, 10.0], 
            [9.0, 5.0, 10.0], 
            [5.0, 9.0, 10.0],

            # --- Ebene 4: Sehr hoch (z = 13.0 m), fast direkt über dem Objekt (Radius 1 m) ---
            [5.0, 4.0, 13.0], 
            [4.0, 5.0, 13.0], 
            [6.0, 5.0, 13.0], 
            [5.0, 6.0, 13.0]
        ])
        
        # ==========================================
        # Schritt 2: Initialisierung
        # ==========================================
        self.current_k = 1

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
        # Schritt 4: Positions-Optimierung
        # ==========================================
        self.get_logger().info("Schritt 4: Innere PSO-Schleife (Suche beste Formation)...")
        current_total_cost = 0.0
        formations = []
        current_cluster_costs = [] # NEU: temporäre Liste für diesen Zyklus
        valid_solution = True
        
        for idx, cluster_drones in enumerate(clusters):
            best_form, cost = self.optimizer.run_pso(cluster_drones, self.start_robot_poses, self.obstacle_coords)
            
            if best_form is None:
                self.get_logger().error(f"PSO fehlgeschlagen für Cluster {idx+1}.")
                valid_solution = False
                break
                
            current_total_cost += cost
            formations.append(best_form)
            current_cluster_costs.append(cost) # NEU: Kosten für dieses Cluster merken
            
        if not valid_solution:
            self.timer.cancel()
            return

        # Nach der Schleife speichern wir die Historie ab:
        self.k_history.append(self.current_k)
        self.total_cost_history.append(current_total_cost)
        self.cluster_costs_history[self.current_k] = current_cluster_costs

        # ==========================================
        # Schritt 5: Berechnung der Gesamtkosten
        # ==========================================
        self.get_logger().info(f"Schritt 5: Gesamtkosten C_total = {current_total_cost:.4f}")

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
            
            # Wir publishen das visualisierte Ergebnis von k-1 ein letztes Mal, 
            # damit RViz das korrekte Endresultat anzeigt.
            if self.best_marker_array is not None:
                self.marker_pub.publish(self.best_marker_array)
                
            self.timer.cancel() # Loop stoppen
            if self.best_marker_array is not None:
                self.marker_pub.publish(self.best_marker_array)
                
            self.timer.cancel() # Loop stoppen
            self.plot_results() # NEU: Grafik am Ende erstellen!
            return

        self.get_logger().info("--> NEIN")
        self.get_logger().info(f"    Erhöhe k = k + 1 (Gehe in der nächsten Runde auf {self.current_k + 1})")
        
        # --- RViz-Visualisierung für den aktuellen, gültigen Schritt aufbauen ---
        marker_array = MarkerArray()
        self.add_obstacle_marker(marker_array)
        
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
    def add_obstacle_marker(self, marker_array):
        obs = self.create_base_marker(0, Marker.CUBE, 5.0, 5.0, 0.5, 1.0, 0.1, 0.1)
        obs.scale.x, obs.scale.y, obs.scale.z = 2.0, 2.0, 1.0
        marker_array.markers.append(obs)

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
        
        # Gesamtkosten als durchgehende schwarze Linie
        plt.plot(self.k_history, self.total_cost_history, 'k-o', linewidth=2, label='Gesamtkosten (C_total)')
        
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

        # Diagramm hübsch machen
        plt.title('Entwicklung der CRLB-Kosten über die Iterationen')
        plt.xlabel('Anzahl der Cluster (k)')
        plt.ylabel('Kosten')
        plt.xticks(self.k_history) # Nur ganze Zahlen auf der x-Achse
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        
        # Direkt in deinem Workspace speichern
        save_path = '/home/vboxuser/map_ws/crlb_plot.png'
        plt.savefig(save_path, dpi=300)
        self.get_logger().info(f"Plot wurde erfolgreich gespeichert unter: {save_path}")
        
        # Fenster öffnen
        plt.show()

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