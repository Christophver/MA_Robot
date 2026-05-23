import rclpy
from rclpy.node import Node
import numpy as np
from visualization_msgs.msg import Marker, MarkerArray
from sklearn.cluster import KMeans

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
        
        # Feste Drohnenpunkte
        self.all_drones = np.array([
            [2.0, 2.0], [2.0, 8.0], 
            [8.0, 2.0], [8.0, 8.0], 
            [5.0, 8.0], [8.0, 5.0]
        ])
        
        # ==========================================
        # Schritt 2: Initialisierung
        # ==========================================
        self.current_k = 1
        self.previous_total_cost = float('inf')
        self.best_marker_array = None # Speicher für die beste RViz-Ausgabe (k-1)
        
        # Farben für die Visualisierung der Cluster
        self.cluster_colors = [
            ((0.1, 1.0, 0.1), (0.1, 0.1, 1.0)), # Grün / Blau
            ((1.0, 1.0, 0.1), (0.1, 1.0, 1.0)), # Gelb / Cyan
            ((1.0, 0.1, 1.0), (1.0, 0.5, 0.0)), # Magenta / Orange
            ((0.5, 0.0, 1.0), (1.0, 0.0, 0.0)), # Lila / Rot
            ((1.0, 1.0, 1.0), (0.5, 0.5, 0.5)), # Weiß / Grau
            ((0.0, 1.0, 0.5), (0.5, 0.2, 0.2))  # Türkis / Braun
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
        valid_solution = True
        
        for idx, cluster_drones in enumerate(clusters):
            best_form, cost = self.optimizer.run_pso(cluster_drones, self.start_robot_poses, self.obstacle_coords)
            
            if best_form is None:
                self.get_logger().error(f"PSO fehlgeschlagen für Cluster {idx+1}.")
                valid_solution = False
                break
                
            current_total_cost += cost
            formations.append(best_form)
            
        if not valid_solution:
            self.timer.cancel()
            return

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
            return

        self.get_logger().info("--> NEIN")
        self.get_logger().info(f"    Erhöhe k = k + 1 (Gehe in der nächsten Runde auf {self.current_k + 1})")
        
        # --- RViz-Visualisierung für den aktuellen, gültigen Schritt aufbauen ---
        marker_array = MarkerArray()
        self.add_obstacle_marker(marker_array)
        for idx, (cluster_drones, form) in enumerate(zip(clusters, formations)):
            d_color, r_color = self.cluster_colors[idx % len(self.cluster_colors)]
            self.add_cluster_markers(marker_array, cluster_drones, form, 
                                     drone_color=d_color, robot_color=r_color, base_id=(idx+1)*100)
        
        # Wir speichern diese Formation zwischen. Wenn im nächsten Schritt (k+1) die
        # Kosten steigen, holen wir genau dieses Bild (k) wieder hervor und stoppen.
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
            drone_marker = self.create_base_marker(base_id + idx, Marker.SPHERE, d[0], d[1], 0.5, *drone_color)
            drone_marker.scale.x, drone_marker.scale.y, drone_marker.scale.z = 0.5, 0.5, 0.5
            marker_array.markers.append(drone_marker)
            
        for i in range(3):
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