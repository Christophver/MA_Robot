import rclpy
from rclpy.node import Node
import numpy as np
import math
import time
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import scipy.ndimage as ndimage
import csv
import os
import sys


# ROS 2 Messages
from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import PoseArray

# Eigene Imports
from multi_robot_optimizer.pso_algorithm import SwarmOptimizer, ESDFMapVectorized

class OptimizerNode(Node):
    def __init__(self):
        super().__init__('formation_optimizer_node')

        # ==========================================
        # Schritt 1: Dateneingabe & Parameter
        # ==========================================
        self.declare_parameter('w_crlb', 1.0)
        self.declare_parameter('w_move', 1.6)
        self.declare_parameter('w_obs', 1.00)
        self.declare_parameter('max_drone_dist', 30.0)
        self.declare_parameter('robot_types', ['A', 'A', 'B', 'B'])
        self.declare_parameter('max_iterations', 50)
        self.declare_parameter('c1', 1.5)
        self.declare_parameter('c2', 1.5)
        self.declare_parameter('inertia_weight', 0.5)
        self.declare_parameter('d_safe', 2.0)
        self.declare_parameter('d_crash', 0.15) 

        # ==========================================
        # STATISTIK-MODUS SCHALTER (An/Aus)
        # ==========================================
        self.enable_batch_evaluation = True  # <--- HIER AN/AUS SCHALTEN (True/False)
        self.target_name = "Objekt_3"
        self.current_run = 1
        self.total_runs = 1             # Anzahl der Durchläufe im Batch-Modus
        self.experiment_results = []         # Speicher für die CSV-Daten
        # ==========================================
        
        params = {
            'w_crlb': self.get_parameter('w_crlb').value,
            'w_move': self.get_parameter('w_move').value,
            'w_obs': self.get_parameter('w_obs').value,
            'max_drone_dist': self.get_parameter('max_drone_dist').value,
            'robot_types': self.get_parameter('robot_types').value,
            'max_iterations': self.get_parameter('max_iterations').value,
            'c1': self.get_parameter('c1').value,
            'c2': self.get_parameter('c2').value,
            'inertia_weight': self.get_parameter('inertia_weight').value,
            'd_safe': self.get_parameter('d_safe').value,
            'd_crash': self.get_parameter('d_crash').value,
        }

        self.optimizer = SwarmOptimizer(params)
        
        # ROS Publisher für die RViz-Visualisierung
        self.marker_pub = self.create_publisher(MarkerArray, 'formation_markers', 10)
        
        # ==========================================
        # Schritt 2: Subscriber & Initialisierung
        # ==========================================
        self.all_drones = np.array([])  # Wird über das Topic gefüllt
        
        self.map_received = False
        self.drones_received = False
        
        self.map_sub = self.create_subscription(OccupancyGrid, '/map', self.map_callback, 10)
        self.drone_sub = self.create_subscription(PoseArray, '/drone_targets', self.drone_callback, 10)

        # Farben für die Cluster-Visualisierung
        self.cluster_colors = [
            (0.122, 0.467, 0.706), (0.682, 0.780, 0.910), (1.000, 0.498, 0.055),
            (1.000, 0.733, 0.471), (0.173, 0.627, 0.173), (0.596, 0.875, 0.541),
            (0.839, 0.153, 0.157), (1.000, 0.596, 0.588), (0.580, 0.404, 0.741),
            (0.773, 0.690, 0.835), (0.549, 0.337, 0.294), (0.769, 0.612, 0.580),
            (0.890, 0.467, 0.761), (0.969, 0.714, 0.824), (0.498, 0.498, 0.498),
            (0.780, 0.780, 0.780), (0.737, 0.741, 0.133), (0.859, 0.859, 0.553),
            (0.090, 0.745, 0.812), (0.620, 0.855, 0.898),
        ]

        # Initiale Posen der Bodenroboter
        self.start_robot_poses = [
            0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 
            0.0, 1.0, 1.0, 1.0, 2.0, 1.0   
        ]
        
        # Initialisierung der PSO-Zustandsvariablen
        self.current_k = 1
        self.optimization_done = False 
        self.k_history = []
        self.total_cost_history = []
        self.cluster_costs_history = {}
        self.previous_total_cost = float('inf')
        self.best_marker_array = None 
        
        self.timer = None
        self.get_logger().info("Warte auf Karte (/map) und Drohnen (/drone_targets)...")
        
        # Prüft sofort, ob evtl. schon Daten anliegen (für schnelle Neustarts)
        self.check_start_condition()

    def map_callback(self, msg):
        """Wird aufgerufen, sobald image_to_ros.py die Karte publiziert."""
        if self.map_received:
            return  # Nur beim ersten Mal ausführen

        self.get_logger().info("Karte empfangen! Verarbeite Grid und berechne ESDF...")

        self.map_resolution = msg.info.resolution
        self.map_origin_x = msg.info.origin.position.x
        self.map_origin_y = msg.info.origin.position.y
        width = msg.info.width
        height = msg.info.height

        # 1. ROS 1D-Array zurück in ein 2D NumPy Array wandeln
        grid_1d = np.array(msg.data, dtype=np.int8)
        grid_2d = grid_1d.reshape((height, width))

        # 2. Binäres Grid für den Bresenham LoS-Check erstellen (0 = Frei, 1 = Wand)
        raw_binary = np.where(grid_2d > 50, 1, 0)
        # NEU: Zwingt das Array in einen sauberen C-Speicherblock für Numba!
        self.binary_grid = np.ascontiguousarray(raw_binary, dtype=np.int8)

        # 3. ESDF berechnen (Distanz zu Wänden)
        free_space = 1 - self.binary_grid
        self.esdf_matrix = ndimage.distance_transform_edt(free_space)

        # 4. Deine neue vektorisierte Klasse initialisieren!
        self.esdf_map = ESDFMapVectorized(self.esdf_matrix, self.map_resolution, self.map_origin_x, self.map_origin_y)

        # 5. Dem Optimizer die Map-Objekte injizieren
        self.optimizer.esdf_map = self.esdf_map
        self.optimizer.binary_grid = self.binary_grid
        self.optimizer.map_resolution = self.map_resolution
        self.optimizer.map_origin_x = self.map_origin_x
        self.optimizer.map_origin_y = self.map_origin_y

        self.map_received = True
        self.get_logger().info("Umgebung erfolgreich initialisiert. Erzeuge RViz Debug-Ansicht...")
        self.check_start_condition()

        # 1. Debug Map dauerhaft an RViz senden (alle 2 Sekunden)
        # So kann RViz die Nachricht nicht mehr verpassen!
        self.debug_timer = self.create_timer(2.0, self.publish_rviz_debug_map)

        # JETZT ERST den Loop starten!
        self.timer = self.create_timer(4.0, self.evaluate_next_k)
        
    def check_start_condition(self):
        """Startet die PSO-Schleife erst, wenn Karte UND Drohnen-Posen existieren."""
        if self.map_received and self.drones_received and self.timer is None:
            self.get_logger().info("🚀 Beide Datenströme bereit. Starte Optimierungsschleife...")
            # Optional: Hier wieder dein rviz debug grid aktivieren falls gewünscht
            self.timer = self.create_timer(4.0, self.evaluate_next_k)

    def drone_callback(self, msg):
        """Empfängt die maßgeschneiderten Drohnen-Positionen direkt von OpenCV."""
        if self.drones_received:
            return

        drones_6d = []
        for pose in msg.poses:
            dx = pose.position.x
            dy = pose.position.y
            dz = pose.position.z
            
            # Rekonstruktion des Yaw aus dem Quaternion
            yaw = 2.0 * math.atan2(pose.orientation.z, pose.orientation.w)
            drones_6d.append([dx, dy, dz, 0.0, 0.0, yaw])

        self.all_drones = np.array(drones_6d)
        self.drones_received = True
        self.get_logger().info(f"✅ {len(self.all_drones)} Ziel-Drohnen erfolgreich registriert!")
        
        # Prüfen, ob wir loslegen können
        self.check_start_condition()

    


    def evaluate_next_k(self):
        
        self.publish_drone_targets()

        start_time = time.perf_counter()

        


        if self.optimization_done:
            if self.best_marker_array is not None:
                self.marker_pub.publish(self.best_marker_array)
            return
        
        if self.current_k > len(self.all_drones):
            self.get_logger().info("Maximale Cluster-Anzahl erreicht. Abbruch.")
            self.timer.cancel()
            
            # --- NEU: Zwingt den Knoten, sich komplett zu beenden ---
            import sys
            sys.exit(0)
            # --------------------------------------------------------
            return

        self.get_logger().info(f"\n------------------------------------------------")
        self.get_logger().info(f"Starte Evaluierung für k = {self.current_k}")
        
        clusters = []
        labels = []  # NEU: Hier speichern wir die Zuweisungen für RViz
            

        if self.current_k == 1:
            clusters.append(self.all_drones.tolist())
            # Bei k=1 sind alle Drohnen im selben Cluster (Index 0)
            labels = [0] * len(self.all_drones) 
        else:
            # PRO-TIPP für deine Arbeit: Nur über X, Y, Z clustern! ([:, :3])
            # Winkel (Yaw) haben eine andere Skalierung als Meter und würden das Clustering verfälschen.
            kmeans = KMeans(n_clusters=self.current_k, random_state=42, n_init=10).fit(self.all_drones[:, :3])
            
            labels = kmeans.labels_  # Das ist unser Array für RViz!
            
            for i in range(self.current_k):
                clusters.append(self.all_drones[kmeans.labels_ == i].tolist())

        self.publish_drone_targets(labels=labels)
                
        current_total_cost = 0.0
        formations = []
        current_cluster_costs = [] 
        valid_solution = True

        # NEU: Sammler für die Einzelkosten dieses Durchlaufs
        total_crlb_cost = 0.0
        total_obs_cost = 0.0
        total_move_cost = 0.0
        
        
        # Dynamisch w_move auslesen, sonst 0.1
        w_move = self.get_parameter('w_move').value if self.has_parameter('w_move') else 0.1

        # ==========================================
        # NEU: w_obs dynamisch auslesen und an PSO senden
        # ==========================================
        w_obs = self.get_parameter('w_obs').value if self.has_parameter('w_obs') else 1.0
        self.optimizer.params['w_obs'] = w_obs
        
        # (Optional) Hier den Namen für die CSV dynamisch anpassen, 
        # damit dein Kuchendiagramm-Skript die w_obs-Sweeps automatisch trennt:
        self.target_name = f"U_Profil_w_obs_{w_obs}"
        # ==========================================
        
        for idx, cluster_drones in enumerate(clusters):
            
            # ACHTUNG: Aufruf geändert! obstacle_coords wird nicht mehr an run_pso übergeben!
            best_form, pso_cost = self.optimizer.run_pso(cluster_drones, self.start_robot_poses)
            
            if best_form is None or pso_cost >= 1e8:
                self.get_logger().warn(f"PSO für Cluster {idx+1} gescheitert (Sichtlinie blockiert oder Singularität).")
                valid_solution = False
                break 
                
            c_move_raw = 0.0
            num_robots_in_form = len(best_form) // 2
            
            for i in range(num_robots_in_form):
                sx = self.start_robot_poses[i*2]
                sy = self.start_robot_poses[i*2+1]
                tx = best_form[i*2]
                ty = best_form[i*2+1]
                c_move_raw += math.hypot(tx - sx, ty - sy)
                
            
            
            cluster_total_cost = pso_cost + (w_move * c_move_raw)
            current_total_cost += cluster_total_cost
            formations.append(best_form)
            current_cluster_costs.append(cluster_total_cost)

            # NEU: Werte für die CSV aufaddieren (über alle Cluster hinweg)
            move_cost = w_move * c_move_raw
            total_crlb_cost += self.optimizer.best_crlb
            total_obs_cost += self.optimizer.best_obs
            total_move_cost += move_cost
            

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        self.get_logger().info(f"⏱️ Evaluierung für k={self.current_k} abgeschlossen in {elapsed_time:.3f} Sekunden.")

        if not valid_solution:
            self.get_logger().warn(f"--> k={self.current_k} ist physikalisch ungültig. Erhöhe k erzwungenermaßen.")
            self.previous_total_cost = float('inf') 
            self.current_k += 1
            return 
        
        self.get_logger().info(f"Schritt 5: Gesamtkosten J_total = {current_total_cost:.4f}")

        self.k_history.append(self.current_k)
        self.total_cost_history.append(current_total_cost)
        self.cluster_costs_history[self.current_k] = current_cluster_costs
        
        if self.current_k > 1 and current_total_cost > self.previous_total_cost:
            self.get_logger().info("--> JA (Stopp)")
            self.get_logger().info(f"    (Aktuell: {current_total_cost:.4f} > Vorher: {self.previous_total_cost:.4f})")
            
            self.get_logger().info(f"\n+++ ERGEBNIS +++")
            self.get_logger().info(f"Beste Formation ermittelt für k = {self.current_k - 1} Cluster.")
            
            # ===== NEUER BATCH-LOGIK BLOCK =====
            if self.enable_batch_evaluation:
                # Schalter ist AN: Daten des besten k speichern und Loop neu starten.
                # WICHTIG: plot_results() wird hier übersprungen, damit der Code nicht pausiert!
                self.finish_run(final_k=self.current_k - 1, final_cost=self.previous_total_cost)
                return 
            else:
                # Schalter ist AUS: Normales Verhalten für Vorführungen
                self.optimization_done = True 
                self.plot_results() 
                return
            # ===================================

        self.get_logger().info("--> NEIN, Erhöhe k = k + 1")
        
        marker_array = MarkerArray()

        if self.best_marker_array is not None:
            self.marker_pub.publish(self.best_marker_array)
                
        
        
        sorted_clusters = []
        for c_drones, form in zip(clusters, formations):
            center = np.mean(c_drones, axis=0)
            angle = np.arctan2(center[1] - 5.0, center[0] - 5.0)
            sorted_clusters.append((angle, c_drones, form))
            
        sorted_clusters.sort(key=lambda item: item[0])

        for idx, (_, cluster_drones, form) in enumerate(sorted_clusters):
            c_color = self.cluster_colors[idx % len(self.cluster_colors)]
            self.add_cluster_markers(marker_array, cluster_drones, form, 
                                     drone_color=c_color, robot_color=c_color, base_id=(idx+1)*100)
        
        self.best_marker_array = marker_array 
        self.marker_pub.publish(marker_array)

        self.previous_total_cost = current_total_cost

        # === HIER IST DER FIX: Die Werte für das aktuell beste k "einfrieren" ===
        self.best_run_crlb = total_crlb_cost
        self.best_run_move = total_move_cost
        self.best_run_obs = total_obs_cost

        self.current_k += 1

    


    def publish_drone_targets(self, labels=None):
        """Sendet die Drohnen an RViz und färbt sie nach Cluster-Zugehörigkeit (k)."""
        marker_array = MarkerArray()
        
        for idx, d in enumerate(self.all_drones):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "drone_targets"
            marker.id = idx
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            
            marker.pose.position.x = float(d[0])
            marker.pose.position.y = float(d[1])
            marker.pose.position.z = float(d[2])
            marker.pose.orientation.w = 1.0
            
            # Form und Größe
            marker.scale.x = 0.5
            marker.scale.y = 0.5
            marker.scale.z = 0.5
            
            # --- FARB-LOGIK ---
            if labels is not None and len(labels) == len(self.all_drones):
                # Wir haben eine Cluster-Zuweisung für diese Drohne!
                cluster_id = int(labels[idx])
                c_color = self.cluster_colors[cluster_id % len(self.cluster_colors)]
                
                marker.color.r = float(c_color[0])
                marker.color.g = float(c_color[1])
                marker.color.b = float(c_color[2])
                marker.color.a = 1.0 
            else:
                # Fallback (z.B. am Anfang, wenn noch nicht geclustert wurde)
                marker.color.r = 0.5
                marker.color.g = 0.5
                marker.color.b = 0.5
                marker.color.a = 0.8
            
            marker_array.markers.append(marker)

        if not hasattr(self, 'drone_target_pub'):
            self.drone_target_pub = self.create_publisher(MarkerArray, 'drone_targets_rviz', 10)
            
        self.drone_target_pub.publish(marker_array)

    def add_cluster_markers(self, marker_array, drones, formation, drone_color, robot_color, base_id):
        for idx, d in enumerate(drones):
            drone_marker = self.create_base_marker(base_id + idx, Marker.SPHERE, d[0], d[1], d[2], *drone_color)
            drone_marker.scale.x, drone_marker.scale.y, drone_marker.scale.z = 0.5, 0.5, 0.5
            marker_array.markers.append(drone_marker)
            
        robot_types = self.get_parameter('robot_types').value if self.has_parameter('robot_types') else ['A', 'A', 'A', 'A', 'B', 'B']

        num_robots = len(formation) // 2
        for i in range(num_robots):
            rx = float(formation[i*2])
            ry = float(formation[i*2+1])
            r_type = robot_types[i] if i < len(robot_types) else 'A'

            if r_type == 'A':
                scale_z = 0.2
                robot_marker = self.create_base_marker(base_id + 50 + i, Marker.CYLINDER, rx, ry, scale_z / 2.0, *robot_color)
                robot_marker.scale.x, robot_marker.scale.y, robot_marker.scale.z = 0.5, 0.5, scale_z
            else:
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
        plt.plot(self.k_history, self.total_cost_history, 'k-o', linewidth=2, label=r'Gesamtkosten ($J_{total}$)')
        
        for k, costs in self.cluster_costs_history.items():
            for idx, c in enumerate(costs):
                c_color = self.cluster_colors[idx % len(self.cluster_colors)]
                offset = (idx - len(costs)/2) * 0.05 
                plt.scatter(k + offset, c, s=100, zorder=5, color=c_color)
                plt.text(k + offset + 0.05, c, f'C{idx+1}', fontsize=9, verticalalignment='center')

        best_k = self.current_k - 1
        plt.axvline(x=best_k, color='green', linestyle='--', alpha=0.5, label=f'Optimales k = {best_k}')

        plt.title(r'Entwicklung der Gesamtkosten ($J_{total}$) über die Iterationen')
        plt.xlabel('Anzahl der Cluster (k)')
        plt.ylabel(r'Kosten ($J_{total}$)')
        plt.xticks(self.k_history) 
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        
        save_path = '/home/vboxuser/map_ws/j_total_plot.png'
        plt.savefig(save_path, dpi=300)
        self.get_logger().info(f"Plot wurde erfolgreich gespeichert unter: {save_path}")
        plt.show()



    def publish_rviz_debug_map(self):
        """Erzeugt eine farbige Kachel-Karte für RViz zur Überprüfung des ESDFs."""
        # Wir fassen 5x5 Pixel zusammen (Downsampling), damit RViz nicht abstürzt
        step = 5 
        
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "esdf_gradient"
        marker.id = 0
        marker.type = Marker.CUBE_LIST
        marker.action = Marker.ADD

        # Größe einer Kachel im Raum
        marker.scale.x = float(self.map_resolution * step)
        marker.scale.y = float(self.map_resolution * step)
        marker.scale.z = 0.05

        points = []
        colors = []

        # Maximalen Distanzwert für die Skalierung des Grüns finden
        max_dist = float(np.max(self.esdf_matrix))
        if max_dist == 0: max_dist = 1.0

        for y in range(0, self.binary_grid.shape[0], step):
            for x in range(0, self.binary_grid.shape[1], step):
                dist = self.esdf_matrix[y, x]
                is_obstacle = self.binary_grid[y, x] > 0

                p = Point()
                # Weltkoordinate für die Kachel berechnen
                p.x = float(self.map_origin_x + (x + step/2.0) * self.map_resolution)
                p.y = float(self.map_origin_y + (y + step/2.0) * self.map_resolution)
                p.z = 0.0
                
                c = ColorRGBA()
                c.a = 0.9 # volle deckkraft
                
                if is_obstacle:
                    # Hindernis = Rot
                    c.r, c.g, c.b = 1.0, 0.0, 0.0
                else:
                    # ESDF = Grüner Gradient! (Je höher die Distanz, desto grüner)
                    intensity = dist / max_dist
                    c.r = 0.0
                    c.g = float(intensity) 
                    c.b = 0.0

                points.append(p)
                colors.append(c)

        marker.points = points
        marker.colors = colors

        if not hasattr(self, 'debug_pub'):
            self.debug_pub = self.create_publisher(Marker, 'debug_grid', 1)
            
        self.debug_pub.publish(marker)
        self.get_logger().info("✅ Debug-Grid (Grüner Gradient) an RViz gesendet!")

    def calculate_drone_pose_facing_wall(self, drone_pos, obs):
        if isinstance(obs, dict):
            cx, cy = obs.get('x', 0.0), obs.get('y', 0.0)
            sx, sy = obs.get('sx', obs.get('s', 2.0)), obs.get('sy', obs.get('s', 2.0))
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

        x_surf = max(cx - sx/2.0, min(dx, cx + sx/2.0))
        y_surf = max(cy - sy/2.0, min(dy, cy + sy/2.0))
        z_surf = max(cz - sz/2.0, min(dz, cz + sz/2.0))

        vx = x_surf - dx
        vy = y_surf - dy
        vz = z_surf - dz

        if vx == 0 and vy == 0 and vz == 0:
            vx, vy, vz = cx - dx, cy - dy, cz - dz

        yaw = math.atan2(vy, vx)
        pitch = math.atan2(vz, math.hypot(vx, vy))
        roll = 0.0 

        return [dx, dy, dz, roll, pitch, yaw]
    



    def finish_run(self, final_k, final_cost):
        """Speichert die Daten des aktuellen Laufs inklusive k-Historie und triggert den nächsten."""
        
        # --- DEBUG-PRINT: Zeigt uns im Terminal, ob die Werte da sind! ---
        val_crlb = getattr(self, 'best_run_crlb', 0.0)
        val_move = getattr(self, 'best_run_move', 0.0)
        val_obs = getattr(self, 'best_run_obs', 0.0)
        
        
        self.experiment_results.append({
            'target': self.target_name,
            'run': self.current_run,
            'k': final_k,
            'cost': final_cost,
            'cost_crlb': val_crlb,
            'cost_move': val_move,
            'cost_obs': val_obs,
            'k_history': list(self.k_history),
            'cost_history': list(self.total_cost_history)
        })
        
        self.get_logger().info(f"🏁 --- Durchlauf {self.current_run}/{self.total_runs} abgeschlossen! (k={final_k}, J={final_cost:.2f}) ---")
        
        if self.current_run < self.total_runs:
            self.current_run += 1
            self.get_logger().info(f"\n================================================")
            self.get_logger().info(f"🚀 STARTE DURCHLAUF {self.current_run} VON {self.total_runs}")
            self.get_logger().info(f"================================================\n")
            
            self.current_k = 1
            self.previous_total_cost = float('inf')
            self.best_marker_array = None 
            self.optimization_done = False
            self.k_history = []
            self.total_cost_history = []
            self.cluster_costs_history = {}
            
            self.timer.reset()
        else:
            self.get_logger().info(f"🎉 Alle {self.total_runs} Durchläufe beendet! Werte Statistik aus...")
            self.evaluate_statistics_and_shutdown()
    # reset_for_next_run bleibt exakt so wie es ist! (Dort setzt du die Listen ja schon auf [] zurück)

    def reset_for_next_run(self):
        """Setzt die Variablen zurück und startet die Schleife von vorn."""
        self.current_k = 1  
        self.previous_total_cost = float('inf')
        self.k_history = []
        self.total_cost_history = []
        self.cluster_costs_history = {}
        
        self.get_logger().info(f"🔄 Starte neuen Durchlauf ({self.current_run}). Setze k=1.")
        
        # Startet den Loop wieder nach 1 Sekunde Pause
        self.timer = self.create_timer(1.0, self.evaluate_next_k)

    def evaluate_statistics_and_shutdown(self):
        """Wertet die Daten aus, schreibt die CSV und beendet ROS."""
        costs = [res['cost'] for res in self.experiment_results]
        ks = [res['k'] for res in self.experiment_results]

        self.get_logger().info("\n=========================================")
        self.get_logger().info("🏆 EXPERIMENT ABGESCHLOSSEN 🏆")
        self.get_logger().info(f"Gesamt-Durchläufe: {len(self.experiment_results)}")
        self.get_logger().info(f"Kosten (J_total): Durchschnitt = {np.mean(costs):.2f}, StdAbw = {np.std(costs):.2f}")
        self.get_logger().info(f"Gewähltes k:      Durchschnitt = {np.mean(ks):.2f}, StdAbw = {np.std(ks):.2f}")
        self.get_logger().info("=========================================\n")

        # CSV Export (Append-Modus für mehrere Messobjekte)
        file_path = os.path.expanduser('~/map_ws/pso_evaluation_results.csv')
        
        # NEU: Prüfen, ob die Datei schon existiert
        file_exists = os.path.isfile(file_path)
        
        try:
            # NEU: mode='a' (append) hängt Daten unten an, statt sie zu überschreiben
            with open(file_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                
                # NEU: Spaltenköpfe NUR schreiben, wenn die Datei neu erstellt wird
                if not file_exists:
                    writer.writerow(['Messobjekt', 'Run_ID', 'Gewaehltes_k', 'Finale_Kosten_J', 'cost_crlb', 'cost_move', 'cost_obs', 'k_Verlauf', 'Kosten_Verlauf'])
                
                for res in self.experiment_results:
                    k_hist_str = str(res['k_history'])
                    cost_hist_str = str(res['cost_history'])
                    
                    # Hier müssen die Keys exakt so heißen wie oben im Dictionary!
                    writer.writerow([
                        res['target'], 
                        res['run'], 
                        res['k'], 
                        res['cost'], 
                        res['cost_crlb'], 
                        res['cost_move'], 
                        res['cost_obs'], 
                        k_hist_str, 
                        cost_hist_str
                    ])

            self.get_logger().info(f"💾 CSV erfolgreich gespeichert/erweitert unter: {file_path}")
        except Exception as e:
            self.get_logger().error(f"Fehler beim Speichern der CSV: {e}")

        if rclpy.ok():
            rclpy.shutdown()

        sys.exit(0)

def main(args=None):
    rclpy.init(args=args)
    node = OptimizerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        # Dieser Block wird aufgerufen, wenn du Strg + C drückst!
        node.get_logger().info("\n⚠️ Experiment manuell abgebrochen (Strg+C)!")
        
        # Prüfen, ob wir im Batch-Modus sind und schon Daten gesammelt haben
        if hasattr(node, 'enable_batch_evaluation') and node.enable_batch_evaluation:
            if len(node.experiment_results) > 0:
                node.get_logger().info(f"Speichere die bisherigen {len(node.experiment_results)} Durchläufe ab...")
                # Führt die Auswertung durch, schreibt die CSV und beendet sich (sys.exit)
                node.evaluate_statistics_and_shutdown()
            else:
                node.get_logger().info("Noch kein Durchlauf vollständig beendet. Beende ohne Speichern.")
    finally:
        node.destroy_node()
        rclpy.try_shutdown()

if __name__ == '__main__':
    main()