import rclpy
from rclpy.node import Node
import numpy as np
import math
import time
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import scipy.ndimage as ndimage

# ROS 2 Messages
from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import OccupancyGrid

# Eigene Imports
from multi_robot_optimizer.pso_algorithm import SwarmOptimizer, ESDFMapVectorized

class OptimizerNode(Node):
    def __init__(self):
        super().__init__('formation_optimizer_node')

        # --- HIER: DIE DEFINITION DES MESSOBJEKTS ---
        # (Füge hier die Zeile ein, die dir dein image_to_ros.py ausspuckt!)
        self.target_obstacles = [[0.0, 0.0, 5.0, 20.0, 15.0, 10.0]]
        
        # avoidance_obstacles WURDEN ENTFERNT -> Das übernimmt jetzt das OccupancyGrid!
        
        # ==========================================
        # Schritt 1: Dateneingabe & Parameter
        # ==========================================
        self.declare_parameter('w_crlb', 1.0)
        self.declare_parameter('w_move', 0.1)
        self.declare_parameter('w_obs', 0.5)
        self.declare_parameter('max_drone_dist', 30.0)
        self.declare_parameter('robot_types', ['A', 'A', 'B', 'B'])
        self.declare_parameter('max_iterations', 50)
        self.declare_parameter('c1', 1.5)
        self.declare_parameter('c2', 1.5)
        self.declare_parameter('inertia_weight', 0.5)
        self.declare_parameter('d_safe', 2.0)
        self.declare_parameter('d_crash', 0.15) 
        
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
        
        # ROS Publisher
        self.marker_pub = self.create_publisher(MarkerArray, 'formation_markers', 10)
        self.obstacle_pub = self.create_publisher(MarkerArray, 'obstacle_markers', 10)

        # NEU: ROS Subscriber für die Map
        self.map_received = False
        self.map_sub = self.create_subscription(OccupancyGrid, '/map', self.map_callback, 10)

        self.cluster_colors = [
            (0.122, 0.467, 0.706), (0.682, 0.780, 0.910), (1.000, 0.498, 0.055),
            (1.000, 0.733, 0.471), (0.173, 0.627, 0.173), (0.596, 0.875, 0.541),
            (0.839, 0.153, 0.157), (1.000, 0.596, 0.588), (0.580, 0.404, 0.741),
            (0.773, 0.690, 0.835), (0.549, 0.337, 0.294), (0.769, 0.612, 0.580),
            (0.890, 0.467, 0.761), (0.969, 0.714, 0.824), (0.498, 0.498, 0.498),
            (0.780, 0.780, 0.780), (0.737, 0.741, 0.133), (0.859, 0.859, 0.553),
            (0.090, 0.745, 0.812), (0.620, 0.855, 0.898),
        ]

        self.start_robot_poses = [
            0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 
            0.0, 1.0, 1.0, 1.0, 2.0, 1.0   
        ]
        
        # --- Drohnen generieren (nur rund um das Zielobjekt!) ---
        generated_drones_3d = []
        y_coords = np.linspace(-7.0, 7.0, 5)
        z_coords = np.linspace(0.5, 9.5, 2)
        
        for x in [-11.0, 11.0]:
            for y in y_coords:
                for z in z_coords:
                    generated_drones_3d.append([float(x), float(y), float(z)])
                    
        x_coords = np.linspace(-9.5, 9.5, 7)
        for y in [-8.5, 8.5]:
            for x in x_coords:
                for z in z_coords:
                    generated_drones_3d.append([float(x), float(y), float(z)])

        self.get_logger().info(f"Test-Szenario generiert: {len(generated_drones_3d)} Drohnen-Messpunkte erzeugt.")
        
        drones_6d = []
        main_obstacle = self.target_obstacles[0] 
        
        for pos in generated_drones_3d:
            pose = self.calculate_drone_pose_facing_wall(pos, main_obstacle)
            drones_6d.append(pose)
            
        self.all_drones = np.array(drones_6d)
        
        # Initialisierung
        self.current_k = 1
        self.optimization_done = False 
        self.k_history = []
        self.total_cost_history = []
        self.cluster_costs_history = {}
        self.previous_total_cost = float('inf')
        self.best_marker_array = None 
        
        # WICHTIG: Timer ist jetzt auf None! Er wird erst gestartet, wenn die Karte da ist.
        self.timer = None
        self.get_logger().info("Warte auf 2.5D Karte vom Topic '/map'...")

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
        self.binary_grid = np.where(grid_2d > 50, 1, 0).astype(np.uint8)

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
        self.get_logger().info("Umgebung erfolgreich initialisiert. Starte PSO...")

        # JETZT ERST den Loop starten!
        self.timer = self.create_timer(4.0, self.evaluate_next_k)


    def evaluate_next_k(self):
        self.publish_obstacles()

        start_time = time.perf_counter()

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
        
        clusters = []
        if self.current_k == 1:
            clusters.append(self.all_drones.tolist())
        else:
            kmeans = KMeans(n_clusters=self.current_k, random_state=42, n_init=10).fit(self.all_drones)
            for i in range(self.current_k):
                clusters.append(self.all_drones[kmeans.labels_ == i].tolist())
                
        current_total_cost = 0.0
        formations = []
        current_cluster_costs = [] 
        valid_solution = True
        
        # Dynamisch w_move auslesen, sonst 0.1
        w_move = self.get_parameter('w_move').value if self.has_parameter('w_move') else 0.1
        
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
                
            max_travel_dist = num_robots_in_form * 50.0 
            norm_move = min(c_move_raw / max_travel_dist, 1.0) if max_travel_dist > 0 else 0.0
            
            cluster_total_cost = pso_cost + (w_move * norm_move)
            current_total_cost += cluster_total_cost
            formations.append(best_form)
            current_cluster_costs.append(cluster_total_cost)
            
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
            
            self.optimization_done = True 
            self.plot_results() 
            return

        self.get_logger().info("--> NEIN, Erhöhe k = k + 1")
        
        marker_array = MarkerArray()

        if self.best_marker_array is not None:
            self.marker_pub.publish(self.best_marker_array)
                
        self.publish_obstacles()
        
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
        self.current_k += 1

    def publish_obstacles(self):
        marker_array = MarkerArray()
        
        def create_markers(obs_list, r, g, b):
            for obs in obs_list:
                marker = Marker()
                marker.header.frame_id = "map"
                marker.header.stamp = self.get_clock().now().to_msg()
                marker.ns = "obstacles"
                marker.id = len(marker_array.markers) 
                marker.type = Marker.CUBE
                marker.action = Marker.ADD
                
                marker.pose.position.x = float(obs[0])
                marker.pose.position.y = float(obs[1])
                marker.pose.position.z = float(obs[2])
                marker.pose.orientation.w = 1.0 

                marker.scale.x = float(obs[3])
                marker.scale.y = float(obs[4])
                marker.scale.z = float(obs[5])

                marker.color.r = r
                marker.color.g = g
                marker.color.b = b
                marker.color.a = 0.7 

                marker_array.markers.append(marker)

        # NUR NOCH DAS ZIELOBJEKT ZEICHNEN (Der Rest ist in der OccupancyGrid Map)
        create_markers(self.target_obstacles, 0.2, 0.8, 0.2)
        
        self.obstacle_pub.publish(marker_array)

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