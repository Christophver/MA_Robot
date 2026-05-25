import numpy as np
import math

class SwarmOptimizer:
    def __init__(self, params):
        self.params = params
        self.num_robots = 3 # Wird in run_pso dynamisch überschrieben

    def run_pso(self, drone_cluster, start_robot_poses, obstacle_coords):
        cluster_center = np.mean(drone_cluster, axis=0) 
        
        # 1. Anzahl der Roboter dynamisch auslesen (x,y pro Roboter)
        self.num_robots = len(start_robot_poses) // 2
        dimensions = self.num_robots * 2
        
        # 2. Dynamische Partikelanzahl: N = 10 * D (mindestens 30)
        num_particles = max(30, 10 * dimensions)
        
        particles = np.random.uniform(-5, 5, (num_particles, dimensions))
        
        # Streuung der Partikel um das Cluster-Zentrum
        for i in range(self.num_robots):
            particles[:, i*2] += cluster_center[0]     
            particles[:, i*2+1] += cluster_center[1]   
            
        best_formation = None
        best_cost = float('inf')
        
        for _ in range(50):
            for p in particles:
                cost = self._calculate_cost(p, drone_cluster, obstacle_coords)
                if cost < best_cost:
                    best_cost = cost
                    best_formation = p.copy()
        
        return best_formation, best_cost

    def _calculate_cost(self, formation, drone_cluster, obstacle_coords, use_minimax=False):
        crlb_values = []
        
        for drone_pos in drone_cluster:
            crlb_val = self._calculate_crlb_3d_from_ground(formation, drone_pos)
            crlb_values.append(crlb_val)
            
        if use_minimax:
            base_cost = max(crlb_values) if crlb_values else float('inf')
        else:
            base_cost = sum(crlb_values)
            
        c_obs = self._calculate_obstacle_penalty(formation, obstacle_coords)
        c_inter = self._calculate_inter_robot_penalty(formation)
        
        return base_cost + c_obs + c_inter

    def _calculate_crlb_3d_from_ground(self, formation, drone_pos):
        J = np.zeros((3, 3))
        
        # Dynamische Schleife über alle verfügbaren Roboter
        for i in range(self.num_robots):
            rx, ry = formation[i*2], formation[i*2+1]
            rz = 0.0 
            
            dx = drone_pos[0] - rx
            dy = drone_pos[1] - ry
            dz = drone_pos[2] - rz 
            
            d = math.sqrt(dx**2 + dy**2 + dz**2)
            
            if d > 30.0:
                return float('inf')
            if d == 0:
                continue 
                
            sigma = 0.2 + 0.3 * d
            variance = sigma ** 2
            
            ux = dx / d
            uy = dy / d
            uz = dz / d
            
            J_i = (1.0 / variance) * np.array([
                [ux**2,   ux*uy,   ux*uz],
                [ux*uy,   uy**2,   uy*uz],
                [ux*uz,   uy*uz,   uz**2]
            ])
            
            J += J_i
            
        det = np.linalg.det(J)
        if det < 1e-6:
            return float('inf')
            
        J_inv = np.linalg.inv(J)
        return np.trace(J_inv)

    def _calculate_obstacle_penalty(self, formation, obstacle_coords):
        penalty = 0.0
        safe_dist = 2.0 
        penalty_weight = 1000.0 
        
        # Dynamische Schleife
        for i in range(self.num_robots):
            rx, ry = formation[i*2], formation[i*2+1]
            for obs in obstacle_coords:
                d = math.hypot(rx - obs[0], ry - obs[1])
                if d < safe_dist:
                    penalty += penalty_weight * ((safe_dist - d) ** 2)
        return penalty

    def _calculate_inter_robot_penalty(self, formation):
        penalty = 0.0
        min_dist = 1.5 
        penalty_weight = 1000.0
        
        # Dynamische Schleife zur Kollisionsprüfung aller Roboter-Paare
        for i in range(self.num_robots):
            for j in range(i+1, self.num_robots):
                rx1, ry1 = formation[i*2], formation[i*2+1]
                rx2, ry2 = formation[j*2], formation[j*2+1]
                
                d = math.hypot(rx1 - rx2, ry1 - ry2)
                if d < min_dist:
                    penalty += penalty_weight * ((min_dist - d) ** 2)
        return penalty