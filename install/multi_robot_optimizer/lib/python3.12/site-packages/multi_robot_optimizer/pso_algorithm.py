import numpy as np
import math

class SwarmOptimizer:
    def __init__(self, params):
        self.params = params

    def run_pso(self, drone_cluster, start_robot_poses, obstacle_coords):
        cluster_center = np.mean(drone_cluster, axis=0)
        num_particles = 30
        # Partikel initialisieren (3 Roboter = 6 Dimensionen)
        particles = np.random.uniform(-5, 5, (num_particles, 6))
        
        for i in range(3):
            particles[:, i*2] += cluster_center[0]
            particles[:, i*2+1] += cluster_center[1]
            
        best_formation = None
        best_cost = float('inf')
        
        for _ in range(50):
            for p in particles:
                # Hier rufen wir nun die korrekte Methode auf
                cost = self._calculate_cost(p, drone_cluster, obstacle_coords)
                if cost < best_cost:
                    best_cost = cost
                    best_formation = p.copy()
        
        return best_formation, best_cost

    def _calculate_cost(self, formation, drone_cluster, obstacle_coords):
        # 1. Hard Constraints prüfen
        if not self._check_hard_constraints(formation, drone_cluster, obstacle_coords):
            return float('inf')
            
        # 2. Kosten berechnen (CRLB als Ersatz für Informations-Metrik)
        total_cost = 0.0
        for drone_pos in drone_cluster:
            # Beispielrechnung: Summe der quadrierten Distanzen
            for i in range(3):
                dist = math.hypot(formation[i*2] - drone_pos[0], formation[i*2+1] - drone_pos[1])
                total_cost += dist 
        return total_cost

    def _check_hard_constraints(self, formation, drone_cluster, obstacle_coords):
        # Inter-Robot-Kollision (Mindestabstand 1.5m)
        for i in range(3):
            for j in range(i+1, 3):
                if math.hypot(formation[i*2] - formation[j*2], formation[i*2+1] - formation[j*2+1]) < 1.5:
                    return False
        # Drohnen-Crash-Prüfung (Mindestabstand 2.0m)
        for i in range(3):
            for d in drone_cluster:
                if math.hypot(formation[i*2] - d[0], formation[i*2+1] - d[1]) < 2.0:
                    return False
        return True