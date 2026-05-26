import numpy as np
import math

class SwarmOptimizer:
    def __init__(self, params):
        self.params = params
        self.num_robots = 3 # Wird in run_pso dynamisch überschrieben
        self.obstacle_coords = []

    def run_pso(self, drone_cluster, start_robot_poses, obstacle_coords):
        cluster_center = np.mean(drone_cluster, axis=0) 
        self.obstacle_coords = obstacle_coords
        
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
        
        # ==========================================
        # 3. SCHWARM-BEWEGUNG MIT ABBRUCHKRITERIEN
        # ==========================================
        best_cost = float('inf')
        stagnation_counter = 0
        patience = 15 # Nach 15 Runden ohne Verbesserung abbrechen
        max_iterations = 200 # Maximales Sicherheitsnetz
        min_improvement = 1e-4 # Was gilt als signifikante Verbesserung?
        
        for iteration in range(max_iterations):
            improved = False
            
            for p in particles:
                cost = self._calculate_cost(p, drone_cluster, obstacle_coords)
                # Prüfen, ob die neue Formation spürbar besser ist
                if cost < best_cost - min_improvement:
                    best_cost = cost
                    best_formation = p.copy()
                    improved = True
            
            # Stagnation hochzählen oder zurücksetzen
            if not improved:
                stagnation_counter += 1
            else:
                stagnation_counter = 0 
                
            # Early Stopping auslösen, wenn wir "feststecken"
            if stagnation_counter >= patience:
                # Hier könntest du später einen log einbauen, um das in ROS zu sehen, z.B.:
                # print(f"PSO Stop durch Stagnation bei Iteration {iteration}")
                break
        
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
            # WICHTIG: Definiere robot_pos HIER, bevor du es benutzt!
            robot_pos = [rx, ry, rz]
            
            dx = drone_pos[0] - rx
            dy = drone_pos[1] - ry
            dz = drone_pos[2] - rz 
            
            d = math.sqrt(dx**2 + dy**2 + dz**2)
            
            # 1. Partielles LoS-Kriterium für Redundanz ---
            # Wenn der Roboter zu weit weg ist, liefert er keine Messdaten (J_i = 0).
            # Wir brechen aber NICHT ab, sondern ignorieren ihn für diese Drohne einfach!
            if d > 30.0 or d == 0:
                continue 
                
            # 2. NEU: LoS-Sichtbarkeits-Check
            if self._is_los_blocked(robot_pos, drone_pos, self.obstacle_coords):
                continue # Roboter sieht Drohne nicht durch Hindernis

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
            
        # Die echte Observabilitäts-Prüfung ---
        # Die Determinante wird (nahe) 0, wenn:
        # a) Weniger als 3 Roboter die Drohne sehen (zu wenig Infos)
        # b) Die Roboter in einer geraden Linie stehen (schlechte Geometrie)
        
        det = np.linalg.det(J)
        if det < 1e-6:
            # Einfach nur zurückgeben, kein Logger-Aufruf hier!
            return float(1e9) # Wir geben einen sehr hohen (aber endlichen) Wert zurück
            
        J_inv = np.linalg.inv(J)
        return np.trace(J_inv)
    
    def _is_los_blocked(self, robot_pos, drone_pos, obstacle_coords):
        """
        Prüft mittels 3D Slab-Algorithmus, ob die Sichtlinie zwischen Roboter 
        und Drohne einen würfelförmigen/quaderförmigen Hinderniskörper (AABB) schneidet.
        """
        # Standardwerte, falls die Daten unvollständig übergeben werden
        DEFAULT_SIZE = 2.0
        DEFAULT_HEIGHT = 5.0
        
        for obs in obstacle_coords:
            # 1. ROBUSTE DATENEXTRAKTION
            # Prüfen, ob wir ein Dictionary {'x': 5, 'y': 5, 's': 2} oder eine Liste [5, 5] haben
            if isinstance(obs, dict):
                # .get() nutzt den Standardwert, falls der Schlüssel nicht existiert
                x = obs.get('x', 0.0)
                y = obs.get('y', 0.0)
                s = obs.get('s', DEFAULT_SIZE)  # Breite/Tiefe
                h = obs.get('h', DEFAULT_HEIGHT) # Höhe
                z = obs.get('z', h / 2.0)       # Z-Zentrum (liegt standardmäßig auf dem Boden)
            else:
                x = obs[0]
                y = obs[1]
                s = DEFAULT_SIZE
                h = DEFAULT_HEIGHT
                z = h / 2.0
                
            # 2. GRENZEN DER BOUNDING BOX (AABB) BERECHNEN
            s_half = s / 2.0
            h_half = h / 2.0
            
            x_min, x_max = x - s_half, x + s_half
            y_min, y_max = y - s_half, y + s_half
            z_min, z_max = z - h_half, z + h_half
            
            # 3. RICHTUNGSVEKTOR DER SICHTLINIE
            dir_x = drone_pos[0] - robot_pos[0]
            dir_y = drone_pos[1] - robot_pos[1]
            dir_z = drone_pos[2] - robot_pos[2]
            
            t_enter = float('-inf')
            t_exit = float('inf')
            
            # 4. SLAB-TEST FÜR ALLE 3 ACHSEN
            
            # X-Achse
            if dir_x != 0:
                tx1 = (x_min - robot_pos[0]) / dir_x
                tx2 = (x_max - robot_pos[0]) / dir_x
                t_enter = max(t_enter, min(tx1, tx2))
                t_exit = min(t_exit, max(tx1, tx2))
            elif robot_pos[0] < x_min or robot_pos[0] > x_max:
                continue # Strahl ist parallel und verfehlt die Box auf der X-Achse
                
            # Y-Achse
            if dir_y != 0:
                ty1 = (y_min - robot_pos[1]) / dir_y
                ty2 = (y_max - robot_pos[1]) / dir_y
                t_enter = max(t_enter, min(ty1, ty2))
                t_exit = min(t_exit, max(ty1, ty2))
            elif robot_pos[1] < y_min or robot_pos[1] > y_max:
                continue # Verfehlt auf der Y-Achse
                
            # Z-Achse
            if dir_z != 0:
                tz1 = (z_min - robot_pos[2]) / dir_z
                tz2 = (z_max - robot_pos[2]) / dir_z
                t_enter = max(t_enter, min(tz1, tz2))
                t_exit = min(t_exit, max(tz1, tz2))
            elif robot_pos[2] < z_min or robot_pos[2] > z_max:
                continue # Verfehlt auf der Z-Achse (z.B. Drohne fliegt weit drüber)
                
            # 5. SCHNITTPUNKT-EVALUIERUNG
            # Wenn t_enter <= t_exit, durchquert der unendliche Strahl den Würfel.
            # Wir müssen nur noch prüfen, ob der Schnittpunkt im Segment zwischen 
            # Roboter (t=0) und Drohne (t=1) liegt.
            if t_enter <= t_exit and t_exit >= 0 and t_enter <= 1:
                return True # Sicht definitiv durch Würfel blockiert!
                
        return False
    

    def _calculate_obstacle_penalty(self, formation, obstacle_coords):
        penalty = 0.0
        safe_margin = 0.5 # Zusätzlicher Sicherheitsabstand zum Würfel
        
        for i in range(self.num_robots):
            rx, ry = formation[i*2], formation[i*2+1]
            for obs in obstacle_coords:
                # Prüfen, ob der Roboter innerhalb der Würfelgrenzen ist
                if (obs['x'] - obs['s']/2 - safe_margin < rx < obs['x'] + obs['s']/2 + safe_margin) and \
                   (obs['y'] - obs['s']/2 - safe_margin < ry < obs['y'] + obs['s']/2 + safe_margin):
                    penalty += 100.0 # Harte Strafe für Kollision
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