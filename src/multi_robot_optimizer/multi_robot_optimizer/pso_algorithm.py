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
        # Initialisierung der Geschwindigkeiten und Partikel-Erinnerungen
        velocities = np.zeros((num_particles, dimensions))
        personal_best_positions = particles.copy()
        personal_best_costs = np.full(num_particles, float('inf'))
        
        global_best_position = None
        best_cost = float('inf')
        
        stagnation_counter = 0
        patience = 15 # Nach 15 Runden ohne Verbesserung abbrechen
        min_improvement = 0.001
        
        # PSO Hyperparameter dynamisch laden
        max_iterations = self.params.get('max_iterations', 50)
        w = self.params.get('inertia_weight', 0.5)  
        c1 = self.params.get('c1', 1.5)             
        c2 = self.params.get('c2', 1.5)             
        
        for iteration in range(max_iterations):
            improved = False
            
            # 1. Fitness evaluieren und pBest/gBest updaten
            for i in range(num_particles):
                cost = self._calculate_cost(particles[i], drone_cluster, obstacle_coords)
                
                # Update Personal Best (lokales Optimum des Partikels)
                if cost < personal_best_costs[i]:
                    personal_best_costs[i] = cost
                    personal_best_positions[i] = particles[i].copy()
                
                # Update Global Best (Schwrm-Optimum)
                if cost < best_cost - min_improvement:
                    best_cost = cost
                    global_best_position = particles[i].copy()
                    improved = True
            
            # --- START PARTIKEL UPDATE ---
            # Zufallsmatrizen r1 und r2 (stochastische Komponente)
            r1 = np.random.rand(num_particles, dimensions)
            r2 = np.random.rand(num_particles, dimensions)
            
            # Geschwindigkeiten berechnen (Kanonische PSO Gleichung)
            velocities = (w * velocities + 
                          c1 * r1 * (personal_best_positions - particles) + 
                          c2 * r2 * (global_best_position - particles))
            
            # Maximale Geschwindigkeit begrenzen (verhindert Schwarm-Explosion)
            v_max = 2.0 
            velocities = np.clip(velocities, -v_max, v_max)
            
            # Positionen aktualisieren
            particles = particles + velocities
            # --- ENDE PARTIKEL UPDATE ---
            
            # Stagnation hochzählen oder zurücksetzen
            if not improved:
                stagnation_counter += 1
            else:
                stagnation_counter = 0 
                
            # Early Stopping
            if stagnation_counter >= patience:
                break
        
        return global_best_position, best_cost
    
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

    def _calculate_crlb_3d_from_ground(self, formation, drone_pose):
        J = np.zeros((6, 6))
        
        # DEFINITION DEINER FLOTTE: 
        # Hier legst du fest, welche Roboter du ins Feld schickst.
        # Beispiel: 4 Roboter, aber insgesamt 6 Tracker (2x Typ A, 2x Typ B)
        # Die Länge dieser Liste muss exakt self.num_robots entsprechen!
        # 1. Parameter dynamisch laden (Fallback auf 6 Roboter, falls YAML fehlt)
        robot_types = self.params.get('robot_types', ['A', 'A', 'A', 'A', 'B', 'B'])
        
        # 2. BULLETPROOF-SICHERHEIT: Liste immer exakt an die Startposen anpassen!
        if len(robot_types) < self.num_robots:
            # Liste ist zu kurz? Mit Typ 'A' auffüllen, bis es passt
            robot_types = list(robot_types) + ['A'] * (self.num_robots - len(robot_types))
        elif len(robot_types) > self.num_robots:
            # Liste ist zu lang? Abschneiden
            robot_types = robot_types[:self.num_robots]
        
        # Liste für alle berechneten Tracker-Positionen
        tracker_positions = []

        # 1. Tracker-Positionen aus den Roboter-Positionen extrahieren
        for i in range(self.num_robots):
            rx, ry = formation[i*2], formation[i*2+1]
            r_type = robot_types[i]
            
            if r_type == 'A':
                # Typ A: 1 Tracker zentral auf dem Roboter
                tracker_positions.append([rx, ry, 0.0])
                
            elif r_type == 'B':
                # Typ B: 2 Tracker mit 20cm Abstand, orthogonal zur Sichtlinie
                dx = drone_pose[0] - rx
                dy = drone_pose[1] - ry
                
                # Distanz am Boden
                dist_ground = math.hypot(dx, dy)
                
                if dist_ground > 0.001:
                    # Normalisierter Richtungsvektor
                    ux = dx / dist_ground
                    uy = dy / dist_ground
                    
                    # Orthogonalvektor (90 Grad gedreht)
                    nx = -uy
                    ny = ux
                    
                    # Basislinie: 10cm in jede Richtung
                    offset = 0.1 
                    
                    t1_x = rx + offset * nx
                    t1_y = ry + offset * ny
                    
                    t2_x = rx - offset * nx
                    t2_y = ry - offset * ny
                    
                    tracker_positions.append([t1_x, t2_y, 0.0])
                    tracker_positions.append([t2_x, t2_y, 0.0])
                else:
                    # Fallback, falls die Drohne exakt senkrecht drüber ist
                    tracker_positions.append([rx + 0.1, ry, 0.0])
                    tracker_positions.append([rx - 0.1, ry, 0.0])

        # Wenn wir durch Typ A/B Kombinationen weniger als 6 Tracker haben, 
        # ist die 6D-Matrix mathematisch nicht voll rangfähig (singulär).
        if len(tracker_positions) < 6:
            self.get_logger().warn("Achtung: Weniger als 6 Tracker im Cluster!")
            return float(1e9)

        # 2. CRLB für JEDEN TRACKER berechnen (nicht mehr pro Roboter!)
        for tracker_pos in tracker_positions:
            tx, ty, tz = tracker_pos
            
            dx = drone_pose[0] - tx
            dy = drone_pose[1] - ty
            dz = drone_pose[2] - tz 
            
            d_3d = math.sqrt(dx**2 + dy**2 + dz**2)
            
            if d_3d > 30.0 or d_3d == 0:
                continue 
                
            # LoS-Check (Hindernisprüfung)
            if self._is_los_blocked([tx, ty, tz], drone_pose[:3], self.obstacle_coords):
                continue 

            # --- POSITIONS-BLOCK ---
            sigma_pos = 0.2 + 0.3 * d_3d
            var_pos = sigma_pos ** 2
            
            ux = dx / d_3d
            uy = dy / d_3d
            uz = dz / d_3d
            
            J_pos = (1.0 / var_pos) * np.array([
                [ux**2,   ux*uy,   ux*uz],
                [ux*uy,   uy**2,   ux*uz],
                [ux*uz,   uy*uz,   uz**2]
            ])
            
            # --- WINKEL-BLOCK ---
            sigma_ang = 0.05 + 0.01 * d_3d 
            inv_var_ang = 1.0 / (sigma_ang ** 2)
            
            # --- MATRIX ZUSAMMENBAUEN ---
            J_j = np.zeros((6, 6))
            J_j[0:3, 0:3] = J_pos
            J_j[3, 3] = inv_var_ang # Roll
            J_j[4, 4] = inv_var_ang # Pitch
            J_j[5, 5] = inv_var_ang # Yaw
            
            # Akkumulation der Information aller Tracker
            J += J_j
            
        # --- BULLETPROOF MATRIX INVERTIERUNG ---
        try:
            # 1. Determinante prüfen (Toleranz etwas strikter setzen)
            det = np.linalg.det(J)
            if det < 1e-5 or math.isnan(det):
                return float(1e9) 
                
            # 2. Invertieren und Spur berechnen
            J_inv = np.linalg.inv(J)
            crlb_trace = np.trace(J_inv)
            
            # 3. DER WICHTIGSTE CHECK: CRLB darf NIEMALS negativ oder NaN sein!
            # Fängt den -10^17 Glitch ab und bestraft die fehlerhafte Formation
            if crlb_trace <= 0 or math.isnan(crlb_trace) or math.isinf(crlb_trace):
                return float(1e9)
                
            return crlb_trace
            
        except np.linalg.LinAlgError:
            # Fängt ab, falls NumPy die Matrix intern als komplett unlösbar einstuft
            return float(1e9)
            
        
    
    def _is_los_blocked(self, robot_pos, drone_pos, obstacle_coords):
        """
        Prüft mittels 3D Slab-Algorithmus, ob die Sichtlinie zwischen Roboter 
        und Drohne einen quaderförmigen Hinderniskörper (AABB) schneidet.
        """
        # Standardwerte für alte Formate
        DEFAULT_SIZE = 2.0
        DEFAULT_HEIGHT = 5.0
        
        for obs in obstacle_coords:
            # 1. ROBUSTE DATENEXTRAKTION (Angepasst für Quader)
            if isinstance(obs, dict):
                cx = obs.get('x', 0.0)
                cy = obs.get('y', 0.0)
                # Versuche sx/sy zu laden, falle sonst auf 's' zurück
                sx = obs.get('sx', obs.get('s', DEFAULT_SIZE)) 
                sy = obs.get('sy', obs.get('s', DEFAULT_SIZE))
                sz = obs.get('h', DEFAULT_HEIGHT) 
                cz = obs.get('z', sz / 2.0)
            else:
                # Prüfen, ob das neue 6D-Format [cx, cy, cz, sx, sy, sz] vorliegt
                if len(obs) == 6:
                    cx, cy, cz, sx, sy, sz = obs
                else:
                    # Fallback für alte [x, y]-Listen
                    cx = obs[0]
                    cy = obs[1]
                    sx = DEFAULT_SIZE
                    sy = DEFAULT_SIZE
                    sz = DEFAULT_HEIGHT
                    cz = sz / 2.0
                
            # 2. GRENZEN DER BOUNDING BOX (AABB) BERECHNEN
            x_min, x_max = cx - (sx / 2.0), cx + (sx / 2.0)
            y_min, y_max = cy - (sy / 2.0), cy + (sy / 2.0)
            z_min, z_max = cz - (sz / 2.0), cz + (sz / 2.0)
            
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
                continue 
                
            # Y-Achse
            if dir_y != 0:
                ty1 = (y_min - robot_pos[1]) / dir_y
                ty2 = (y_max - robot_pos[1]) / dir_y
                t_enter = max(t_enter, min(ty1, ty2))
                t_exit = min(t_exit, max(ty1, ty2))
            elif robot_pos[1] < y_min or robot_pos[1] > y_max:
                continue 
                
            # Z-Achse
            if dir_z != 0:
                tz1 = (z_min - robot_pos[2]) / dir_z
                tz2 = (z_max - robot_pos[2]) / dir_z
                t_enter = max(t_enter, min(tz1, tz2))
                t_exit = min(t_exit, max(tz1, tz2))
            elif robot_pos[2] < z_min or robot_pos[2] > z_max:
                continue 
                
            # 5. SCHNITTPUNKT-EVALUIERUNG
            if t_enter <= t_exit and t_exit >= 0 and t_enter <= 1:
                return True 
                
        return False
    

    def _calculate_obstacle_penalty(self, formation, obstacle_coords):
        penalty = 0.0
        
        # Werte für deine Parameter (siehe LaTeX-Notizen)
        d_safe = self.params.get('d_safe', 2.0)
        d_crash = self.params.get('d_crash', 0.15)

        for i in range(self.num_robots):
            rx, ry = formation[i*2], formation[i*2+1]
            
            # Finde die geringste Distanz des Roboters zu IRGENDEINEM Hindernis
            min_d_obs = float('inf')
            
            for obs in obstacle_coords:
                # Robuste Datenextraktion (wie in _is_los_blocked)
                if isinstance(obs, dict):
                    ox = obs.get('x', 0.0)
                    oy = obs.get('y', 0.0)
                    os_val = obs.get('s', 2.0)
                else:
                    ox, oy = obs[0], obs[1]
                    os_val = 2.0
                
                # Euklidische Distanz zur nächstgelegenen Außenkante des Quaders
                dx = max(0.0, abs(rx - ox) - (os_val / 2.0))
                dy = max(0.0, abs(ry - oy) - (os_val / 2.0))
                d_obs = math.hypot(dx, dy)
                
                if d_obs < min_d_obs:
                    min_d_obs = d_obs
            
            # Umsetzung der stückweise definierten Funktion aus der Thesis
            if min_d_obs <= d_crash:
                # Partikel ist physisch im Hindernis -> unendliche Kosten
                return float('inf') 
            elif min_d_obs < d_safe:
                # Partikel ist im Potenzialfeld -> exponentielle Strafe
                penalty += ((1.0 / min_d_obs) - (1.0 / d_safe)) ** 2
                
        return penalty

    def _calculate_inter_robot_penalty(self, formation):
        penalty = 0.0
        
        # Parameter analog zur Hindernisvermeidung
        d_min = 1.5    # Sicherheitsabstand (Repulsive Kraft beginnt)
        d_crash = 0.6  # 2x 0.3m Roboter-Radius -> Physischer Crash
        
        # Dynamische Schleife zur Kollisionsprüfung aller Roboter-Paare
        for i in range(self.num_robots):
            for j in range(i+1, self.num_robots):
                rx1, ry1 = formation[i*2], formation[i*2+1]
                rx2, ry2 = formation[j*2], formation[j*2+1]
                
                dist = math.hypot(rx1-rx2, ry1-ry2)
                
                if dist <= d_crash:
                    # Physisch unmöglich -> harte Restriktion (Partikel verwerfen)
                    return float('inf') 
                elif dist < d_min:
                    # Exponentieller Anstieg der Abstoßungskraft, je näher sie kommen
                    penalty += ((1.0 / dist) - (1.0 / d_min)) ** 2
                    
        return penalty