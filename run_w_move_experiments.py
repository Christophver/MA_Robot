import subprocess
import time

def run_w_move_experiments():
    # Deine Parameter für den w_move Sweep (exakt aus deiner LaTeX-Tabelle)
    w_move_values = [10.0]
    runs_per_value = 30  # Laut deiner Tabelle waren es 15 Evaluierungsläufe (kannst du natürlich auf 30 erhöhen)
    
    # Deine spezifischen ROS 2 Paket- und Knoten-Namen
    opt_package = "multi_robot_optimizer" 
    opt_node = "optimizer_node"
    
    img_package = "multi_robot_optimizer"
    img_node = "image_to_ros"
    
    # WICHTIG: Für den w_move Test nutzen wir laut deiner Tabelle Szenario 1
    scenario = 1

    print("🚀 Starte vollautomatische Testreihe für w_move...")
    total_runs = len(w_move_values) * runs_per_value
    current_run = 0

    for w in w_move_values:
        print(f"\n=======================================================")
        print(f" STARTE SWEEP: w_move = {w}")
        print(f"=======================================================")
        
        for i in range(1, runs_per_value + 1):
            current_run += 1
            print(f" ---> [w_move={w}] Durchlauf {i}/{runs_per_value} (Gesamtfortschritt: {current_run}/{total_runs})")
            
            # --- SCHRITT 1: Optimierer mit w_move starten ---
            cmd_opt = [
                "ros2", "run", opt_package, opt_node,
                "--ros-args", "-p", f"w_move:={w}"
            ]
            opt_process = subprocess.Popen(cmd_opt)
            
            # --- SCHRITT 2: Kurz warten, bis der Optimierer zuhört ---
            time.sleep(2.5)
            
            # --- SCHRITT 3: Image_to_ros starten (Triggert Szenario 1) ---
            cmd_img = [
                "ros2", "run", img_package, img_node,
                "--ros-args", "-p", f"scenario:={scenario}"
            ]
            img_process = subprocess.Popen(cmd_img)
            
            # --- SCHRITT 4: Warten, bis sys.exit(0) im Optimierer aufgerufen wird ---
            opt_process.wait()
            
            # --- SCHRITT 5: Aufräumen ---
            img_process.terminate()
            img_process.wait() 
            
            time.sleep(1)

    print("\n✅ Alle w_move Durchläufe erfolgreich abgeschlossen!")

if __name__ == "__main__":
    run_w_move_experiments()