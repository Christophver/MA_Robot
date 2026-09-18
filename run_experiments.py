import subprocess
import time

def run_final_evaluation():
    # Deine 4 Masterarbeits-Szenarien
    scenarios = [1, 2, 3, 4]
    runs_per_scenario = 1
    
    # Deine gefundenen Sweet Spots aus den Parameter-Sweeps
    w_move_fixed = 1.6
    w_obs_fixed = 1.0
    
    opt_package = "multi_robot_optimizer" 
    opt_node = "optimizer_node"
    img_package = "multi_robot_optimizer"
    img_node = "image_to_ros"

    total_runs = len(scenarios) * runs_per_scenario
    print(f"🚀 Starte FINALE EVALUIERUNG ({total_runs} Durchläufe)...")
    current_run = 0

    for s in scenarios:
        print(f"\n=======================================================")
        print(f" STARTE SZENARIO {s} (Fixierte Parameter: w_move={w_move_fixed}, w_obs={w_obs_fixed})")
        print(f"=======================================================")
        
        for i in range(1, runs_per_scenario + 1):
            current_run += 1
            print(f" ---> [Szenario {s}] Durchlauf {i}/{runs_per_scenario} (Gesamtfortschritt: {current_run}/{total_runs})")
            
            # --- SCHRITT 1: Optimierer starten ---
            # Wir übergeben hier deine Sweet Spots und zusätzlich das Szenario für den CSV-Namen!
            cmd_opt = [
                "ros2", "run", opt_package, opt_node,
                "--ros-args", 
                "-p", f"w_obs:={w_obs_fixed:.2f}",
                "-p", f"w_move:={w_move_fixed:.2f}",
                "-p", f"scenario_name:={s}"
            ]
            opt_process = subprocess.Popen(cmd_opt)
            
            time.sleep(2.5) 
            
            # --- SCHRITT 2: Bild_Knoten starten ---
            cmd_img = [
                "ros2", "run", img_package, img_node,
                "--ros-args", "-p", f"scenario:={s}"
            ]
            img_process = subprocess.Popen(cmd_img)
            
            opt_process.wait()
            
            img_process.terminate()
            img_process.wait() 
            time.sleep(1)

    print(f"\n✅ Alle {total_runs} finalen Durchläufe erfolgreich abgeschlossen!")

if __name__ == "__main__":
    run_final_evaluation()