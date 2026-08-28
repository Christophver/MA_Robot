import subprocess
import time

def run_all_experiments():
    # Deine Parameter für den Sweep
    w_obs_values = [1.0,10.0,100.0]
    runs_per_value = 30
    
    # Deine spezifischen ROS 2 Paket- und Knoten-Namen
    opt_package = "multi_robot_optimizer" 
    opt_node = "optimizer_node"
    
    img_package = "multi_robot_optimizer"
    img_node = "image_to_ros"
    scenario = 3

    print("🚀 Starte vollautomatische Testreihe (Workflow: Opt -> Wait -> Img)...")
    total_runs = len(w_obs_values) * runs_per_value
    current_run = 0

    for w in w_obs_values:
        print(f"\n=======================================================")
        print(f" STARTE SWEEP: w_obs = {w}")
        print(f"=======================================================")
        
        for i in range(1, runs_per_value + 1):
            current_run += 1
            print(f" ---> [w_obs={w}] Durchlauf {i}/{runs_per_value} (Gesamtfortschritt: {current_run}/{total_runs})")
            
            # --- SCHRITT 1: Optimierer im Hintergrund starten ---
            cmd_opt = [
                "ros2", "run", opt_package, opt_node,
                "--ros-args", "-p", f"w_obs:={w}"
            ]
            # Popen startet den Prozess und lässt das Skript sofort weiterlaufen
            opt_process = subprocess.Popen(cmd_opt)
            
            # --- SCHRITT 2: Kurz warten, bis der Optimierer zuhört ---
            time.sleep(2.5) # 2.5 Sekunden sollten für den ROS-Start locker reichen
            
            # --- SCHRITT 3: Image_to_ros starten (Triggert die Daten) ---
            cmd_img = [
                "ros2", "run", img_package, img_node,
                "--ros-args", "-p", f"scenario:={scenario}"
            ]
            img_process = subprocess.Popen(cmd_img)
            
            # --- SCHRITT 4: Skript pausieren, bis der Optimierer GANZ FERTIG ist ---
            # Hier wartet Python, bis in deinem Knoten sys.exit(0) aufgerufen wird
            opt_process.wait()
            
            # --- SCHRITT 5: Aufräumen ---
            # Wir beenden den image_to_ros Knoten, damit beim nächsten Durchlauf 
            # alles wieder bei 0 anfängt.
            img_process.terminate()
            img_process.wait() 
            
            # 1 Sekunde Pause, damit ROS 2 die Ports sauber freigibt
            time.sleep(1)

    print(f"\n✅ Alle {total_runs} Durchläufe erfolgreich abgeschlossen!")
    print("Du kannst jetzt deine Auswertung starten!")

if __name__ == "__main__":
    run_all_experiments()