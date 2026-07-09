import cv2
import numpy as np
import math
import time

# ROS 2 Imports
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Header

# ==========================================
# KONFIGURATION (Hier anpassen!)
# ==========================================
IMAGE_PATH = '/home/vboxuser/map_ws/Mein_Luftbild.png'  # Trage hier dein Testbild ein
PIXEL_TO_METER = 0.1              # Skalierung: 1 Pixel = 10 cm in der echten Welt
BUILDING_HEIGHT = 10.0            # Angenommene Standardhöhe aller Gebäude in Metern

# Globale Variablen für den Human-in-the-Loop
selected_target_idx = 0
contours = []
image_display = None

def mouse_callback(event, x, y, flags, param):
    """Reagiert auf Mausklicks im Safety-Check-Fenster."""
    global selected_target_idx, contours, image_display
    
    if event == cv2.EVENT_LBUTTONDOWN:
        for idx, cnt in enumerate(contours):
            if cv2.pointPolygonTest(cnt, (x, y), False) >= 0:
                selected_target_idx = idx
                print(f"--> Neues Messobjekt manuell ausgewählt: Index {idx}")
                draw_and_show()
                break

def draw_and_show():
    """Zeichnet die Gebäude und markiert das Ziel grün, den Rest rot."""
    global image_display, contours, selected_target_idx
    display = image_display.copy()

    for idx, cnt in enumerate(contours):
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        if idx == selected_target_idx:
            cv2.drawContours(display, [box], 0, (0, 255, 0), 3)
            cv2.putText(display, "MESSOBJEKT", (int(rect[0][0])-40, int(rect[0][1])-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.drawContours(display, [box], 0, (0, 0, 255), 2)

    cv2.imshow("Safety Check - Klicke zur Korrektur, druecke ENTER zum Bestaetigen", display)

def main():
    global contours, image_display, selected_target_idx

    # 1. Bild laden
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        print("Fehler: Bild nicht gefunden. Pfad überprüfen!")
        return

    image_display = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. & 3. Kantenerkennung & Morphologie
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 100)
    
    kernel = np.ones((7,7), np.uint8)
    closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=3)
    
    # 4. Konturen extrahieren
    found_contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in found_contours if cv2.contourArea(c) > 100]

    if not contours:
        print("Keine Gebäude im Bild gefunden!")
        return

    # 5. Geometrische Heuristik
    h, w = img.shape[:2]
    center_x, center_y = w // 2, h // 2
    min_dist = float('inf')
    
    for idx, cnt in enumerate(contours):
        M = cv2.moments(cnt)
        if M['m00'] != 0:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            dist = math.hypot(cx - center_x, cy - center_y)
            if dist < min_dist:
                min_dist = dist
                selected_target_idx = idx

    print(f"Heuristik: Gebäude {selected_target_idx} ist am nächsten zur Bildmitte.")

    # 6. Safety Check
    cv2.namedWindow("Safety Check - Klicke zur Korrektur, druecke ENTER zum Bestaetigen", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Safety Check - Klicke zur Korrektur, druecke ENTER zum Bestaetigen", 1280, 720)
    cv2.setMouseCallback("Safety Check - Klicke zur Korrektur, druecke ENTER zum Bestaetigen", mouse_callback)

    draw_and_show()

    print("\n--- SAFETY CHECK ---")
    print("Prüfe das Bildfenster! Ist das GRÜNE Objekt das korrekte Messobjekt?")
    print("JA   -> Drücke die ENTER-Taste im Bildfenster")
    print("NEIN -> Klicke mit der Maus auf das korrekte Gebäude, danach ENTER")

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 13 or key == 10: 
            break

    cv2.destroyAllWindows()

    # ==========================================
    # 7. Zielobjekt extrahieren (Für den Drohnen-Start)
    # ==========================================
    target_obstacles = []

    for idx, cnt in enumerate(contours):
        if idx == selected_target_idx:
            x, y, bw, bh = cv2.boundingRect(cnt)
            real_w = bw * PIXEL_TO_METER
            real_h = bh * PIXEL_TO_METER
            
            real_cx = ((x + bw/2.0) - center_x) * PIXEL_TO_METER
            real_cy = ((center_y) - (y + bh/2.0)) * PIXEL_TO_METER 
            
            obs_data = [round(real_cx, 2), round(real_cy, 2), BUILDING_HEIGHT/2.0, round(real_w, 2), round(real_h, 2), BUILDING_HEIGHT]
            target_obstacles.append(obs_data)

    # ==========================================
    # 8. Binäre Karte für ROS erstellen (OccupancyGrid)
    # ==========================================
    print("\nErstelle 2.5D OccupancyGrid...")
    obstacle_grid = np.zeros((h, w), dtype=np.uint8)
    
    # Gebäude gefüllt einzeichnen (255 = Wand)
    cv2.drawContours(obstacle_grid, contours, -1, 255, thickness=cv2.FILLED)

    # Y-Achse spiegeln (OpenCV -> ROS Konvention)
    flipped_grid = np.flipud(obstacle_grid)

    # ROS Standard: 0 = Frei, 100 = Hindernis
    flat_grid = flipped_grid.flatten()
    ros_grid = np.where(flat_grid > 0, 100, 0).astype(np.int8)

    # ==========================================
    # 9. ROS 2 Map Publisher
    # ==========================================
    rclpy.init()
    node = rclpy.create_node('map_publisher_node')
    map_pub = node.create_publisher(OccupancyGrid, '/map', 10)

    grid_msg = OccupancyGrid()
    grid_msg.header = Header(frame_id="map", stamp=node.get_clock().now().to_msg())
    grid_msg.info.resolution = PIXEL_TO_METER
    grid_msg.info.width = w
    grid_msg.info.height = h

    # Ursprung setzen
    grid_msg.info.origin.position.x = - (center_x * PIXEL_TO_METER)
    grid_msg.info.origin.position.y = - (center_y * PIXEL_TO_METER)
    grid_msg.info.origin.position.z = 0.0

    grid_msg.data = ros_grid.tolist()

    print("Publiziere Map auf Topic '/map'...")
    
    # Sende die Karte mehrmals, um sicherzustellen, dass die Optimizer-Node sie empfängt
    for _ in range(5):
        map_pub.publish(grid_msg)
        time.sleep(0.5)

    print("\n\n==========================================")
    print("Kopiere DIES in deine optimizer_node.py (__init__):")
    print("==========================================")
    print(f"self.target_obstacles = {target_obstacles}")
    print("# self.avoidance_obstacles WIRD NICHT MEHR BENÖTIGT (Das macht jetzt die ROS-Map!)")
    
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()