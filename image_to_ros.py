import cv2
import numpy as np
import math
import time

# ROS 2 Imports
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Header
from sklearn.cluster import KMeans
from geometry_msgs.msg import PoseArray, Pose
import scipy.ndimage as ndimage

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
    cv2.drawContours(obstacle_grid, contours, -1, 255, thickness=cv2.FILLED)
    flipped_grid = np.flipud(obstacle_grid)

    flat_grid = flipped_grid.flatten()
    ros_grid = np.where(flat_grid > 0, 100, 0).astype(np.int8)

    # ==========================================
    # NEU: Drohnen-Ring NUR um das Messobjekt berechnen
    # ==========================================
    print("Berechne maßgeschneiderte Drohnen-Positionen an der Iso-Kontur...")
    target_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(target_mask, [contours[selected_target_idx]], -1, 255, thickness=cv2.FILLED)
    flipped_target = np.flipud(target_mask)

    # Lokales ESDF nur für das Zielobjekt in Metern
    target_esdf = ndimage.distance_transform_edt(flipped_target == 0) * PIXEL_TO_METER
    ring_mask = (target_esdf > 2.8) & (target_esdf < 3.2)

    # Nullpunkt für die Weltkoordinaten
    origin_x = - (center_x * PIXEL_TO_METER)
    origin_y = - (center_y * PIXEL_TO_METER)

    valid_points = []
    for y in range(ring_mask.shape[0]):
        for x in range(ring_mask.shape[1]):
            if ring_mask[y, x]:
                wx = origin_x + (x * PIXEL_TO_METER)
                wy = origin_y + (y * PIXEL_TO_METER)
                valid_points.append([wx, wy])

    valid_points = np.array(valid_points)
    drone_poses_list = []

    if len(valid_points) >= 16:
        kmeans = KMeans(n_clusters=16, random_state=42, n_init=10).fit(valid_points)
        z_levels = [1.0, BUILDING_HEIGHT - 1.0] # 1m über Boden, 1m unter Dach
        
        # Bounding Box Zentrum des Zielobjekts für den Blickwinkel (Facing Wall)
        x_box, y_box, bw, bh = cv2.boundingRect(contours[selected_target_idx])
        tcx = ((x_box + bw/2.0) - center_x) * PIXEL_TO_METER
        tcy = ((center_y) - (y_box + bh/2.0)) * PIXEL_TO_METER

        for center in kmeans.cluster_centers_:
            for z in z_levels:
                # Blickwinkel zur Gebäudemitte berechnen (Yaw)
                vx = tcx - center[0]
                vy = tcy - center[1]
                yaw = math.atan2(vy, vx)
                drone_poses_list.append([center[0], center[1], z, yaw])
    else:
        print("WARNUNG: Zielkontur zu klein für stabilen Drohnen-Ring!")

    # ==========================================
    # 9. ROS 2 Node initialisieren und publizieren
    # ==========================================
    rclpy.init()
    node = rclpy.create_node('map_publisher_node')
    map_pub = node.create_publisher(OccupancyGrid, '/map', 10)
    drone_pub = node.create_publisher(PoseArray, '/drone_targets', 10)

    # OccupancyGrid Nachricht
    grid_msg = OccupancyGrid()
    grid_msg.header = Header(frame_id="map", stamp=node.get_clock().now().to_msg())
    grid_msg.info.resolution = PIXEL_TO_METER
    grid_msg.info.width = w
    grid_msg.info.height = h
    grid_msg.info.origin.position.x = origin_x
    grid_msg.info.origin.position.y = origin_y
    grid_msg.data = ros_grid.tolist()

    # PoseArray Nachricht für Drohnen
    pose_array_msg = PoseArray()
    pose_array_msg.header = Header(frame_id="map", stamp=node.get_clock().now().to_msg())
    
    for dp in drone_poses_list:
        pose = Pose()
        pose.position.x = float(dp[0])
        pose.position.y = float(dp[1])
        pose.position.z = float(dp[2])
        # Orientierung aus Yaw-Winkel generieren (Quaternion)
        pose.orientation.z = math.sin(dp[3] / 2.0)
        pose.orientation.w = math.cos(dp[3] / 2.0)
        pose_array_msg.poses.append(pose)

    print("Publiziere Map und Drohnen-Messpunkte...")
    for _ in range(5):
        map_pub.publish(grid_msg)
        drone_pub.publish(pose_array_msg)
        time.sleep(0.5)

    print("✅ Übertragung erfolgreich abgeschlossen.")
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()