import cv2
import numpy as np
import math

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
        # Prüfe, in welcher Kontur geklickt wurde
        for idx, cnt in enumerate(contours):
            # pointPolygonTest gibt +1 zurück, wenn der Klick innerhalb der Kontur liegt
            if cv2.pointPolygonTest(cnt, (x, y), False) >= 0:
                selected_target_idx = idx
                print(f"--> Neues Messobjekt manuell ausgewählt: Index {idx}")
                draw_and_show() # Fenster mit neuen Farben aktualisieren
                break

def draw_and_show():
    """Zeichnet die Gebäude und markiert das Ziel grün, den Rest rot."""
    global image_display, contours, selected_target_idx
    display = image_display.copy()

    for idx, cnt in enumerate(contours):
        # Wir zeichnen zur Veranschaulichung die rotierte Box (OBB)
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        if idx == selected_target_idx:
            # Zielobjekt = Grün
            cv2.drawContours(display, [box], 0, (0, 255, 0), 3)
            cv2.putText(display, "MESSOBJEKT", (int(rect[0][0])-40, int(rect[0][1])-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            # Störobjekt = Rot
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

    # ==========================================
    # 2. & 3. Kantenerkennung (Canny) statt Binarisierung
    # ==========================================
    # Leichtes Weichzeichnen, um kleine Störpixel (Rauschen) zu ignorieren
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Canny Edge Detection: Findet nur die harten Gebäude-Umrisse
    edges = cv2.Canny(blurred, 30, 100)
    
    # Die gefundenen Kanten sind nur dünne Linien. Wir blähen sie auf (Dilation) 
    # und schließen sie (Closing), damit sie zu soliden Kontur-Blöcken verschmelzen.
    kernel = np.ones((7,7), np.uint8)
    closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=3)
    
    # 4. Konturen extrahieren
    found_contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter: Rauschen entfernen (alles unter 100 Pixel Fläche wird ignoriert)
    contours = [c for c in found_contours if cv2.contourArea(c) > 100]

    if not contours:
        print("Keine Gebäude im Bild gefunden!")
        return

    # ==========================================
    # 5. Geometrische Heuristik (Zentralität)
    # ==========================================
    h, w = img.shape[:2]
    center_x, center_y = w // 2, h // 2
    min_dist = float('inf')
    
    for idx, cnt in enumerate(contours):
        # Berechne den Schwerpunkt (Centroid) der Kontur
        M = cv2.moments(cnt)
        if M['m00'] != 0:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            dist = math.hypot(cx - center_x, cy - center_y)
            
            # Das Gebäude am nächsten zur Bildmitte wird der Standard-Kandidat
            if dist < min_dist:
                min_dist = dist
                selected_target_idx = idx

    print(f"Heuristik: Gebäude {selected_target_idx} ist am nächsten zur Bildmitte.")

    # ==========================================
    # 6. Human-in-the-Loop (Safety Check)
    # ==========================================
    # cv2.WINDOW_NORMAL erlaubt das freie Skalieren des Fensters mit der Maus
    cv2.namedWindow("Safety Check - Klicke zur Korrektur, druecke ENTER zum Bestaetigen", cv2.WINDOW_NORMAL)
    
    # Fenster auf eine angenehme Startgröße zwingen (z.B. HD-Auflösung 1280x720)
    cv2.resizeWindow("Safety Check - Klicke zur Korrektur, druecke ENTER zum Bestaetigen", 1280, 720)
    
    cv2.setMouseCallback("Safety Check - Klicke zur Korrektur, druecke ENTER zum Bestaetigen", mouse_callback)

    draw_and_show()

    print("\n--- SAFETY CHECK ---")
    print("Prüfe das Bildfenster! Ist das GRÜNE Objekt das korrekte Messobjekt?")
    print("JA   -> Drücke die ENTER-Taste im Bildfenster")
    print("NEIN -> Klicke mit der Maus auf das korrekte Gebäude, danach ENTER")

    # Warten auf ENTER-Taste (Keycode 13)
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 13 or key == 10: 
            break

    cv2.destroyAllWindows()

    # ==========================================
    # 7. ROS-Konvertierung (OpenCV -> ROS Map Frame)
    # ==========================================
    target_obstacles = []
    avoidance_obstacles = []

    for idx, cnt in enumerate(contours):
        # Wir extrahieren die achsenparallele Bounding Box (AABB)
        x, y, bw, bh = cv2.boundingRect(cnt)

        # Umrechnung von Pixel in Meter
        real_w = bw * PIXEL_TO_METER
        real_h = bh * PIXEL_TO_METER

        # MAGIE: Koordinatensystem an ROS anpassen!
        # OpenCV: (0,0) ist oben links, +Y geht nach unten.
        # ROS 2: (0,0) ist in der Mitte, +Y geht nach oben.
        real_cx = ((x + bw/2.0) - center_x) * PIXEL_TO_METER
        real_cy = ((center_y) - (y + bh/2.0)) * PIXEL_TO_METER 

        # Format: [cx, cy, cz, sx, sy, sz]
        obs_data = [round(real_cx, 2), round(real_cy, 2), BUILDING_HEIGHT/2.0, round(real_w, 2), round(real_h, 2), BUILDING_HEIGHT]

        if idx == selected_target_idx:
            target_obstacles.append(obs_data)
        else:
            avoidance_obstacles.append(obs_data)

    print("\n\n==========================================")
    print("Kopiere dies in deine optimizer_node.py (__init__):")
    print("==========================================")
    print(f"self.target_obstacles = {target_obstacles}")
    print(f"self.avoidance_obstacles = {avoidance_obstacles}")

if __name__ == '__main__':
    main()