import cv2
import numpy as np
import os

def create_blank_map(size=200):
    """Erzeugt eine weiße Leinwand (200x200 Pixel)."""
    return np.ones((size, size), dtype=np.uint8) * 255

def save_map(img, filename):
    filepath = os.path.expanduser(f'~/map_ws/{filename}')
    cv2.imwrite(filepath, img)
    print(f"✅ Gespeichert: {filepath}")

# ==========================================
# Szenario 1: Simple Form (Rechteck / Quadrat)
# ==========================================
map1 = create_blank_map()
# cv2.rectangle(bild, (x_start, y_start), (x_ende, y_ende), farbe, dicke(-1 = gefüllt))
cv2.rectangle(map1, (80, 80), (120, 120), 0, -1)
save_map(map1, 'scenario_1_simple.png')

# ==========================================
# Szenario 2: Simple Form mit Hindernis (Wand)
# ==========================================
map2 = create_blank_map()
cv2.rectangle(map2, (80, 80), (120, 120), 0, -1) # Basis-Objekt
# Massive Wand davor (auf Y-Höhe 150)
cv2.rectangle(map2, (40, 150), (160, 160), 0, -1)
save_map(map2, 'scenario_2_simple_obs.png')

# ==========================================
# Szenario 3: Komplexe Form (Breites U-Profil)
# ==========================================
map3 = create_blank_map()
# Linker Balken
cv2.rectangle(map3, (50, 60), (70, 140), 0, -1)
# Unterer Balken
cv2.rectangle(map3, (50, 120), (150, 140), 0, -1)
# Rechter Balken
cv2.rectangle(map3, (130, 60), (150, 140), 0, -1)
save_map(map3, 'scenario_3_complex.png')

# ==========================================
# Szenario 4: Komplexe Form mit vielen Hindernissen (Käfig)
# ==========================================
map4 = create_blank_map()
# Das breite U-Profil
cv2.rectangle(map4, (50, 60), (70, 140), 0, -1)
cv2.rectangle(map4, (50, 120), (150, 140), 0, -1)
cv2.rectangle(map4, (130, 60), (150, 140), 0, -1)

# Hindernis 1: Wand unten (etwas breiter und tiefer gesetzt)
cv2.rectangle(map4, (30, 170), (170, 180), 0, -1)

# Hindernis 2 & 3: Diagonale Wände/Ecken (weiter nach außen geschoben)
cv2.line(map4, (10, 20), (40, 50), 0, thickness=8)
cv2.line(map4, (190, 20), (160, 50), 0, thickness=8)
save_map(map4, 'scenario_4_complex_obs.png')