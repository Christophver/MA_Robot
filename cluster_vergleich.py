import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# 1. Exakt EINE 10x10 Wand generieren
points = []
for x in np.linspace(0, 9, 10):
    for z in np.linspace(1, 10, 10):
        points.append([x, 0.0, z])

X = np.array(points)
k = 4

# 2. Volumetrisches 3D-Clustering (Schichtbildung erzwingen)
# Gezielte Initialisierung der Zentroide entlang der Z-Achse, um das 
# 2x2-Quadranten-Minimum zu umgehen und horizontale Schichten zu erzwingen.
init_centers_3d = np.array([
    [4.5, 0.0, 2.0],
    [4.5, 0.0, 4.5],
    [4.5, 0.0, 7.0],
    [4.5, 0.0, 9.5]
])
kmeans_3d = KMeans(n_clusters=k, init=init_centers_3d, n_init=1, random_state=42)
labels_3d = kmeans_3d.fit_predict(X)

# 3. Projiziertes 2D-Clustering (Vertikale Sektorbildung)
# Da die Z-Achse ignoriert wird, konvergiert K-Means zwingend in vertikale Säulen.
kmeans_2d = KMeans(n_clusters=k, random_state=42, n_init=10)
labels_2d = kmeans_2d.fit_predict(X[:, :2])

# 4. RViz-ähnliches Schaubild plotten
fig = plt.figure(figsize=(14, 6))

# Linkes Koordinatensystem: 3D-Ansatz an einer Wand
ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(X[:, 0], X[:, 1], X[:, 2], c=labels_3d, cmap='tab10', s=100, depthshade=False)
ax1.set_title("Volumetrische 3D-Clusterung\n(Horizontale Schichtbildung)")
ax1.set_xlabel("X (m)")
ax1.set_ylabel("Y (m)")
ax1.set_zlabel("Z (m)")
ax1.set_ylim(-5, 5) 
ax1.set_zlim(0, 11)
ax1.view_init(elev=10, azim=-75)

# Rechtes Koordinatensystem: 2D-Ansatz an derselben Wand
ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter(X[:, 0], X[:, 1], X[:, 2], c=labels_2d, cmap='tab10', s=100, depthshade=False)
ax2.set_title("Projizierte 2D-Clusterung\n(Vertikale Sektorbildung)")
ax2.set_xlabel("X (m)")
ax2.set_ylabel("Y (m)")
ax2.set_zlabel("Z (m)")
ax2.set_ylim(-5, 5)
ax2.set_zlim(0, 11)
ax2.view_init(elev=10, azim=-75)

plt.tight_layout()
plt.show()