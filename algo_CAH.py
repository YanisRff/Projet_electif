from fonctions import *

if __name__ == '__main__':

    points = [
        (1, 1),
        (1, 2),
        (1, 5),
        (3, 4),
        (4, 3),
        (6, 2),
        (0, 4)
    ]
    labels = ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7']


    (X_min, Y_min), d_min = dist_min(points, dist_euclidienne)
    print(f"Paire la plus proche : {X_min} – {Y_min}  (distance = {d_min:.2f})")

    (X_m1, Y_m1), d_m1 = dist_min(points, dist_euclidienne)
    print(f"Paire la plus proche (distance Manhattan) : {X_m1} – {Y_m1}  (d = {d_m1:.2f})")


    Z = cluster_hierarchique(points, method='single')


    dendrogramme_dessin(Z, labels=labels, title='Dendrogramme CAH ')