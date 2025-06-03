from fonctions import *
import csv

if __name__ == '__main__':
    """
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
    """

    labels = []
    points = []

    with open('DATA.csv', 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # Ignorer l'en-tête

        for row in reader:
            label = row[0]  # Individu XX
            coords = list(map(float, row[1:]))  # Convertir les 9 valeurs en float

            labels.append(label)
            points.append(tuple(coords))  # 9 dimensions

    # Exemple d'affichage
    print("labels =", labels)
    print("points =", points)


    #(X_min, Y_min), d_min = dist_min(points, dist_euclidienne)
    #print(f"Paire la plus proche : {X_min} – {Y_min}  (distance = {d_min:.2f})")


    #(X_m1, Y_m1), d_m1 = dist_min(points, dist_euclidienne)
    #print(f"Paire la plus proche (distance Manhattan) : {X_m1} – {Y_m1}  (d = {d_m1:.2f})")

    (X_m1, Y_m1), d_m1 = dist_min(points, dist_euclidienne)

    print(f"Paire la plus proche (distance Manhattan) : {X_m1} – {Y_m1}  (d = {d_m1:.2f})")



    matrice = remplissage_matrice(points)

    print(cluster_hierarchique(points, method='ward', seuil=2.0))



    repaire(points)
