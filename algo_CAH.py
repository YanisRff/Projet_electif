from fonctions import *
import csv
import argparse
from sklearn.metrics import silhouette_score
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Mon programme avec options")
    parser.add_argument("--data", choices=["test", "final"], nargs="?", default="test", help="Data à utiliser pour l'exécution du programme")
    parser.add_argument("--k", type=int, nargs="?", default=1, help="Identifiant de la data à utiliser (ex : 1, 2...)")
    parser.add_argument("--reddim", choices=["PCA", "TSNE", "compare"],nargs="?", default="PCA", help="Fonction de réduction de dimension à utiliser")
    args = parser.parse_args()

    if args.data == "test":
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
    elif args.data == "final":
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


    (X_m1, Y_m1), d_m1 = dist_min(points, dist_euclidienne)

    print(f"Paire la plus proche (distance Manhattan) : {X_m1} – {Y_m1}  (d = {d_m1:.2f})")


    Z, clusters_cah, seuil = cluster_hierarchique(points, method='ward',k=args.k)
    print("\nRésultats CAH:")
    print("Z : ",Z)
    print("Seuil utilisé:", seuil)
    print("Clusters:", clusters_cah)
    print(cluster_hierarchique(points, method='ward',k=args.k))
    print(clustering(points))
    
    #heatmap(remplissage matrice)

    if len(set(clusters_cah)) > 1:
        print("Silhouette Score:", round(silhouette_score(np.array(points), clusters_cah), 3))
        print(clusters_cah)
    else:
        print("Silhouette Score non calculable (un seul cluster)")

    if args.reddim == "PCA":
        repaire(points, args.k, "PCA")
    elif args.reddim == "TSNE":
        repaire(points, args.k, "TSNE")
    elif args.reddim == "compare":
        repaire(points, args.k, "PCA")
        if args.data == "final":
            repaire(points, args.k, "TSNE")


    if args.data == "final":
        if args.reddim == "PCA":
            results_kmeans = kmeans_clustering(points, labels, k=args.k, red_dim="PCA")
            print("\nClusters with kmeans_clustering\n")
            for label, cluster in results_kmeans:
                print(f"{label} → Cluster {cluster}")
            results_dbscan = dbscan_clustering(points=points, labels=labels, eps=1.5, min_samples=3, show_plot=True, red_dim="PCA")
            print("\nClusters with DBSCAN\n")
            for label, cluster in results_dbscan:
                print(f"{label} → Cluster {cluster}")
        elif args.reddim == "TSNE":
            results_kmeans = kmeans_clustering(points, labels, k=args.k, red_dim="TSNE")
            print("\nClusters with kmeans_clustering\n")
            for label, cluster in results_kmeans:
                print(f"{label} → Cluster {cluster}")
            results_dbscan = dbscan_clustering(points=points, labels=labels, eps=1.5, min_samples=3, show_plot=True, red_dim="TSNE")
            print("\nClusters with DBSCAN\n")
            for label, cluster in results_dbscan:
                print(f"{label} → Cluster {cluster}")
        elif args.reddim == "compare":
            results = kmeans_clustering(points, labels, k=args.k, red_dim="PCA")
            results = kmeans_clustering(points, labels, k=args.k, red_dim="TSNE")
        for label, cluster in results:
            print(f"{label} → Cluster {cluster}")
            results_kmeans = kmeans_clustering(points, labels, k=args.k, red_dim="PCA")
            results_kmeans = kmeans_clustering(points, labels, k=args.k, red_dim="TSNE")
            print("\nClusters with kmeans_clustering\n")
            for label, cluster in results_kmeans:
                print(f"{label} → Cluster {cluster}")
            results_dbscan = dbscan_clustering(points=points, labels=labels, eps=1.5, min_samples=3, show_plot=True, red_dim="PCA")
            results_dbscan = dbscan_clustering(points=points, labels=labels, eps=1.5, min_samples=3, show_plot=True, red_dim="TSNE")
            print("\nClusters with DBSCAN\n")
            for label, cluster in results_dbscan:
                print(f"{label} → Cluster {cluster}")

    repaire(points)