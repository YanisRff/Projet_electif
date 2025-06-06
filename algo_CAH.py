from fonctions import *
import csv
import argparse
from sklearn.metrics import silhouette_score
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Mon programme avec options")
    parser.add_argument("--data", choices=["test", "final"], nargs="?", default="test", help="Data à utiliser pour l'exécution du programme")
    parser.add_argument("--k", type=int, nargs="?", default=1, help="Identifiant de la data à utiliser (ex : 1, 2...)")
    parser.add_argument("--reddim", choices=["PCA", "TSNE", "compare"],nargs="?", default="PCA", help="Fonction de réduction de dimension à utiliser")
    parser.add_argument("--hm", action='store_true', help="Afficher les heatmaps à chaque étape")
    parser.add_argument("--clean", action='store_true', help="Nettoyer les fichiers ou dossiers temporaires")
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


    Z, clusters_cah, seuil = cluster_hierarchique(points, method='ward',k=args.k)
    if args.clean == False:
        print("\n📊 Résultats de la CAH (méthode: Ward)")
        print(f"- Seuil utilisé : {seuil:.2f}")
        print(f"- Nombre de clusters détectés : {len(set(clusters_cah))}\n")
        print("| Point | Cluster |")
        print("|-------|---------|")
        for i, c in enumerate(clusters_cah):
            print(f"|  {i:<5} |   {c:<7} |")

    cluster1, matrice, indices = clustering(points, args.hm, args.k,args.clean)
    for i in range(len(indices)):
        print("point ", i, "correspond au cluster ", indices[i])
    if args.clean == False:
        print(cluster1)
        print(matrice)

    heatmap(remplissage_matrice(points))
    k = max(2, args.k)
    results_kmeans = kmeans_clustering(points, labels, k=k, show_plot=False, red_dim="PCA")
    kmeans_labels = [cluster for _, cluster in results_kmeans]

    if len(set(kmeans_labels)) > 1:
        evaluate_clusters(points, kmeans_labels)

    else:
        print("Évaluation impossible : un seul cluster détecté par KMeans.")

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
            if args.clean == False:
                print("\n🔢 Affectation des individus (K-Means)\n")
                for label, cluster in results_kmeans:
                    print(f"{label} → Cluster {cluster}")
            results_dbscan = dbscan_clustering(points=points, labels=labels, eps=1.5, min_samples=3, show_plot=True, red_dim="PCA")
            if args.clean == False:
                print("\n⚠️  DBSCAN")
                for label, cluster in results_dbscan:
                    cluster_str = "Bruit" if cluster == -1 else f"Cluster {cluster}"
                    print(f"{label} → {cluster_str}")

        elif args.reddim == "TSNE":
            results_kmeans = kmeans_clustering(points, labels, k=args.k, red_dim="TSNE")
            if args.clean == False:
                print("\nClusters with kmeans_clustering\n")
                for label, cluster in results_kmeans:
                    print(f"{label} → Cluster {cluster}")
            results_dbscan = dbscan_clustering(points=points, labels=labels, eps=1.5, min_samples=3, show_plot=True, red_dim="TSNE")
            if args.clean == False:
                print("\nClusters with DBSCAN\n")
                for label, cluster in results_dbscan:
                    print(f"{label} → Cluster {cluster}")
        elif args.reddim == "compare":
            results_kmeans = kmeans_clustering(points, labels, k=args.k, red_dim="PCA")
            results_kmeans = kmeans_clustering(points, labels, k=args.k, red_dim="TSNE")
            if args.clean == False:
                print("\nClusters with kmeans_clustering\n")
                for label, cluster in results_kmeans:
                    print(f"{label} → Cluster {cluster}")
            results_dbscan = dbscan_clustering(points=points, labels=labels, eps=1.5, min_samples=3, show_plot=True, red_dim="PCA")
            results_dbscan = dbscan_clustering(points=points, labels=labels, eps=1.5, min_samples=3, show_plot=True, red_dim="TSNE")
            if args.clean == False:
                print("\nClusters with DBSCAN\n")
                for label, cluster in results_dbscan:
                    print(f"{label} → Cluster {cluster}")
        
        stat_desc(points, labels, args.k)
