
from math import sqrt,inf
from scipy.stats import ttest_ind,ttest_1samp,t
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_samples, silhouette_score
from scipy.spatial.distance import cdist, pdist

def mean(arr):
    #Calcule la moyenne en prenant un tableau de int en entrée
    sum=0
    for i in range(len(arr)):
        sum+= arr[i]
    return sum/len(arr)

def mediane(arr):
    arr = sorted(arr)
    n = len(arr)
    if n == 0:
        return None  # ou lever une erreur, selon ce que tu veux
    if n % 2 == 1:  # taille impair
        return arr[n // 2]
    else:  # taille pair
        return (arr[n//2 - 1] + arr[n//2]) / 2

def variance(arr):
    # Calculate the variance of an array of integers
    if len(arr) <= 1:
        return 0  # Return 0 or handle this case as appropriate for your application

    moy = mean(arr)
    var = 0
    for i in range(len(arr)):
        var += (arr[i] - moy) ** 2
    return var / (len(arr) - 1)


def ecart_type(variance):
    # Calcule la ecart-type en prenant une variance en entrée
    return sqrt(variance)

def v_min(arr):
    """Retourne la valeur minimale d'un tableau"""
    min = inf
    for i in range(len(arr)):
        if arr[i] < min:
            min = arr[i]

    return min

def v_max(arr):
    """Retourne la valeur maximale d'un tableau"""
    max = -inf
    for i in range(len(arr)):
        if arr[i] > max:
            max = arr[i]
    return max

def etendue(max,min):
    return max-min


def covariance(X,Y):
    """Prend deux tableaux,X et Y en entrée et retourne la covariance"""
    moy_x = mean(X)
    moy_y = mean(Y)
    sum = 0
    for i in range(len(X)):
        sum+= (X[i]-moy_x)*(Y[i]-moy_y)
    return sum/(len(X)-1)

def coefficient_regression_b1(cov,var):
    return cov/var

def coefficient_regression_b0(mean_y,b1,mean_x):
    return mean_y-b1*mean_x

def droite_regression(b0,b1,X):
    y = []
    for i in range(len(X)):
        y.append(b0+b1*X[i])
    return y



def coefficient_determination_r2(SCE,SCT):
    return 1-(SCE/SCT)

def SCE(Y,droite_regression):
    """Retourne la somme des carrés des erreyrs"""
    sum = 0
    for i in range(len(Y)):
        sum+= (Y[i]-droite_regression[i])*(Y[i]-droite_regression[i])
    return sum

def SCT(Y,mean_y):
    """Retourne la somme des carrés totaux"""
    sum = 0
    for i in range(len(Y)):
        sum+= (Y[i]-mean_y)*(Y[i]-mean_y)
    return sum

def SCR(mean_y,droite_regression):
    """Retourne la somme des carrés de la régression"""
    sum=0
    for i in range(len(droite_regression)):
        sum+= (droite_regression[i]-mean_y)*(droite_regression[i]-mean_y)
    return sum

def residus(Y,droite_regression):
    """Calcul les résidus, l'écart entre les valeurs observées et les valeurs prédites"""
    y=[]
    for i in range(len(Y)):
        y.append(Y[i]-droite_regression[i])
    return y

def mean_squared_error(SCE,nombre_observation):
    """Retourne la somme des carrés des erreurs"""
    return SCE/(nombre_observation-2)

def RMSE(MSE):
    """Retourne la racine de la somme des carrés des erreurs"""
    return sqrt(MSE)


def standard_error(X,RMSE,mean):
    """calcule l'erreur standard de la pente"""
    sum=0
    for i in range(len(X)):
        sum+= (X[i]-mean)**2
    return RMSE/(sqrt(sum))


def standard_error_origin(X,rmse,nombre_observation,mean):
    """calcule l'erreur standard de l'ordonnée à l'origine"""
    sum = 0
    for i in range(len(X)):
        sum += (X[i] - mean) ** 2

    return rmse*sqrt((1/nombre_observation) + (mean**2)/sum)


def statistique_t(b1,std_error):
    return b1/std_error



"""--------------------PARTIE 5------------------"""

def dist_euclidienne(p1, p2):
    """
    Distance euclidienne entre deux points
    """
    dist = 0
    for i in range (len(p1)):
        dist += (p2[i] - p1[i]) ** 2
    return sqrt(dist)


def dist_manhattan(p1, p2):
    """
    Distance de Manhattan (L1) entre deux points
    """
    dist = 0
    for i in range (len(point1)):
        dist += abs(point2[i] - point1[i])
    return dist


def dist_chebyshev(p1, p2):
    """
    Distance Chebyshev entre deux points
    """
    dist = max(abs(p1[0] - p2[0]), abs(p1[1]- p2[1]))

    return dist


def dist_min(points,distance_func):
    """
    Paire de points la plus proche dans la liste (X,Y)
    Retourne les deux points les plus proches
    """
    min_distance = inf
    pair = (None, None)
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            d = distance_func(points[i],points[j])
            if d < min_distance:
                min_distance = d
                pair = (points[i],points[j])

    return pair, min_distance


def repaire(full_points,n_clusters=1, red_dim="PCA"):

    """
    Utilise les différentes méthodes de clustering afin de réaliser un repaire étape par étape de l'avancé des différents
    algorithmes
    """
    if len(full_points[0]) > 2:
        if red_dim =="PCA":
            pca = PCA(n_components=2)
            points = pca.fit_transform(full_points)
            points = points.tolist()
        elif red_dim == "TSNE":
            full_points_np = np.array(full_points)
            n_samples = full_points_np.shape[0]
            perplexity = min(30, n_samples - 1)
            tsne = TSNE(n_components=2, perplexity=perplexity)
            points = tsne.fit_transform(full_points_np)
            points = points.tolist()
    else:
        points = full_points[:]

    X = [p[0] for p in points]
    Y = [p[1] for p in points]

    plt.scatter(X,Y,color='black')

    while len(points)>1+n_clusters-1:
        (p1,p2),distance = dist_min(points,dist_euclidienne)
        new_point = [(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2]
        plt.pause(0.5)
        points.append(new_point)
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], color='blue')
        if(p1):
            points.remove(p1)
        if(p2):
            points.remove(p2)
    plt.show()


def seuil_max_gap(Z,k=1):
    """
    calcule le seuil pour le dendrogramme, affiche une ligne en pointillé
    """
    distances = Z[:, 2]
    sorted_distances = np.sort(distances)[::-1]

    if k <= 1 or k > len(Z) + 1:
        return None
    seuil = (sorted_distances[k - 2] + sorted_distances[k - 1]) / 2
    return seuil

def remplissage_matrice(ensemble_points):
    matrice = []
    for i in range(len(ensemble_points)):
        ligne = []
        for j in range(len(ensemble_points)):
            ligne.append(round(dist_euclidienne(ensemble_points[i], ensemble_points[j])**2, 2))
        matrice.append(ligne)
    return matrice


def clustering(points, show_plot = True,k=1, clean = True):
    pt = points[:]
    clusters=[]
    matrice = remplissage_matrice(pt)
    count=1
    dim = len(points[0])
    n = len(points)
    cluster_ids = list(range(n))  # Chaque point initial a son propre identifiant
    pt_indices = list(range(n))
    count_max = len(matrice)
    while len(matrice)>k:

        min_dist = inf
        indice_x = 0
        indice_y = 0
        for i in range(len(matrice)):
            for j in range(len(matrice[i])):
                if matrice[i][j] < min_dist and matrice[i][j] !=0:
                    min_dist = matrice[i][j]
                    indice_x = i
                    indice_y = j


        clusters.append({pt[indice_x], pt[indice_y],min_dist})

        id_x = cluster_ids[pt_indices[indice_x]]
        id_y = cluster_ids[pt_indices[indice_y]]
        new_id = min(id_x, id_y)
        old_id = max(id_x, id_y)
        for i in range(n):
            if cluster_ids[i] == old_id:
                cluster_ids[i] = new_id

        centre = ()
        for i in range(len(points[0])):
            centre += (((pt[indice_x][i] + pt[indice_y][i])/2),)

        pt[indice_x] = centre
        pt.remove(pt[indice_y])

        pt_indices.pop(indice_y)

        matrice = remplissage_matrice(pt)
        if clean == False:
            print(f"\n📌 Étape {count - 1} de la CAH")
            print(f"🔗 Fusion des points aux indices {indice_x} et {indice_y}")
            print(f"   ↳ Distance : {round(min_dist, 2)}")
            print(f"   ↳ Nouveau centre : {centre}\n")
    
            print("📐 Matrice des distances (carrées) :")
            header = "       " + "".join([f"{i:^8}" for i in range(len(matrice))])
            print(header)
            print("-" * len(header))
            for i, row in enumerate(matrice):
                row_str = f"{i:^7}" + "".join([f"{val:^8.2f}" for val in row])
                print(row_str)
            print("-" * len(header))

        count +=1
        if show_plot:
            heatmap(matrice)

    return matrice,clusters, cluster_ids




def cluster_hierarchique(points, method='single', k=1):
    """
    Classification Ascendante Hiérarchique (CAH) avec seuil automatique si non fourni.
    """
    data = np.array(points)
    Z = linkage(data, method=method, metric='euclidean')

    seuil = seuil_max_gap(Z, k)

    dendrogram(Z)
    if seuil != None:
        plt.axhline(y=seuil, color='black', linestyle='--')
    plt.title('Dendrogramme CAH (seuil auto)' if seuil else 'Dendrogramme CAH')
    plt.xlabel('Points')
    plt.ylabel('Distance')
    plt.show()
    if seuil != None:
        clusters = fcluster(Z, t=seuil, criterion='distance')
    else:
        clusters = None
    return Z, clusters, seuil

def kmeans_clustering(points, labels, k=3, show_plot=True, red_dim="PCA"):
    points = np.array(points)
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(points)
    cluster_labels = kmeans.labels_

    results = list(zip(labels, cluster_labels))

    if show_plot:
        if red_dim.upper() == "TSNE":
            perplexity = min(30, max(5, len(points) // 3))
            reducer = TSNE(n_components=2, random_state=42, perplexity=perplexity)
            points_2D = reducer.fit_transform(points)
            centers_2D = None
        else:
            reducer = PCA(n_components=2)
            points_2D = reducer.fit_transform(points)
            centers_2D = reducer.transform(kmeans.cluster_centers_)

        plt.figure(figsize=(8, 6))

        for i in range(k):
            cluster_points = [pt for pt, c in zip(points_2D, cluster_labels) if c == i]
            xs = [pt[0] for pt in cluster_points]
            ys = [pt[1] for pt in cluster_points]
            plt.scatter(xs, ys, label=f'Cluster {i}', s=50)

        if centers_2D is not None:
            plt.scatter(centers_2D[:, 0], centers_2D[:, 1], c='black', marker='X', s=200, label='Centres')

        for (x, y), name in zip(points_2D, labels):
            plt.text(x, y, name, fontsize=8)

        plt.title(f"K-Means clustering (projection {red_dim.upper()})")
        plt.xlabel(f"{red_dim.upper()} 1")
        plt.ylabel(f"{red_dim.upper()} 2")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return results

def heatmap(dist_matrix):
    sns.heatmap(dist_matrix, cmap='viridis')
    plt.title("Matrice des distances entre points 9D")
    plt.show()


def dbscan_clustering(points, labels, eps=1.5, min_samples=3, show_plot=True, red_dim="PCA"):
    points = np.array(points)

    # Standardisation (important pour DBSCAN)
    scaler = StandardScaler()
    points_scaled = scaler.fit_transform(points)

    # Application de DBSCAN
    db = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_labels = db.fit_predict(points_scaled)

    results = list(zip(labels, cluster_labels))

    if show_plot:
        # Réduction de dimension selon l'argument red_dim
        if red_dim.upper() == "TSNE":
            perplexity = min(30, max(5, len(points) // 3))
            reducer = TSNE(n_components=2, random_state=42, perplexity=perplexity)
            points_2D = reducer.fit_transform(points_scaled)
        else:  # par défaut ou si "PCA"
            reducer = PCA(n_components=2)
            points_2D = reducer.fit_transform(points_scaled)

        # Tracer les clusters
        plt.figure(figsize=(8, 6))
        unique_labels = sorted(set(cluster_labels))
        colors = plt.cm.get_cmap('tab10', len(unique_labels))

        for idx, label in enumerate(unique_labels):
            cluster_points = [pt for pt, c in zip(points_2D, cluster_labels) if c == label]
            xs = [pt[0] for pt in cluster_points]
            ys = [pt[1] for pt in cluster_points]
            if label == -1:
                plt.scatter(xs, ys, label='Bruit', c='grey', marker='x', s=50)
            else:
                plt.scatter(xs, ys, label=f'Cluster {label}', color=colors(idx), s=50)

        for (x, y), name in zip(points_2D, labels):
            plt.text(x, y, name, fontsize=8)

        plt.title(f"DBSCAN clustering (projection {red_dim.upper()})")
        plt.xlabel(f"{red_dim.upper()} 1")
        plt.ylabel(f"{red_dim.upper()} 2")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return results


#Zone evaluation
def dunn_index(X, labels):
    """
    Calcule l'indice de Dunn.
    """
    unique_clusters = np.unique(labels)
    if len(unique_clusters) < 2:
        return None

    X = np.array(X)
    centroids = [np.mean(X[labels == k], axis=0) for k in unique_clusters]

    #distances inter-clusters (entre centroïdes)
    inter_dists = []
    for i in range(len(centroids)):
        for j in range(i + 1, len(centroids)):
            inter_dists.append(np.linalg.norm(centroids[i] - centroids[j]))

    #distances intra-clusters
    intra_dists = []
    for k in unique_clusters:
        cluster_points = X[labels == k]
        if len(cluster_points) >= 2:
            intra_dists.append(np.max(pdist(cluster_points)))
        else:
            intra_dists.append(0)  #aucun écart si 1 seul point

    max_intra = max(intra_dists)
    if max_intra == 0:
        return None

    return min(inter_dists) / max_intra
def cohesion(points, labels):
    """
    Calcule la moyenne des distances intra-cluster.
    """
    clusters = set(labels)
    total = 0
    count = 0
    for c in clusters:
        pts = [p for p, lab in zip(points, labels) if lab == c]
        if len(pts) > 1:
            dist = cdist(pts, pts, metric='euclidean')
            total += np.sum(dist) / (len(pts) * (len(pts) - 1))
            count += 1
    return total / count if count > 0 else 0

def evaluate_clusters(X, labels, verbose=True):
    """
    Évalue un clustering via :
    - silhouette globale
    - silhouette moyenne par cluster
    - indice de Dunn
    - cohésion intra-cluster
    """
    if len(set(labels)) < 2:
        print("Impossible de calculer les indices : au moins deux clusters sont nécessaires.")
        return

    X = np.array(X)
    silhouette_vals = silhouette_samples(X, labels)
    silhouette_avg = silhouette_score(X, labels)
    dunn = dunn_index(X, labels)
    coh = cohesion(X, labels)

    cluster_ids = np.unique(labels)
    cluster_stats = []

    for cluster in cluster_ids:
        cluster_sil_vals = silhouette_vals[labels == cluster]
        mean_score = cluster_sil_vals.mean()
        size = len(cluster_sil_vals)
        cluster_stats.append({
            "cluster": cluster,
            "mean_silhouette": mean_score,
            "size": size
        })

    if verbose:
        print(f"\nSilhouette moyenne globale : {silhouette_avg:.4f}")
        print("Silhouette moyenne par cluster :")
        for stat in cluster_stats:
            print(f" - Cluster {stat['cluster']}: {stat['mean_silhouette']:.4f} (taille: {stat['size']})")
        if dunn is not None:
            print(f"Indice de Dunn : {dunn:.4f}")
        else:
            print("Indice de Dunn non calculable (clusters trop petits).")
        print(f"Cohésion moyenne intra-cluster : {coh:.4f}")

    return {
        "silhouette_global": silhouette_avg,
        "dunn_index": dunn,
        "cohesion": coh,
        "clusters": cluster_stats
    }

def stat_desc(points, labels, k):

    def transpose(matrix):
        return list(map(list, zip(*matrix)))

    print("Pour le CAH : ")
    cluster1, matrice, indices = clustering(points, False, k, True)
    print("Indices des clusters:", indices)  # Ajout pour le débogage

    # Initialiser le dictionnaire avec toutes les clés possibles
    tab = {i: [] for i in range(k)}

    for i, cluster_index in enumerate(indices):
        if cluster_index < k:  # Assurez-vous que l'indice est dans la plage attendue
            tab[cluster_index].append(points[i])

    for i in range(k):
        print(f"Cluster {i} :")
        if tab[i]:  # Vérifiez si le cluster n'est pas vide
            dimensions = transpose(tab[i])
            for j, dim in enumerate(dimensions):
                print(f"Dimension {j} :")
                print("Moyenne : ", mean(dim))
                print("Médiane : ", mediane(dim))
                print("Écart-type : ", ecart_type(variance(dim)))
                print()
        else:
            print(f"Aucun point de données pour le cluster {i}")
        print()

    print("Pour le K-Means : ")
    results_kmeans = kmeans_clustering(points, labels, k, show_plot=False)
    cluster_labels = [label for (_, label) in results_kmeans]
    for i in range(k):
        k_points = [pt for idx, pt in enumerate(points) if cluster_labels[idx] == i]
        print(f"Cluster {i} :")
        if k_points:  # Vérifiez si le cluster n'est pas vide
            dimensions = transpose(k_points)
            for j, dim in enumerate(dimensions):
                print(f"Dimension {j} :")
                print("Moyenne : ", mean(dim))
                print("Médiane : ", mediane(dim))
                print("Écart-type : ", ecart_type(variance(dim)))
                print()
        else:
            print(f"Aucun point de données pour le cluster {i}")
        print()

    print("Pour le DBSCAN : ")
    results_dbscan = dbscan_clustering(points, labels, show_plot=False)
    db_labels = [label for (_, label) in results_dbscan]
    clusters = sorted(set(db_labels))
    for i in clusters:
        d_points = [pt for idx, pt in enumerate(points) if db_labels[idx] == i]
        print(f"Cluster {i} :")
        if d_points:  # Vérifiez si le cluster n'est pas vide
            dimensions = transpose(d_points)
            for j, dim in enumerate(dimensions):
                print(f"Dimension {j} :")
                print("Moyenne : ", mean(dim))
                print("Médiane : ", mediane(dim))
                print("Écart-type : ", ecart_type(variance(dim)))
                print()
        else:
            print(f"Aucun point de données pour le cluster {i}")
        print()

