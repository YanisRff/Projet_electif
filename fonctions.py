
from math import sqrt,inf
from scipy.stats import ttest_ind,ttest_1samp,t
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def mean(arr):
    #Calcule la moyenne en prenant un tableau de int en entrée
    sum=0
    for i in range(len(arr)):
        sum+= arr[i]
    return sum/len(arr)

def mediane(arr):
    """Calcule la médiane en prenant en entrée un talbeau d'entier"""
    n = len(arr)
    if n%2==0:
        """Si la taille du tableau est paire, on fait la moyenne des deux valeurs au centre"""
        mediane = (arr[n//(2-1)] + arr[(n//2) + 1])/2
    else:

        mediane = arr[(n+1)//2]
    return mediane
def variance(arr):
    # Calcule la variance en prenant un tableau de int en entrée ainsi que la moyenne de ce même tableau
    moy = mean(arr)
    var = 0
    for i in range(len(arr)):
        var+= (arr[i]-moy)**2
    return var/(len(arr)-1)

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

    dist = sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
    return dist
    dist = 0
    for i in range (len(point1)):
        dist += (point2[i] - point1[i]) ** 2
    return sqrt(dist)


def dist_manhattan(p1, p2):
    """
    Distance de Manhattan (L1) entre deux points
    """
    dist = abs(p2[0] - p1[0]) + abs(p2[1] - p1[1])

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
        points = full_points

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
    print(full_points)
    plt.show()

def seuil_max_gap(Z):
    """
    calcule le seuil pour le dendrogramme, affiche une ligne en pointillé
    """
    distances = Z[:, 2]
    gaps = np.diff(distances)
    index_max_gap = np.argmax(gaps)
    seuil = (distances[index_max_gap] + distances[index_max_gap + 1]) / 2
    return seuil

def clustering(points):
    """
    Effectue la méthode de clustering de classification ascendante hiérarchique
    calcule la distance la plus faible entre deux points pour les fusionner
    réecris la matrice avec les points fusionnés puis re calcule les distance
    """
    #n = len(points)
    #distance_min = inf
    i_min, j_min = -1, -1
    while len(points) > 1 :
        distance_min = inf
        n = len(points)
        # Étape 1 : trouver la paire la plus proche
        for i in range(n):
            for j in range(i + 1, n):
                d = dist_euclidienne(points[i], points[j])
                if d < distance_min:
                    distance_min = d
                    i_min, j_min = i, j

        print(f"Fusion des clusters {i_min} et {j_min}, distance = {distance_min:.2f}")

        # Étape 2 : barycentre du cluster entre les deux points
        new_point = ((np.array(points[i_min]) + np.array(points[j_min])) / 2).tolist()

        # Étape 3 : mettre à jour la liste de points
        new_points = []
        for k in range(n):
            if k not in (i_min, j_min):
                new_points.append(points[k])
        new_points.append(new_point)

        # Étape 4 : Création matrice dim n-1
        points = new_points
        new_n = len(points)
        # Création d'une matrice de zéros
        matrice = []
        for i in range(new_n):
            ligne = []
            for j in range(new_n):
                ligne.append(0)
            matrice.append(ligne)
        # Remplissage avec les distances après clustering
        for i in range(new_n):
            for j in range(new_n):
                if i != j:
                    matrice[i][j] = round(dist_euclidienne(points[i], points[j]), 2)
    return points, matrice




def cluster_hierarchique(points, method='single', seuil=None):
    """
    Classification Ascendante Hiérarchique (CAH) avec seuil automatique si non fourni.
    """
    data = np.array(points)
    Z = linkage(data, method=method, metric='euclidean')

    if seuil is None:
        seuil = seuil_max_gap(Z)

    dendrogram(Z)
    plt.axhline(y=seuil, color='black', linestyle='--')
    plt.title('Dendrogramme CAH (seuil auto)' if seuil else 'Dendrogramme CAH')
    plt.xlabel('Points')
    plt.ylabel('Distance')
    plt.show()

    clusters = fcluster(Z, t=seuil, criterion='distance')
    return Z, clusters, seuil

def kmeans_clustering(points, labels, k=3, show_plot=True, red_dim="PCA"):
    points = np.array(points)
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(points)
    cluster_labels = kmeans.labels_

    results = list(zip(labels, cluster_labels))

    if show_plot:
        # Réduction de dimension selon l'argument red_dim
        if red_dim.upper() == "TSNE":
            perplexity = min(30, max(5, len(points) // 3))
            reducer = TSNE(n_components=2, random_state=42, perplexity=perplexity)
            points_2D = reducer.fit_transform(points)
            centers_2D = None  # t-SNE ne permet pas de transformer de nouveaux points comme les centres
        else:  # par défaut ou si "PCA"
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


def silhouette_score_custom(points, labels):
    """
    Calcule l'indice de silhouette pour chaque point et retourne le score moyen.
    Version simplifiée pour comprendre le principe.
    """
    n = len(points)
    s = np.zeros(n)

    for i in range(n):
        # Points du même cluster
        cluster_points = [points[j] for j in range(n) if labels[j] == labels[i] and j != i]
        if not cluster_points:
            s[i] = 0
            continue

        # Distance moyenne intra-cluster (a_i)
        a_i = np.mean([dist_euclidienne(points[i], p) for p in cluster_points])

        # Distance moyenne au cluster le plus proche (b_i)
        other_clusters = set(labels) - {labels[i]}
        b_i = inf
        for cluster in other_clusters:
            cluster_points = [points[j] for j in range(n) if labels[j] == cluster]
            mean_dist = np.mean([dist_euclidienne(points[i], p) for p in cluster_points])
            if mean_dist < b_i:
                b_i = mean_dist

        s[i] = (b_i - a_i) / max(a_i, b_i) if max(a_i, b_i) > 0 else 0

    return np.mean(s)