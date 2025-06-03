
from math import sqrt,inf
from scipy.stats import ttest_ind,ttest_1samp,t
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
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

def dist_euclidienne(X, Y):
    """
    Distance euclidienne entre deux points
    """
    distance_euclidienne = []
    for i in range(len(X)-1) :
        dist = sqrt((X[i] - X[i+1]) ** 2 + (Y[i] - Y[i+1]) ** 2)
        distance_euclidienne.append(dist)
    return distance_euclidienne

def dist_Manhattan(X, Y):
    """
    Distance de Manhattan (L1) entre deux points
    """
    distance_Manhattan = []
    for i in range(len(X)-1):
        dist = abs(X[i] - Y[i]) + abs(X[i+1] - Y[i+1])
        distance_Manhattan.append(dist)
    return distance_Manhattan


def dist_chebyshev(X, Y):
    """
    Distance Chebyshev entre deux points
    """
    distance_chebyshev = []
    for i in range (len(X)-1):
        dist = max(abs(X[i] - Y[i]), abs(X[i+1] - Y[i+1]))
        distance_chebyshev.append(dist)
    return distance_chebyshev


points = [(1, 1), (1, 2), (1, 5), (3, 4), (4, 3), (6, 2), (0, 4)]
def dist_min(points,distance_func):
    """
    Paire de points la plus proche dans la liste (X,Y)
    En gros on connait les 2 points les plus proches
    """
    min_distance = inf
    pair = (None, None)
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            d = distance_func(points[i],points[j])
            if d < min_distance:
                min_d = d
                pair = (points[i],points[j])
    return pair, min_d


def cluster_hierarchique(points, method='single'):
    """
   Classification Ascendante Hiérarchique (CAH) sur la liste de points."""
    data = np.array(points)
    Z = linkage(data, method=method, metric='euclidienn')
    return Z


def dendrogramme_dessin(Z, labels=None, title='Dendrogramme'):
    """
    Trace un dendrogramme à partir de la matrice
"""
    plt.figure(figsize=(8, 5))
    dendrogram(Z, labels=labels)
    plt.title(title)
    plt.xlabel('Points')
    plt.ylabel('Distance')
    plt.tight_layout()
    plt.show()


