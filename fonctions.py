
from math import sqrt
from math import inf
from scipy.stats import ttest_ind,ttest_1samp,t
import numpy as np
import matplotlib.pyplot as plt

def mean(arr):
    #Calcule la moyenne en prenant un tableau de int en entrée
    sum=0
    for i in range(len(arr)):
        sum+= arr[i]
    return sum/len(arr)

def mediane(arr):
    n = len(arr)
    if n%2==0:
        mediane = (arr[n//(2-1)] + arr[(n//2) + 1])/2
    else:
        mediane = arr[(n+1)//2]
    return mediane
def variance(arr,mean):
    # Calcule la variance en prenant un tableau de int en entrée ainsi que la moyenne de ce même tableau
    var = 0
    for i in range(len(arr)):
        var+= (arr[i]-mean)*(arr[i]-mean)
    return var/(len(arr)-1)

def ecart_type(variance):
    # Calcule la ecart-type en prenant une variance en entrée
    return sqrt(variance)

def v_min(arr):
    min = inf
    for i in range(len(arr)):
        if arr[i] < min:
            min = arr[i]

    return min

def v_max(arr):
    max = -inf
    for i in range(len(arr)):
        if arr[i] > max:
            max = arr[i]
    return max

def etendue(max,min):
    return max-min


def covariance(X,Y,mean_x,mean_y):
    sum = 0
    for i in range(len(X)):
        sum+= (X[i]-mean_x)*(Y[i]-mean_y)
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

    """R^2 \approx 0{,}05 signifie que seulement 5 % de la variabilité des y est expliquée par la variable x
	•	Cela signifie que la relation linéaire est très faible
	•	Le modèle linéaire n’explique pratiquement rien des variations observées → le nuage de points est probablement très dispersé ou non linéaire##"""



def coefficient_determination_r2(SCE,SCT):
    return 1-(SCE/SCT)

def SCE(Y,droite_regression):
    sum = 0
    for i in range(len(Y)):
        sum+= (Y[i]-droite_regression[i])*(Y[i]-droite_regression[i])
    return sum

def SCT(Y,mean_y):
    sum = 0
    for i in range(len(Y)):
        sum+= (Y[i]-mean_y)*(Y[i]-mean_y)
    return sum

def SCR(mean_y,droite_regression):
    sum=0
    for i in range(len(droite_regression)):
        sum+= (droite_regression[i]-mean_y)*(droite_regression[i]-mean_y)
    return sum
def residus(Y,droite_regression):
    y=[]
    for i in range(len(Y)):
        y.append(Y[i]-droite_regression[i])
    return y

def mean_squared_error(SCE,nombre_observation):
    return SCE/(nombre_observation-2)

def RMSE(MSE):
    return sqrt(MSE)


def standard_error(X,RMSE,mean):
    sum=0
    for i in range(len(X)):
        sum+= (X[i]-mean)**2
    return RMSE/(sqrt(sum))


def standard_error_origin(X,rmse,nombre_observation,mean):
    sum = 0
    for i in range(len(X)):
        sum += (X[i] - mean) ** 2

    return rmse*sqrt((1/nombre_observation) + (mean**2)/sum)


def statistique_t(b1,std_error):
    return b1/std_error


from math import sqrt, inf
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt



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


def dist_min(ensemble_de_points,distance_func=dist):
    """
    Paire de points la plus proche dans la liste (X,Y)
    En gros on connait les 2 points les plus proches
    """
    min_distance = inf
    pair = (None, None)
    n = len(ensemble_de_points)
    for i in range(n):
        for j in range(i + 1, n):
            d = distance_func(ensemble_de_points[i],ensemble_de_points[j])
            if d < min_distance:
                min_d = d
                pair = (ensemble_de_points[i], ensemble_de_points[j])
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


