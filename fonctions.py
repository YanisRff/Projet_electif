
from math import sqrt
from math import inf
from scipy.stats import ttest_ind
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



"""-----------PARTIE 5---------"""
def dist(X, Y):
    """
    Distance euclidienne entre deux points
    """
    for i in range(len(X)-1) :
        distance = sqrt((X[i] - Y[i+1]) ** 2 + (Y[i] - Y[i+1]) ** 2)

    return distance

def dist1(X, Y):
    """
    Distance de Manhattan (L1) entre deux points
    """
    return abs(X[0] - Y[0]) + abs(X[1] - Y[1])


def dist_inf(X, Y):
    """
    Distance Chebyshev entre deux points
    """
    return max(abs(X[0] - Y[0]), abs(X[1] - Y[1]))


def dist_min(points, distance_func=dist):
    """
    Paire de points la plus proche dans la liste 'points',

    """
    min_d = inf
    pair = (None, None) """ou (0,0) je sais pas """
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            d = distance_func(points[i], points[j])
            if d < min_d:
                min_d = d
                pair = (points[i], points[j])
    return pair, min_d


def hierarchic_cluster(points, method='single'):
    """
   Classification Ascendante Hiérarchique (CAH) sur la liste de points."""
    data = np.array(points)
    Z = linkage(data, method=method, metric='euclidienn')
    return Z


def plot_dendrogramme(Z, labels=None, title='Dendrogramme'):
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


    (X_min, Y_min), d_min = dist_min(points, distance_func=dist)
    print(f"Paire la plus proche : {X_min} – {Y_min}  (distance = {d_min:.2f})")

    (X_m1, Y_m1), d_m1 = dist_min(points, distance_func=dist1)
    print(f"Paire la plus proche (distance Manhattan) : {X_m1} – {Y_m1}  (d = {d_m1:.2f})")


    Z = hierarchic_cluster(points, method='single')


    plot_dendrogramme(Z, labels=labels, title='Dendrogramme CAH ')