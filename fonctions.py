from math import sqrt
from math import inf
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
        mediane = (arr[n//2] + arr[(n//2) + 1])/2
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

def min(arr):
    min = inf
    for i in range(len(arr)):
        if arr[i] < min:
            min = arr[i]

    return min

def max(arr):
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
        sum+= (X[i]-mean_x)*(Y[i]-mean_y)/(len(X)-1)
    return sum

def coefficient_regression_b1(cov,var):
    return cov/var

def coefficient_regression_b0(mean_y,b1,mean_x):
    return mean_y-b1*mean_x

def droite_regression(b0,b1,X):
    y = []
    for i in range(len(X)):
        y.append(b0+b1*X)
    return y

