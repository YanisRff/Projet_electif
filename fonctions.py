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
    return var/len(arr)

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