from fonctions import *


if __name__ == '__main__':

    X = []
    Y = []
    M_1 = (1, 1)
    M_2 = (1, 2)
    M_3 = (1, 5)
    M_4 = (3, 4)
    M_5 = (4, 3)
    M_6 = (6, 2)
    M_7 = (0, 4)

    ensemble_points = [M_1, M_2, M_3, M_4, M_5, M_6, M_7]

    for i in range(len(ensemble_points)):
        """Le tableau ensemble_points est un talbeau de points bi-dimensionnels, on répartit donc les valeurs de
        X et de Y dans des tableau différents afin de pouvoir les étudier"""
        X.append(ensemble_points[i][0])
        Y.append(ensemble_points[i][1])

    print("X: ",X)
    print("Y: ",Y)
    moyenne_x = mean(X)
    moyenne_y = mean(Y)

    print("Moyenne X :",moyenne_x)
    print("Moyenne Y : ",moyenne_y)

    mediane_x = mediane(X)
    mediane_y = mediane(Y)

    print("Médiane X : ",mediane_x)
    print("Médiane Y : ",mediane_y)

    variance_x = variance(X,moyenne_x)
    variance_y = variance(Y,moyenne_y)

    print("Variance X: ",variance_x)
    print("Variance Y: ",variance_y)

    ecart_type_x = ecart_type(variance_x)
    ecart_type_y = ecart_type(variance_y)

    print("Ecart type X : ",ecart_type_x)
    print("Ecart type Y : ",ecart_type_y)

    min_x = min(X)
    max_x = max(X)

    min_y = min(Y)
    max_y = max(Y)

    print("Maximum de X : ",max_x, ", Minimum de X :",min_x)
    print("Maximum de Y : ", max_y, ", Minimum de Y :", min_y)

    etendue_x = etendue(max_x,min_x)
    etendue_y = etendue(max_y,min_y)

    print("Etendue de X : ",etendue_x)
    print("Etendue de Y :",etendue_y)


    plt.plot(X, Y, "ob")
    plt.show()

    covariance_xy = covariance(X,Y,moyenne_x,moyenne_y)

    print("Covariance xy : ",covariance_xy)

    b1 = coefficient_regression_b1(covariance_xy, variance_x)

    print("Coefficient regression b1 : ",b1)

    b0 = coefficient_regression_b0(moyenne_y,b1,moyenne_x)

    print("Coefficient regression b0 :",b0)