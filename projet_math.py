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
    nombre_observation = len(X)


    "----------------------------------- PARTIE 1 -----------------------------------"
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

    min_x = v_min(X)
    max_x = v_max(X)

    min_y = v_min(Y)
    max_y = v_max(Y)

    print("Maximum de X : ",max_x, ", Minimum de X :",min_x)
    print("Maximum de Y : ", max_y, ", Minimum de Y :", min_y)

    etendue_x = etendue(max_x,min_x)
    etendue_y = etendue(max_y,min_y)

    print("Etendue de X : ",etendue_x)
    print("Etendue de Y :",etendue_y)


    plt.plot(X, Y, "ob")

    "----------------------------------- PARTIE 2 -----------------------------------"
    covariance_xy = covariance(X,Y,moyenne_x,moyenne_y)

    print("Covariance xy : ",covariance_xy)

    b1 = coefficient_regression_b1(covariance_xy, variance_x)

    print("Coefficient regression b1 : ",b1)

    b0 = coefficient_regression_b0(moyenne_y,b1,moyenne_x)

    print("Coefficient regression b0 :",b0)

    droite_reg = droite_regression(b0,b1,X)

    print("Points droite regression : ",droite_reg)

    ligne_x = np.linspace(v_min(X), v_max(X), 100)
    droite_regression = b0 + b1 * ligne_x
    plt.plot(ligne_x, droite_regression, color='red', label='Droite de régression')
    plt.show()

    "----------------------------------- PARTIE 3 -----------------------------------"
    residu = residus(Y,droite_reg)
    print("Résidus :" ,residu)
    sce = SCE(Y,droite_reg)
    sct = SCT(Y,moyenne_y)
    scr = SCR(moyenne_y,droite_reg)

    R2 =coefficient_determination_r2(sce,sct)

    print("SCE :", sce)
    print("SCT : ",sct)
    print("SCR : ",scr)
    print("SCR/SCT : ",scr/sct)
    print("Coefficient de détermination : ", R2)

    mse = mean_squared_error(sce,nombre_observation)

    print("MSE : ",mse)

    rmse = RMSE(mse)

    print("RMSE : ",rmse)

    "----------------------------------- PARTIE 4 -----------------------------------"

    std_error = standard_error(X,rmse,moyenne_x)

    print("Erreurs standards de la pende SEb1: ",std_error)

    std_error_origin = standard_error_origin(X,rmse,nombre_observation,moyenne_x)

    print("Erreur standard de l'ordonnée à l'origine : ",std_error_origin)

    statistique_test_b1 = statistique_t(b1,std_error)

    print("statistique de test b1 : ",statistique_test_b1)

    statistique_test_b0 = statistique_t(b0,std_error_origin)

    print("statistique de test b0 : ", statistique_test_b0)

    print("intervalle de confiance : [ ",b1-2.571*std_error,";",b1+2.571*std_error,"]")
    #La pente n'est pas significative (0 contenu dans l'intervalle de confiance) on ne rejette pas h0

    print("intervalle de confiance : [ ",b0-2.571*std_error_origin,";",b0+2.571*std_error_origin,"]")
    #La pente à l'origine est significativement différente de 0, on rejette h0

    p_value_b1 = 2*t.sf(abs(statistique_test_b1), df=5) #on multiplie par 2 car c'est un test bilatéral
    print("P-valeur pour l'hypothèse H0: b1=0 : ",p_value_b1)
    print("La P-valeur est supérieure à alpha (0.05) on ne rejette donc pas H0")
    p_value_b0 = 2*t.sf(abs(statistique_test_b0), df=5)
    print("La P-valeur pour l'hypothèse H0:B0=0 : ",p_value_b0)
    print("La P-valeur est inférieure à 0.05, on rejette donc H0")



