


points_x = []
points_y = []


def mean(arr):
    sum=0
    for i in range(len(arr)):
        sum+= arr[i]
    return sum/len(arr)
def variance(arr,mean):
    var = 0
    for i in range(len(arr)):
        var+= (arr[i]-mean)*(arr[i]-mean)
    return var/len(arr)

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
        X.append(ensemble_points[i][0])
        Y.append(ensemble_points[i][1])

    mean_x = mean(X)
    mean_y = mean(Y)

    


