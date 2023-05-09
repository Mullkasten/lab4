import numpy as np
import pandas as pd
import math
import scipy.stats


def sred_znach_j(z, N, p):
    z_average = [i for i in range(0, p)]    #среднее значение j-го признака
    for j in range(0, p):
        for i in range(0, N):
            z_average[j] += z[i][j]
        z_average[j] /= N
    return z_average


def ocenka_Dj(z, z_average, N, p):
    S_variance = [i for i in range(0, p)]   #оценка дисперсии j-го столбца
    for j in range(0, p):
        for i in range(0, N):
            S_variance[j] += (z[i][j] - z_average[j])**2
        S_variance[j] /= N
    return S_variance


def get_r(z, z_average, N, p):
    sigma = [[0] * p for i in range(p)]     #ковариационная матрица
    for i in range(0, p):
        for j in range(0, p):
            for k in range(0, N):
                sigma[i][j] += (z[k][i] - z_average[i]) * (z[k][j] - z_average[j])
            sigma[i][j] /= N
    return sigma


def standart_X(z, z_average, S_variance, N, p):
    X = [[0] * p for i in range(N)]         #стандартизованная матрица
    for i in range(0, N):
        for j in range(0, p):
            X[i][j] = (z[i][j] - z_average[j]) / ((S_variance[j])**(0.5))
    return X


def f_d(sigma, N, p):
    d = 0
    for i in range(0, p):
        for j in range(0, p):
            if i != j:
                d += sigma[i][j]**2
    d *= N
    return d


def f_jacobi(A, n, eps): 
    #Шаг 1
    T = np.eye(n, dtype=float)          #Задаем T как единичную матрица порядка n

    #Шаг 2
    sum_1 = 0                           #Находим первую преграду alfa
    for j in range(0, n):
        for i in range(0, n):
            if i != j:
                sum_1 += A[i][j]**2
    alfa_0 = math.sqrt(sum_1) / n        #Первая преграда
    a_k = alfa_0

    flag = True
  
    while flag:
        #Шаг 3
        p = -1
        q = -1
        max = 0
        #Находим наибольший по модулю внедиагональный элемент
        for i in range(0, n):
            for j in range(0, n):
                if i != j and abs(A[i][j]) > abs(max):
                    p = i
                    q = j
                    max = A[i][j]
        
        if abs(max) > a_k:        
            #Шаг 4
            y = (A[p][p] - A[q][q]) / 2             #Анализируем найденный элемент alfa
            
            x = 0
            if y == 0:
                x = -1
            else:
                x = -np.sign(y) * (A[p][q] / math.sqrt(A[p][q]**2 + y**2))

            s = x / math.sqrt(2 * (1 + math.sqrt(1 - x**2)))

            c = math.sqrt(1 - s**2)

            for i in range(0, n):                   #Преобразуем строки и столбцы матрицы A[][]
                if i != p and i != q:
                    z_1 = A[i][p]
                    z_2 = A[i][q]
                    A[q][i] = z_1*s + z_2*c
                    A[i][q] = A[q][i]
                    A[i][p] = z_1*c - z_2*s
                    A[p][i] = A[i][p]
            z_5 = s*s
            z_6 = c*c
            z_7 = s*c
            v_1 = A[p][p]
            v_2 = A[p][q]
            v_3 = A[q][q]

            A[p][p] = v_1*z_6 + v_3*z_5 - 2*v_2*z_7     #В результате находим матрицу 
            A[q][q] = v_1*z_5 + v_3*z_6 + 2*v_2*z_7
            A[p][q] = (v_1 - v_3)*z_7 + v_2*(z_6 - z_5)
            A[q][p] = A[p][q]

            for i in range(0, n):
                z_3 = T[i][p]
                z_4 = T[i][q]
                T[i][q] = z_3*s + z_4*c
                T[i][p] = z_3*c - z_4*s

        #Шаг 5
        flag = False                                    #Находим новую преграду и повторяем вычисления с шага 3
        for i in range(0, n):
            for j in range(0, n):
                if i !=j and (abs(A[i][j]) >= eps*alfa_0):
                    flag = True
        a_k = a_k / (n**2)

    lamdas = []
    for i in range(0, n):
        lamdas.append(A[i][i])
    
    return [lamdas, T]

#функция сортировки по-убыванию столбцов матрицы
def sort_matrix(A, n, m):
    for j in range(0, m):
        tmp_list = []
        for i in range(0, n):
            tmp_list.append(A[i][j])
        tmp_list.sort(reverse=True)
        for i in range(0, n):
            A[i][j] = tmp_list[i]
    return A

#функция вычисления дисперсий матрицы по столбцам
def variance_by_column(A, n, m):
    var_A = []
    for j in range(0, m):
        tmp_list = []
        for i in range(0, n):
            tmp_list.append(A[i][j])
        var_A.append(np.var(tmp_list))
    
    return var_A


def number_of_new_features(sobst_chisla, p):
    new_p = 1    #число новых признаков
    for i in range(0, p):
        sum_lamda = 0
        for j in range(0, i):
            sum_lamda += sobst_chisla[j]
        new_p = i
        part = sum_lamda / sum(sobst_chisla)
        if part > 0.95:
            break
    return new_p


def diagonal_matrix(A, n):
    B = np.eye(n, dtype=float)    #диагональная квадратная матрица
    for i in range(0, n):
        B[i][i] = A[i]
    return B


if __name__ == '__main__':
    z = pd.read_csv("lab4.csv", sep=";").to_numpy()    #исходная матрица
    print("\n\tИсходная матрица\n", pd.DataFrame(z))
    N = 58    #размерность матрицы z
    p = 10

    z_j = sred_znach_j(z, N, p)
    print("\n\tСреднее значение j-го признака\n", pd.DataFrame(z_j))

    S_j = ocenka_Dj(z, z_j, N, p)
    print("\n\tОценка дисперсии j-го столбца\n", pd.DataFrame(S_j))

    X = standart_X(z, z_j, S_j, N, p)
    print("\n\tСтандартизованная матрица\n", pd.DataFrame(X))

    x_average = sred_znach_j(X, N, p)
    r = get_r(X, x_average, N, p)
    print("\n\tКовариационная матрица\n", pd.DataFrame(r))

    #значимо ли отличается от единичной матрицы
    #корреляционная матрица исходных стандартизованных данных    
    d = f_d(r, N, p)
    degree_freedom = p * (p - 1) / 2
    hi = 2.0141034
    print("d:\t", d)
    print("Степень свободы: ", degree_freedom)
    print("Хи-квадрат: ", scipy.stats.chi2.ppf(0.95, degree_freedom), "\n")
    if d <= hi:
        print("Принимаем гипотезу H0 - МГК не целесообразен\n")
    else:
        print("Принимаем гипотезу H1 - МГК целесообразен\n")

    #тестовый пример
    #test_A = np.array([[1.00, 0.42, 0.54, 0.66], [0.42, 1.00, 0.32, 0.44], [0.54, 0.32, 1.00, 0.22], [0.66, 0.44, 0.22, 1.00]])
    #print("\n\tТестовая матрица А:\n", pd.DataFrame(test_A))
    #
    #test = f_jacobi(test_A, 4, 0.0001) 
    #print("\n\tЛамбды:\n", test[0])
    #print("\n\tМатрица T:\n", test[1])

    #исходная задача
    property_R = f_jacobi(r, p, 0.0001) #По методу Якоби

    sobst_chisla = sorted(property_R[0], reverse=True)    #Лямбды
    sobsts_vect = sort_matrix(property_R[1], p, p)      #Матрица T

    print("\n\tСобственные числа корреляционной матрицы:\n", pd.DataFrame(sobst_chisla)) 
    print("\n\tСобственные вектора корреляционной матрицы:\n", pd.DataFrame(sobsts_vect))

    mgk = diagonal_matrix(sobst_chisla, p)
    print("\n\tКовариационная матрица главных компонент:\n", pd.DataFrame(mgk))

    c_average = sred_znach_j(sobsts_vect, p, p)
    c_variance = ocenka_Dj(sobsts_vect, c_average, p, p)
    
    load_matrix = standart_X(sobsts_vect, c_average, c_variance, p, p)
    print("\n\tМатрица нагрузок на главные компоненты:\n", pd.DataFrame(load_matrix))

    Y = np.dot(X, sobsts_vect)
    print("\n\tПроекции объектов на главные компоненты:\n", pd.DataFrame(Y))

    disp_x = variance_by_column(X, N, p)
    disp_y = variance_by_column(Y, N, p)

    print("\nВыборочная дисперсия исходных признаков Х: ", disp_x)
    print("Сумма: \n", sum(np.around(disp_x)))
    print("\nВыборочная дисперсия проекций объектов на главные компоненты Y: ", disp_y)
    print("Сумма: \n", sum(np.around(disp_x)))

    mk = diagonal_matrix(disp_y, p)
    print("\n\tМатрица ковариации для проекций объектов на главные компоненты:\n", pd.DataFrame(mk))

    new_p = number_of_new_features(sobst_chisla, p)    #число новых признаков, удовлетворяющих условию I(p') > 0.95
    print("\nЧисло новых признаков: ", new_p)
    
    I = sum(sobst_chisla[0:new_p]) / sum(sobst_chisla)
    print("\nI(p'): ", I)
    print("\n\tПервая главная компонента:\n", pd.DataFrame(sobsts_vect[:, 0]))
    print("\n\tВторая главная компонента:\n", pd.DataFrame(sobsts_vect[:, 1]))


