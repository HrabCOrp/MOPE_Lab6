import math
import random
from _decimal import Decimal
from itertools import compress
from scipy.stats import f, t
import numpy as np
from functools import reduce

N = 14
m = 3
l = 1.73

min_x1, max_x1, min_x2, max_x2, min_x3, max_x3 = -20, 15, 10, 60, 15, 35,
mean_Xmin = (min_x1 + min_x2 + min_x3) / 3  # середнє значення Xmin
mean_Xmax = (max_x1 + max_x2 + max_x3) / 3  # середнє значення Xmax


def equation_solver(x1, x2, x3, coefficients, importance=[True] * 11):
    factors_array = [1, x1, x2, x3, x1 * x2, x1 * x3, x2 * x3, x1 * x2 * x3, x1 ** 2, x2 ** 2,
                     x3 ** 2]  # підставляємо фактори
    return sum([el[0] * el[1] for el in compress(zip(coefficients, factors_array), importance)])  # роз'язок рівняння


def fx_coefficients(x1, x2, x3):
    coefficients = [3.2, 9.1, 4.1, 3.7, 2.1, 0.8, 4.9, 3.7, 0.5, 1.0, 0.3]  # задані коефіцієнти
    return equation_solver(x1, x2, x3, coefficients)


def generate_y(m, factors_table):
    return [[round(fx_coefficients(row[0], row[1], row[2]) + random.randint(-5, 5), 3) for _ in range(m)]
            for row in factors_table]  # генерація у


def gen_matrix(array):  # формування таблиці факторів
    raw_list = [row + [row[0] * row[1], row[0] * row[2], row[1] * row[2], row[0] * row[1] * row[2]] + list(
        map(lambda x: x ** 2, row)) for row in array]  # формуємо комбінації факторів даного масиву (по рядках)
    return list(
        map(lambda row: list(map(lambda el: round(el, 3), row)), raw_list))  # округлюємо елементи в кожному рядку


# виведення матриці
def print_matrix(m, N, factors, y_vals, factors_type=":"):
    headers = list(map(lambda x: x.ljust(10),
                       ["x1", "x2", "x3", "x12", "x13", "x23", "x123", "x1^2", "x2^2", "x3^2"] + [
                           "y{}".format(i + 1) for i in range(m)]))  # формуємо список заголовків матриці
    rows = [list(factors[i]) + list(y_vals[i]) for i in range(N)]  # формуємо список значень матриці
    print("\nМатриця планування" + factors_type)
    print(" ".join(headers))  # формуємо рядок заголовків матриці
    # формуємо рядки значень матриці
    print("\n".join([" ".join(map(lambda j: "{:<+10}".format(j), rows[i])) for i in range(len(rows))]))


# виведення рівняння
def print_equation(coefficients, importance=[True] * 11):  # до 2ї стат. перевірки виводимо з усіма коефіцієнтами
    x_i_names = list(compress(["", "x1", "x2", "x3", "x12", "x13", "x23", "x123", "x1^2", "x2^2", "x3^2"], importance))
    coefficients_to_print = list(compress(coefficients, importance))  # коефіцієнти з урахуванням значущості
    equation = " ".join(
        ["".join(i) for i in zip(list(map(lambda x: "{:+.2f}".format(x), coefficients_to_print)), x_i_names)])
    print("Рівняння регресії: y = " + equation)


def x_i(i, x_coded):
    with_null_x = list(map(lambda x: [1] + x, gen_matrix(x_coded)))
    res = [row[i] for row in with_null_x]
    return np.array(res)


def m_ij(*arrays):
    return np.average(reduce(lambda accum, el: accum * el, list(map(lambda el: np.array(el), arrays))))


# знаходження коефіцієнтів
def find_coef(factors, y_vals):
    coefficients = [[m_ij(x_i(column, factors), x_i(row, factors)) for column in range(11)] for row in
                    range(11)]  # коефіцієнти рівняння
    y_i = list(map(lambda row: np.average(row), y_vals))
    free_values = [m_ij(y_i, x_i(i, factors)) for i in range(11)]
    beta_coef = np.linalg.solve(coefficients, free_values)
    return list(beta_coef)


def cochran_value(f1, f2, q):
    partResult1 = q / f2  # (f2 - 1)
    params = [partResult1, f1, (f2 - 1) * f1]
    fisher = f.isf(*params)
    result = fisher / (fisher + (f2 - 1))
    return Decimal(result).quantize(Decimal('.0001'))


def student_value(f3, q):
    return Decimal(abs(t.ppf(q / 2, f3))).quantize(Decimal('.0001'))


def fisher_value(f3, f4, q):
    return Decimal(abs(f.isf(q, f4, f3))).quantize(Decimal('.0001'))


def cochran_criterion(m, N, y_matrix, p=0.95):
    print("\nКритерій Кохрена: ")
    yVariance = [np.var(i) for i in y_matrix]  # пошук дисперсії за допомогою numpy
    yVar_max = max(yVariance)
    Gp = yVar_max / sum(yVariance)
    f1 = m - 1
    f2 = N
    q = 1 - p
    Gt = cochran_value(f1, f2, q)
    print(f"Gp = {Gp:.3f}; Gt = {Gt:.3f}")
    if Gp < Gt:
        print("Gp < Gt => Дисперсія однорідна.")
        return True
    else:
        print("Gp > Gt => Дисперсія неоднорідна.")
        return False


def student_criterion(m, N, y_matrix, beta_coef):
    print("\nКритерій Стьюдента: ")

    f3 = (m - 1) * N
    q = 0.05
    T = student_value(f3, q)

    # Оцінка генеральної дисперсії відтворюваності
    Sb = np.average(list(map(np.var, y_matrix)))
    Sbs_2 = Sb / (N * m)
    Sbs = math.sqrt(Sbs_2)
    t_i = np.array([abs(beta_coef[i]) / Sbs for i in range(len(beta_coef))])

    # Перевірка значущості коефіцієнтів
    importance = [True if el > T else False for el in list(t_i)]

    beta_i = ["b0", "b1", "b2", "b3", "b12", "b13", "b23", "b123", "b11", "b22", "b33"]
    importance_to_print = ["Суттєвий" if i else "Несуттєвий" for i in importance]
    to_print = map(lambda x: x[0] + " " + x[1], zip(beta_i, importance_to_print))
    print(*to_print, sep="; ")
    print_equation(beta_coef, importance)  # рівняння без несуттєвих коефіцієнтів
    return importance


def fisher_criterion(m, N, d, x_matrix, y_matrix, b_coefficients, importance):  # критерій Фішера
    print("\nКритерій Фішера")
    f3 = (m - 1)
    f4 = N - d
    q = 0.05
    yTheor = np.array(
        [equation_solver(row[0], row[1], row[2], b_coefficients) for row in x_matrix])  # теоретичне у
    meanY = np.array(list(map(lambda el: np.average(el), y_matrix)))  # середнє у
    # Дисперсія адекватності
    Sad = m / (N - d) * sum((yTheor - meanY) ** 2)
    y_variations = np.array(list(map(np.var, y_matrix)))
    s_v = np.average(y_variations)
    Fp = float(Sad / s_v)
    Ft = fisher_value(f3, f4, q)
    theoretical_values_to_print = list(zip(map(lambda x: "x1 = {0[1]:<10} x2 = {0[2]:<10} x3 = {0[3]:<10}"
                                               .format(x), x_matrix), yTheor))
    print("\nПорівняймо отримані значення у з середніми:")
    print("Теоретичні значення y:")
    print("\n".join(["{arr[0]}: y = {arr[1]}".format(arr=el) for el in theoretical_values_to_print]))
    print("\nСередні значення у:")
    print("\n".join(["y = {arr}".format(arr=el) for el in meanY]))
    print("\nFp = {}, Ft = {}".format(Fp, Ft))
    print("Fp < Ft => модель адекватна" if Fp < Ft else "Fp > Ft => модель неадекватна")
    return True if Fp < Ft else False


x0 = [(max_x1 + min_x1) / 2, (max_x2 + min_x2) / 2, (max_x3 + min_x3) / 2]  # X0
detx = [(min_x1 - x0[0]), (min_x2 - x0[1]), (min_x3 - x0[2])]  # дельта X

# нормовані значення факторів
x_coded = [[-1, -1, -1],
           [-1, +1, +1],
           [+1, -1, +1],
           [+1, +1, -1],
           [-1, -1, +1],
           [-1, +1, -1],
           [+1, -1, -1],
           [+1, +1, +1],
           [-1.73, 0, 0],
           [+1.73, 0, 0],
           [0, -1.73, 0],
           [0, +1.73, 0],
           [0, 0, -1.73],
           [0, 0, +1.73]]

# натуралізовані значення факторів
x_natur = [[min_x1, min_x2, min_x3],
           [min_x1, max_x2, max_x3],
           [max_x1, min_x2, max_x3],
           [max_x1, max_x2, min_x3],
           [min_x1, min_x2, max_x3],
           [min_x1, max_x2, min_x3],
           [max_x1, min_x2, min_x3],
           [max_x1, max_x2, max_x3],
           [-l * detx[0] + x0[0], x0[1], x0[2]],
           [l * detx[0] + x0[0], x0[1], x0[2]],
           [x0[0], -l * detx[1] + x0[1], x0[2]],
           [x0[0], l * detx[1] + x0[1], x0[2]],
           [x0[0], x0[1], -l * detx[2] + x0[2]],
           [x0[0], x0[1], l * detx[2] + x0[2]]]

xMatrix_natur = gen_matrix(x_natur)
y_values = generate_y(m, x_natur)
while not cochran_criterion(m, N, y_values):  # збільшення m
    m += 1
    y_values = generate_y(m, xMatrix_natur)

print_matrix(m, N, xMatrix_natur, y_values, " для натуралізованих факторів:")
coefficients = find_coef(xMatrix_natur, y_values)
print_equation(coefficients)

importance = student_criterion(m, N, y_values, coefficients)

d = len(list(filter(None, importance)))
fisher_criterion(m, N, d, xMatrix_natur, y_values, coefficients, importance)
