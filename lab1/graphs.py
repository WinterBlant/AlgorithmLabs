from matplotlib import pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

def linearithmic(x, a, b, c):
    return a*x*np.log2(b*x)+c

def plot_graph(func):
    elapsed_time = []
    with open(func + ".txt", 'r') as f:
        for i in range(5):
            elapsed_time.append([])
            for elem in f.readline()[1:-2].split(','):
                elapsed_time[i].append(float(elem))
    elapsed_time = np.array(elapsed_time)
    average_elapsed_time = []
    for i in range(2000):
        average_elapsed_time.append(sum(elapsed_time[:, i]) / 5)
    x = np.arange(1, 2001)
    if func == "constant":
        p = np.polyfit(x, average_elapsed_time, 0)
        ya = np.polyval(p, x)
    elif func == "elements_sum" or func == "elements_multiplicative" or func == "horner_polynomial":
        p = np.polyfit(x, average_elapsed_time, 1)
        ya = np.polyval(p, x)
    elif func == "naive_polynomial" or func == "bubble_sort":
        p = np.polyfit(x, average_elapsed_time, 2)
        ya = np.polyval(p, x)
    elif func == "matrix":
        p = np.polyfit(x, average_elapsed_time, 3)
        ya = np.polyval(p, x)
    else:
        params, povt = curve_fit(linearithmic, x, average_elapsed_time)
        ya = []
        for elem in x:
            ya.append(linearithmic(elem, *params))
    plt.figure(figsize=[20, 12])
    plt.xlabel("Время (секунды)")
    plt.ylabel("Размерность вектора v")
    plt.plot(x, average_elapsed_time, label="Экспериментальные результаты")
    plt.plot(x, ya, label="Аппроксимация на основе теоретических оценок", color="red")
    plt.legend()
    plt.savefig(func + ".png", dpi=1000, quality=95)
    plt.show()


plot_graph("constant")
plot_graph("elements_sum")
plot_graph("elements_multiplicative")
plot_graph("naive_polynomial")
plot_graph("horner_polynomial")
plot_graph("bubble_sort")
plot_graph("quick_sort")
plot_graph("tim_sort")
plot_graph("matrix")
