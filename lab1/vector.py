import time
import numpy as np
import decimal
decimal.getcontext().prec = 100


def constant(arr):
    return 1


def elements_sum(arr):
    return sum(arr)


def elements_multiplicative(arr):
    mult = 1
    for elem in arr:
        mult *= elem
    return mult


def naive_polynomial(arr):
    poly = decimal.Decimal(0)
    for i in range(len(arr)):
        poly += decimal.Decimal(arr[i].item())*(decimal.Decimal(1.5)**decimal.Decimal(i))
    return poly


def horner_polynomial(arr):
    poly = decimal.Decimal(arr[len(arr)-1].item())
    for i in range(len(arr)-2, -1, -1):
        poly = decimal.Decimal(1.5)*poly + decimal.Decimal(arr[i].item())
    return poly


def bubble_sort(arr):
    n = len(arr)
    for i in range(n - 1):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr


def quick_sort(arr):
    return quick_sort_rec(arr, 0, len(arr)-1)


def quick_sort_rec(arr, start, end):
    if start >= end:
        return
    div = divide(arr, start, end)
    quick_sort_rec(arr, start, div - 1)
    quick_sort_rec(arr, div + 1, end)


def divide(arr, start, end):
    pivot = arr[start]
    low = start + 1
    high = end
    while True:
        while low <= high and arr[high] >= pivot:
            high = high - 1
        while low <= high and arr[low] <= pivot:
            low = low + 1
        if low <= high:
            arr[low], arr[high] = arr[high], arr[low]
        else:
            break
    arr[start], arr[high] = arr[high], arr[start]
    return high


def tim_sort(arr):
    return sorted(arr)


def generate_random_array(length):
    rng = np.random.default_rng()
    return rng.integers(100, size=length)


def main(func):
    elapsed_time = []
    for n in range(1, 2001):
        arr = generate_random_array(n)
        t0 = time.perf_counter()
        func(arr)
        elapsed_time.append(time.perf_counter() - t0)
        print("finished ", n)
    with open(func.__name__+".txt", 'a') as f:
        f.write(str(elapsed_time)+'\n')


main(constant)
main(elements_sum)
main(elements_multiplicative)
main(naive_polynomial)
main(horner_polynomial)
main(bubble_sort)
main(quick_sort)
main(tim_sort)



