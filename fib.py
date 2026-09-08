

# what is time complexity of fibonacci sequence

import matplotlib.pyplot as plt
import time
from functools import cache


def naive_fibonacci(n):
    if n == 0:
        return 0
    if n == 1:
        return 1

    return fibonacci(n-1) + fibonacci(n-2)


# @cache
cch = dict()
def fibonacci(n):
    if n == 0:
        return 0
    if n == 1:
        return 1

    if n not in cch:
        cch[n] = fibonacci(n-1) + fibonacci(n-2)

    return cch[n]


def f_time(n):
    start = time.time()
    f_n = fibonacci(n)
    return time.time() - start



ns = list(range(0,40))
times = [f_time(n) for n in ns]

plt.plot(ns, times)
plt.show()
