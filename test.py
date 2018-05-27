import matplotlib.pyplot as plt
import numpy as np
import functools

a = [1, 3, 2]



def foo(i, j):
    if i < j:
        return -1
    elif i == j:
        return 0
    else:
        return 1

a.sort(key=functools.cmp_to_key(foo))

print(a)

