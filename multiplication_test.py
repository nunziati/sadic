import numpy as np
import time

a = np.random.normal(size=(100000, 3))

start_time = time.time()
b = (a ** 2).sum(axis=1)
print(time.time() - start_time)
print("")

start_time = time.time()
c = a @ a.transpose()
print(time.time() - start_time)
print("")