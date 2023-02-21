import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

true_depth_indexes_ref = np.load("true_depth_index_64.npy")

true_depth_indexes_test = np.load("10A_depth_index_64.npy")

"""diff = np.abs(true_depth_indexes_ref - true_depth_indexes_test)

print(diff)

print("mean =", np.mean(diff))
print("rms =", np.sqrt(np.mean(np.square(diff))))
print("max =", np.max(diff))

# plot histogram of diff
plt.hist(diff, bins=30)
plt.show()"""

#my_depth_indexes = 2. - np.load("test_depth_index.npy")

pietro_depth_indexes = pd.read_csv("1a0i_di.txt", sep='\t')

true_depth_indexes_test = 2. - true_depth_indexes_test
#diff = np.abs(true_depth_indexes_test - pietro_depth_indexes[[str(i) + ".000" for i in range(1, 12)]].to_numpy().transpose())
diff = np.abs(true_depth_indexes_test - pietro_depth_indexes["10.000"].to_numpy().transpose())

print("mean =", np.mean(diff))
print("rms =", np.sqrt(np.mean(np.square(diff))))
print("max =", np.max(diff))

plt.hist(diff, bins=30)
plt.show()