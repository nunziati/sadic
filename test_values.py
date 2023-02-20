import numpy as np
import pandas as pd

my_true_depth_indexes = np.load("true_depth_index.npy")

my_depth_indexes = 2. - np.load("test_depth_index.npy")

pietro_depth_indexes = pd.read_csv("1a0i_di.txt", sep='\t')


diff = np.abs(my_depth_indexes - pietro_depth_indexes[[str(i) + ".000" for i in range(1, 12)]].to_numpy().transpose())

print("mean =", np.round(np.mean(diff, axis=1), 3))
print("norm =", np.round(np.linalg.norm(diff, axis=1), 3))
print("max =", np.round(np.max(diff, axis=1), 3))