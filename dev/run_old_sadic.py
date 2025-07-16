import sadic
import numpy as np

input_arg = "7ljs"

input_mode = "code"

result = sadic.sadic(input_arg, input_mode=input_mode)

result.save_pdb("7ljs_sadic_output.pdb")

di = result.get_depth_index()
print(len(di))
print(di[1])
np.save("1ubq.npy", di[1])