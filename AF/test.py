
import numpy as np

path = r"D:\database\AF\wave\segments\1\mimic_perform_non_af_001_0.npy"
record = np.load(path)
print(len(record[0]))
