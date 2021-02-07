import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

dataset = pd.read_csv('Pizza.csv', header = 0)
x = dataset.iloc[:,1:8].values
x_std = StandardScaler().fit_transform(x)

cov_mat = np.cov(x_std.T)
eig_vals, eig_vecs = np.linalg.eig(cov_mat)

eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i]) for i in range(len(eig_vals))]

tot = sum(eig_vals)
var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse = True)]
cum_var_exp = np.cumsum(var_exp)

plt.figure(figsize=(8,8))
plt.bar(range(7), var_exp, alpha=0.5, align='center', label='Individual Variance Explained')
plt.step(range(7), cum_var_exp, where='mid', label ='Cumulative Variance Explained')
plt.ylabel('Explained Variance ratio')
plt.xlabel('Principal components')
plt.legend(loc='best')
plt.tight_layout()
plt.show()