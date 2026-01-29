from random import random
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
import numpy as np

fig, ax = plt.subplots(1, 1, figsize=(6, 3))
ax.set(xlabel='dimensions (m)', ylabel='log(dmax/dmin)', title='dmax/dmin vs. dimensionality')
line_styles = {0: 'ro-', 1: 'b^-', 2: 'gs-', 3: 'cv-'}

for idx, num_samples in enumerate([2, 23, 56, 98]):
    feature_range = np.arange(1,101)
    ratios = []
    for num_features in feature_range:
        X, Y = make_classification(n_samples=num_samples, n_features=num_features, n_informative=1, n_redundant=0, n_clusters_per_class=1, random_state=1)
        
        query_point = np.random.randint(0, len(X))

        updatedX = np.delete(X, query_point, axis = 0)

        distances = np.linalg.norm(updatedX - query_point, axis = 1)
        ratio = np.max(distances) / np.min(distances)
        ratios.append(ratio)

    ax.plot(feature_range, np.log(ratios), line_styles[idx], label=f'N={num_samples:,}')

plt.legend()
plt.tight_layout()
plt.grid(True)
plt.show()