from flow_matching_utils import train_moon_gen
from sklearn.mixture import GaussianMixture
import torch
x_1, y = train_moon_gen(batch_size=10000, device='cuda', is_pretrain=False, mode='up_down_shift') # sample data

gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=0)
gmm.fit(x_1)
labels = gmm.predict(x_1)
# Plot result points by their gmm labels
import matplotlib.pyplot as plt
plt.scatter(x_1[:, 0], x_1[:, 1], c=labels)
plt.show()
plt.savefig("gmm_labels.png")