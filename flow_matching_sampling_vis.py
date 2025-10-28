import torch
import numpy as np
data_sample = torch.randn(1, 2)
random_sample = torch.randn(100, 2)
random_t = torch.rand(100)
hidden_z_list = np.array([])
velocity_list = np.array([])
for i in range(100):
    hidden_z = (1-random_t[i]) * data_sample + random_sample[i].unsqueeze(0) * random_t[i]
    print(hidden_z)
    hidden_z_list = np.append(hidden_z_list, hidden_z.detach().numpy())
    velocity_list = np.append(velocity_list, (data_sample - random_sample[i].unsqueeze(0)).detach().numpy())
hidden_z_list = hidden_z_list.reshape(100, 2)
velocity_list = velocity_list.reshape(100, 2)
# Visualize hidden_z_list and data sample with different colors. Each hidden_z point need a velocity arrow pointing to the data sample.
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 8))
plt.scatter(hidden_z_list[:, 0], hidden_z_list[:, 1], c='blue', label='Hidden z')
plt.scatter(data_sample[0, 0], data_sample[0, 1], c='red', label='Data sample')
# Let the arrow shorter
plt.quiver(
    hidden_z_list[:, 0], hidden_z_list[:, 1],
    velocity_list[:, 0] * 0.2, velocity_list[:, 1] * 0.2,  # scale down arrow length
    angles='xy', scale_units='xy', scale=1, color='green', width=0.003
)
plt.xlim(-4, 4)
plt.ylim(-4, 4)
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Hidden z and Velocity Vectors')
plt.legend()
plt.grid()
plt.savefig('hidden_z_velocity.png')
plt.show()
