import matplotlib.pyplot as plt
import torch
import numpy as np

data = torch.load('fom_pred.pt')
for i in range(data.shape[0]):
    print(i)
    print(data[i].shape)
    snap = torch.from_numpy(data[i])
    snap = snap.permute(1, 0)
    x = snap[0]
    y = snap[1]
    z = snap[2]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(z, y, x, c='r', marker='o')
    plt.savefig(f'Viz/Water_full/snap_{i}.png')
    plt.close(fig)



