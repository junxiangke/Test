import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def plot_3D(data):
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(data[:, 2], data[:, 0], data[:, 1], alpha=0.2)
    ax.set_xlim3d(0, 10)
    ax.set_ylim3d(0, 10)
    ax.set_zlim3d(0, 10)
    plt.show()


def generate():
    x = np.arange(5)
    y = np.arange(10)
    x, y, z = np.meshgrid(x, y, x)
    x = np.reshape(x, (-1, 1))
    y = np.reshape(y, (-1, 1))
    z = np.reshape(z, (-1, 1))
    data = np.concatenate([x, y, z], axis=1)
    return data


if __name__ == "__main__":
    # x = np.arange(3)
    # plt.plot(x, x)
    # plt.show()
    data = generate()
    plot_3D(data)