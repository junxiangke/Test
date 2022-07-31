import numpy as np
import matplotlib.pyplot as plt


def allocate_probe(grid_size, volume):
    # grid_size (x_dim, y_dim, z_dim)
    # volume (x_min, x_max, y_min, y_max, z_min, z_max)
    #
    # out: (x, y, z, pos)
    grid_size_x, grid_size_y, grid_size_z = grid_size
    x_min, x_max, y_min, y_max, z_min, z_max = volume
    x = np.linspace(x_min, x_max, grid_size_x)
    y = np.linspace(y_min, y_max, grid_size_y)
    z = np.linspace(z_min, z_max, grid_size_z)
    out = np.zeros((8, 8, 8, 3))
    for i in range(grid_size_x):
        for j in range(grid_size_y):
            for k in range(grid_size_z):
                out[i, j, k] = np.array([x[i], y[j], z[k]])
    return out


def coor_map_probe(x, x_normal, grid_size, volume, probe_pos, probe_color, probe_normal):
    # x (point, 3)
    # grid_size (x_dim, y_dim, z_dim)
    # volume (x_min, x_max, y_min, y_max, z_min, z_max)
    # probe_pos (x, y, z, pos)
    # probe_color (x, y, z, h, w, c)
    # probe_normal (h, w, norm)
    # output feat (point, x(2), y(2), z(2), 3)
    # output grid_coor (point, dim3, min_max2)
    grid_size_x, grid_size_y, grid_size_z = grid_size
    x_min, x_max, y_min, y_max, z_min, z_max = volume
    delta_x = (x_max - x_min) / (grid_size_x - 1)
    delta_y = (y_max - y_min) / (grid_size_y - 1)
    delta_z = (z_max - z_min) / (grid_size_z - 1)
    id_x = (x[:, 0] - x_min) / delta_x
    id_y = (x[:, 1] - y_min) / delta_y
    id_z = (x[:, 2] - z_min) / delta_z
    index = np.vstack([id_x, id_y, id_z])
    index = index.T.astype(np.int32)
    # print(index)
    grid_coor = np.zeros((x.shape[0], 3, 2))  # (point, xyz_dim(3), min_max(2))
    grid_coor[:, :, 0] = probe_pos[index[:, 0], index[:, 1], index[:, 2]]  # (point, min_xyz(3))
    grid_coor[:, :, 1] = probe_pos[index[:, 0] + 1, index[:, 1] + 1, index[:, 2] + 1]  # (point, max_xyz(3))


    # pick up the closest normal
    # print(np.argmax(np.sum(x_normal[0] * probe_normal, axis=-1)))
    probe_normal = probe_normal.reshape((1, -1, 3))  # (1, ray_num, 3)
    x_normal = x_normal[:, None, :]  # (point, 1, 3)
    tmp = x_normal * probe_normal  # (point, ray_num, 3)
    # print(tmp.shape)
    tmp = np.sum(tmp, axis=-1)  # (point, ray_num)
    ray_idx = np.argmax(tmp, axis=1)  # most close ray index for all point
    # print(ray_idx)

    # cal feat
    feat = np.zeros((x.shape[0], 2, 2, 2, 3))
    sh = probe_color.shape  # probe_color (x, y, z, h, w, c)
    probe_color = np.reshape(probe_color, (sh[0], sh[1], sh[2], -1, 3))  # (x, y, z, h*w, c)
    print(probe_color.shape)
    for i in range(2):
        for j in range(2):
            for k in range(2):
                feat[:, i, j, k, :] = probe_color[index[:, 0]+i, index[:, 1]+j, index[:, 2]+k, ray_idx, :]
    return grid_coor, feat



if __name__ == "__main__":
    grid_size = (8, 8, 8)
    volume = (-1.45, 1.45, -0.45, 2.45, -2.05, 0.95)
    out = allocate_probe(grid_size, volume)
    probe_pos = out
    ax = plt.axes(projection="3d")
    out = out.reshape((-1, 3))
    ax.scatter3D(out[:, 0], out[:, 1], out[:, 2])
    # ax.scatter3D(0, 0, 0, color=(1, 0, 0))
    x = np.random.rand(2, 3) * 0.5
    x_normal = np.random.rand(2, 3)
    x_normal = x_normal / np.linalg.norm(x_normal, axis=1, keepdims=True)
    probe_color = np.random.rand(*grid_size, 11, 11, 3)
    probe_normal = np.random.rand(11, 11, 3)
    grid_coor, feat = coor_map_probe(x, x_normal, grid_size, volume, probe_pos, probe_color, probe_normal)
    # res = probe[index[:, 0], index[:, 1], index[:, 2]]
    # print(res)
    # ax.scatter3D(res[:, 0], res[:, 1], res[:, 2], color=(1, 0, 0))
    # ax.scatter3D(x[:, 0], x[:, 1], x[:, 2], color=(0, 0, 0))
    # ax.scatter3D(grid_coor[:, :, 0][:, 0], grid_coor[:, :, 0][:, 1], grid_coor[:, :, 0][:, 2], color=(1, 0, 0))
    # ax.scatter3D(grid_coor[:, :, 1][:, 0], grid_coor[:, :, 1][:, 1], grid_coor[:, :, 1][:, 2], color=(1, 0, 0))
    # plt.show()

