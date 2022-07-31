import numpy as np
import taichi as ti
from my_ray_tracing_models import Ray, Camera, Hittable_list, Sphere, PI, random_in_unit_sphere, refract, reflect, reflectance, random_unit_vector
import cv2


ti.init(arch=ti.cuda)

# Canvas
aspect_ratio = 1.0
image_width = 800
image_height = int(image_width / aspect_ratio)
canvas = ti.Vector.field(3, dtype=ti.f32, shape=(image_width, image_height))
hit_point_flag = ti.field(dtype=ti.i32, shape=(image_width, image_height))
hit_point_coor = ti.Vector.field(3, dtype=ti.f32, shape=(image_width, image_height))


@ti.kernel
def ray_intersect():
    for i, j in canvas:
        u = (i + ti.random()) / image_width
        v = (j + ti.random()) / image_height
        # color = ti.Vector([0.0, 0.0, 0.0])
        ray = camera.get_ray(u, v)
        xy_range = ti.Vector([-1.4, 1.4, 1, 2])
        flag = intersect_z_plane(ray.origin, ray.direction, 0, xy_range)
        if flag > 0:
            canvas[i, j] = ti.Vector([1.0, 1.0, 1.0])
            hit_point_flag[i, j] = 1
            hit_point_coor[i, j] = intersect_z_plane_point(ray.origin, ray.direction, 0, xy_range)


@ti.func
def intersect_z_plane(ray_o, ray_d, z_plane, xy_range):
    # ray_o and ray_d shape (3, )
    # z_plane value
    # xy_range (xmin, y_min, x_max, y_max)
    flag = 1
    # hit_point = np.zeros(3)
    if ray_d[2] < 1e-6:
        flag = -1
    else:
        t = (z_plane - ray_o[2]) / ray_d[2]
        hit_point = ray_o + t * ray_d
        if hit_point[0] < xy_range[0] or hit_point[0] > xy_range[1] \
            or hit_point[1] < xy_range[2] or hit_point[1] > xy_range[3]:
            flag = -1
    return flag

@ti.func
def intersect_z_plane_point(ray_o, ray_d, z_plane, xy_range):
    # ray_o and ray_d shape (3, )
    # z_plane value
    # xy_range (xmin, y_min, x_max, y_max)
    flag = 1
    # hit_point = np.zeros(3)
    res = ti.Vector([0.0, 0.0, 0.0])
    if ray_d[2] < 1e-6:
        flag = -1
    else:
        t = (z_plane - ray_o[2]) / ray_d[2]
        hit_point = ray_o + t * ray_d
        if hit_point[0] < xy_range[0] or hit_point[0] > xy_range[1] \
            or hit_point[1] < xy_range[2] or hit_point[1] > xy_range[3]:
            flag = -1
        else:
            res = hit_point
    return res


if __name__ == "__main__":
    camera = Camera()
    gui = ti.GUI("Ray Tracing", res=(image_width, image_height))
    canvas.fill(0)
    hit_point_flag.fill(0)
    hit_point_coor.fill(0)
    ray_intersect()
    # while gui.running:
    #     gui.set_image(canvas.to_numpy())
    #     gui.show()
    hit_point_flag_numpy = hit_point_flag.to_numpy()
    hit_point_flag_numpy = np.rot90(hit_point_flag_numpy)
    idx_x, idx_y = np.where(hit_point_flag_numpy == 1)
    print(len(idx_x))
    
    # cv2.imshow("image", hit_point_flag_numpy)
