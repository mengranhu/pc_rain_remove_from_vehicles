import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import math
import cv2
import os
from mypypcd import pypcd
import torch
import glob

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

tv = None
try:
    import cumm.tensorview as tv
except:
    pass


class VoxelGeneratorWrapper():
    def __init__(self, vsize_xyz, coors_range_xyz, num_point_features, max_num_points_per_voxel, max_num_voxels):
        try:
            from spconv.utils import VoxelGeneratorV2 as VoxelGenerator
            self.spconv_ver = 1
        except:
            try:
                from spconv.utils import VoxelGenerator
                self.spconv_ver = 1
            except:
                from spconv.utils import Point2VoxelCPU3d as VoxelGenerator
                self.spconv_ver = 2

        if self.spconv_ver == 1:
            self._voxel_generator = VoxelGenerator(
                voxel_size=vsize_xyz,
                point_cloud_range=coors_range_xyz,
                max_num_points=max_num_points_per_voxel,
                max_voxels=max_num_voxels
            )
        else:
            self._voxel_generator = VoxelGenerator(
                vsize_xyz=vsize_xyz,
                coors_range_xyz=coors_range_xyz,
                num_point_features=num_point_features,
                max_num_points_per_voxel=max_num_points_per_voxel,
                max_num_voxels=max_num_voxels
            )

    def generate(self, points):
        if self.spconv_ver == 1:
            voxel_output = self._voxel_generator.generate(points)
            if isinstance(voxel_output, dict):
                voxels, coordinates, num_points = \
                    voxel_output['voxels'], voxel_output['coordinates'], voxel_output['num_points_per_voxel']
            else:
                voxels, coordinates, num_points = voxel_output
        else:
            assert tv is not None, f"Unexpected error, library: 'cumm' wasn't imported properly."
            voxel_output = self._voxel_generator.point_to_voxel(tv.from_numpy(points))
            tv_voxels, tv_coordinates, tv_num_points = voxel_output
            # make copy with numpy(), since numpy_view() will disappear as soon as the generator is deleted
            voxels = tv_voxels.numpy()
            # tv_voxels.numpy_view()
            coordinates = tv_coordinates.numpy()
            num_points = tv_num_points.numpy()
        return voxels, coordinates, num_points


def get_points(src_lidar_file, trans_axis=False):
    pc = pypcd.PointCloud.from_path(src_lidar_file)
    np_x = (np.array(pc.pc_data['x'], dtype=np.float32)).astype(np.float32)
    np_y = (np.array(pc.pc_data['y'], dtype=np.float32)).astype(np.float32)
    np_z = (np.array(pc.pc_data['z'], dtype=np.float32)).astype(np.float32)

    intensity = (np.array(pc.pc_data['intensity'], dtype=np.float32)).astype(np.float32).reshape(-1, 1)
    # elongation = (np.array(pc.pc_data['elongation'], dtype=np.float32)).astype(np.float32).reshape(-1, 1)

    if trans_axis:
        np_y = -1 * np_y
        objpoints = np.transpose(np.vstack((np_z, np_y, np_x)))
    else:
        objpoints = np.transpose(np.vstack((np_x, np_y, np_z)))
    # return objpoints, intensity, elongation
    return objpoints, intensity


pcd_path = "./"
xyz_all = np.array([0, 0, 0])

voxel_size = 1.0

VOXEL_SIZE = [1.5, 0.1, 0.3]
POINT_CLOUD_RANGE = [0, -1.0, 10.0, 1.5, 5.0, 50.0]
NUM_POINT_FEATURES = 4
MAX_POINTS_PER_VOXEL = 32
MAX_NUMBER_OF_VOXELS = 150000

trans = [6.88965565, 0.00000000, 0.00000000]

pcd_files = glob.glob(os.path.join(pcd_path, '*.pcd'))
for filename in pcd_files:
    if filename.endswith('.pcd'):
        points, intensity = get_points(filename)
        voxel_generator = VoxelGeneratorWrapper(vsize_xyz=VOXEL_SIZE,
                                                coors_range_xyz=POINT_CLOUD_RANGE,
                                                num_point_features=NUM_POINT_FEATURES,
                                                max_num_points_per_voxel=MAX_POINTS_PER_VOXEL,
                                                max_num_voxels=MAX_NUMBER_OF_VOXELS, )
        xyzi = np.hstack((points, intensity.astype(int)))

        voxel_output = voxel_generator.generate(xyzi)
        # coordinates order z->y->x
        voxels, coordinates, num_points = voxel_output

        y_list = []
        z_list = []
        voxel_fea_list = []
        voxel_fea_std_list = []
        voxel_clean_list = []
        idx = 0
        for i, coordinate in enumerate(coordinates):
            voxel = voxels[i]
            # print("voxel: ", voxel)
            voxel_clean = [voxel[j] for j in range(len(voxel)) if voxel[j][0] > 0]
            # print("after voxel: ", voxel_clean)
            if len(voxel_clean) > 0:
                idx = idx + 1
                y = POINT_CLOUD_RANGE[1] + VOXEL_SIZE[1] * coordinate[1]
                z = POINT_CLOUD_RANGE[2] + VOXEL_SIZE[2] * coordinate[0]
                y_list.append(y)
                z_list.append(z)
                voxel_clean = np.array(voxel_clean)
                voxel_clean_list.append(voxel_clean)
                intensity_average = sum(voxel_clean[:, 3]) / len(voxel_clean[:, 3])
                intensity_std = np.std(voxel_clean[:, 3])
                voxel_fea_list.append(intensity_average)
                voxel_fea_std_list.append(intensity_std)
                # print('y:', y)
                # print('z:', z)
                # print('i:', idx)
                # print("voxel clean:", voxel_clean)
        # print('y_list:', y_list)
        # print('z_list:', z_list)
        # print('voxel_fea_list:', voxel_fea_list)
        print('voxel_fea_std_list:', voxel_fea_std_list)

        # ax.scatter(y_list, z_list, voxel_fea_list)
        ax.scatter(y_list, z_list, voxel_fea_std_list)
        # ax.scatter(y_list, z_list, voxel_fea_list)
        ax.set_xlabel('Y', fontdict={'size': 15, 'color': 'red'})
        ax.set_ylabel('Z', fontdict={'size': 15, 'color': 'red'})
        ax.set_zlabel('Intensity', fontdict={'size': 15, 'color': 'red'})
        plt.show()

        remove_rain_xyzi = np.array([1, 0, 0, 0])
        for i, voxel_clean in enumerate(voxel_clean_list):
            if voxel_fea_std_list[i] > 20:
                remove_rain_xyzi = np.vstack((remove_rain_xyzi, voxel_clean))

        remove_rain_xyzi = remove_rain_xyzi[1:, :]

        print(remove_rain_xyzi)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array(remove_rain_xyzi)[:, 0: 3])
        o3d.io.write_point_cloud("./remove_rain_xyz.pcd", pcd)
