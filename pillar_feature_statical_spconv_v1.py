import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import math
import cv2
import os
from mypypcd import pypcd
import torch

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
    return objpoints, intensity,


class Pillar:
    def __init__(self, idx):
        # Parameters
        self.idx = idx  # [m]
        self.voxel_size = 0.01  # [m]
        self.col_idx = 0  # [m]
        self.row_idx = 0  # [m]
        self.pt_num = 0
        self.pt_idx = []
        self.intensity_avg = 0.0
        self.intensity_std = 0.0
        # self.intensity_avg = 0.001
        # self.intensity_std = 0.001

    def fea_updates(self, xyzi):
        self.pt_num = len(self.pt_idx)
        intensity_list = [xyzi[3] for _ in range(len(xyzi))]
        self.intensity_avg = sum(intensity_list) / len(intensity_list)
        self.intensity_std = np.std(intensity_list)


pcd_path = "./"
xyz_all = np.array([0, 0, 0])

voxel_size = 1.0

# VOXEL_SIZE = [1.0, 0.1, 0.1]
# POINT_CLOUD_RANGE = [-2.0, 8.0, -10.0, 30.0, 0.0, 60.0]
# NUM_POINT_FEATURES = 4
# MAX_POINTS_PER_VOXEL = 10
# MAX_NUMBER_OF_VOXELS = 1500000

VOXEL_SIZE = [1.0, 0.1, 0.3]
POINT_CLOUD_RANGE = [0, -10.0, 11.0, 8.0, 30.0, 60.0]
NUM_POINT_FEATURES = 4
MAX_POINTS_PER_VOXEL = 32
MAX_NUMBER_OF_VOXELS = 150000

x_offset = VOXEL_SIZE[0] / 2 + POINT_CLOUD_RANGE[0]
y_offset = VOXEL_SIZE[1] / 2 + POINT_CLOUD_RANGE[1]
z_offset = VOXEL_SIZE[2] / 2 + POINT_CLOUD_RANGE[2]

rotation = np.array([0.93913256, 0.02410406, -0.34270838, 6.88965565,
                     -0.02264348, 0.99970946, 0.00826306, 0.00000000,
                     0.34280798, -0.00000000, 0.93940550, 0.00000000,
                     0.00000000, 0.00000000, 0.00000000, 1.00000000])

# rotation = np.array([0.93913256, 0.02410406, -0.34270838,
#                     -0.02264348, 0.99970946, 0.00826306,
#                      0.34280798, -0.00000000, 0.93940550])
rotation = rotation.reshape(4, 4)
print(rotation)

trans = [6.88965565, 0.00000000, 0.00000000]

for filename in os.listdir(pcd_path):
    whole_path = os.path.join(pcd_path, filename)
    print(whole_path)
    if whole_path.find(".pcd") != 0:
        points, intensity = get_points(whole_path)

        xyz_Invertible = np.hstack((points, np.ones(intensity.shape)))
        print(xyz_Invertible)
        points_trans = np.array(xyz_Invertible.dot(np.transpose(rotation)))
        print(points_trans)
        xyzi = np.hstack((points_trans[:, 0:3], intensity))

        print(xyzi)

        voxel_generator = VoxelGeneratorWrapper(vsize_xyz=VOXEL_SIZE,
                                                coors_range_xyz=POINT_CLOUD_RANGE,
                                                num_point_features=NUM_POINT_FEATURES,
                                                max_num_points_per_voxel=MAX_POINTS_PER_VOXEL,
                                                max_num_voxels=MAX_NUMBER_OF_VOXELS, )

        # points = data_dict['points'][:, 0: self.num_point_features].copy()
        # 假设一份点云数据是N*4，那么经过pillar生成后会得到三份数据
        # voxels代表了每个生成的voxel数据，维度是[M, 32, 4]
        # coordinates代表了每个生成的voxel所在的【zyx】轴坐标，维度是[M,3]
        # num_points代表了每个生成的voxel中有多少个有效的点维度是[m,]，因为不满5会被0填充
        voxel_output = voxel_generator.generate(xyzi)
        voxels, coordinates, num_points = voxel_output
        print("voxels", voxels)
        print("coordinates", coordinates)
        print("num_points", num_points)


        voxel_idx = []
        voxel_fea = []
        voxel_fea_std = []

        # f_center = torch.zeros_like(voxels[:, :, :3])
        for i, voxel in enumerate(voxels):
            voxel_idx.append(i)
            intensity_np = voxel[:, 3][voxel[:, 3] != 0]
            # print("after:", intensity_np)
            intensity_aveage = sum(intensity_np) / len(intensity_np)
            intensity_std = np.std(intensity_np)
            voxel_fea.append(intensity_aveage)
            voxel_fea_std.append(intensity_std)
            # coordinates[:, 3].to(voxels.dtype).unsqueeze(1) * VOXEL_SIZE[0] + x_offset
            # coordinates[:, 2].to(voxels.dtype).unsqueeze(1) * VOXEL_SIZE[1] + y_offset
            # coordinates[:, 1].to(voxels.dtype).unsqueeze(1) * VOXEL_SIZE[2] + z_offset

        exit()
