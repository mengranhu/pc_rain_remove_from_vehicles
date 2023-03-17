from mypypcd import pypcd

def get_points(src_lidar_file, trans_axis=False):
    pc = pypcd.PointCloud.from_path(src_lidar_file)
    np_x = (np.array(pc.pc_data['x'], dtype=np.float32)).astype(np.float32)
    np_y = (np.array(pc.pc_data['y'], dtype=np.float32)).astype(np.float32)
    np_z = (np.array(pc.pc_data['z'], dtype=np.float32)).astype(np.float32)

    intensity = (np.array(pc.pc_data['intensity'], dtype=np.float32)).astype(np.float32).reshape(-1,1)
    elongation = (np.array(pc.pc_data['elongation'], dtype=np.float32)).astype(np.float32).reshape(-1,1)

    if trans_axis:
        np_y = -1 * np_y
        objpoints = np.transpose(np.vstack((np_z, np_y, np_x)))
    else:
        objpoints = np.transpose(np.vstack((np_x, np_y, np_z)))
    return objpoints, intensity, elongation
