import torch
import numpy as np
import torchgeometry as tgm
import math

torch.set_printoptions(sci_mode=False, precision=2)

joint_names = {
    0: 'Pelvis',  
    1: 'Left Hip',  
    2: 'Right Hip', 
    3: 'Spine1',  
    4: 'Left Knee',  
    5: 'Right Knee', 
    6: 'Spine2',  
    7: 'Left Ankle', 
    8: 'Right Ankle',  
    9: 'Spine3',  
    10: 'Left Foot', 
    11: 'Ritht Foot',  
    12: 'Neck',  
    13: 'Left Collar',  
    14: 'Right Collar',  
    15: 'Head', 
    16: 'Left Shoulder', 
    17: 'Right Shoulder', 
    18: 'Left Elbow',  
    19: 'Right Elbow', 
    20: 'Left Wrist',  
    21: 'Right Wrist',         
    22: 'Left Hand',  
    23: 'Right Hand'
}

def quaternion_to_tait_bryan_angle(quaternion):
    """Convert a quaternion to Tait-Bryan Angles.

    Args:
        quaternion (torch.Tensor): tensor with quaternion.

    Return:
        torch.Tensor: tensor with quaternion.

    Shape:
        - Input: :math:`(*, 4)` where `*` means, any number of dimensions
        - Output: :math:`(*, 3)`

    Example:
        >>> angle_axis = torch.rand(2, 4)  # Nx4
        >>> tb_angle = quaternion_to_tait_bryan_angle(quaternion)  # Nx3
    """
    q_w = quaternion[..., 0:1]
    q_x = quaternion[..., 1:2]
    q_y = quaternion[..., 2:3]
    q_z = quaternion[..., 3:4]

    yaw = torch.arctan2(2 * (q_w * q_z + q_x * q_y), 1 - 2 * (q_y * q_y + q_z * q_z))
    theta = torch.arcsin(2 * (q_w * q_y - q_z * q_x))
    roll = torch.arctan2(2 * (q_w * q_x + q_y * q_z), 1 - 2 * (q_x * q_x + q_y * q_y))
    tb_angle = torch.cat((yaw, theta, roll), dim = -1)
    return tb_angle

def compute_info(path, sample_num, part=False):

    data = np.load(path)

    pose_data = data['pose']
    pose_data = pose_data.reshape(-1, 24, 3)  # (frame_num, joint_num, 3) 
    T = pose_data.shape[0]
    sample_indices = np.linspace(0, T - 1, sample_num, dtype=int) # sampling
    sample_data = pose_data[sample_indices]

    part_indices = [0, 1, 2, 4, 5, 6, 7, 8, 12, 15, 16, 17, 18, 19, 20, 21]
    if part:
        part_data = sample_data[:, part_indices, :]
    else:
        part_data = sample_data[:, :, :]
    quat_data = tgm.angle_axis_to_quaternion(torch.tensor(part_data))

    tb_angle_data = quaternion_to_tait_bryan_angle(quat_data)
    tb_degree_data = tgm.rad2deg(tb_angle_data)  # (sample_num, J, 3)

    pelvis_pos = data['pred_joints'][sample_indices, 0, :].reshape(-1, 1, 3)  #(sample_num, 1, 3)
    tb_degree_data = np.concatenate((pelvis_pos, tb_degree_data), axis=1)
    
    return tb_degree_data


    


