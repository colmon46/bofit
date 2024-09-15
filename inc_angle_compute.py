import numpy as np
import os
import pandas as pd

angles_name_en = ['Angle between top of the body and ground', 'angle between left upper arm and top of the body', 'angle between left upper arm and front of the body',
                    'angle of left elbow', 'angle between right upper arm and top of the body', 'angle between right upper arm and front of the body', 'angle of right elbow',
                    'angle between upper body and top of the body', 'angle between left thigh and top of the body', 'angle between left thigh and front of the body', 'angle of left knee',
                    'angle between right thigh and top of the body', 'angle between right thigh and front of the body', 'angle of right knee',
                    'twist or rotate angle of upper body', 'distance off the ground with feet', 'distance moved forward relative to initial state', 'distance moved to left relative to initial state']

def rounding(n):
    return round(n / 5) * 5

def ANGLE(a, b):
    cos = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    if cos > 1:
        cos = 1
    elif cos < -1:
        cos = -1
    return np.degrees(np.arccos(cos))

def compute(path, sample_num):

    data = numpy.load(path)
    joints = {'hip': 0, 'rhip': 1, 'rknee': 2, 'rfoot': 3,
            'lhip': 4, 'lknee': 5, 'lfoot': 6,
            'spine': 7, 'thorax': 8, 'neck': 9, 'head': 10,
            'lshoulder': 11, 'lelbow': 12, 'lwrist': 13,
            'rshoulder': 14, 'relbow': 15, 'rwrist': 16}  

    for idx in range(len(data)):
        for joint_idx in range(17):
            x_, y_, z_ = data[idx, joint_idx, :]
            data[idx, joint_idx, :] = np.array([-x_, -z_, -y_ ])

    indices = np.linspace(0, len(data)-1, sample_num, dtype=int)

    features = []   

    for idx in indices:
        frame = data[idx]
        
        axis_z = frame[joints['spine']] - frame[joints['hip']] 
        axis_z = axis_z / np.linalg.norm(axis_z)
        axis_y = frame[joints['lhip']] - frame[joints['hip']]  
        axis_y = axis_y / np.linalg.norm(axis_y)
        axis_x = np.cross(axis_y, axis_z)  

        if idx == 0:
            shoulder_line_0 = frame[joints['lshoulder']] - frame[joints['rshoulder']]
            dis_off_ground_0 = min(abs(frame[joints['lfoot']][2]), abs(frame[joints['rfoot']][2]))
            hip_0 = frame[joints['hip']][:2]
            axis_x_0 = axis_x

        z_ground = np.degrees(np.arcsin(np.dot(axis_z, np.array([0, 0, 1]))))

        lshoulder_elbow = frame[joints['lshoulder']] - frame[joints['lelbow']]
        lshoulder_z = ANGLE(-lshoulder_elbow, axis_z)
        lshoulder_x = ANGLE(-lshoulder_elbow, axis_x)


        lelbow_lwrist = frame[joints['lelbow']] - frame[joints['lwrist']]
        lelbow_angle = ANGLE(lshoulder_elbow, -lelbow_lwrist)


        rshoulder_elbow = frame[joints['rshoulder']] - frame[joints['relbow']]
        rshoulder_z = ANGLE(-rshoulder_elbow, axis_z)
        rshoulder_x = ANGLE(-rshoulder_elbow, axis_x)


        relbow_rwrist = frame[joints['relbow']] - frame[joints['rwrist']]
        relbow_angle = ANGLE(rshoulder_elbow, -relbow_rwrist)

        trunk = frame[joints['neck']] - frame[joints['spine']]
        trunk_z = ANGLE(trunk, axis_z)


        lknee_lhip = frame[joints['lknee']] - frame[joints['lhip']]
        lhip_z = ANGLE(lknee_lhip, axis_z)
        lhip_x = ANGLE(lknee_lhip, axis_x)


        lfoot_lknee = frame[joints['lfoot']] - frame[joints['lknee']]
        lknee_angle = ANGLE(-lfoot_lknee, lknee_lhip)


        rknee_rhip = frame[joints['rknee']] - frame[joints['rhip']]
        rhip_z = ANGLE(rknee_rhip, axis_z)
        rhip_x = ANGLE(rknee_rhip, axis_x)

        rfoot_rknee = frame[joints['rfoot']] - frame[joints['rknee']]
        rknee_angle = ANGLE(-rfoot_rknee, rknee_rhip)

        shoulder_line = frame[joints['lshoulder']] - frame[joints['rshoulder']]
        rotation = ANGLE(shoulder_line, shoulder_line_0)


        dis_off_ground = min(abs(frame[joints['lfoot']][2]), abs(frame[joints['rfoot']][2]))
        distance = abs(dis_off_ground_0 - dis_off_ground) 


        distance_xy = frame[joints['hip']][:2] - hip_0
        distance_x = np.dot(distance_xy, axis_x[:2]) / np.linalg.norm(axis_x[:2])

        distance_y = np.dot(distance_xy, axis_y[:2]) / np.linalg.norm(axis_y[:2])


        feature = [z_ground, lshoulder_z, lshoulder_x, lelbow_angle, 
                rshoulder_z, rshoulder_x, relbow_angle, trunk_z,
                lhip_z, lhip_x, lknee_angle, rhip_z, rhip_x, rknee_angle,
                rotation, distance, distance_x, distance_y]
        for i in range(len(feature)):
            feature[i] = rounding(feature[i])
        features.append(feature)
    return features
    
