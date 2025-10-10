import numpy as np
from scipy import linalg

def _make_homogeneous_rep_matrix(R, t):
    P = np.zeros((4,4))
    P[:3,:3] = R
    P[:3, 3] = t.reshape(3)
    P[3,3] = 1
    return P

def DLT_multi(P_list, point_list):
    """
    Performs Direct Linear Transformation (DLT) for multi-view triangulation (N >= 2).
    
    P_list: list of 3x4 projection matrices (numpy arrays).
    point_list: list of 2D points (uv coordinates) from each view.
    """
    A = []
    
    for P, point in zip(P_list, point_list):
        u, v = point[0], point[1]
        
        if u == -1 or v == -1: 
            continue
            
        A.append(u * P[2, :] - P[0, :])
        A.append(v * P[2, :] - P[1, :])

    A = np.array(A)
    
    if A.shape[0] < 4:
        return np.array([-1, -1, -1]) 
    
    U, s, Vh = linalg.svd(A, full_matrices = False)

    triangulated_point = Vh[-1]
    
    if triangulated_point[3] != 0:
        return triangulated_point[0:3] / triangulated_point[3]
    else:
        return np.array([-1, -1, -1]) 

def read_camera_parameters(camera_id):
    inf = open('camera_parameters/c' + str(camera_id) + '.dat', 'r')
    cmtx = []
    dist = []
    line = inf.readline()
    for _ in range(3):
        line = inf.readline().split()
        line = [float(en) for en in line]
        cmtx.append(line)
    line = inf.readline()
    line = inf.readline().split()
    line = [float(en) for en in line]
    dist.append(line)
    return np.array(cmtx), np.array(dist)

def read_rotation_translation(camera_id, savefolder = 'camera_parameters/'):
    inf = open(savefolder + 'rot_trans_c'+ str(camera_id) + '.dat', 'r')
    inf.readline()
    rot = []
    trans = []
    for _ in range(3):
        line = inf.readline().split()
        line = [float(en) for en in line]
        rot.append(line)
    inf.readline()
    for _ in range(3):
        line = inf.readline().split()
        line = [float(en) for en in line]
        trans.append(line)
    inf.close()
    return np.array(rot), np.array(trans)

def _convert_to_homogeneous(pts):
    pts = np.array(pts)
    if len(pts.shape) > 1:
        w = np.ones((pts.shape[0], 1))
        return np.concatenate([pts, w], axis = 1)
    else:
        return np.concatenate([pts, [1]], axis = 0)

def get_projection_matrix(camera_id):
    cmtx, dist = read_camera_parameters(camera_id)
    rvec, tvec = read_rotation_translation(camera_id)
    P = cmtx @ _make_homogeneous_rep_matrix(rvec, tvec)[:3,:]
    return P

def write_keypoints_to_disk(filename, kpts):
    fout = open(filename, 'w')
    for frame_kpts in kpts:
        for kpt in frame_kpts:
            if len(kpt) == 2:
                fout.write(str(kpt[0]) + ' ' + str(kpt[1]) + ' ')
            else:
                fout.write(str(kpt[0]) + ' ' + str(kpt[1]) + ' ' + str(kpt[2]) + ' ')
        fout.write('\n')
    fout.close()