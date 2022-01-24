import torch
import torch.nn.functional as F


def symmetric_orthogonalization(x):
    """Maps 9D input vectors onto SO(3) via symmetric orthogonalization.

      x: should have size [batch_size, 9]

      Output has size [batch_size, 3, 3], where each inner 3x3 matrix is in SO(3).
      """
    m = x.reshape(-1, 3, 3)
    u, s, v = torch.svd(m)
    vt = torch.transpose(v, 1, 2)
    det = torch.det(torch.matmul(u, vt))
    det = det.reshape(-1, 1, 1)
    vt = torch.cat((vt[:, :2, :], vt[:, -1:, :] * det), 1)
    r = torch.matmul(u, vt)
    return r


def angle_axis_to_rotation_matrix(angle_axis):
    """Convert batch of 3D angle-axis vectors into a batch of 3D rotation matrices

    Arguments:
        angle_axis: (b, 3) Torch tensor,
            batch of 3D angle-axis vectors

    Return Values:
        rotation_matrix: (b, 3, 3) Torch tensor,
            batch of 3D rotation matrices
    """
    assert angle_axis.shape[-1] == 3, "Angle-axis vector must be a (*, 3) tensor, received {}".format(
        angle_axis.shape)

    def angle_axis_to_rotation_matrix_rodrigues(angle_axis, theta2):
        theta = torch.sqrt(theta2).unsqueeze(-1)  # bx1
        r = angle_axis / theta  # bx3
        rx = r[..., 0]  # b
        ry = r[..., 1]  # b
        rz = r[..., 2]  # b
        r_skew = torch.zeros_like(r).unsqueeze(-1).repeat_interleave(3, dim=-1)  # bx3x3
        r_skew[..., 2, 1] = rx
        r_skew[..., 1, 2] = -rx
        r_skew[..., 0, 2] = ry
        r_skew[..., 2, 0] = -ry
        r_skew[..., 1, 0] = rz
        r_skew[..., 0, 1] = -rz
        R = torch.eye(3, dtype=r.dtype, device=r.device).unsqueeze(0) \
            + theta.sin().unsqueeze(-1) * r_skew \
            + (1.0 - theta.cos().unsqueeze(-1)) * torch.matmul(r_skew, r_skew)  # bx3x3
        return R

    def angle_axis_to_rotation_matrix_taylor(angle_axis):
        rx, ry, rz = torch.chunk(angle_axis, 3, dim=-1)
        ones = torch.ones_like(rx)
        R = torch.cat([ones, -rz, ry, rz, ones, -rx, -ry, rx, ones], dim=1).view(-1, 3, 3)
        return R

    theta2 = torch.einsum('bi,bi->b', (angle_axis, angle_axis))

    eps = 1e-6
    if (theta2 > eps).all():
        rotation_matrix = angle_axis_to_rotation_matrix_rodrigues(angle_axis, theta2)
    else:
        rotation_matrix = angle_axis_to_rotation_matrix_taylor(angle_axis)
        rotation_matrix_rodrigues = angle_axis_to_rotation_matrix_rodrigues(angle_axis, theta2)
        # Iterate over batch dimension
        # Note: cannot use masking or torch.where because any NaNs in the gradient
        # of the unused branch get propagated
        # See: https://github.com/pytorch/pytorch/issues/9688
        for b in range(angle_axis.shape[0]):
            if theta2[b, ...] > eps:
                rotation_matrix[b, ...] = rotation_matrix_rodrigues[b:(b + 1), ...]
    return rotation_matrix


def normalise_points(p):
    return F.normalize(p, p=2, dim=-1)


def transform_points(p, R, t):
    return torch.einsum('brs,bms->bmr', (R, p)) + t.unsqueeze(-2)


def transform_and_normalise_points(p, R, t):
    return normalise_points(transform_points(p, R, t))


def transform_points_by_theta(p, theta):
    R = angle_axis_to_rotation_matrix(theta[..., :3])
    t = theta[..., 3:]
    t = F.pad(t, (0, 1), "constant", 0.0)
    return transform_points(p, R, t)


def transform_and_normalise_points_by_theta(p, theta):
    return normalise_points(transform_points_by_theta(p, theta))


def project_points_by_theta(p, theta, K=None, b2p=True, scale=False):
    bs = p.size(0)
    pts3d_in = p
    if scale:
        s_3d = pts3d_in.reshape(bs, -1).std(1)[:, None]
        mu_3d = pts3d_in.mean(1)
        pts3d_in = (pts3d_in - mu_3d[:, None]) / s_3d.unsqueeze(-1)
    p_transformed = transform_points_by_theta(pts3d_in, theta)
    if b2p:
        return bearings_to_points(p_transformed, K)
    else:
        T = K[:, 2:]
        T = torch.cat([T, torch.zeros_like(T[:, :1])], dim=-1)
        p_transformed = p_transformed * K[:, 0][:, None, None] + T[:, None]
        return p_transformed


def points_to_bearings(p):
    """
    Arguments:
        p: (b, n, 2) Torch tensor,
            batch of 2D point-sets
        K: (b, 4) Torch tensor or None,
            batch of camera intrinsic parameters (fx, fy, cx, cy),
            set to None if points are already K-normalised
    """
    bearings = F.pad(p, (0, 1), "constant", 1.0)
    return F.normalize(bearings, p=2, dim=-1)


def bearings_to_points(bearings, K=None):
    """
    Arguments:
        bearings: (b, n, 3) Torch tensor,
            batch of bearing vector sets
        K: (b, 4) Torch tensor or None,
            batch of camera intrinsic parameters (fx, fy, cx, cy),
            set to None if points are already K-normalised
    """
    points = bearings[:, :, :2]  # / bearings[:, :, 2:3]
    if K is not None:  # p is in image coordinates, apply (px - cx) / fx
        K = K.unsqueeze(-2)  # (b, 1, 4)
        points = points * K[:, :, :2] + K[:, :, 2:]
    return points


