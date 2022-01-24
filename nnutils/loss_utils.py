"""
Loss Utils.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import kornia
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_batch_svd import svd
import lpips


def rand_rot(N, max_rot_angle=float(math.pi), axes=(1, 1, 1)):
    rand_axis = torch.zeros((N, 3)).float().normal_()

    axes = torch.Tensor(axes).float()
    rand_axis = axes[None, :] * rand_axis

    rand_axis = F.normalize(rand_axis, dim=1, p=2)
    rand_angle = torch.ones(N).float().uniform_(0, max_rot_angle)
    R_ss_rand = rand_axis * rand_angle[:, None]
    return R_ss_rand


def texture_dt_loss_v(texture_flow, dist_transf, reduce=True):
    V = texture_flow.size(1)
    flow_grid = texture_flow.view(-1, V, 1, 2)
    dist_transf = torch.nn.functional.grid_sample(dist_transf, flow_grid, padding_mode='border', align_corners=True)
    if reduce:
        return dist_transf.mean()
    else:
        dist_transf = dist_transf.mean(-1).mean(-1).squeeze(1)
        return dist_transf


def bds_loss(verts, bds, pix_to_face, reduce=True, n_samples=None, k=1):
    bt, nv, _ = verts.shape
    _, H, W, _ = pix_to_face.shape
    if n_samples is not None:
        indices = torch.randperm(bds.shape[1])[:n_samples]
        bds_v = bds[..., indices, :-1]
        bds_m = bds[..., indices, -1]
    else:
        bds_v = bds[..., :-1]
        bds_m = bds[..., -1]

    dist = torch.cdist(bds_v, verts) ** 2
    min_dists = dist.topk(k, largest=False, sorted=False)[0]
    bds_m = bds_m[:, :, None]
    loss = (min_dists * bds_m).mean(-1).mean(-1)
    if reduce:
        return loss.mean()
    else:
        return loss


def bds_loss_mask(verts, bds, faces, pix_to_face, reduce=True, n_samples=1000, k=1):
    bt, nv, _ = verts.shape
    _, H, W, _ = pix_to_face.shape
    indices = torch.randperm(bds.shape[1])[:n_samples]
    bds_v = bds[..., indices, :-1]
    bds_m = bds[..., indices, -1]
    fi_maps = pix_to_face[..., 0].reshape(bt, -1).detach()
    visible_vertices_ = torch.zeros(bt * nv, device=faces.device)
    mask_faces = faces < 0
    faces_ = faces + torch.arange(faces.shape[0], device=faces.device)[:, None, None] * nv
    faces_[mask_faces] = -1
    faces_ = faces_.reshape(-1, 3)
    fmu_ = fi_maps[fi_maps >= 0]
    fmu_ = fmu_.long()
    sel_faces = faces_[fmu_].long()
    sel_faces = sel_faces.reshape(-1).unique(dim=0).long()
    visible_vertices_.scatter_(0, sel_faces, 1)
    visible_vertices = visible_vertices_.reshape(bt, nv).detach()
    dist = torch.cdist(bds_v, verts) ** 2
    dist = (1 - visible_vertices[:, None]) * 1000 + visible_vertices[:, None] * dist
    min_dists = dist.topk(k, largest=False, sorted=False)[0]
    bds_m = bds_m[:, :, None]
    loss = (min_dists * bds_m).mean(-1).sum(-1)
    if reduce:
        return loss.mean()
    else:
        return loss


class Boundaries_Loss(nn.Module):
    def forward(self, verts, bds, pix_to_face, reduce=True, n_samples=1000):
        return bds_loss(verts, bds, pix_to_face, reduce=reduce, n_samples=n_samples)


def edt_loss(mask_rendered, edt, reduce=True):
    bsize = mask_rendered.shape[0]
    mask_con_err = edt * mask_rendered[:, None]
    loss = mask_con_err.reshape(bsize, -1).mean(-1)
    if reduce:
        return loss.mean()
    else:
        return loss


def cycle_loss(points_2d, points_proj, mask=None):
    bt, nv, _ = points_proj.shape
    if mask is not None:
        visible = F.grid_sample(mask[:, None], points_proj[:, None], padding_mode='border', align_corners=True,
                                mode='nearest')
        visible = (visible.squeeze(1).squeeze(1) > 0).float()
        visible = visible.clone().detach()
        loss = visible * (torch.norm(points_proj - points_2d, p=2, dim=-1) ** 2)
        loss = (loss / (visible.sum(-1, keepdim=True) + 1e-4)).sum(-1).mean(-1)
    else:
        loss = (torch.norm(points_proj - points_2d, p=2, dim=-1) ** 2).mean(-1).mean(-1)
    return loss


def vis_loss(vis_pred, pix_to_face, faces):
    bt, nv = vis_pred.shape
    fi_maps = pix_to_face[..., 0].reshape(bt, -1).detach()
    visible_vertices_ = torch.zeros(bt * nv, device=faces.device)
    faces_ = faces + torch.arange(faces.shape[0], device=faces.device)[:, None, None] * nv
    faces_ = faces_.reshape(-1, 3)
    fmu_ = fi_maps[fi_maps >= 0]
    fmu_ = fmu_.long()
    sel_faces = faces_[fmu_].long()
    sel_faces = sel_faces.reshape(-1).unique(dim=0).long()
    visible_vertices_.scatter_(0, sel_faces, 1)
    visible_vertices = visible_vertices_.reshape(bt, nv).detach()
    loss = F.mse_loss(vis_pred, visible_vertices)
    return loss, visible_vertices


def entropy_loss(A):
    entropy = -torch.sum(A * torch.log(A), 1)
    return torch.mean(entropy)


def kp_l1_loss(kp_pred, kp_gt, reduction='mean'):
    criterion = torch.nn.L1Loss(reduction='none')
    vis = (kp_gt[:, :, 2, None] > 0).float()
    loss = criterion(vis * kp_pred, vis * kp_gt[:, :, :2])
    loss = loss.mean(-1).mean(-1)
    if reduction == 'mean':
        return loss.mean()
    else:
        return loss


def locally_rigid_fn(meshes, mesh_template):
    N = len(meshes)
    edges_packed = meshes.edges_packed()
    verts_packed = meshes.verts_packed()
    verts_edges = verts_packed[edges_packed]
    num_edges_per_mesh = meshes.num_edges_per_mesh()
    edge_to_mesh_idx = meshes.edges_packed_to_mesh_idx()
    weights = num_edges_per_mesh.gather(0, edge_to_mesh_idx)
    weights = 1.0 / weights.float()

    v0, v1 = verts_edges.unbind(1)
    mesh_dist = ((v0 - v1).norm(dim=1, p=2))
    edges_packed_t = mesh_template.edges_packed()
    verts_packed_t = mesh_template.verts_packed()
    verts_edges_t = verts_packed_t[edges_packed_t]
    v0_t, v1_t = verts_edges_t.unbind(1)
    mesh_template_dist = ((v0_t - v1_t).norm(dim=1, p=2))
    loss = (mesh_dist - mesh_template_dist) ** 2.0
    loss = loss * weights
    return loss.sum() / N


class Locally_Rigid(nn.Module):
    def forward(self, meshes, mesh_template):
        return locally_rigid_fn(meshes, mesh_template)


class PerceptualTextureLoss(object):
    def __init__(self):
        self.loss_fn_alex = lpips.LPIPS(net='alex', lpips=True, spatial=False).cuda()

    def __call__(self, img_pred, img_gt, mask_pred, mask_gt, reduce=True):
        """
        Input:
          img_pred, img_gt: B x 3 x H x W
        mask_pred, mask_gt: B x H x W
        """
        mask_pred = mask_pred.unsqueeze(1)
        mask_gt = mask_gt.unsqueeze(1)
        pred = img_pred * mask_gt
        target = img_gt * mask_gt
        pred = 2 * pred - 1
        target = 2 * target - 1
        dist = self.loss_fn_alex(pred, target)
        if reduce:
            return dist.mean()
        else:
            return dist


def rotationErrors(R, R_gt, eps=1e-7):
    # Gradient is non-differentiable at acos(1) and acos(-1)
    max_dot_product = 1.0 - eps
    return (0.5 * ((R * R_gt).sum(dim=(-2, -1)) - 1.0)).clamp_(-max_dot_product, max_dot_product).acos()


def rotationErrorsTheta(theta, theta_gt, eps=1e-7):
    R = kornia.angle_axis_to_rotation_matrix(theta[..., :3])
    R_gt = kornia.angle_axis_to_rotation_matrix(theta_gt[..., :3])
    return rotationErrors(R, R_gt, eps)


def translationErrors(t, t_gt):
    return (t - t_gt).norm(dim=-1)


def translationErrorsTheta(theta, theta_gt):
    t = theta[..., 3:]
    t_gt = theta_gt[..., 3:]
    return translationErrors(t, t_gt)


def cam_flip_loss(K, theta, K_flip, theta_flip):
    K_loss = F.mse_loss(K, K_flip)
    rot_loss = rotationErrorsTheta(theta, theta_flip).mean()
    trans_loss = translationErrorsTheta(theta, theta_flip).mean()
    return (K_loss + rot_loss + trans_loss) / 3


def laplacian_cot(meshes, nbrs_indices_pre=None):
    verts_packed = meshes.verts_packed()
    faces_packed = meshes.faces_packed()
    edges_packed = meshes.edges_packed()

    V, F = verts_packed.shape[0], faces_packed.shape[0]

    face_verts = verts_packed[faces_packed]
    v0, v1, v2 = face_verts[:, 0], face_verts[:, 1], face_verts[:, 2]

    A = (v1 - v2).norm(dim=1)
    B = (v0 - v2).norm(dim=1)
    C = (v0 - v1).norm(dim=1)

    s = 0.5 * (A + B + C)

    area = (s * (s - A) * (s - B) * (s - C)).clamp_(min=1e-12).sqrt()
    A2, B2, C2 = A * A, B * B, C * C
    cota = (B2 + C2 - A2) / area
    cotb = (A2 + C2 - B2) / area
    cotc = (A2 + B2 - C2) / area
    cot = torch.stack([cota, cotb, cotc], dim=1)
    cot /= 4.0

    e0, e1 = edges_packed.unbind(1)
    idx01 = torch.stack([e0, e1], dim=1)
    idx10 = torch.stack([e1, e0], dim=1)
    idx_adj = torch.cat([idx01, idx10], dim=0).t()
    L = torch.sparse.FloatTensor(idx_adj, torch.ones_like(idx_adj[0]), (V, V))
    deg = torch.sparse.sum(L, dim=1).to_dense()
    n_nbrs = deg.max()

    n_pnts = deg.shape[0]

    ii = faces_packed[:, [1, 2, 0]]
    jj = faces_packed[:, [2, 0, 1]]
    idx = torch.stack([ii, jj], dim=0).view(2, F * 3)
    cot[cot < 0] = 0
    L = torch.sparse.FloatTensor(idx, cot.view(-1), (V, V))
    L = L + L.t()
    L = L.coalesce()

    c_flat = L._values()
    L_indices = L._indices().t()
    nbrs_val = L_indices[:, 1]
    if nbrs_indices_pre is None:
        nbrs_idx = torch.zeros(n_pnts, dtype=int, device=verts_packed.device)
        nbrs_indices = []
        for ii in L_indices:
            nbrs_indices.append(nbrs_idx[ii[0]].clone())
            nbrs_idx[ii[0]] = nbrs_idx[ii[0]] + 1
        nbrs_indices = torch.tensor(nbrs_indices, device=c_flat.device)
        nbrs_indices = torch.stack([L_indices[:, 0], nbrs_indices])
        nbrs_indices_pre = nbrs_indices
    else:
        nbrs_indices = nbrs_indices_pre

    nbrs_s = torch.sparse.LongTensor(nbrs_indices, nbrs_val, (n_pnts, n_nbrs)).to_dense()
    wgts_s = torch.sparse.FloatTensor(nbrs_indices, c_flat, (n_pnts, n_nbrs)).to_dense()
    norm_w = torch.sparse.sum(L, dim=1).to_dense().view(-1, 1)
    idx = norm_w > 0
    norm_w[idx] = 1.0 / norm_w[idx]

    return L, deg, nbrs_s, wgts_s, nbrs_indices_pre, norm_w


def arap_loss(mesh_def, mesh_base, precomputed):
    if precomputed is not None:
        [weights, wgts, nbrs] = precomputed
    else:
        with torch.no_grad():
            weights, deg, nbrs, wgts, nbrs_indices_pre, norm_w = laplacian_cot(mesh_base, None)
    V = mesh_base.verts_packed()
    V_def = mesh_def.verts_packed()
    mesh_hood = (V[nbrs] - V[:, None]) * (wgts[..., None] != 0).float().detach()
    mesh_hood = mesh_hood.permute(0, 2, 1)
    tru_hood = ((V[nbrs] - V[:, None]) * wgts[..., None]).permute(0, 2, 1)
    rot_hood = (V_def[nbrs] - V_def[:, None]).permute(0, 2, 1)
    tmp = rot_hood @ tru_hood.permute(0, 2, 1)
    U, s, Vt = svd(tmp)
    R = (Vt @ U.permute(0, 2, 1)).permute(0, 2, 1)
    dets = torch.det(R)
    last_Vt = Vt[:, -1, :]
    last_Vt = last_Vt * dets[:, None]
    Vt = torch.cat([Vt[:, :-1], last_Vt[:, None]], dim=1)
    R = (Vt @ U.permute(0, 2, 1)).permute(0, 2, 1)
    R_tru = R @ mesh_hood
    arap_norm = (rot_hood - R_tru).norm(dim=1, p=2).pow(2) * wgts * ((wgts != 0).float().detach())
    arap_norm = arap_norm.sum(-1).mean()
    return arap_norm, [weights, wgts, nbrs]


class Arap(nn.Module):
    def __init__(self):
        super(Arap, self).__init__()
        self.precomputed = None

    def forward(self, mesh_def, mesh_base):
        l_val, self.precomputed = arap_loss(mesh_def, mesh_base, self.precomputed)
        return l_val
