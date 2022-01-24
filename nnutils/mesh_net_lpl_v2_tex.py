"""
Mesh net model.
"""

import warnings

import numpy as np
import torch
import torch.nn as nn
import torchvision
from absl import flags
from pytorch3d.ops import SubdivideMeshes
from pytorch3d.structures import Meshes
from torch.autograd import grad as grad_fn

from . import geometry_utilities as geo
from . import net_blocks as nb

warnings.filterwarnings('ignore')

# -------------- flags -------------#
# ----------------------------------#
flags.DEFINE_integer('nz_feat', 200, 'Encoded feature size')

flags.DEFINE_boolean('texture', True, 'if true uses texture!')
flags.DEFINE_integer('tex_subdivision', 1, 'Texture resolution per face')

flags.DEFINE_integer('subdivide', 3, '# to subdivide icosahedron, 3=642verts, 4=2562 verts')

flags.DEFINE_boolean('use_deconv', False, 'If true uses Deconv')
flags.DEFINE_string('upconv_mode', 'bilinear', 'upsample mode')

flags.DEFINE_boolean('learn_mean', False, 'Learn mean shape')
flags.DEFINE_boolean('learnable_kp', False, 'If true, only the meanshape is symmetric')
flags.DEFINE_integer('basis_k', 5, 'number of basis components')

def conv1x1(in_planes, out_planes, std=0.01):
    """1x1 convolution"""
    cnv = nn.Conv2d(in_planes, out_planes, bias=True, kernel_size=1)

    cnv.weight.data.normal_(0., std)
    if cnv.bias is not None:
        cnv.bias.data.fill_(0.)

    return cnv


def batch_mm(matrix, matrix_batch):
    """
    :param matrix: Sparse or dense matrix, size (m, n).
    :param matrix_batch: Batched dense matrices, size (b, n, k).
    :return: The batched matrix-matrix product, size (m, n) x (b, n, k) = (b, m, k).
    """
    batch_size = matrix_batch.shape[0]
    # Stack the vector batch into columns. (b, n, k) -> (n, b, k) -> (n, b*k)
    vectors = matrix_batch.transpose(0, 1).reshape(matrix.shape[1], -1)

    # A matrix-matrix product is a batched matrix-vector product of the columns.
    # And then reverse the reshaping. (m, n) x (n, b*k) = (m, b*k) -> (m, b, k) -> (b, m, k)
    return matrix.mm(vectors).reshape(matrix.shape[0], batch_size, -1).transpose(1, 0)


def solve_basis(u, V, B, cam, K, L=None, visibility=None, lpl_weight=torch.tensor(-1.), beta_weight=torch.tensor(-1.),
                detach_B=False):
    # Least squares solver for deformation coefficients
    batch_size = u.shape[0]
    K_basis = B.shape[1]
    alpha = u - geo.project_points_by_theta(V, cam, K, scale=False)
    if visibility is not None:
        alpha = visibility.unsqueeze(-1) * alpha
    cam_ = cam.clone()
    K_ = K.clone()
    cam_[:, -2:] *= 0
    K_[:, -2:] *= 0
    cam_ext = cam_[:, None].repeat(1, K_basis, 1).reshape(-1, *cam.shape[1:])
    K_ext = K_[:, None].repeat(1, K_basis, 1).reshape(-1, *K.shape[1:])
    B_resh = B.reshape(-1, B.shape[-1])
    B_resh = B_resh.reshape(B_resh.shape[0], -1, 3)
    if detach_B == True:
        B_resh = B_resh.detach()

    if L is not None:
        B_ = B[0].reshape(K_basis, -1, 3)
        LB = batch_mm(L, B_)
        LTLB = batch_mm(L.t(), LB)
        BTLTLB = B_.reshape(K_basis, -1) @ LTLB.permute(2, 1, 0).reshape(-1, K_basis)
    gamma = geo.project_points_by_theta(B_resh, cam_ext, K_ext, scale=False)
    gamma = gamma.reshape(batch_size, K_basis, -1, 2)
    if visibility is not None:
        gamma = visibility[:, None, :, None] * gamma
    gamma = gamma.reshape(batch_size, K_basis, -1)
    if L is not None:
        eye = torch.eye(K_basis, device=u.device)
        betas_final = torch.inverse(
            torch.exp(beta_weight) * eye[None] + torch.exp(lpl_weight) * BTLTLB[None] + gamma.bmm(
                gamma.permute(0, 2, 1))).bmm(
            gamma.bmm(alpha.reshape(batch_size, -1, 1)))
    else:
        eye = torch.eye(K_basis, device=u.device)
        betas_final = torch.inverse(torch.exp(beta_weight) * eye[None] + gamma.bmm(gamma.permute(0, 2, 1))).bmm(
            gamma.bmm(alpha.reshape(batch_size, -1, 1)))
    betas_final = betas_final.squeeze(-1)
    new_shape = betas_final[..., None] * B
    new_shape = new_shape.sum(1).reshape(new_shape.shape[0], -1, 3)
    shape = V + new_shape
    return betas_final, shape

def ReprojectionError(p2d, p3d, P_13d, K, W):
    b = p2d.size(0)
    pts2d_pro = geo.project_points_by_theta(p3d, P_13d, K)
    if W is not None:
        error = torch.norm((W * (pts2d_pro - p2d)).reshape(b, -1), p=2, dim=-1) ** 2
    else:
        error = torch.norm(pts2d_pro.reshape(b, -1) - p2d.reshape(b, -1), p=2, dim=-1) ** 2
    return error


def ReprojectionError_flat(p2d, p3d, P_13d, K, W):
    b = p2d.size(0)
    pts2d_pro = geo.project_points_by_theta(p3d.view(b, -1, 3), P_13d, K)
    if W is not None:
        error = torch.norm((W.unsqueeze(-1) * (pts2d_pro - p2d.view(b, -1, 2))).reshape(b, -1), p=2, dim=-1) ** 2
    else:
        error = torch.norm(pts2d_pro.reshape(b, -1) - p2d, p=2, dim=-1) ** 2
    return error


def Dy(f, x, y, z, k, w):
    """
    Dy(x) = -(D_YY^2 f(x, y))^-1 D_XY^2 f(x, y)
    Lemma 4.3 from
    Stephen Gould, Richard Hartley, and Dylan Campbell, 2019
    "Deep Declarative Networks: A New Hope", arXiv:1909.04866
    Arguments:
        f: (b, ) Torch tensor, with gradients
            batch of objective functions evaluated at (x, y)
        x: (b, n) Torch tensor, with gradients
            batch of input vectors
        y: (b, m) Torch tensor, with gradients
            batch of minima of f
    Return Values:
        Dy(x): (b, m, n) Torch tensor,
            batch of gradients of y with respect to x
    """
    grad_outputs = torch.ones_like(f)
    DYf = grad_fn(f, y, grad_outputs=grad_outputs, create_graph=True)[0]  # bxm
    DYYf = torch.empty_like(DYf.unsqueeze(-1).expand(-1, -1, y.size(-1)))  # bxmxm
    DZYf = torch.empty_like(DYf.unsqueeze(-1).expand(-1, -1, z.size(-1)))  # bxmxn
    DKYf = torch.empty_like(DYf.unsqueeze(-1).expand(-1, -1, k.size(-1)))  # bxmxn
    if w is not None:
        DWYf = torch.empty_like(DYf.unsqueeze(-1).expand(-1, -1, w.size(-1)))  # bxmxn
    grad_outputs = torch.ones_like(DYf[:, 0])
    for i in range(DYf.size(-1)):  # [0,m)
        DYf_i = DYf[:, i]  # b
        DYYf[:, i:(i + 1), :] = grad_fn(DYf_i, y, grad_outputs=grad_outputs, create_graph=True)[
            0].unsqueeze(1)  # bx1xm
        if w is not None:
            out = grad_fn(DYf_i, [z, k, w], grad_outputs=grad_outputs, create_graph=True)
        else:
            out = grad_fn(DYf_i, [z, k], grad_outputs=grad_outputs, create_graph=True)
        DZYf[:, i:(i + 1), :] = out[0].unsqueeze(1)  # bx1xn
        DKYf[:, i:(i + 1), :] = out[1].unsqueeze(1)  # bx1xn
        if w is not None:
            DWYf[:, i:(i + 1), :] = out[2].unsqueeze(1)  # bx1xn

    DYYf = DYYf.detach()
    DZYf = DZYf.detach()
    if w is not None:
        DWYf = DWYf.detach()
    DYYf = 0.5 * (DYYf + DYYf.transpose(1, 2))  # In case of floating point errors

    try:
        U = torch.cholesky(DYYf, upper=True)
        Dy_at_z = torch.cholesky_solve(-1.0 * DZYf, U, upper=True)  # bxmxn
        Dy_at_k = torch.cholesky_solve(-1.0 * DKYf, U, upper=True)  # bxmxn
        if w is not None:
            Dy_at_w = torch.cholesky_solve(-1.0 * DWYf, U, upper=True)  # bxmxn

    except:
        Dy_at_z = torch.empty_like(DZYf)
        Dy_at_k = torch.empty_like(DKYf)
        if w is not None:
            Dy_at_w = torch.empty_like(DWYf)

        for i in range(DYYf.size(0)):  # For some reason, doing this in a loop doesn't crash
            try:
                U = torch.cholesky(DYYf[i, ...], upper=True)
                Dy_at_z[i, ...] = torch.cholesky_solve(-1.0 * DZYf[i, ...], U, upper=True)
                Dy_at_k[i, ...] = torch.cholesky_solve(-1.0 * DKYf[i, ...], U, upper=True)
                if w is not None:
                    Dy_at_w[i, ...] = torch.cholesky_solve(-1.0 * DWYf[i, ...], U, upper=True)

            except:
                Dy_at_z[i, ...], _ = torch.solve(-1.0 * DZYf[i, ...], DYYf[i, ...])
                Dy_at_k[i, ...], _ = torch.solve(-1.0 * DKYf[i, ...], DYYf[i, ...])
                if w is not None:
                    Dy_at_w[i, ...], _ = torch.solve(-1.0 * DWYf[i, ...], DYYf[i, ...])
    # Set NaNs to 0:
    if torch.isnan(Dy_at_z).any():
        Dy_at_z[torch.isnan(Dy_at_z)] = 0.0  # In-place
        Dy_at_k[torch.isnan(Dy_at_k)] = 0.0  # In-place
        if w is not None:
            Dy_at_w[torch.isnan(Dy_at_w)] = 0.0  # In-place

    # Clip gradient norms:
    max_norm = 100.0
    Dy_norm2 = Dy_at_z.norm(dim=-2, keepdim=True)  # bxmxn
    Dy_norm3 = Dy_at_k.norm(dim=-2, keepdim=True)  # bxmxn
    if w is not None:
        Dy_norm4 = Dy_at_w.norm(dim=-2, keepdim=True)  # bxmxn

    if (Dy_norm2 > max_norm).any():
        clip_coef = (max_norm / (Dy_norm2 + 1e-6)).clamp_max_(1.0)
        Dy_at_z = clip_coef * Dy_at_z
    if (Dy_norm3 > max_norm).any():
        clip_coef = (max_norm / (Dy_norm3 + 1e-6)).clamp_max_(1.0)
        Dy_at_k = clip_coef * Dy_at_k
    if w is not None:
        if (Dy_norm4 > max_norm).any():
            clip_coef = (max_norm / (Dy_norm4 + 1e-6)).clamp_max_(1.0)
            Dy_at_w = clip_coef * Dy_at_w
    else:
        Dy_at_w = None
    return None, Dy_at_z, Dy_at_k, Dy_at_w


class BPnP_n(torch.autograd.Function):
    '''
    BPnP adaptation from "Deep Declarative Networks: A New Hope", arXiv:1909.04866
    '''
    @staticmethod
    def forward(ctx, pts2d_in, pts3d_in, K, W=None, ini_pose=None, ite=5):
        pts2d_in, pts3d_in = pts2d_in.detach(), pts3d_in.detach()
        K = K.detach()
        if W is not None:
            W = W.detach()
        bs = pts2d_in.size(0)
        device = pts2d_in.device
        t_n = torch.zeros(bs, 2, device=device, dtype=pts2d_in.dtype)
        M_n = torch.rand(bs, 3, device=device, dtype=pts2d_in.dtype)
        if ini_pose is not None:
            t_i = ini_pose[:, 3:]
            M_i = ini_pose[:, :3]
            t_n = t_i
            M_n = M_i
        P_13d = torch.cat([M_n, t_n], dim=-1).clone().detach().requires_grad_(True)

        optimizer = torch.optim.LBFGS([P_13d], lr=1., max_iter=100,
                                      tolerance_grad=1e-07, tolerance_change=1e-09,
                                      line_search_fn='strong_wolfe')

        def closure():
            optimizer.zero_grad()
            loss = ReprojectionError(pts2d_in, pts3d_in, P_13d, K, W).mean()
            loss.backward()
            return loss

        optimizer.step(closure)
        ctx.save_for_backward(pts2d_in.detach(), pts3d_in.detach(), P_13d.detach(), K.detach(), W, ini_pose)
        return P_13d

    @staticmethod
    def backward(ctx, grad_output):
        pts2d_in, pts3d_in, P_13d_in, K, W, ini_pose = ctx.saved_tensors
        bs = pts2d_in.size(0)
        with torch.enable_grad():
            pts2d = pts2d_in.reshape(bs, -1).detach().requires_grad_(True)
            pts3d = pts3d_in.reshape(bs, -1).detach().requires_grad_(True)
            P_13d = P_13d_in.detach().requires_grad_(True)
            K_d = K.detach().requires_grad_(True)
            if W is not None:
                W_d = W.reshape(bs, -1).detach().requires_grad_(True)
            else:
                W_d = None
            fn_at_theta = ReprojectionError_flat(pts2d, pts3d, P_13d, K_d, W_d)  # b
            Dtheta, Dz, Dk, Dwd = Dy(fn_at_theta, pts3d, P_13d, pts2d, K_d, W_d)  # bx6xmn
            grad_3d = None
            grad_input = torch.einsum("ab,abc->ac", (grad_output, Dz))  # bx6 * bx6xmn-> bxmn
            grad_2d = grad_input.reshape(*pts2d_in.shape)
            grad_input = torch.einsum("ab,abc->ac", (grad_output, Dk))  # bx6 * bx6xmn-> bxmn
            grad_K = grad_input.reshape(*K.shape)
            if W_d is not None:
                grad_input = torch.einsum("ab,abc->ac", (grad_output, Dwd))  # bx6 * bx6xmn-> bxmn
                grad_W = grad_input.reshape(*W.shape)
            else:
                grad_W = None
        return grad_2d, grad_3d, grad_K, grad_W, None


class ResNetConv(nn.Module):
    def __init__(self, n_blocks=4):
        super(ResNetConv, self).__init__()
        self.resnet = torchvision.models.resnet18(pretrained=True)
        self.n_blocks = n_blocks

    def forward(self, x):
        n_blocks = self.n_blocks
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        if n_blocks >= 1:
            x = self.resnet.layer1(x)
        if n_blocks >= 2:
            x = self.resnet.layer2(x)
        if n_blocks >= 3:
            x = self.resnet.layer3(x)
        if n_blocks >= 4:
            x = self.resnet.layer4(x)
        return x


class Encoder(nn.Module):
    """
    Current:
    Resnet with 4 blocks (x32 spatial dim reduction)
    Another conv with stride 2 (x64)
    This is sent to 2 fc layers with final output nz_feat.
    """

    def __init__(self, input_shape, nz_feat=100, batch_norm=True):
        super(Encoder, self).__init__()
        self.resnet_conv = ResNetConv(n_blocks=4)
        self.enc_conv1 = nb.conv2d(batch_norm, 512, 256, stride=2, kernel_size=4)
        nc_input = 256 * (input_shape[0] // 64) * (input_shape[1] // 64)
        self.enc_fc = nb.fc_stack(nc_input, nz_feat, 2)

        nb.net_init(self.enc_conv1)

    def forward(self, img):
        resnet_feat = self.resnet_conv.forward(img)
        out_enc_conv1 = self.enc_conv1(resnet_feat)
        out_enc_conv2 = out_enc_conv1.view(img.size(0), -1)
        feat = self.enc_fc.forward(out_enc_conv2)

        return feat, out_enc_conv1



class TexturePredictor_2d(nn.Module):
    """
    Outputs mesh texture
    """

    def __init__(self, num_verts):
        super(TexturePredictor_2d, self).__init__()
        self.res_color_net = conv1x1(256 * 4 * 4, num_verts * 3)
        self.res_color_net.weight.data *= 0.01
        self.res_color_net.bias.data.fill_(0.0)

    def forward(self, feat):
        feat = feat.reshape(feat.shape[0], -1)
        vert_colors = self.res_color_net(feat[:, :, None, None]).squeeze(-1).squeeze(-1)
        return vert_colors.reshape(feat.shape[0], -1, 3)


class BasisGenerator(nn.Module):
    def __init__(self, num_verts, K):
        super(BasisGenerator, self).__init__()
        self.w = nn.Parameter(torch.rand(K, num_verts * 3) * 0.01)

    def forward(self):
        out = self.w
        return out


# ------------ Mesh Net ------------#
# ----------------------------------#
class MeshNet(nn.Module):
    def __init__(self, input_shape, opts, nz_feat=100, sfm_mean_shape=None, shapenet_mesh=None,
                 kp_dict=None):
        # Input shape is H x W of the image.
        super(MeshNet, self).__init__()
        self.opts = opts
        self.pred_texture = opts.texture

        # Mean shape.
        if shapenet_mesh is not None:
            sfm_mean_shape = shapenet_mesh

        verts, faces = sfm_mean_shape[0], sfm_mean_shape[1]
        verts = verts.astype(np.float32)
        num_verts = verts.shape[0]
        if opts.learn_mean:
            self.mean_v = nn.Parameter(torch.tensor(verts))
        else:
            self.mean_v = torch.tensor(verts).cuda()
        self.num_output = num_verts

        self.faces = torch.tensor(faces).long().cuda()
        self.num_verts = num_verts

        self.encoder = Encoder(input_shape, nz_feat=nz_feat)
        self.cam_solver = BPnP_n.apply
        mesh_template = Meshes(verts=[self.get_mean_shape().cuda()], faces=[self.faces])
        self.sdivide = SubdivideMeshes(mesh_template)

        # prep subdivision class for multiple subdivisions
        mesh_subdivision = []
        template_tmp = mesh_template
        for i in range(opts.tex_subdivision):
            sdivide_new = SubdivideMeshes(template_tmp)
            template_tmp = sdivide_new(template_tmp)
            mesh_subdivision.append(sdivide_new)
        self.mesh_subdivision = mesh_subdivision
        res = self.get_subdivision(mesh_template)
        verts_full = res.verts_padded().shape[1]

        if kp_dict:
            # if kp_dict is provided
            vert2kp_init = torch.zeros(len(kp_dict), verts_full).float()
            ms = self.get_mean_shape()
            res_v = res.verts_list()[0]
            for i_kp, k in enumerate(kp_dict):
                idx = kp_dict[k]
                ref_v = ms[idx]
                diffs = ((ref_v[None] - res_v) ** 2).sum(-1)
                diffs_amin = diffs.argmin()
                print(k, diffs_amin.item())
                vert2kp_init[i_kp, diffs_amin] = 1
                self.vert2kp = vert2kp_init.cuda()

        self.texture_predictor_p = TexturePredictor_2d(verts_full)
        self.basis = BasisGenerator(verts_full, opts.basis_k)
        self.beta_weight = nn.Parameter(torch.tensor([0.]))
        self.lpl_weight = nn.Parameter(torch.tensor([0.]))

    def forward(self, img):
        img_feat, res_feats = self.encoder.forward(img)
        return img_feat, res_feats

    def get_mean_shape(self):
        return self.mean_v

    def get_subdivision(self, mesh):
        res = mesh
        for ms in self.mesh_subdivision:
            res = ms(res)
        return res
