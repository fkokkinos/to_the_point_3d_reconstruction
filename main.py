"""
Script for the bird shape, pose and texture experiment.

The model takes imgs, outputs the deformation to the mesh & camera parameters
Loss consists of:
- keypoint reprojection loss
- mask reprojection loss
- smoothness/laplacian priors on triangles
- texture reprojection losses

example usage : python -m cmr.experiments.shape --name=bird_shape --plot_scalars --save_epoch_freq=1 --batch_size=8 --display_visuals --display_freq=2000
"""

import os.path as osp
import pickle as pkl
from collections import OrderedDict

import gdist
import kornia as K
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pytorch3d.io as p3dio
import scipy.io as sio
import torch
import torchvision
import trimesh
from absl import app
from absl import flags
from kornia.geometry.conversions import angle_axis_to_quaternion, quaternion_to_angle_axis
from pytorch3d.loss import mesh_laplacian_smoothing, mesh_normal_consistency
from pytorch3d.renderer.mesh import TexturesUV, TexturesVertex
from pytorch3d.structures import Meshes
from pytorch3d.vis.texture_vis import texturesuv_image_matplotlib
from skimage.transform import resize

from data import cub as cub_data
from data import p3d as p3d_data
from nnutils import geom_utils
from nnutils import geometry_utilities as geo
from nnutils import loss_utils
from nnutils import mesh_net_lpl_v2_tex as mesh_net
from nnutils import train_utils
from nnutils.mesh_net_lpl_v2_tex import solve_basis
from nnutils.nmr import NeuralRenderer
from utils import bird_vis
from utils import image as image_utils

torch.manual_seed(0)
flags.DEFINE_string('dataset', 'cub', 'cub or pascal or p3d')
flags.DEFINE_integer('def_steps', 1, 'number of deformation steps')
flags.DEFINE_string('mesh_dir', 'meshes/bird.obj', 'tmp dir to extract dataset')
flags.DEFINE_string('kp_dict', 'meshes/bird_kp_dictionary.pkl', 'tmp dir to extract dataset')
flags.DEFINE_string('basis_pretrained', None, 'filepath to pretrained basis')
# Weights:
flags.DEFINE_float('kp_loss_wt', 30., 'keypoint loss weight')
flags.DEFINE_float('mask_loss_wt', 1., 'mask loss weight')
flags.DEFINE_float('triangle_reg_wt', 0., 'weights to triangle smoothness prior')
flags.DEFINE_float('vert2kp_loss_wt', .16, 'reg to vertex assignment')
flags.DEFINE_float('tex_loss_wt', .5, 'weights to tex loss')
flags.DEFINE_float('tex_dt_loss_wt', .5, 'weights to tex dt loss')
flags.DEFINE_boolean('use_gtpose', True, 'if true uses gt pose for projection, but camera still gets trained.')
flags.DEFINE_float('rigid_wt', 0.5, 'mask loss weight')
flags.DEFINE_float('bdt_reg_wt', 0.01, 'reg to deformation')
flags.DEFINE_float('vis_loss_wt', 0., 'visibility loss weight')
flags.DEFINE_float('def_loss_wt', 0., 'deformation regularization weight')
flags.DEFINE_float('betas_loss_wt', 0., 'betas regularization weight')
flags.DEFINE_float('arap_reg_wt', 0., 'ARAP regularization weight')
flags.DEFINE_float('tri_basis_wt', 0., 'laplacian regularization weight')
flags.DEFINE_float('sil_loss_wt', 0., 'silhouette loss weight')
flags.DEFINE_float('normal_loss_wt', 0., 'normal loss weight')
flags.DEFINE_float('basis_cycle_loss_wt', 0., 'basis cycle loss weight')
flags.DEFINE_float('arap_basis_loss_wt', 0., 'ARAP basis regularization weight')

flags.DEFINE_float('equiv_loss_wt', 1., 'equivariance loss weight')
flags.DEFINE_float('equiv_cam_loss_wt', 0.1, 'equivariance loss weight for camera estimation')
flags.DEFINE_float('equiv_degrees', 40., 'equivariance rotation maximum degrees')
flags.DEFINE_float('equiv_t', 0.15, 'equivariance translation ratio')
flags.DEFINE_float('equiv_s_min', 0.7, 'equivariance scale minimum multiplier')
flags.DEFINE_float('equiv_s_max', 1.2, 'equivariance scale maximum multiplier')

flags.DEFINE_boolean('weighted_camera', False, 'whether to print scalars')
flags.DEFINE_boolean('flip_train', False, 'enforce flip equivariance')
flags.DEFINE_boolean('dev', False, 'dev flag')
flags.DEFINE_boolean('color_gdist', False, 'use gdist for coloring points')

opts = flags.FLAGS

curr_path = osp.dirname(osp.abspath(__file__))
cache_path = osp.join(curr_path, '..', 'misc', 'cachedir')


def mirror_camera(K, pose):
    # mirror camera around axis
    angles = pose[:, :3]
    quat = angle_axis_to_quaternion(angles)
    quat_offset = torch.FloatTensor([1, 1, -1, -1]).view(1, -1).to(pose.device)
    new_quat = quat_offset * quat
    angles_new = quaternion_to_angle_axis(new_quat)

    t_mirror = torch.tensor([-1., 1.], device=pose.device)
    t_mirror = torch.stack([t_mirror] * pose.shape[0])
    t_new = pose[:, 3:] * t_mirror
    K_new = K * torch.tensor([1., 1., -1., 1], device=pose.device)
    pose_new = torch.cat([angles_new, t_new], dim=-1)
    return K_new, pose_new


def mirror_sample(img, mask_pred, mask, K, pose):
    # flip sample horizontally
    img_flip = torch.flip(img, dims=(3,))
    mask_pred_flip = torch.flip(mask_pred, dims=(2,))
    mask_flip = torch.flip(mask, dims=(2,))
    K_new, pose_new = mirror_camera(K, pose)
    return img_flip, mask_pred_flip, mask_flip, K_new, pose_new


class ShapeTrainer(train_utils.Trainer):
    def define_model(self):
        opts = self.opts

        # ----------
        # Options
        # ----------
        anno_sfm_path = osp.join(opts.cub_cache_dir, 'sfm', 'anno_train.mat')
        anno_sfm = sio.loadmat(anno_sfm_path, struct_as_record=False, squeeze_me=True)
        sfm_mean_shape = (np.transpose(anno_sfm['S']), anno_sfm['conv_tri'] - 1)

        kp_dict = pkl.load(open(opts.kp_dict, 'rb'))
        mesh_loaded = p3dio.load_obj(opts.mesh_dir)
        v, f = mesh_loaded[0].numpy(), mesh_loaded[1].verts_idx.numpy()
        # scale = 2. / torch.max(torch.nn.functional.pdist(torch.from_numpy(v))).numpy()
        # v = v * scale
        # v = v - v.mean(0)
        # v = torch.from_numpy(v)[None]
        shapenet_mesh = [v, f]

        img_size = (opts.img_size, opts.img_size)
        self.model = mesh_net.MeshNet(
            img_size, opts, nz_feat=opts.nz_feat, sfm_mean_shape=sfm_mean_shape,
            shapenet_mesh=shapenet_mesh, kp_dict=kp_dict)

        if opts.num_pretrain_epochs > 0:
            self.load_network(self.model, 'pred', opts.num_pretrain_epochs)

        if opts.basis_pretrained is not None:
            basis_tmp = torch.load(opts.basis_pretrained)
            self.model.basis.w.data = basis_tmp
        self.model = self.model.cuda(device=opts.gpu_id)

        # For renderering.
        faces = self.model.faces.view(1, -1, 3)
        self.faces_orig = faces.repeat(opts.batch_size, 1, 1)
        self.renderer = NeuralRenderer(opts.img_size)
        self.renderer_predcam = NeuralRenderer(opts.img_size)
        if opts.texture:
            self.tex_renderer = NeuralRenderer(opts.img_size)
            self.tex_renderer.ambient_light_only()

        # For visualization
        mesh_template = Meshes(verts=[self.model.get_mean_shape()], faces=[self.faces_orig[0]])
        mesh_up = self.model.get_subdivision(mesh_template)
        self.faces = mesh_up.faces_packed()
        num_verts_up = mesh_up.verts_list()[0].shape[0]
        self.vis_rend = bird_vis.VisRenderer(opts.img_size, v.shape[0], self.faces[:1].data.cpu().numpy())
        self.vis_rend_up = bird_vis.VisRenderer(opts.img_size, num_verts_up, self.faces[:1].data.cpu().numpy())
        self.equiv_augm = K.augmentation.RandomAffine(p=1, degrees=opts.equiv_degrees,
                                                      translate=(opts.equiv_t, opts.equiv_t),
                                                      scale=(opts.equiv_s_min, opts.equiv_s_max), return_transform=True,
                                                      padding_mode=1)
        import pylab
        cm = pylab.get_cmap('gist_rainbow')
        if opts.color_gdist:
            distance_gd = gdist.local_gdist_matrix(
                mesh_up.verts_list()[0].data.cpu().numpy().astype(np.float64),
                mesh_up.faces_list()[0].cpu().numpy().astype(np.int32)
            )

            c_dist = distance_gd.toarray()[0]
            c_dist = c_dist / c_dist.max()
            c_nodes = np.array([cm(c)[:3] for c in c_dist])
            self.c_nodes = c_nodes
        else:
            mesh = trimesh.Trimesh(mesh_up.verts_list()[0].data.cpu().numpy(),
                                   mesh_up.faces_list()[0].data.cpu().numpy(),
                                   process=False)
            graph = mesh.vertex_adjacency_graph
            edges = nx.bfs_edges(graph, 0)
            nodes = [0] + [v for u, v in edges]
            colors = 255 * np.array([cm(float(i) / num_verts_up)[:3] for i in range(num_verts_up)])
            c_nodes = np.zeros_like(colors)
            for i, n in enumerate(nodes):
                c_nodes[n] = colors[i]
            self.c_nodes = c_nodes
        return

    def init_dataset(self):
        opts = self.opts

        if opts.dataset == 'cub':
            self.data_module = cub_data
        elif opts.dataset == 'p3d':
            self.data_module = p3d_data
        else:
            print('Unknown dataset %d!' % opts.dataset)
        if opts.dev:
            self.dataloader = self.data_module.data_loader(opts, shuffle=False)
        else:
            self.dataloader = self.data_module.data_loader(opts, shuffle=True)

        self.resnet_transform = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])

    def define_criterion(self):
        self.projection_loss = loss_utils.kp_l1_loss
        self.mask_loss_fn = torch.nn.L1Loss()
        self.boundaries_fn = loss_utils.bds_loss
        self.entropy_loss = loss_utils.entropy_loss
        self.vis_loss_fn = loss_utils.vis_loss
        self.equiv_cam_loss_fn = loss_utils.cam_flip_loss
        self.arap_loss_mesh_fn = loss_utils.Arap()
        self.arap_loss_basis_fn = loss_utils.Arap()
        self.cycle_loss_fn = loss_utils.cycle_loss

        self.locally_rigid_fn = loss_utils.Locally_Rigid()
        if self.opts.texture:
            self.texture_loss = loss_utils.PerceptualTextureLoss()
            self.texture_dt_loss_fn = loss_utils.texture_dt_loss_v

    def set_input(self, batch):
        # Prepare batch data and horizontally flipped version
        opts = self.opts
        input_img_tensor = batch['img'].type(torch.FloatTensor).clone()
        self.input_imgs = self.resnet_transform(input_img_tensor).cuda()
        self.imgs = batch['img'].type(torch.FloatTensor).cuda()
        self.masks = batch['mask'].type(torch.FloatTensor).cuda()
        self.kps = batch['kp'].type(torch.FloatTensor).cuda()
        self.cams = batch['sfm_pose'].type(torch.FloatTensor).cuda()
        mask_dts = np.stack([image_utils.compute_dt_barrier(m) for m in batch['mask']])
        dt_tensor = torch.tensor(mask_dts).float().cuda()
        self.dts_barrier = dt_tensor.unsqueeze(1)
        self.boundaries = image_utils.compute_inner_points(self.masks.cpu().numpy(), 500)
        self.boundaries = torch.tensor(self.boundaries).float().cuda()
        mask_edt = np.stack([image_utils.compute_dt(m) for m in batch['mask']])
        self.edts_barrier = torch.tensor(mask_edt).float().unsqueeze(1).cuda()

        if opts.flip_train:
            input_img_tensor_flip = batch['img_flip'].type(torch.FloatTensor).clone()
            self.input_imgs_flip = self.resnet_transform(input_img_tensor_flip).cuda()
            self.imgs_flip = batch['img_flip'].type(torch.FloatTensor).cuda()
            self.masks_flip = batch['mask_flip'].type(torch.FloatTensor).cuda()
            self.kps_flip = batch['kp_flip'].type(torch.FloatTensor).cuda()
            self.cams_flip = batch['sfm_pose_flip'].type(torch.FloatTensor).cuda()
            mask_dts = np.stack([image_utils.compute_dt_barrier(m) for m in batch['mask_flip']])
            dt_tensor = torch.tensor(mask_dts).float().cuda()
            self.dts_barrier_flip = dt_tensor.unsqueeze(1)

            self.boundaries_flip = image_utils.compute_inner_points(self.masks_flip.cpu().numpy(), 500)
            self.boundaries_flip = torch.tensor(self.boundaries_flip).float().cuda()
            mask_edt = np.stack([image_utils.compute_dt(m) for m in batch['mask_flip']])
            self.edts_barrier_flip = torch.tensor(mask_edt).float().unsqueeze(1).cuda()
            self.input_imgs = torch.cat([self.input_imgs, self.input_imgs_flip], dim=0)
            self.imgs = torch.cat([self.imgs, self.imgs_flip], dim=0)
            self.masks = torch.cat([self.masks, self.masks_flip], dim=0)
            self.kps = torch.cat([self.kps, self.kps_flip], dim=0)
            self.dts_barrier = torch.cat([self.dts_barrier, self.dts_barrier_flip], dim=0)
            self.boundaries = torch.cat([self.boundaries, self.boundaries_flip], dim=0)
            self.edts_barrier = torch.cat([self.edts_barrier, self.edts_barrier_flip], dim=0)

    def forward(self, drop_deform=False, init_basis=False):
        opts = self.opts
        self.drop_deform = drop_deform
        img_feat, self.res_feats = self.model(self.input_imgs)
        batch_size = img_feat.shape[0]
        faces = self.faces_orig[:1].repeat(batch_size, 1, 1)

        self.mean_shape = self.model.get_mean_shape()
        if opts.learnable_kp:
            self.vert2kp = torch.nn.functional.softmax(self.model.vert2kp, dim=1)
        else:
            self.vert2kp = self.model.vert2kp
        self.mean_v = self.mean_shape[None].repeat(batch_size, 1, 1)

        mesh_3d = Meshes(verts=self.mean_v, faces=faces)
        mesh_3d_up = self.model.get_subdivision(mesh_3d)
        self.mean_v = mesh_3d_up.verts_padded()
        self.faces = mesh_3d_up.faces_padded()
        self.tex_flow = self.model.texture_predictor_p(self.res_feats)
        self.visibility = torch.sigmoid(self.tex_flow[..., -1])  # visibility in [0,1]
        self.tex_flow = torch.tanh(self.tex_flow[..., :2])
        self.tex_flow_uv = (self.tex_flow + 1) / 2
        self.tex_flow_uv[..., 1] = 1 - self.tex_flow_uv[..., 1]
        self.textures = TexturesUV(maps=self.imgs.permute(0, 2, 3, 1), verts_uvs=self.tex_flow_uv,
                                   faces_uvs=self.faces)

        s_2d = self.tex_flow.reshape(batch_size, -1).std(1)[:, None]
        mu_2d = self.tex_flow.mean(1)
        self.K = torch.cat([s_2d, s_2d, mu_2d], dim=-1)
        shape_tmp = self.mean_v
        # Compute pose and deformation
        def_steps = 1 if drop_deform else opts.def_steps
        for def_step in range(def_steps):
            ini_pose = None
            if def_step > 0:
                ini_pose = self.cam_pred
            if opts.weighted_camera:
                self.bpnp_weight = self.visibility[..., None]
                self.cam_pred = self.model.cam_solver(self.tex_flow, shape_tmp, self.K, self.bpnp_weight, ini_pose)
            else:
                self.cam_pred = self.model.cam_solver(self.tex_flow, shape_tmp, self.K, None, ini_pose)

            # Deform mean shape:
            if drop_deform:
                self.pred_v = shape_tmp
                self.pred_v_detach = self.pred_v
            else:
                all_basis = self.model.basis()[None].repeat(self.faces.shape[0], 1, 1)
                if init_basis:
                    betas, shape = solve_basis(self.tex_flow.detach(), self.mean_v, all_basis, self.cam_pred.detach(),
                                               self.K.detach(), L=None, visibility=None,
                                               lpl_weight=self.model.lpl_weight,
                                               beta_weight=self.model.beta_weight)
                else:
                    betas, shape = solve_basis(self.tex_flow, self.mean_v, all_basis, self.cam_pred,
                                               self.K, L=None, visibility=None, lpl_weight=self.model.lpl_weight,
                                               beta_weight=self.model.beta_weight)
                self.pred_v = shape
                new_shape = betas[..., None].detach() * all_basis
                new_shape = new_shape.sum(1).reshape(new_shape.shape[0], -1, 3)
                self.pred_v_detach = self.mean_v + new_shape
            shape_tmp = self.pred_v

        proj_cam = self.cam_pred

        # Project keypoints
        trans_verts = geo.project_points_by_theta(self.pred_v, proj_cam, self.K)
        trans_verts_mean = geo.project_points_by_theta(self.mean_v, proj_cam, self.K)
        if opts.learnable_kp:
            vert2kp = torch.nn.functional.softmax(self.model.vert2kp, dim=1)
        else:
            vert2kp = self.model.vert2kp
        self.kp_verts = torch.matmul(vert2kp, trans_verts)
        self.kp_verts_pred_v = torch.matmul(vert2kp, trans_verts)
        self.kp_verts_mean_v = torch.matmul(vert2kp, trans_verts_mean)
        self.kp_verts_regr_points = torch.matmul(vert2kp, self.tex_flow)
        self.kp_pred_transformed = self.kp_verts_pred_v[..., :2]
        self.kp_meanv_transformed = self.kp_verts_mean_v[..., :2]

        self.mask_pred, self.pix_to_face, depth = self.renderer(self.pred_v, self.faces, proj_cam, self.K)
        self.texture_pred, _, self.pix_to_face_up, self.depth_up = self.tex_renderer(self.pred_v, self.faces,
                                                                                     proj_cam,
                                                                                     self.K,
                                                                                     textures=self.textures)
        self.texture_pred = self.texture_pred[:, 0:3, :, :]
        self.imgs_flip, self.masks_flip, self.mask_pred_flip, self.K_flip, self.proj_cam_flip = mirror_sample(
            self.imgs, self.masks, self.mask_pred, self.K, proj_cam)
        self.texture_pred_flip, self.mask_rend_flip, _, _ = self.tex_renderer(self.pred_v.detach(), self.faces,
                                                                              self.proj_cam_flip.detach(),
                                                                              self.K_flip.detach(),
                                                                              textures=self.textures)
        self.texture_pred_flip = self.texture_pred_flip[:, 0:3, :, :]

        # Compute all losses
        self.kp_loss = self.projection_loss(self.kp_pred_transformed, self.kps)
        self.mask_loss = self.mask_loss_fn(self.mask_pred, self.masks)
        if opts.texture:
            self.tex_loss = 0.5 * self.texture_loss(self.texture_pred, self.imgs, self.mask_pred, self.masks)
            self.tex_loss += 0.5 * self.texture_loss(self.texture_pred_flip, self.imgs_flip, self.mask_pred_flip,
                                                     self.masks_flip)
            self.tex_loss_dt = self.texture_dt_loss_fn(self.tex_flow, self.edts_barrier)
            self.vis_loss, self.visible_verts = self.vis_loss_fn(self.visibility, self.pix_to_face.detach(), self.faces)

        # Equivariance Loss
        self.equiv_loss = torch.tensor(0.)
        if opts.equiv_loss_wt > 0:
            self.imgs_augm, self.augm_mt = self.equiv_augm(self.imgs)
            self.input_imgs_augm = self.resnet_transform(self.imgs_augm)
            _, self.res_feats_augm = self.model(self.input_imgs_augm)
            self.tex_flow_augm = self.model.texture_predictor_p(self.res_feats_augm)
            self.tex_flow_augm = torch.tanh(self.tex_flow_augm[..., :2])
            M_3x3 = K.convert_affinematrix_to_homography(self.augm_mt[:, :-1])
            self.dst_norm_trans_src_norm = K.normalize_homography(M_3x3, self.imgs.shape[-2:], self.imgs.shape[-2:])
            self.src_norm_trans_dst_norm = torch.inverse(self.dst_norm_trans_src_norm)[:, None]
            self.dst_norm_trans_src_norm = self.dst_norm_trans_src_norm[:, None]
            self.tex_flow_trans = self.dst_norm_trans_src_norm[:, :, :2, :2].matmul(
                self.tex_flow.unsqueeze(-1)).squeeze(-1) + self.dst_norm_trans_src_norm[:, :, :2, -1]
            self.tex_flow_trans = self.tex_flow_trans.clone().detach()
            self.equiv_loss = torch.nn.functional.mse_loss(self.tex_flow_augm, self.tex_flow_trans)
            if opts.flip_train:
                K_, proj_cam_ = self.K[:batch_size // 2], proj_cam[:batch_size // 2]
                K_mirror, proj_cam_mirror = mirror_camera(K_, proj_cam_)
                K_flip, proj_cam_flip = self.K[batch_size // 2:], proj_cam[batch_size // 2:]
                self.mask_pred_equiv, _, _ = self.renderer(self.pred_v[:batch_size // 2].detach(),
                                                           self.faces[:batch_size // 2].detach(),
                                                           proj_cam_mirror, K_mirror)
                self.equiv_cam_loss = self.equiv_cam_loss_fn(K_mirror, proj_cam_mirror, K_flip, proj_cam_flip)
                self.equiv_loss += opts.equiv_cam_loss_wt * self.equiv_cam_loss

        trans_verts = geo.project_points_by_theta(self.pred_v.detach(), proj_cam.detach(), self.K.detach())
        mesh_3d_proj = Meshes(verts=trans_verts, faces=self.faces)
        mesh_flow = Meshes(verts=self.tex_flow, faces=self.faces)
        self.rigid_loss = self.cycle_loss_fn(self.tex_flow, trans_verts.detach(), self.masks)
        self.rigid_loss += self.locally_rigid_fn(mesh_3d_proj, mesh_flow)
        if not drop_deform:
            self.basis_cycle_loss = self.cycle_loss_fn(self.tex_flow.detach(), trans_verts)
            self.deform_loss = (self.pred_v - self.mean_v).pow(2).mean()
        else:
            self.basis_cycle_loss = torch.tensor([0.])
            self.deform_loss = torch.tensor([0.])
        self.betas_loss = betas.norm(dim=-1, p=2).mean() if not drop_deform else torch.tensor([0.])

        if opts.sil_loss_wt:
            trans_verts = geo.project_points_by_theta(self.pred_v, proj_cam, self.K)
            self.bdt_loss = self.boundaries_fn(trans_verts, self.boundaries, self.faces, self.pix_to_face)
            self.sil_loss = self.bdt_loss
        else:
            self.sil_loss = torch.tensor([0.])

        B = self.model.basis()
        K_basis = B.shape[0]
        B_ = B.reshape(K_basis, -1, 3)
        mesh_basis = Meshes(verts=self.mean_v[:1].repeat(K_basis, 1, 1) + B_,
                            faces=self.faces[:1].repeat(K_basis, 1, 1))
        mesh_def = Meshes(verts=self.pred_v, faces=self.faces)
        mesh_mean = Meshes(verts=self.mean_v, faces=self.faces)
        self.triangle_loss = mesh_laplacian_smoothing(mesh_def,
                                                      method="cot") if opts.triangle_reg_wt > 0 and not drop_deform else torch.tensor(
            [0.])
        self.normal_loss = mesh_normal_consistency(
            mesh_def) if opts.normal_loss_wt > 0 and not drop_deform else torch.tensor([0.])

        if opts.arap_reg_wt > 0 and not drop_deform:
            self.arap_loss = self.arap_loss_mesh_fn(mesh_def, mesh_mean)
        else:
            self.arap_loss = torch.tensor([0.])

        if opts.arap_basis_loss_wt > 0 and not drop_deform:
            mesh_arap = Meshes(verts=self.mean_v[:1].repeat(K_basis, 1, 1) + B_,
                               faces=self.faces[:1].repeat(K_basis, 1, 1))
            mesh_tmpl = Meshes(verts=self.mean_v[:1].repeat(K_basis, 1, 1), faces=self.faces[:1].repeat(K_basis, 1, 1))
            self.arap_basis_loss = self.arap_loss_basis_fn(mesh_arap, mesh_tmpl)
        else:
            self.arap_basis_loss = torch.tensor([0.])

        self.tri_basis_loss = mesh_laplacian_smoothing(mesh_basis, method="uniform")

        self.vert2kp_loss = self.entropy_loss(vert2kp) if opts.learnable_kp else torch.tensor([0.])

        self.bdt_loss = self.boundaries_fn(self.tex_flow, self.boundaries, self.pix_to_face)

        # Compute total loss
        self.total_loss = 0
        if opts.mask_loss_wt > 0 and not init_basis:
            self.total_loss += opts.mask_loss_wt * self.mask_loss
        if opts.kp_loss_wt > 0 and not init_basis:
            self.total_loss += opts.kp_loss_wt * self.kp_loss
        if opts.tex_loss_wt > 0 and not init_basis:
            self.total_loss += opts.tex_loss_wt * self.tex_loss
        if opts.bdt_reg_wt > 0 and not init_basis:
            self.total_loss += opts.bdt_reg_wt * self.bdt_loss
        if opts.vert2kp_loss_wt > 0 and opts.learnable_kp and not init_basis:
            self.total_loss += opts.vert2kp_loss_wt * self.vert2kp_loss
        if opts.rigid_wt > 0 and not init_basis:
            self.total_loss += opts.rigid_wt * self.rigid_loss
        if opts.triangle_reg_wt > 0 and not drop_deform:
            self.total_loss += opts.triangle_reg_wt * self.triangle_loss
        if opts.tri_basis_wt > 0 and not drop_deform:
            self.total_loss += opts.tri_basis_wt * self.tri_basis_loss
        if opts.arap_reg_wt > 0 and not drop_deform:
            self.total_loss += opts.arap_reg_wt * self.arap_loss
        if opts.sil_loss_wt > 0:
            self.total_loss += opts.sil_loss_wt * self.sil_loss
        if opts.equiv_loss_wt > 0 and not init_basis:
            self.total_loss += opts.equiv_loss_wt * self.equiv_loss
        if opts.def_loss_wt > 0 and not drop_deform:
            self.total_loss += opts.def_loss_wt * self.deform_loss
        if opts.normal_loss_wt > 0 and not drop_deform:
            self.total_loss += opts.normal_loss_wt * self.normal_loss
        if opts.tex_dt_loss_wt > 0 and not init_basis:
            self.total_loss += opts.tex_dt_loss_wt * self.tex_loss_dt
        if opts.vis_loss_wt > 0 and not init_basis:
            self.total_loss += opts.vis_loss_wt * self.vis_loss
        if opts.betas_loss_wt > 0 and not drop_deform:
            self.total_loss += opts.betas_loss_wt * self.betas_loss
        if opts.basis_cycle_loss_wt > 0 and not drop_deform:
            self.total_loss += opts.basis_cycle_loss_wt * self.basis_cycle_loss
        if opts.arap_basis_loss_wt > 0 and not drop_deform:
            self.total_loss += opts.arap_basis_loss_wt * self.arap_basis_loss

    def get_current_visuals(self):
        # Visualize progress and samples from a batch
        vis_dict = {}
        mask_concat = torch.cat([self.masks, self.mask_pred], 2)

        num_show = min(2, self.masks.shape[0])
        # Visualize images and 2D locations
        for i in range(num_show):
            input_img = bird_vis.kp2im(self.kps[i].data, self.imgs[i].data)
            pred_kp_img = bird_vis.kp2im(self.kp_meanv_transformed[i].data, self.imgs[i].data)
            pred_transformed_kp_img = bird_vis.kp2im(self.kp_pred_transformed[i].data, self.imgs[i].data)
            pred_texflow_kp_img = bird_vis.kp2im(self.kp_verts_regr_points[i].data, self.imgs[i].data)
            pred_transformed_kp_mask = bird_vis.kp2im(self.kp_pred_transformed[i].data, self.mask_pred[i].data)
            pts2d_pro = geo.project_points_by_theta(self.pred_v[i][None].cpu(), self.cam_pred[i][None].cpu(),
                                                    self.K[i][None].cpu(),
                                                    scale=False)[0]
            pred_transformed_reproj = bird_vis.kp2im(pts2d_pro[:, :2].data, self.imgs[i].data,
                                                     manual_colors=self.c_nodes)

            img_sample_pos = bird_vis.kp2im(self.tex_flow[i].data, self.imgs[i].data, manual_colors=self.c_nodes)
            boundaries = bird_vis.kp2im(self.boundaries[i].data, self.imgs[i].data)
            masks = bird_vis.tensor2mask(mask_concat[i].data)

            if self.opts.texture:
                tex_img = bird_vis.tensor2im(self.texture_pred[i].data)
                imgs = np.hstack((input_img, pred_kp_img, pred_transformed_kp_img, pred_texflow_kp_img, img_sample_pos,
                                  pred_transformed_reproj, boundaries, tex_img))
            else:
                imgs = np.hstack((input_img, pred_transformed_kp_img, img_sample_pos))

            vis_dict['%d' % i] = np.hstack((imgs, masks, pred_transformed_kp_mask))
        vis_dict['masked_img %d' % 0] = bird_vis.tensor2im((self.imgs[0] * self.masks[0]).data)
        vis_dict['barrier %d' % 0] = bird_vis.tensor2im((self.edts_barrier[0] / self.edts_barrier[0].max()).data)

        # Visualization of Equivariance loss
        if self.opts.equiv_loss_wt > 0:
            i = 0
            imgs_augm = bird_vis.tensor2im(self.imgs_augm[i].data)
            input_img_augm = bird_vis.kp2im(self.tex_flow_augm[i].data, self.imgs_augm[i].data,
                                            manual_colors=self.c_nodes)
            input_img_augm_2 = bird_vis.kp2im(self.tex_flow_trans[i].data, self.imgs_augm[i].data,
                                              manual_colors=self.c_nodes)
            input_img_orig = bird_vis.kp2im(self.tex_flow[i].data, self.imgs[i].data, manual_colors=self.c_nodes)

            imgs_flip = np.hstack((imgs_augm, input_img_augm, input_img_augm_2, input_img_orig))
            vis_dict['equiv_img %d' % i] = imgs_flip

            if opts.flip_train:
                batch_size = self.imgs.shape[0]
                im, im_f = self.imgs[:batch_size // 2].detach(), self.imgs[batch_size // 2:].detach()
                tim, tim_f = self.tex_flow[:batch_size // 2].detach(), self.tex_flow[batch_size // 2:].detach()
                im = bird_vis.kp2im(tim[i].data, im[i].data, manual_colors=self.c_nodes)
                im_f = bird_vis.kp2im(tim_f[i].data, im_f[i].data, manual_colors=self.c_nodes)
                t, t_f = self.texture_pred[:batch_size // 2].detach(), self.texture_pred[batch_size // 2:].detach()
                t, t_f = bird_vis.tensor2im(t[i].data), bird_vis.tensor2im(t_f[i].data)
                m, m_f = self.mask_pred[:batch_size // 2].detach(), self.mask_pred[batch_size // 2:].detach()
                m, m_f = bird_vis.tensor2mask(m[i].data), bird_vis.tensor2mask(m_f[i].data)
                foo = bird_vis.tensor2mask(self.mask_pred_equiv[i].data)
                imgs_flip = np.hstack((im, m, t, foo, im_f, m_f, t_f))
                vis_dict['flip_train_img %d' % i] = imgs_flip
        vis_dict['depth %d' % 0] = self.pix_to_face_up[0, :, :, :1].float().data.cpu().numpy()

        fig = plt.figure(figsize=(6, 6))
        ax1 = fig.add_subplot(1, 1, 1)
        texturesuv_image_matplotlib(self.textures[0], subsample=None)
        ax1.axis('off')
        fig.tight_layout(pad=0, w_pad=0, h_pad=0)
        fig.canvas.draw()
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        tex_viz = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        tex_viz = resize(tex_viz, self.masks.shape[-2:]) * 255
        plt.close(fig)
        vis_dict['tex_viz'] = tex_viz

        return vis_dict

    def get_current_points(self):
        # Plot meshes to visdom
        color = geom_utils.sample_textures_v(self.tex_flow, self.imgs)
        textures = TexturesVertex(verts_features=color[:1])

        mesh = Meshes(
            verts=[self.pred_v[0]],
            faces=[self.faces[0]],
            textures=textures
        )
        mean_textures = TexturesVertex(verts_features=torch.from_numpy(self.c_nodes[None] / 255).to(self.pred_v.device))
        mean_mesh = Meshes(
            verts=[self.mean_v[0]],
            faces=[self.faces[0]],
            textures=mean_textures
        )
        to_ret = {
            'mean_shape': mean_mesh,
            'pred_mesh': mesh,
        }
        if opts.weighted_camera:
            ms = geo.project_points_by_theta(self.mean_v, self.cam_pred, self.K, b2p=False)

            color_w = torch.ones_like(color[:1])
            color_w *= self.visibility[:1, :, None]
            weight_texture = TexturesVertex(verts_features=color_w)

            weight_mesh = Meshes(
                verts=ms[:1],
                faces=self.faces[:1],
                textures=weight_texture
            )
            to_ret['weight_mesh'] = weight_mesh
            color_w = torch.ones_like(color[:1])
            color_w *= self.visible_verts[:1, :, None]
            weight_texture = TexturesVertex(verts_features=color_w)

            weight_mesh = Meshes(
                verts=ms[:1],
                faces=self.faces[:1],
                textures=weight_texture
            )
            to_ret['weight_gt_mesh'] = weight_mesh
        return to_ret


    def get_current_scalars(self):
        # Plot losses to visdom
        sc_dict = OrderedDict([
            ('smoothed_total_loss', self.smoothed_total_loss),
            ('total_loss', self.total_loss.item()),
            ('kp_loss', (opts.kp_loss_wt * self.kp_loss).item()),
            ('mask_loss', (opts.mask_loss_wt * self.mask_loss).item()),
            ('rigid_loss', (opts.rigid_wt * self.rigid_loss).item()),
            ('vert2kp_loss', (opts.vert2kp_loss_wt * self.vert2kp_loss).item()),
            ('tri_loss', (opts.triangle_reg_wt * self.triangle_loss).item()),
            ('bdt_loss', (opts.bdt_reg_wt * self.bdt_loss).item()),
            ('equiv_loss', (opts.equiv_loss_wt * self.equiv_loss).item()),
            ('vis_loss', (opts.vis_loss_wt * self.vis_loss).item()),
            ('def_loss', (opts.def_loss_wt * self.deform_loss).item()),
            ('betas_loss', (opts.betas_loss_wt * self.betas_loss).item()),
            ('arap_loss', (opts.arap_reg_wt * self.arap_loss).item()),
            ('sil_loss', (opts.sil_loss_wt * self.sil_loss).item()),
            ('normal_loss', (opts.normal_loss_wt * self.normal_loss).item()),
            ('basis_cycle_loss', (opts.basis_cycle_loss_wt * self.basis_cycle_loss).item()),
            ('tri_basis_loss', (opts.tri_basis_wt * self.tri_basis_loss).item()),
            ('arap_basis_loss', (opts.arap_basis_loss_wt * self.arap_basis_loss).item()),
        ])
        if self.opts.texture:
            sc_dict['tex_loss'] = (opts.tex_loss_wt * self.tex_loss).item()
            sc_dict['tex_dt_loss'] = (opts.tex_dt_loss_wt * self.tex_loss_dt).item()
        return sc_dict


def main(_):
    torch.manual_seed(0)
    trainer = ShapeTrainer(opts)
    trainer.init_training()
    trainer.train()


if __name__ == '__main__':
    app.run(main)
