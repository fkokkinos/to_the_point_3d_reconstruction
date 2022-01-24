"""
Takes an image, returns stuff.
"""
import os
import os.path as osp
import pickle as pkl

import numpy as np
import pytorch3d.io as p3dio
import scipy.io as sio
import torch
import torchvision
from absl import flags
from pytorch3d.renderer.mesh import TexturesUV
from pytorch3d.structures import Meshes
from torch.autograd import Variable

from nnutils import geometry_utilities as geo
from nnutils import mesh_net_lpl_v2_tex as mesh_net
from nnutils.geom_utils import mesh_laplacian
from nnutils.mesh_net_lpl_v2_tex import solve_basis
from nnutils.nmr import NeuralRenderer

# These options are off by default, but used for some ablations reported.
flags.DEFINE_boolean('ignore_pred_delta_v', False, 'Use only mean shape for prediction')
flags.DEFINE_boolean('use_sfm_ms', False, 'Uses sfm mean shape for prediction')
flags.DEFINE_boolean('use_sfm_camera', False, 'Uses sfm mean camera')
flags.DEFINE_string('mesh_dir', 'meshes/bird.obj', 'tmp dir to extract dataset')
flags.DEFINE_string('kp_dict', 'meshes/bird_kp_dictionary.pkl', 'tmp dir to extract dataset')
flags.DEFINE_integer('num_lbs', 15, 'keypoint loss weight')
flags.DEFINE_boolean('weighted_camera', False, 'whether to print scalars')
flags.DEFINE_boolean('drop_deform', False, 'whether to drop deformation')
flags.DEFINE_integer('def_steps', 1, 'number of deformation steps')
flags.DEFINE_float('scale_template', 1., 'gradient reversal layer weight')


class MeshPredictor(object):
    def __init__(self, opts):
        self.opts = opts
        print('Setting up model..')
        anno_sfm_path = osp.join(opts.cub_cache_dir, 'sfm', 'anno_train.mat')
        anno_sfm = sio.loadmat(anno_sfm_path, struct_as_record=False, squeeze_me=True)
        sfm_mean_shape = (np.transpose(anno_sfm['S']), anno_sfm['conv_tri'] - 1)

        kp_dict = pkl.load(open(opts.kp_dict, 'rb'))
        mesh_loaded = p3dio.load_obj(opts.mesh_dir)
        v, f = mesh_loaded[0].numpy(), mesh_loaded[1].verts_idx.numpy()
        # scale = 2. / torch.max(torch.nn.functional.pdist(torch.from_numpy(v))).numpy()
        # v = v * scale
        # v = v - v.mean(0)
        # shapenet_mesh = [v, f]
        v = torch.from_numpy(v)[None]
        self.s_3d = v.reshape(1, -1).std(1)[:, None]
        self.mu_3d = v.mean(1)
        v = (v - self.mu_3d[:, None]) / self.s_3d.unsqueeze(-1)
        v = v[0].numpy() * opts.scale_template
        shapenet_mesh = [v, f]
        img_size = (opts.img_size, opts.img_size)
        self.model = mesh_net.MeshNet(img_size, opts, nz_feat=opts.nz_feat, sfm_mean_shape=sfm_mean_shape,
                                      shapenet_mesh=shapenet_mesh, kp_dict=kp_dict)

        self.load_network(self.model, 'pred', self.opts.num_train_epoch)
        self.model.eval()
        self.model = self.model.cuda(device=self.opts.gpu_id)

        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)

        print('Number of parameters:', count_parameters(self.model))
        self.renderer = NeuralRenderer(opts.img_size)

        if opts.texture:
            self.tex_renderer = NeuralRenderer(opts.img_size)
            self.tex_renderer.ambient_light_only()

        if opts.use_sfm_ms:
            anno_sfm_path = osp.join(opts.cub_cache_dir, 'sfm', 'anno_testval.mat')
            anno_sfm = sio.loadmat(anno_sfm_path, struct_as_record=False, squeeze_me=True)
            sfm_mean_shape = torch.Tensor(np.transpose(anno_sfm['S'])).cuda(device=opts.gpu_id)
            self.sfm_mean_shape = Variable(sfm_mean_shape, requires_grad=False)
            self.sfm_mean_shape = self.sfm_mean_shape.unsqueeze(0).repeat(opts.batch_size, 1, 1)
            sfm_face = torch.LongTensor(anno_sfm['conv_tri'] - 1).cuda(device=opts.gpu_id)
            self.sfm_face = Variable(sfm_face, requires_grad=False)
            faces = self.sfm_face.view(1, -1, 3)
        else:
            faces = self.model.faces.view(1, -1, 3)
        self.faces = faces.repeat(opts.batch_size, 1, 1)
        mesh_template = Meshes(verts=[self.model.get_mean_shape()], faces=[self.faces[0]])
        mesh_up = self.model.get_subdivision(mesh_template)
        self.faces_up = mesh_up.faces_packed()
        self.L = mesh_laplacian(mesh_template, 'uniform')

        self.resnet_transform = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def load_network(self, network, network_label, epoch_label):
        save_filename = '{}_net_{}.pth'.format(network_label, epoch_label)
        network_dir = os.path.join(self.opts.checkpoint_dir, self.opts.name)
        save_path = os.path.join(network_dir, save_filename)
        print('loading {}..'.format(save_path))
        try:
            network.load_state_dict(torch.load(save_path))
        except Exception as e:
            print(e)
            network.load_state_dict(torch.load(save_path), strict=False)
        return

    def set_input(self, batch):
        opts = self.opts

        img_tensor = batch['img'].clone().type(torch.FloatTensor)

        input_img_tensor = batch['img'].type(torch.FloatTensor).clone()
        for b in range(input_img_tensor.size(0)):
            input_img_tensor[b] = self.resnet_transform(input_img_tensor[b])

        self.input_imgs = input_img_tensor.cuda(device=opts.gpu_id)
        self.imgs = img_tensor.cuda(device=opts.gpu_id)
        if opts.use_sfm_camera:
            cam_tensor = batch['sfm_pose'].type(torch.FloatTensor)
            self.sfm_cams = cam_tensor.cuda(device=opts.gpu_id)
        self.frames_idx = batch['inds'].cuda(device=opts.gpu_id)

    def predict(self, batch):
        """
        batch has B x C x H x W numpy
        """
        self.set_input(batch)
        self.forward()
        return self.collect_outputs()

    def forward(self):
        opts = self.opts
        faces = self.faces[:1].repeat(self.input_imgs.shape[0], 1, 1)
        img_feat, self.res_feats = self.model(self.input_imgs)
        self.mean_shape = self.model.get_mean_shape()
        self.mean_v = self.mean_shape[None].repeat(self.imgs.shape[0], 1, 1).clone()
        self.pred_v = self.mean_v

        mesh_3d = Meshes(verts=self.pred_v, faces=faces)
        mesh_3d_up = self.model.get_subdivision(mesh_3d)
        self.mean_v = mesh_3d_up.verts_padded()
        self.faces_up = mesh_3d_up.faces_padded()
        self.tex_flow = self.model.texture_predictor_p(self.res_feats)
        self.visibility = torch.sigmoid(self.tex_flow[..., -1])
        self.tex_flow = torch.tanh(self.tex_flow[..., :2])
        self.tex_flow_uv = (self.tex_flow + 1) / 2
        self.tex_flow_uv[..., 1] = 1 - self.tex_flow_uv[..., 1]
        self.textures = TexturesUV(maps=self.imgs.permute(0, 2, 3, 1), verts_uvs=self.tex_flow_uv,
                                   faces_uvs=self.faces_up)
        s_2d = self.tex_flow.reshape(self.input_imgs.shape[0], -1).std(1)[:, None]
        mu_2d = self.tex_flow.mean(1)
        self.K = torch.cat([s_2d, s_2d, mu_2d], dim=-1)

        shape_tmp = self.mean_v
        def_steps = opts.def_steps
        for def_step in range(def_steps):
            ini_pose = None
            if def_step > 0:
                ini_pose = self.cam_pred
            if opts.weighted_camera:
                self.bpnp_weight = self.visibility[..., None]
                self.cam_pred = self.model.cam_solver(self.tex_flow, shape_tmp, self.K, self.bpnp_weight, ini_pose)
            else:
                self.cam_pred = self.model.cam_solver(self.tex_flow, shape_tmp, self.K, None, ini_pose)
            if opts.drop_deform:
                self.pred_v = self.mean_v
            else:
                all_basis = self.model.basis()[None].repeat(self.faces_up.shape[0], 1, 1)
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
        if opts.learnable_kp:
            vert2kp = torch.nn.functional.softmax(self.model.vert2kp, dim=1)
        else:
            vert2kp = self.model.vert2kp

        self.kp_verts = torch.matmul(vert2kp, trans_verts)
        self.kp_verts_pred_v = torch.matmul(vert2kp, trans_verts)
        self.kp_verts_transformed = self.kp_verts_pred_v
        self.kp_pred_transformed = self.kp_verts_pred_v[..., :2]
        self.kp_pred = self.kp_verts_pred_v[..., :2]
        self.kp_verts_regr_points = torch.matmul(vert2kp, self.tex_flow)

        # Render alpha and RGB channels
        self.mask_pred, pix_to_face, _ = self.renderer(self.pred_v, self.faces_up, proj_cam, self.K)
        self.texture_pred, _, _, _ = self.tex_renderer(self.pred_v, self.faces_up, proj_cam, self.K,
                                                       textures=self.textures)
        self.texture_pred = self.texture_pred[:, 0:3, :, :]

    def collect_outputs(self):
        outputs = {
            'kp_pred': self.kp_pred.data,
            'kp_verts_regr_points': self.kp_verts_regr_points.data,
            'verts': self.pred_v.data,
            'verts_up': self.pred_v.data,
            'faces_up': self.faces_up.data,
            'kp_verts': self.kp_verts.data,
            'cam_pred': self.cam_pred.data,
            'K': self.K.data,
            'mask_pred': self.mask_pred.data,
        }
        if self.opts.texture and not self.opts.use_sfm_ms:
            outputs['texture'] = self.textures
            outputs['tex_flow'] = self.tex_flow
            outputs['visibility'] = self.visibility
            outputs['texture_pred'] = self.texture_pred.data

        return outputs
