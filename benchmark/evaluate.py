"""
Script for testing on CUB.

Sample usage:
python -m cmr.benchmark.evaluate --split val --name <model_name> --num_train_epoch <model_epoch>
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp

import gdist
import networkx as nx
import numpy as np
import scipy.io as sio
import torch
import trimesh
from absl import app
from absl import flags
from pytorch3d.ops import SubdivideMeshes
from pytorch3d.renderer import (
    PointLights
)
from pytorch3d.structures import Meshes
from skimage import io

from data import cub as cub_data
from nnutils import geometry_utilities as geo
from nnutils import predictor_global_lpl_mesh_tex as pred_utils
from nnutils import test_utils
from utils import bird_vis
from utils.bird_vis import convert2np

flags.DEFINE_string('basis_pretrained', None, 'filepath to pretrained basis')
flags.DEFINE_string('dataset', 'cub', 'dataset selection')
flags.DEFINE_boolean('visualize', False, 'if true visualizes things')
flags.DEFINE_boolean('color_v2', False, 'if true visualizes things')

opts = flags.FLAGS


class ShapeTester(test_utils.Tester):
    def __init__(self, opts):
        super().__init__(opts)
        self.sdivide = None
        self.faces = None
        self.save_counter = None
        self.vis_counter = None

    def define_model(self):
        device = 'cuda'
        opts = self.opts

        self.predictor = pred_utils.MeshPredictor(opts)
        if opts.basis_pretrained is not None:
            basis_tmp = torch.load(opts.basis_pretrained)
            basis_tmp = basis_tmp.to(self.predictor.model.basis.w.device)
            self.predictor.model.basis.w.data = basis_tmp
        # for visualization
        self.vis_counter = 0
        self.save_counter = 0

        faces = self.predictor.model.faces.view(1, -1, 3)
        self.faces = faces.repeat(opts.batch_size, 1, 1)
        mesh_template = Meshes(verts=[self.predictor.model.get_mean_shape()], faces=[self.faces[0]])
        self.sdivide = SubdivideMeshes(mesh_template)
        # For visualization
        self.faces_up = self.sdivide(mesh_template).faces_packed()
        mesh_up = self.sdivide(mesh_template)
        num_verts_up = self.sdivide(mesh_template).verts_packed().shape[0]
        lights = PointLights(device=device, location=[[0.0, -1.0, 2.0]], ambient_color=((0.8, 0.8, 0.8),),
                             diffuse_color=((0.4, 0.4, 0.4),))

        self.vis_rend_up = bird_vis.VisRenderer(512, num_verts_up, background_color=(1, 1, 1), lights=lights)
        import pylab
        cm = pylab.get_cmap('gist_rainbow')
        if opts.color_v2:
            distance_gd = gdist.local_gdist_matrix(
                mesh_up.verts_list()[0].data.cpu().numpy().astype(np.float64),
                mesh_up.faces_list()[0].cpu().numpy().astype(np.int32)
            )

            c_dist = distance_gd.toarray()[0]
            c_dist = c_dist / c_dist.max()
            c_nodes = 255 * np.array([cm(c)[:3] for c in c_dist])
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

    def init_dataset(self):
        torch.manual_seed(0)
        if self.opts.dataset == 'cub':
            self.data_module = cub_data
        else:
            print('Unknown dataset %d!' % self.opts.dataset)
        self.dataloader = self.data_module.data_loader(self.opts, shuffle=False)

    def evaluate(self, outputs, batch):
        """
        Compute IOU and keypoint error
        """
        bs = self.opts.batch_size

        # compute iou
        mask_gt = batch['mask'].view(bs, -1).numpy()
        mask_pred = outputs['mask_pred'].cpu().view(bs, -1).type_as(
            batch['mask']).numpy()
        intersection = mask_gt * mask_pred
        union = mask_gt + mask_pred - intersection
        iou = intersection.sum(1) / union.sum(1)

        # Compute pck
        padding_frac = self.opts.padding_frac
        # The [-1,1] coordinate frame in which keypoints corresponds to:
        #    (1+2*padding_frac)*max_bbox_dim in image coords
        # pt_norm = 2* (pt_img - trans)/((1+2*pf)*max_bbox_dim)
        # err_pt = 2*err_img/((1+2*pf)*max_bbox_dim)
        # err_pck_norm = err_img/max_bbox_dim = err_pt*(1+2*pf)/2
        # so the keypoint error in the canonical fram should be multiplied by:
        err_scaling = (1 + 2 * padding_frac) / 2.0
        kps_gt = batch['kp'].cpu().numpy()

        kps_vis = kps_gt[:, :, 2]
        kps_gt = kps_gt[:, :, 0:2]
        kps_pred = outputs['kp_pred'].cpu().type_as(batch['kp']).numpy()
        kp_verts_regr_points = outputs['kp_verts_regr_points'].cpu().type_as(batch['kp']).numpy()
        kps_err = kps_pred - kps_gt
        kps_regr_err = kp_verts_regr_points - kps_gt
        kps_err = np.sqrt(np.sum(kps_err * kps_err, axis=2)) * err_scaling
        kps_regr_err = np.sqrt(np.sum(kps_regr_err * kps_regr_err, axis=2)) * err_scaling

        return iou, kps_err, kps_regr_err, kps_vis

    def visualize(self, outputs, batch, num_angles=6):
        directory = 'results_viz/' + self.opts.name + '_' + self.opts.split + '/'
        if not os.path.exists(directory):
            os.makedirs(directory)
        batch_size = outputs['verts'].shape[0]
        hq_rend_b = self.vis_rend_up(outputs['verts'], outputs['faces_up'], outputs['cam_pred'],
                                     outputs['K']).detach().cpu()
        hq_rend_tex_b = self.vis_rend_up(outputs['verts'], outputs['faces_up'], outputs['cam_pred'], outputs['K'],
                                         outputs['texture']).detach().cpu()
        for b in range(batch_size):
            if not os.path.exists(directory + '/' + str(self.vis_counter)):
                os.makedirs(directory + '/' + str(self.vis_counter))

            vert_up = outputs['verts_up'][b]
            faces_up = outputs['faces_up'][b]
            tex_flow = outputs['tex_flow'][b]
            cam = outputs['cam_pred'][b]
            K = outputs['K'][b]
            visibility = outputs['visibility'][b].detach()
            img = batch['img'][b].clone().type(torch.FloatTensor)
            mask = convert2np(batch['mask'][b])

            hq_rend = np.transpose(convert2np(hq_rend_b[b]), (1, 2, 0))
            hq_rend_tex = np.transpose(convert2np(hq_rend_tex_b[b]), (1, 2, 0))
            img_sample_pos = bird_vis.kp2im(tex_flow.data, img.data, manual_colors=self.c_nodes)
            img_pred = outputs['texture_pred'][b][:3].permute(1, 2, 0).detach().cpu().numpy()
            img = np.transpose(convert2np(batch['img'][b]), (1, 2, 0))
            img_ = np.hstack([img, img_pred])
            io.imsave(directory + '/' + str(self.vis_counter) + '/img.jpg', (img * 255).astype(np.uint8))
            io.imsave(directory + '/' + str(self.vis_counter) + '/mask.jpg', (mask * 255).astype(np.uint8))

            io.imsave(directory + '/' + str(self.vis_counter) + '/rend_hq.jpg', (hq_rend * 255).astype(np.uint8))
            io.imsave(directory + '/' + str(self.vis_counter) + '/rend_tex_hq.jpg',
                      (hq_rend_tex * 255).astype(np.uint8))
            io.imsave(directory + '/' + str(self.vis_counter) + '/rend.jpg', (img_pred * 255).astype(np.uint8))
            io.imsave(directory + '/' + str(self.vis_counter) + '/tiled.jpg', (img_ * 255).astype(np.uint8))
            io.imsave(directory + '/' + str(self.vis_counter) + '/flow.jpg', img_sample_pos.astype(np.uint8))
            pts2d_pro = geo.project_points_by_theta(vert_up[None].cpu(), cam[None].cpu(), K[None].cpu(), scale=False)[0]
            img_reproj = bird_vis.kp2im(pts2d_pro[:, :2].data, batch['img'][b].clone().type(torch.FloatTensor).data,
                                        manual_colors=self.c_nodes)
            io.imsave(directory + '/' + str(self.vis_counter) + '/proj_mesh.jpg', img_reproj.astype(np.uint8))
            mesh_ = trimesh.Trimesh(vert_up.detach().cpu().numpy(), faces_up.detach().cpu().numpy())
            mesh_.export(directory + '/' + str(self.vis_counter) + '/mesh.ply')
            color_w = torch.ones(visibility.shape[0], 3)
            color_w *= visibility[:, None].to(color_w.device)
            color_w = color_w.detach().cpu().numpy()
            mesh_ = trimesh.Trimesh(vert_up.detach().cpu().numpy(), faces_up.detach().cpu().numpy(),
                                    vertex_colors=color_w)
            mesh_.export(directory + '/' + str(self.vis_counter) + '/mesh_vis.ply')
            # pkl.dump({'outputs': outputs, 'batch': batch},
            #          open(directory + '/' + str(self.vis_counter) + '/predictions.pkl', 'wb'))
            self.vis_counter += 1

        self.save_counter += 1

    def test(self):
        opts = self.opts
        bench_stats = {'ious': [], 'kp_errs': [], 'kp_vis': [], 'kps_regr_err': []}

        if opts.ignore_pred_delta_v:
            result_path = osp.join(opts.results_dir, 'results_meanshape.mat')
        elif opts.use_sfm_ms:
            result_path = osp.join(opts.results_dir,
                                   'results_sfm_meanshape.mat')
        else:
            result_path = osp.join(opts.results_dir, 'results.mat')

        if opts.use_sfm_camera:
            result_path = result_path.replace('.mat', '_sfm_camera.mat')

        print('Writing to %s' % result_path)

        n_iter = len(self.dataloader)
        for i, batch in enumerate(self.dataloader):
            if i % 100 == 0:
                print('{}/{} evaluation iterations.'.format(i, n_iter))
            if opts.max_eval_iter > 0 and (i >= opts.max_eval_iter):
                break
            outputs = self.predictor.predict(batch)
            if opts.visualize:
                self.visualize(outputs, batch)
            iou, kp_err, kps_regr_err, kp_vis = self.evaluate(outputs, batch)

            bench_stats['ious'].append(iou)
            bench_stats['kp_errs'].append(kp_err)
            bench_stats['kps_regr_err'].append(kps_regr_err)
            bench_stats['kp_vis'].append(kp_vis)

        bench_stats['kp_errs'] = np.concatenate(bench_stats['kp_errs'], axis=0)
        bench_stats['kps_regr_err'] = np.concatenate(bench_stats['kps_regr_err'], axis=0)
        bench_stats['kp_vis'] = np.concatenate(bench_stats['kp_vis'], axis=0)
        bench_stats['ious'] = np.concatenate(bench_stats['ious'], axis=0)
        sio.savemat(result_path, bench_stats)

        # Report numbers.
        mean_iou = bench_stats['ious'].mean()
        n_vis_p = np.sum(bench_stats['kp_vis'], axis=0)
        n_correct_p_pt1 = np.sum(
            (bench_stats['kp_errs'] < 0.1) * bench_stats['kp_vis'], axis=0)
        n_correct_p_pt15 = np.sum(
            (bench_stats['kp_errs'] < 0.15) * bench_stats['kp_vis'], axis=0)
        pck1 = (n_correct_p_pt1 / n_vis_p).mean()
        pck15 = (n_correct_p_pt15 / n_vis_p).mean()
        print('%s mean iou %.3g, pck.1 %.3g, pck.15 %.3g' %
              (osp.basename(result_path), mean_iou, pck1, pck15))
        n_correct_p_pt1 = np.sum(
            (bench_stats['kps_regr_err'] < 0.1) * bench_stats['kp_vis'], axis=0)
        n_correct_p_pt15 = np.sum(
            (bench_stats['kps_regr_err'] < 0.15) * bench_stats['kp_vis'], axis=0)
        pck1 = (n_correct_p_pt1 / n_vis_p).mean()
        pck15 = (n_correct_p_pt15 / n_vis_p).mean()
        print('Regression: pck.1 %.3g, pck.15 %.3g' %
              (pck1, pck15))


def main(_):
    opts.n_data_workers = 0
    opts.results_dir = osp.join(opts.results_dir_base, '%s' % opts.split,
                                opts.name, 'epoch_%d' % opts.num_train_epoch)
    if not osp.exists(opts.results_dir):
        print('writing to %s' % opts.results_dir)
        os.makedirs(opts.results_dir)

    torch.manual_seed(0)
    tester = ShapeTester(opts)
    tester.init_testing()
    tester.test()


if __name__ == '__main__':
    app.run(main)
