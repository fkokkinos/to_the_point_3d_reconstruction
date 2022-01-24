from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
# rendering components
from pytorch3d.renderer import (
    look_at_view_transform, RasterizationSettings, MeshRasterizer, BlendParams,
    SoftSilhouetteShader, DirectionalLights,
    OrthographicCameras
)
from pytorch3d.renderer.mesh.shader import SoftPhongShader
from pytorch3d.structures import Meshes
from torch import nn

from . import geometry_utilities as geo


class MeshRenderer(nn.Module):
    def __init__(self, rasterizer, shader):
        super().__init__()
        self.rasterizer = rasterizer
        self.shader = shader

    def to(self, device):
        self.rasterizer.to(device)
        self.shader.to(device)

    def forward(self, meshes_world, **kwargs):
        fragments = self.rasterizer(meshes_world, **kwargs)
        images = self.shader(fragments, meshes_world, **kwargs)
        return images, fragments


class NeuralRenderer(torch.nn.Module):

    def __init__(self, img_size=256, background_color=0, lights=None):
        super(NeuralRenderer, self).__init__()
        self.img_size = img_size

        # Setup pytorch3d renderer parameters
        self.cameras = OrthographicCameras()
        self.blend_params = BlendParams(sigma=1e-4, gamma=1e-4, background_color=background_color)
        self.blend_params_tex = BlendParams(background_color=background_color)
        self.tex_raster_settings = RasterizationSettings(
            image_size=img_size,
            blur_radius=np.log(1. / 1e-4 - 1.) * self.blend_params.sigma,
            faces_per_pixel=10, bin_size=None)

        if lights is None:
            self.lights = DirectionalLights(ambient_color=((0.8, 0.8, 0.8),),
                                            diffuse_color=((0., 0., 0.),),
                                            specular_color=((0., 0., 0.),),
                                            direction=((1., 0., 0.),))
        else:
            self.lights = lights

        self.raster_settings_of = RasterizationSettings(
            image_size=img_size,
            blur_radius=0.0,
            faces_per_pixel=1
        )
        self.rasterizer = MeshRasterizer(
            cameras=self.cameras,
            raster_settings=self.raster_settings_of
        )

        self.proj_fn = geo.project_points_by_theta

        self.offset_z = 5.

    def ambient_light_only(self):
        return

    def set_bgcolor(self, color):
        return

    def project_points(self, verts, cams, K):
        proj = self.proj_fn(verts, cams, K)
        return proj

    def forward(self, vertices, faces, cams, K, textures=None):
        eye = torch.tensor([[0, 0, -2.732]], device=vertices.device)
        verts = self.proj_fn(vertices, cams, K, b2p=False)
        vs = verts
        vs[:, :, 1] *= -1  # flip axis to match renderer's orientation
        R, T = look_at_view_transform(eye=eye, device=vertices.device)
        R[:, 0, 0] *= -1
        if textures is None:
            # render without texture
            mesh = Meshes(verts=vs, faces=faces)
            blend_params = BlendParams(sigma=1e-4, gamma=1e-4, background_color=0)
            cameras = OrthographicCameras(device=vs.device)
            sil_raster_settings = RasterizationSettings(
                image_size=self.img_size,
                blur_radius=np.log(1. / 1e-4 - 1.) * blend_params.sigma,
                faces_per_pixel=20, bin_size=None
            )

            silhouette_renderer = MeshRenderer(
                rasterizer=MeshRasterizer(
                    cameras=cameras,
                    raster_settings=sil_raster_settings
                ),
                shader=SoftSilhouetteShader(blend_params=blend_params)
            )

            masks, fragments = silhouette_renderer(meshes_world=mesh, R=R, T=T)
            masks = masks[..., -1]
            pix_to_face = fragments.pix_to_face
            depth = fragments.zbuf[..., 0]
            return masks, pix_to_face, depth
        else:
            # render with texture
            mesh = Meshes(verts=vs, faces=faces, textures=textures)
            cameras = OrthographicCameras(device=vs.device)
            phong_renderer = MeshRenderer(
                rasterizer=MeshRasterizer(
                    cameras=cameras,
                    raster_settings=self.tex_raster_settings
                ),
                shader=SoftPhongShader(device=vs.device, cameras=cameras, lights=self.lights.clone().to(vs.device),
                                       blend_params=self.blend_params_tex)
            )

            imgs, fragments = phong_renderer(meshes_world=mesh, R=R, T=T)
            pix_to_face = fragments.pix_to_face
            sil = imgs[..., -1]
            imgs = imgs[..., :-1]
            imgs = imgs.permute(0, 3, 1, 2)
            depth = fragments.zbuf[..., 0]
            return imgs, sil, pix_to_face, depth
