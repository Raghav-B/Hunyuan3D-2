# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.


import logging
import numpy as np
import os
import torch
from PIL import Image
from typing import List, Union, Optional
import sys
import cv2

from .differentiable_renderer.mesh_render import MeshRender
from .utils.dehighlight_utils import Light_Shadow_Remover
from .utils.multiview_utils import Multiview_Diffusion_Net
from .utils.imagesuper_utils import Image_Super_Net
from .utils.uv_warp_utils import mesh_uv_wrap

logger = logging.getLogger(__name__)


class Hunyuan3DTexGenConfig:

    def __init__(self, light_remover_ckpt_path, multiview_ckpt_path, subfolder_name):
        self.device = 'cuda'
        self.light_remover_ckpt_path = light_remover_ckpt_path
        self.multiview_ckpt_path = multiview_ckpt_path

        self.camera_poses = []
        self.candidate_camera_azims = []
        self.candidate_camera_elevs = []
        self.candidate_view_weights = []

        self.multiview_resolution = 512
        self.render_size = 2048
        self.texture_size = 2048
        self.bake_exp = 4
        self.merge_method = 'fast'

        self.pipe_dict = {'hunyuan3d-paint-v2-0': 'hunyuanpaint', 'hunyuan3d-paint-v2-0-turbo': 'hunyuanpaint-turbo'}
        self.pipe_name = self.pipe_dict[subfolder_name]
    
    def set_multiview_options(self, multiview_res, multiview_image_keys):
        self.multiview_resolution = multiview_res
        self.multiview_image_keys = multiview_image_keys

        candidate_camera_poses = [
            { "elev": 0, "azim": 0, "weight": 1.0, "name": "front" },
            { "elev": 0, "azim": 90, "weight": 0.1, "name": "right" },
            { "elev": 0, "azim": 180, "weight": 0.5, "name": "back" },
            { "elev": 0, "azim": 270, "weight": 0.1, "name": "left" },
            { "elev": 90, "azim": 0, "weight": 0.05, "name": "top" },
            { "elev": -90, "azim": 180, "weight": 0.05, "name": "bottom" },
            { "elev": 0, "azim": 45, "weight": 0.25, "name": "front-right" },
            { "elev": 0, "azim": 135, "weight": 0.25, "name": "back-right" },
            { "elev": 0, "azim": 225, "weight": 0.25, "name": "back-left" },
            { "elev": 0, "azim": 315, "weight": 0.25, "name": "front-left" },
        ]

        self.camera_poses = []
        for pose in candidate_camera_poses:
            if self.multiview_image_keys[pose['name']]:
                self.camera_poses.append(pose)
        
        self.candidate_camera_azims = [pose['azim'] for pose in self.camera_poses]
        self.candidate_camera_elevs = [pose['elev'] for pose in self.camera_poses]
        self.candidate_view_weights = [pose['weight'] for pose in self.camera_poses]


class Hunyuan3DPaintPipeline:
    
    @classmethod
    def from_pretrained(cls, model_path, subfolder='hunyuan3d-paint-v2-0'):        
        if not os.path.exists(model_path):
            print(f"Model path {model_path} not found, quitting...")
            sys.exit()

        delight_model_path = os.path.join(model_path, 'hunyuan3d-delight-v2-0')
        multiview_model_path = os.path.join(model_path, subfolder)
        return cls(Hunyuan3DTexGenConfig(delight_model_path, multiview_model_path, subfolder))

    def __init__(self, config:Hunyuan3DTexGenConfig):
        self.config = config
        self.models = {}
        self.render = MeshRender(
            default_resolution=self.config.render_size,
            texture_size=self.config.texture_size)

        self.load_models()

    def set_multiview_options(self, multiview_res, multiview_image_keys):
        self.config.set_multiview_options(multiview_res, multiview_image_keys)

    def load_models(self):
        # empty cude cache
        torch.cuda.empty_cache()
        # Load model
        # self.models['delight_model'] = Light_Shadow_Remover(self.config)
        self.models['multiview_model'] = Multiview_Diffusion_Net(self.config)
        # self.models['super_model'] = Image_Super_Net(self.config)

    def enable_model_cpu_offload(self, gpu_id: Optional[int] = None, device: Union[torch.device, str] = "cuda"):
        self.models['delight_model'].pipeline.enable_model_cpu_offload(gpu_id=gpu_id, device=device)
        self.models['multiview_model'].pipeline.enable_model_cpu_offload(gpu_id=gpu_id, device=device)

    def render_normal_multiview(self, camera_elevs, camera_azims, use_abs_coor=True):
        normal_maps = []
        for elev, azim in zip(camera_elevs, camera_azims):
            normal_map = self.render.render_normal(
                elev, azim, use_abs_coor=use_abs_coor, return_type='pl')
            normal_maps.append(normal_map)

        return normal_maps

    def render_position_multiview(self, camera_elevs, camera_azims):
        position_maps = []
        for elev, azim in zip(camera_elevs, camera_azims):
            position_map = self.render.render_position(
                elev, azim, return_type='pl')
            position_maps.append(position_map)

        return position_maps

    def bake_from_multiview(self, views, camera_elevs,
                            camera_azims, view_weights, method='graphcut'):
        project_textures, project_weighted_cos_maps = [], []
        project_boundary_maps = []
        for view, camera_elev, camera_azim, weight in zip(
            views, camera_elevs, camera_azims, view_weights):
            project_texture, project_cos_map, project_boundary_map = self.render.back_project(
                view, camera_elev, camera_azim)
            project_cos_map = weight * (project_cos_map ** self.config.bake_exp)
            project_textures.append(project_texture)
            project_weighted_cos_maps.append(project_cos_map)
            project_boundary_maps.append(project_boundary_map)

        if method == 'fast':
            texture, ori_trust_map = self.render.fast_bake_texture(
                project_textures, project_weighted_cos_maps)
        else:
            raise f'no method {method}'
        return texture, ori_trust_map > 1E-8

    def texture_inpaint(self, texture, mask):

        texture_np = self.render.uv_inpaint(texture, mask)
        texture = torch.tensor(texture_np / 255).float().to(texture.device)

        return texture

    def recenter_image(self, image, border_ratio=0.2):
        if image.mode == 'RGB':
            return image
        elif image.mode == 'L':
            image = image.convert('RGB')
            return image

        alpha_channel = np.array(image)[:, :, 3]
        non_zero_indices = np.argwhere(alpha_channel > 0)
        if non_zero_indices.size == 0:
            raise ValueError("Image is fully transparent")

        min_row, min_col = non_zero_indices.min(axis=0)
        max_row, max_col = non_zero_indices.max(axis=0)

        cropped_image = image.crop((min_col, min_row, max_col + 1, max_row + 1))

        width, height = cropped_image.size
        border_width = int(width * border_ratio)
        border_height = int(height * border_ratio)

        new_width = width + 2 * border_width
        new_height = height + 2 * border_height

        square_size = max(new_width, new_height)

        new_image = Image.new('RGBA', (square_size, square_size), (255, 255, 255, 0))

        paste_x = (square_size - new_width) // 2 + border_width
        paste_y = (square_size - new_height) // 2 + border_height

        new_image.paste(cropped_image, (paste_x, paste_y))
        return new_image

    @torch.no_grad()
    def __call__(self, mesh, image, multiview_res, multiview_image_keys):
        self.config.set_multiview_options(multiview_res, multiview_image_keys)
        self.models['multiview_model'].update_config(self.config)

        if not isinstance(image, List):
            image = [image]

        images_prompt = []
        for i in range(len(image)):
            if isinstance(image[i], str):
                image_prompt = Image.open(image[i])
            else:
                image_prompt = image[i]
            images_prompt.append(image_prompt)
            
        # Save image prompt
        image_prompt.save('image_prompt.png')
    
        images_prompt = [self.recenter_image(image_prompt, border_ratio=0) for image_prompt in images_prompt]
        print(f"Image size: {images_prompt[0].size}")


        # images_prompt = [self.models['delight_model'](image_prompt) for image_prompt in images_prompt]

        mesh = mesh_uv_wrap(mesh)

        self.render.load_mesh(mesh)

        selected_camera_elevs, selected_camera_azims, selected_view_weights = \
            self.config.candidate_camera_elevs, self.config.candidate_camera_azims, self.config.candidate_view_weights

        normal_maps = self.render_normal_multiview(
            selected_camera_elevs, selected_camera_azims, use_abs_coor=True)
        position_maps = self.render_position_multiview(
            selected_camera_elevs, selected_camera_azims)

        camera_info = [(((azim // 30) + 9) % 12) // {-20: 1, 0: 1, 20: 1, -90: 3, 90: 3}[elev] + 
                       {-20: 0, 0: 12, 20: 24, -90: 36, 90: 40}[elev] for azim, elev in
                       zip(selected_camera_azims, selected_camera_elevs)]
        
        
        # Split the multiviews into batches to avoid OOM
        batch_size = 6
        multiviews = []
        for i in range(0, len(camera_info), batch_size):
            normal_map = normal_maps[i:i+batch_size]
            position_map = position_maps[i:i+batch_size]
            camera_info_batch = camera_info[i:i+batch_size]

            multiview = self.models['multiview_model'](images_prompt, normal_map + position_map, camera_info_batch)
            temparray = np.array(multiview)
            print(temparray.shape, "multiviews generated for batch", i // batch_size)
            
            multiviews.extend(multiview)

        # multiviews = self.models['multiview_model'](images_prompt, normal_maps + position_maps, camera_info)

        temparray = np.array(multiviews)
        print(temparray.shape, "multiviews generated")

        for i in range(len(multiviews)):
            # multiviews[i] = self.models['super_model'](multiviews[i])
            multiviews[i] = multiviews[i].resize(
                (self.config.render_size, self.config.render_size))

            with open(f'multiviews_{i}.png', 'wb') as f:
                multiviews[i].save(f)
                

        texture, mask = self.bake_from_multiview(multiviews,
                                                 selected_camera_elevs, selected_camera_azims, selected_view_weights,
                                                 method=self.config.merge_method)

        mask_np = (mask.squeeze(-1).cpu().numpy() * 255).astype(np.uint8)
        mask_cv = np.array(mask_np, dtype=np.uint8)
        cv2.imwrite('mask.png', mask_cv)
        print_tex = (texture.cpu().numpy() * 255).astype(np.uint8)
        print_tex = np.clip(print_tex, 0, 255).astype(np.uint8)
        print_tex = cv2.cvtColor(print_tex, cv2.COLOR_RGB2BGR)
        cv2.imwrite('texture.png', print_tex)

        texture = self.texture_inpaint(texture, mask_np)

        print_tex = (texture.cpu().numpy() * 255).astype(np.uint8)
        print_tex = np.clip(print_tex, 0, 255).astype(np.uint8)
        print_tex = cv2.cvtColor(print_tex, cv2.COLOR_RGB2BGR)
        cv2.imwrite('texture_inpainted.png', print_tex)

        self.render.set_texture(texture)
        textured_mesh = self.render.save_mesh()

        return textured_mesh
