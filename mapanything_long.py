import argparse
import glob
import json
import os
import sys
import threading
import warnings
import gc
from datetime import datetime

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

from LoopModels.LoopModel import LoopDetector
from LoopModelDBoW.retrieval.retrieval_dbow import RetrievalDBOW
# from loop_utils.visual_util import segment_sky, download_file_from_url

matplotlib.use('Agg')

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    
from mapanything.models import MapAnything
from mapanything.utils.image import preprocess_inputs

from loop_utils.sim3loop import Sim3LoopOptimizer
from loop_utils.sim3utils import *

from loop_utils.config_utils import load_config

def remove_duplicates(data_list):
    """
        data_list: [(67, (3386, 3406), 48, (2435, 2455)), ...]
    """
    seen = {} 
    result = []
    
    for item in data_list:
        if item[0] == item[2]:
            continue

        key = (item[0], item[2])
        
        if key not in seen.keys():
            seen[key] = True
            result.append(item)
    
    return result

class LongSeqResult:
    def __init__(self):
        self.combined_extrinsics = []
        self.combined_intrinsics = []
        self.combined_depth_maps = []
        self.combined_depth_confs = []
        self.combined_world_points = []
        self.combined_world_points_confs = []
        self.all_camera_poses = []

class MapAnything_Long:
    def __init__(self, image_dir, save_dir, config, config_path=None):
        self.config = config
        self.config_dir = (
            os.path.dirname(os.path.abspath(config_path))
            if config_path is not None
            else os.getcwd()
        )

        self.chunk_size = self.config['Model']['chunk_size']
        self.overlap = self.config['Model']['overlap']
        self.conf_threshold = 1.5
        self.seed = 42
        if torch.cuda.is_available():
            self.device = "cuda"
            major_cc = torch.cuda.get_device_capability()[0]
            self.dtype = torch.bfloat16 if major_cc >= 8 else torch.float16
        else:
            self.device = "cpu"
            self.dtype = torch.float32
        self.use_amp = self.device == "cuda"
        if self.device == "cuda":
            self.amp_dtype = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
        else:
            self.amp_dtype = "fp32"
        self.sky_mask = False
        self.useDBoW = self.config['Model']['useDBoW']
        self.memory_efficient_inference = self.config['Model'].get('memory_efficient_inference', False)
        self.image_resize_mode = self.config['Model'].get('image_resize_mode', 'fixed_mapping')
        self.image_resize_size = self.config['Model'].get('image_resize_size')
        self.image_resolution_set = self.config['Model'].get('image_resolution_set', 518)
        self.image_norm_type = self.config['Model'].get('image_norm_type', 'dinov2')
        self.image_patch_size = self.config['Model'].get('image_patch_size', 14)
        camera_cfg = self.config.get('Camera') or {}
        intrinsics_cfg = camera_cfg.get('intrinsics') or {}
        self._intrinsics_map = {}
        self._default_intrinsics = self._parse_intrinsics_matrix(
            intrinsics_cfg.get('default')
        )
        json_path = intrinsics_cfg.get('json_path')
        if json_path:
            self._load_intrinsics_mapping(json_path)
        self._use_intrinsics = bool(self._intrinsics_map) or (
            self._default_intrinsics is not None
        )

        self.img_dir = image_dir
        self.img_list = None
        self.output_dir = save_dir

        self.result_unaligned_dir = os.path.join(save_dir, '_tmp_results_unaligned')
        self.result_aligned_dir = os.path.join(save_dir, '_tmp_results_aligned')
        self.result_loop_dir = os.path.join(save_dir, '_tmp_results_loop')
        self.pcd_dir = os.path.join(save_dir, 'pcd')
        os.makedirs(self.result_unaligned_dir, exist_ok=True)
        os.makedirs(self.result_aligned_dir, exist_ok=True)
        os.makedirs(self.result_loop_dir, exist_ok=True)
        os.makedirs(self.pcd_dir, exist_ok=True)
        
        self.all_camera_poses = []
        
        self.delete_temp_files = self.config['Model']['delete_temp_files']

        if self.config['Model'].get('using_sim3', True):
            print("MapAnything provides metric scale. Switching to SE(3) alignment.")
            self.config['Model']['using_sim3'] = False

        print('Loading MapAnything model...')

        weights_cfg = self.config.get('Weights', {})
        default_hf_repo = 'facebook/map-anything'
        configured_mapanything = weights_cfg.get('MapAnything', default_hf_repo)
        expanded_path = os.path.expanduser(configured_mapanything)
        is_path_like = (
            configured_mapanything.startswith(('.', '/', '~'))
            or os.path.sep in configured_mapanything
        )

        if is_path_like and os.path.isdir(expanded_path):
            mapanything_source = os.path.abspath(expanded_path)
            print(f"Loading MapAnything from local directory: {mapanything_source}")
        elif is_path_like:
            print(
                f"Warning: Local MapAnything path '{configured_mapanything}' not found. "
                f"Falling back to HuggingFace repo '{default_hf_repo}'."
            )
            mapanything_source = default_hf_repo
        else:
            mapanything_source = configured_mapanything

        self.mapanything_model_name = mapanything_source
        try:
            self.model = MapAnything.from_pretrained(self.mapanything_model_name).to(self.device).eval()
        except Exception as exc:
            raise RuntimeError(f"Failed to load MapAnything weights '{self.mapanything_model_name}': {exc}") from exc
        
        self.chunk_indices = None # [(begin_idx, end_idx), ...]

        self.loop_list = [] # e.g. [(1584, 139), ...]

        self.loop_optimizer = Sim3LoopOptimizer(self.config)

        self.sim3_list = [] # [(s [1,], R [3,3], T [3,]), ...]

        self.loop_sim3_list = [] # [(chunk_idx_a, chunk_idx_b, s [1,], R [3,3], T [3,]), ...]

        self.loop_predict_list = []

        self.loop_enable = self.config['Model']['loop_enable']

        if self.loop_enable:
            if self.useDBoW:
                self.retrieval = RetrievalDBOW(config=self.config)
            else:
                loop_info_save_path = os.path.join(save_dir, "loop_closures.txt")
                self.loop_detector = LoopDetector(
                    image_dir=image_dir,
                    output=loop_info_save_path,
                    config=self.config
                )

        print('init done.')

    def get_loop_pairs(self):

        if self.useDBoW: # DBoW2
            for frame_id, img_path in tqdm(enumerate(self.img_list)):
                image_ori = np.array(Image.open(img_path))
                if len(image_ori.shape) == 2:
                    # gray to rgb
                    image_ori = cv2.cvtColor(image_ori, cv2.COLOR_GRAY2RGB)

                frame = image_ori # (height, width, 3)
                frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
                self.retrieval(frame, frame_id)
                cands = self.retrieval.detect_loop(thresh=self.config['Loop']['DBoW']['thresh'], 
                                                   num_repeat=self.config['Loop']['DBoW']['num_repeat'])

                if cands is not None:
                    (i, j) = cands # e.g. cands = (812, 67)
                    self.retrieval.confirm_loop(i, j)
                    self.retrieval.found.clear()
                    self.loop_list.append(cands)

                self.retrieval.save_up_to(frame_id)

        else: # DNIO v2
            self.loop_detector.run()
            self.loop_list = self.loop_detector.get_loop_list()

    def process_single_chunk(self, range_1, chunk_idx=None, range_2=None, is_loop=False):
        start_idx, end_idx = range_1
        chunk_image_paths = self.img_list[start_idx:end_idx]
        if range_2 is not None:
            start_idx, end_idx = range_2
            chunk_image_paths += self.img_list[start_idx:end_idx]

        views = self._prepare_views(chunk_image_paths)
        print(f"Loaded {len(views)} images")

        torch.cuda.empty_cache()
        outputs = self.model.infer(
            views,
            memory_efficient_inference=self.memory_efficient_inference,
            use_amp=self.use_amp,
            amp_dtype=self.amp_dtype,
            apply_mask=True,
            mask_edges=True,
            apply_confidence_mask=False,
        )
        chunk_predictions = self._prepare_chunk_predictions(outputs)
        torch.cuda.empty_cache()

        if is_loop:
            save_dir = self.result_loop_dir
            filename = f"loop_{range_1[0]}_{range_1[1]}_{range_2[0]}_{range_2[1]}.npy"
        else:
            if chunk_idx is None:
                raise ValueError("chunk_idx must be provided when is_loop is False")
            save_dir = self.result_unaligned_dir
            filename = f"chunk_{chunk_idx}.npy"

        save_path = os.path.join(save_dir, filename)

        if not is_loop and range_2 is None:
            extrinsics = chunk_predictions['camera_poses']
            chunk_range = self.chunk_indices[chunk_idx]
            self.all_camera_poses.append((chunk_range, extrinsics))

        np.save(save_path, chunk_predictions)

        return chunk_predictions if is_loop or range_2 is not None else None

    @staticmethod
    def _tensor_to_numpy(value):
        if value is None:
            return None
        if torch.is_tensor(value):
            value = value.detach().cpu().numpy()
        return np.asarray(value)

    @staticmethod
    def _as_float32(array):
        return np.ascontiguousarray(array.astype(np.float32, copy=False))

    def _resolve_path(self, path):
        if path is None:
            return None
        expanded = os.path.expanduser(path)
        if os.path.isabs(expanded):
            return expanded
        candidate = os.path.join(self.config_dir, expanded)
        if os.path.exists(candidate):
            return candidate
        return os.path.abspath(expanded)

    @staticmethod
    def _normalize_intrinsics_key(key):
        return key.replace("\\", "/")

    @staticmethod
    def _parse_intrinsics_matrix(value):
        if value is None:
            return None
        arr = np.asarray(value, dtype=np.float32)
        if arr.shape == (3, 3):
            return arr
        flat = arr.flatten()
        if flat.size == 4:
            fx, fy, cx, cy = flat
            return np.array(
                [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32
            )
        raise ValueError(
            f"Intrinsics must be 3x3 matrix or [fx, fy, cx, cy] list, got shape {arr.shape}"
        )

    def _load_intrinsics_mapping(self, path):
        resolved = self._resolve_path(path)
        if not os.path.isfile(resolved):
            warnings.warn(f"Intrinsics json_path '{path}' not found. Skipping.")
            return
        with open(resolved, 'r') as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(
                f"Intrinsics file '{resolved}' must contain a JSON object mapping image keys to matrices."
            )
        for key, value in data.items():
            matrix = self._parse_intrinsics_matrix(value)
            if matrix is None:
                continue
            if key == "_default":
                self._default_intrinsics = matrix
            else:
                normalized_key = self._normalize_intrinsics_key(key)
                self._intrinsics_map[normalized_key] = matrix

    def _lookup_intrinsics(self, image_path):
        if not self._use_intrinsics:
            return None
        candidates = [
            self._normalize_intrinsics_key(image_path),
            self._normalize_intrinsics_key(
                os.path.relpath(image_path, self.img_dir)
            ),
            os.path.basename(image_path),
        ]
        for key in candidates:
            if key in self._intrinsics_map:
                return self._intrinsics_map[key].copy()
        if self._default_intrinsics is not None:
            return self._default_intrinsics.copy()
        return None

    def _prepare_views(self, image_paths):
        raw_views = []
        for idx, path in enumerate(image_paths):
            with Image.open(path) as img:
                img_rgb = img.convert('RGB')
                raw_view = {
                    "img": np.array(img_rgb),
                    "idx": idx,
                    "instance": os.path.basename(path),
                }
            intrinsics = self._lookup_intrinsics(path)
            if intrinsics is not None:
                raw_view["intrinsics"] = intrinsics
                print(f"[Intrinsics] {os.path.basename(path)} ->\n{intrinsics}")
            raw_views.append(raw_view)

        return preprocess_inputs(
            raw_views,
            resize_mode=self.image_resize_mode,
            size=self.image_resize_size,
            norm_type=self.image_norm_type,
            patch_size=self.image_patch_size,
            resolution_set=self.image_resolution_set,
            verbose=False,
        )

    def _prepare_chunk_predictions(self, outputs):
        points, confs, images = [], [], []
        poses, intrinsics, depth_maps, masks = [], [], [], []

        for pred in outputs:
            pts = self._tensor_to_numpy(pred.get('pts3d'))
            if pts is None:
                raise ValueError("MapAnything output missing 'pts3d'.")
            points.append(self._as_float32(pts[0]))

            conf = self._tensor_to_numpy(pred.get('conf'))
            if conf is not None:
                confs.append(self._as_float32(conf[0]))
            else:
                confs.append(np.ones(points[-1].shape[:2], dtype=np.float32))

            img = self._tensor_to_numpy(pred.get('img_no_norm'))
            if img is None:
                raise ValueError("MapAnything output missing 'img_no_norm'.")
            img = self._as_float32(img[0])
            images.append(self._as_float32(np.transpose(img, (2, 0, 1))))

            pose = self._tensor_to_numpy(pred.get('camera_poses'))
            if pose is None:
                raise ValueError("MapAnything output missing 'camera_poses'.")
            poses.append(self._as_float32(pose[0]))

            intr = self._tensor_to_numpy(pred.get('intrinsics'))
            if intr is not None:
                intrinsics.append(self._as_float32(intr[0]))

            depth = self._tensor_to_numpy(pred.get('depth_z'))
            if depth is not None:
                depth_maps.append(self._as_float32(depth[0, ..., 0]))

            mask = self._tensor_to_numpy(pred.get('mask'))
            if mask is not None:
                masks.append(np.ascontiguousarray(mask[0, ..., 0] > 0.5))

        chunk_data = {
            'points': self._as_float32(np.stack(points, axis=0)),
            'conf': self._as_float32(np.stack(confs, axis=0)),
            'images': self._as_float32(np.stack(images, axis=0)),
            'camera_poses': self._as_float32(np.stack(poses, axis=0)),
        }

        if intrinsics:
            chunk_data['intrinsics'] = self._as_float32(np.stack(intrinsics, axis=0))
        if depth_maps:
            chunk_data['depth_z'] = self._as_float32(np.stack(depth_maps, axis=0))
        if masks:
            chunk_data['mask'] = np.stack(masks, axis=0)

        return chunk_data
    
    def process_long_sequence(self):
        if self.overlap >= self.chunk_size:
            raise ValueError(f"[SETTING ERROR] Overlap ({self.overlap}) must be less than chunk size ({self.chunk_size})")
        if len(self.img_list) <= self.chunk_size:
            num_chunks = 1
            self.chunk_indices = [(0, len(self.img_list))]
        else:
            step = self.chunk_size - self.overlap
            num_chunks = (len(self.img_list) - self.overlap + step - 1) // step
            self.chunk_indices = []
            for i in range(num_chunks):
                start_idx = i * step
                end_idx = min(start_idx + self.chunk_size, len(self.img_list))
                self.chunk_indices.append((start_idx, end_idx))
        
        if self.loop_enable:
            print('Loop SIM(3) estimating...')
            loop_results = process_loop_list(self.chunk_indices, 
                                             self.loop_list, 
                                             half_window = int(self.config['Model']['loop_chunk_size'] / 2))
            loop_results = remove_duplicates(loop_results)
            print(loop_results)
            # return e.g. (31, (1574, 1594), 2, (129, 149))
            for item in loop_results:
                single_chunk_predictions = self.process_single_chunk(item[1], range_2=item[3], is_loop=True)

                self.loop_predict_list.append((item, single_chunk_predictions))
                print(item)
            

        print(f"Processing {len(self.img_list)} images in {num_chunks} chunks of size {self.chunk_size} with {self.overlap} overlap")

        for chunk_idx in range(len(self.chunk_indices)):
            print(f'[Progress]: {chunk_idx}/{len(self.chunk_indices)}')
            self.process_single_chunk(self.chunk_indices[chunk_idx], chunk_idx=chunk_idx)
            torch.cuda.empty_cache()

        del self.model # Save GPU Memory
        torch.cuda.empty_cache()

        print("Aligning all the chunks...")
        for chunk_idx in range(len(self.chunk_indices)-1):

            print(f"Aligning {chunk_idx} and {chunk_idx+1} (Total {len(self.chunk_indices)-1})")
            chunk_data1 = np.load(os.path.join(self.result_unaligned_dir, f"chunk_{chunk_idx}.npy"), allow_pickle=True).item()
            chunk_data2 = np.load(os.path.join(self.result_unaligned_dir, f"chunk_{chunk_idx+1}.npy"), allow_pickle=True).item()
            
            point_map1 = chunk_data1['points'][-self.overlap:]
            point_map2 = chunk_data2['points'][:self.overlap]
            conf1 = chunk_data1['conf'][-self.overlap:]
            conf2 = chunk_data2['conf'][:self.overlap]

            conf_threshold = min(np.median(conf1), np.median(conf2)) * 0.1
            s, R, t = weighted_align_point_maps(point_map1, 
                                                conf1, 
                                                point_map2, 
                                                conf2, 
                                                conf_threshold=conf_threshold,
                                                config=self.config)
            print("Estimated Scale:", s)
            print("Estimated Rotation:\n", R)
            print("Estimated Translation:", t)

            self.sim3_list.append((s, R, t))


        if self.loop_enable:
            for item in self.loop_predict_list:
                chunk_idx_a = item[0][0]
                chunk_idx_b = item[0][2]
                chunk_a_range = item[0][1]
                chunk_b_range = item[0][3]

                print('chunk_a align')
                point_map_loop = item[1]['points'][:chunk_a_range[1] - chunk_a_range[0]]
                conf_loop = item[1]['conf'][:chunk_a_range[1] - chunk_a_range[0]]
                chunk_a_rela_begin = chunk_a_range[0] - self.chunk_indices[chunk_idx_a][0]
                chunk_a_rela_end = chunk_a_rela_begin + chunk_a_range[1] - chunk_a_range[0]
                print(self.chunk_indices[chunk_idx_a])
                print(chunk_a_range)
                print(chunk_a_rela_begin, chunk_a_rela_end)
                chunk_data_a = np.load(os.path.join(self.result_unaligned_dir, f"chunk_{chunk_idx_a}.npy"), allow_pickle=True).item()
                
                point_map_a = chunk_data_a['points'][chunk_a_rela_begin:chunk_a_rela_end]
                conf_a = chunk_data_a['conf'][chunk_a_rela_begin:chunk_a_rela_end]
            
                conf_threshold = min(np.median(conf_a), np.median(conf_loop)) * 0.1
                s_a, R_a, t_a = weighted_align_point_maps(point_map_a, 
                                                          conf_a, 
                                                          point_map_loop, 
                                                          conf_loop, 
                                                          conf_threshold=conf_threshold,
                                                          config=self.config)
                print("Estimated Scale:", s_a)
                print("Estimated Rotation:\n", R_a)
                print("Estimated Translation:", t_a)

                print('chunk_a align')
                point_map_loop = item[1]['points'][-chunk_b_range[1] + chunk_b_range[0]:]
                conf_loop = item[1]['conf'][-chunk_b_range[1] + chunk_b_range[0]:]
                chunk_b_rela_begin = chunk_b_range[0] - self.chunk_indices[chunk_idx_b][0]
                chunk_b_rela_end = chunk_b_rela_begin + chunk_b_range[1] - chunk_b_range[0]
                print(self.chunk_indices[chunk_idx_b])
                print(chunk_b_range)
                print(chunk_b_rela_begin, chunk_b_rela_end)
                chunk_data_b = np.load(os.path.join(self.result_unaligned_dir, f"chunk_{chunk_idx_b}.npy"), allow_pickle=True).item()
                
                point_map_b = chunk_data_b['points'][chunk_b_rela_begin:chunk_b_rela_end]
                conf_b = chunk_data_b['conf'][chunk_b_rela_begin:chunk_b_rela_end]
            
                conf_threshold = min(np.median(conf_b), np.median(conf_loop)) * 0.1
                s_b, R_b, t_b = weighted_align_point_maps(point_map_b, 
                                                          conf_b, 
                                                          point_map_loop, 
                                                          conf_loop, 
                                                          conf_threshold=conf_threshold,
                                                          config=self.config)
                print("Estimated Scale:", s_b)
                print("Estimated Rotation:\n", R_b)
                print("Estimated Translation:", t_b)

                print('a -> b SIM 3')
                s_ab, R_ab, t_ab = compute_sim3_ab((s_a, R_a, t_a), (s_b, R_b, t_b))
                print("Estimated Scale:", s_ab)
                print("Estimated Rotation:\n", R_ab)
                print("Estimated Translation:", t_ab)

                self.loop_sim3_list.append((chunk_idx_a, chunk_idx_b, (s_ab, R_ab, t_ab)))


        if self.loop_enable:
            input_abs_poses = self.loop_optimizer.sequential_to_absolute_poses(self.sim3_list)
            self.sim3_list = self.loop_optimizer.optimize(self.sim3_list, self.loop_sim3_list)
            optimized_abs_poses = self.loop_optimizer.sequential_to_absolute_poses(self.sim3_list)

            def extract_xyz(pose_tensor):
                poses = pose_tensor.cpu().numpy()
                return poses[:, 0], poses[:, 1], poses[:, 2]
            
            x0, _, y0 = extract_xyz(input_abs_poses)
            x1, _, y1 = extract_xyz(optimized_abs_poses)

            # Visual in png format
            plt.figure(figsize=(8, 8))
            plt.plot(x0, y0, 'o--', alpha=0.45, label='Before Optimization')
            plt.plot(x1, y1, 'o-', label='After Optimization')
            for i, j, _ in self.loop_sim3_list:
                plt.plot([x0[i], x0[j]], [y0[i], y0[j]], 'r--', alpha=0.25, label='Loop (Before)' if i == 5 else "")
                plt.plot([x1[i], x1[j]], [y1[i], y1[j]], 'g-', alpha=0.35, label='Loop (After)' if i == 5 else "")
            plt.gca().set_aspect('equal')
            plt.title("Sim3 Loop Closure Optimization")
            plt.xlabel("x")
            plt.ylabel("z")
            plt.legend()
            plt.grid(True)
            plt.axis("equal")
            save_path = os.path.join(self.output_dir, 'sim3_opt_result.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()

        print('Apply alignment')
        self.sim3_list = accumulate_sim3_transforms(self.sim3_list)
        for chunk_idx in range(len(self.chunk_indices)-1):
            print(f'Applying {chunk_idx+1} -> {chunk_idx} (Total {len(self.chunk_indices)-1})')
            s, R, t = self.sim3_list[chunk_idx]
            
            chunk_data = np.load(os.path.join(self.result_unaligned_dir, f"chunk_{chunk_idx+1}.npy"), allow_pickle=True).item()
            
            chunk_data['points'] = apply_sim3_direct(chunk_data['points'], s, R, t)
            
            aligned_path = os.path.join(self.result_aligned_dir, f"chunk_{chunk_idx+1}.npy")
            np.save(aligned_path, chunk_data)
            
            if chunk_idx == 0:
                chunk_data_first = np.load(os.path.join(self.result_unaligned_dir, f"chunk_0.npy"), allow_pickle=True).item()
                np.save(os.path.join(self.result_aligned_dir, "chunk_0.npy"), chunk_data_first)
            
            aligned_chunk_data = np.load(os.path.join(self.result_aligned_dir, f"chunk_{chunk_idx}.npy"), allow_pickle=True).item() if chunk_idx > 0 else chunk_data_first
            
            points = aligned_chunk_data['points'].reshape(-1, 3)
            colors = (aligned_chunk_data['images'].transpose(0, 2, 3, 1).reshape(-1, 3) * 255).astype(np.uint8)
            confs = aligned_chunk_data['conf'].reshape(-1)
            ply_path = os.path.join(self.pcd_dir, f'{chunk_idx}_pcd.ply')
            save_confident_pointcloud_batch(
                points=points,              # shape: (H, W, 3)
                colors=colors,              # shape: (H, W, 3)
                confs=confs,          # shape: (H, W)
                output_path=ply_path,
                conf_threshold=np.mean(confs) * self.config['Model']['Pointcloud_Save']['conf_threshold_coef'],
                sample_ratio=self.config['Model']['Pointcloud_Save']['sample_ratio']
            )

        self.save_camera_poses()
        
        print('Done.')

    
    def run(self):
        print(f"Loading images from {self.img_dir}...")
        self.img_list = sorted(glob.glob(os.path.join(self.img_dir, "*.jpg")) + 
                                glob.glob(os.path.join(self.img_dir, "*.png")))
        # print(self.img_list)
        if len(self.img_list) == 0:
            raise ValueError(f"[DIR EMPTY] No images found in {self.img_dir}!")
        print(f"Found {len(self.img_list)} images")

        if self.loop_enable:
            self.get_loop_pairs()

            if self.useDBoW:
                self.retrieval.close() # Save CPU Memory
                gc.collect()
            else:
                del self.loop_detector # Save GPU Memory
        torch.cuda.empty_cache()

        self.process_long_sequence()

    def save_camera_poses(self):
        '''
        Save camera poses from all chunks to txt and ply files
        - txt file: Each line contains a 4x4 C2W matrix flattened into 16 numbers
        - ply file: Camera poses visualized as points with different colors for each chunk
        '''
        chunk_colors = [
            [255, 0, 0],    # Red
            [0, 255, 0],    # Green
            [0, 0, 255],    # Blue
            [255, 255, 0],  # Yellow
            [255, 0, 255],  # Magenta
            [0, 255, 255],  # Cyan
            [128, 0, 0],    # Dark Red
            [0, 128, 0],    # Dark Green
            [0, 0, 128],    # Dark Blue
            [128, 128, 0],  # Olive
        ]
        print("Saving all camera poses to txt file...")
        
        all_poses = [None] * len(self.img_list)
        
        first_chunk_range, first_chunk_extrinsics = self.all_camera_poses[0]
        for i, idx in enumerate(range(first_chunk_range[0], first_chunk_range[1])):

            c2w = first_chunk_extrinsics[i] # camera pose of MapAnything is C2W while it is W2C in VGGT!
            all_poses[idx] = c2w

        for chunk_idx in range(1, len(self.all_camera_poses)):
            chunk_range, chunk_extrinsics = self.all_camera_poses[chunk_idx]
            s, R, t = self.sim3_list[chunk_idx-1]   # When call self.save_camera_poses(), all the sim3 are aligned to the first chunk.
            
            S = np.eye(4)
            S[:3, :3] = s * R
            S[:3, 3] = t

            for i, idx in enumerate(range(chunk_range[0], chunk_range[1])):

                c2w = chunk_extrinsics[i] # camera pose of MapAnything is C2W while it is W2C in VGGT!

                transformed_c2w = S @ c2w  # Be aware of the left multiplication!

                all_poses[idx] = transformed_c2w

        poses_path = os.path.join(self.output_dir, 'camera_poses.txt')
        with open(poses_path, 'w') as f:
            for pose in all_poses:
                flat_pose = pose.flatten()
                f.write(' '.join([str(x) for x in flat_pose]) + '\n')
        
        print(f"Camera poses saved to {poses_path}")
        
        ply_path = os.path.join(self.output_dir, 'camera_poses.ply')
        with open(ply_path, 'w') as f:
            # Write PLY header
            f.write('ply\n')
            f.write('format ascii 1.0\n')
            f.write(f'element vertex {len(all_poses)}\n')
            f.write('property float x\n')
            f.write('property float y\n')
            f.write('property float z\n')
            f.write('property uchar red\n')
            f.write('property uchar green\n')
            f.write('property uchar blue\n')
            f.write('end_header\n')
            
            color = chunk_colors[0]
            for pose in all_poses:
                position = pose[:3, 3]
                f.write(f'{position[0]} {position[1]} {position[2]} {color[0]} {color[1]} {color[2]}\n')
        
        print(f"Camera poses visualization saved to {ply_path}")
    
    def close(self):
        '''
            Clean up temporary files and calculate reclaimed disk space.
            
            This method deletes all temporary files generated during processing from three directories:
            - Unaligned results
            - Aligned results
            - Loop results
            
            ~50 GiB for 4500-frame KITTI 00, 
            ~35 GiB for 2700-frame KITTI 05, 
            or ~5 GiB for 300-frame short seq.
        '''
        if not self.delete_temp_files:
            return
        
        total_space = 0

        print(f'Deleting the temp files under {self.result_unaligned_dir}')
        for filename in os.listdir(self.result_unaligned_dir):
            file_path = os.path.join(self.result_unaligned_dir, filename)
            if os.path.isfile(file_path):
                total_space += os.path.getsize(file_path)
                os.remove(file_path)

        print(f'Deleting the temp files under {self.result_aligned_dir}')
        for filename in os.listdir(self.result_aligned_dir):
            file_path = os.path.join(self.result_aligned_dir, filename)
            if os.path.isfile(file_path):
                total_space += os.path.getsize(file_path)
                os.remove(file_path)

        print(f'Deleting the temp files under {self.result_loop_dir}')
        for filename in os.listdir(self.result_loop_dir):
            file_path = os.path.join(self.result_loop_dir, filename)
            if os.path.isfile(file_path):
                total_space += os.path.getsize(file_path)
                os.remove(file_path)
        print('Deleting temp files done.')

        print(f"Saved disk space: {total_space/1024/1024/1024:.4f} GiB")


import shutil
def copy_file(src_path, dst_dir):
    try:
        os.makedirs(dst_dir, exist_ok=True)
        
        dst_path = os.path.join(dst_dir, os.path.basename(src_path))
        
        shutil.copy2(src_path, dst_path)
        print(f"config yaml file has been copied to: {dst_path}")
        return dst_path
        
    except FileNotFoundError:
        print("File Not Found")
    except PermissionError:
        print("Permission Error")
    except Exception as e:
        print(f"Copy Error: {e}")

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='MapAnything-Long')
    parser.add_argument('--image_dir', type=str, required=True,
                        help='Image path')
    parser.add_argument('--config', type=str, required=False, default='./configs/base_config.yaml',
                        help='Config path')
    args = parser.parse_args()

    config = load_config(args.config)

    image_dir = args.image_dir
    path = image_dir.split("/")
    current_datetime = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    exp_dir = './exps'


    save_dir = os.path.join(
            exp_dir, image_dir.replace("/", "_"), current_datetime
        )
    
    # save_dir = os.path.join(
    #     exp_dir, path[-3] + "_" + path[-2] + "_" + path[-1], current_datetime
    # )

    if not os.path.exists(save_dir): 
        os.makedirs(save_dir)
        print(f'The exp will be saved under dir: {save_dir}')
        copy_file(args.config, save_dir)

    if config['Model']['align_method'] == 'numba':
        warmup_numba()

    mapanything_long = MapAnything_Long(image_dir, save_dir, config, config_path=args.config)
    mapanything_long.run()
    mapanything_long.close()

    del mapanything_long
    torch.cuda.empty_cache()
    gc.collect()

    all_ply_path = os.path.join(save_dir, f'pcd/combined_pcd.ply')
    input_dir = os.path.join(save_dir, f'pcd')
    print("Saving all the point clouds")
    merge_ply_files(input_dir, all_ply_path)
    print('MapAnything Long done.')
    sys.exit()
