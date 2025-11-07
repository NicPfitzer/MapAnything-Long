import logging
import os
import shutil
import subprocess
import tempfile
import warnings
from dataclasses import dataclass
from itertools import count
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from mapanything.utils.colmap import get_camera_matrix, qvec2rotmat, read_model
from mapanything.utils.geometry import closed_form_pose_inverse

logger = logging.getLogger(__name__)


@dataclass
class ColmapView:
    """Container for per-image calibration results."""

    original_path: str
    intrinsics: np.ndarray
    pose: np.ndarray
    camera_id: int
    image_id: int

    def to_serializable(self) -> Dict:
        return {
            "image_path": self.original_path,
            "camera_id": self.camera_id,
            "image_id": self.image_id,
            "intrinsics": self.intrinsics.tolist(),
            "pose": self.pose.tolist(),
        }


@dataclass
class ColmapChunkResult:
    chunk_name: str
    registered_views: Dict[str, ColmapView]
    unregistered_paths: List[str]
    workspace: str
    sparse_model_path: Optional[str]

    @property
    def num_registered(self) -> int:
        return len(self.registered_views)

    def summary(self) -> str:
        total = self.num_registered + len(self.unregistered_paths)
        return (
            f"[COLMAP] Chunk '{self.chunk_name}': {self.num_registered}/{total} images registered."
        )

    def to_serializable(self) -> Dict:
        return {
            "chunk_name": self.chunk_name,
            "workspace": self.workspace,
            "sparse_model_path": self.sparse_model_path,
            "registered": [
                view.to_serializable() for view in self.registered_views.values()
            ],
            "unregistered": self.unregistered_paths,
        }


class ColmapRunner:
    """
    Lightweight wrapper that runs the COLMAP sparse pipeline for a list of images and
    returns intrinsics/poses suitable for MapAnything inputs.
    """

    def __init__(
        self,
        workspace_root: Optional[str] = None,
        *,
        colmap_binary: str = "colmap",
        matcher: str = "sequential",
        sequential_overlap: int = 5,
        use_gpu: bool = True,
        camera_model: str = "SIMPLE_RADIAL",
        single_camera: bool = True,
        max_image_size: Optional[int] = 2048,
        min_num_matches: int = 15,
        num_threads: Optional[int] = None,
        keep_workspaces: bool = False,
        verbose: bool = False,
    ):
        self.colmap_binary = colmap_binary
        self.matcher = matcher.lower()
        if self.matcher not in {"sequential", "exhaustive"}:
            raise ValueError("matcher must be either 'sequential' or 'exhaustive'")
        self.sequential_overlap = sequential_overlap
        self.use_gpu = use_gpu
        self.camera_model = camera_model
        self.single_camera = single_camera
        self.max_image_size = max_image_size
        self.min_num_matches = min_num_matches
        self.num_threads = num_threads
        self.keep_workspaces = keep_workspaces
        self.verbose = verbose
        self._chunk_counter = count(0)
        self.workspace_root = (
            os.path.abspath(workspace_root)
            if workspace_root
            else tempfile.mkdtemp(prefix="mapanything_colmap_")
        )
        os.makedirs(self.workspace_root, exist_ok=True)
        if shutil.which(self.colmap_binary) is None:
            raise FileNotFoundError(
                f"COLMAP binary '{self.colmap_binary}' not found on PATH."
            )

    def run_for_chunk(
        self, image_paths: Sequence[str], chunk_name: Optional[str] = None
    ) -> ColmapChunkResult:
        if not image_paths:
            raise ValueError("image_paths cannot be empty.")
        abs_paths = [os.path.abspath(path) for path in image_paths]
        chunk_name = chunk_name or f"chunk_{next(self._chunk_counter):04d}"
        chunk_dir = os.path.join(self.workspace_root, chunk_name)
        self._prepare_workspace(chunk_dir)
        images_dir = os.path.join(chunk_dir, "images")
        os.makedirs(images_dir, exist_ok=True)
        mapping = self._materialize_images(abs_paths, images_dir)
        registered: Dict[str, ColmapView] = {}
        unregistered = list(abs_paths)
        sparse_model_path = None

        if len(mapping) >= 2:
            database_path = os.path.join(chunk_dir, "colmap.db")
            sparse_dir = os.path.join(chunk_dir, "sparse")
            os.makedirs(sparse_dir, exist_ok=True)
            self._run_feature_extractor(database_path, images_dir)
            self._run_matcher(database_path)
            sparse_model_path = self._run_mapper(database_path, images_dir, sparse_dir)
            registered, unregistered = self._parse_model(sparse_model_path, mapping)
        else:
            self._log(
                f"Skipping COLMAP for chunk '{chunk_name}' because it only has {len(mapping)} image(s)."
            )

        if not self.keep_workspaces:
            self._cleanup_workspace(chunk_dir)

        result = ColmapChunkResult(
            chunk_name=chunk_name,
            registered_views=registered,
            unregistered_paths=unregistered,
            workspace=chunk_dir,
            sparse_model_path=sparse_model_path,
        )
        self._log(result.summary())
        return result

    def _prepare_workspace(self, chunk_dir: str) -> None:
        if os.path.exists(chunk_dir):
            shutil.rmtree(chunk_dir)
        os.makedirs(chunk_dir, exist_ok=True)

    def _materialize_images(
        self, image_paths: Sequence[str], images_dir: str
    ) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        for idx, src in enumerate(image_paths):
            if not os.path.isfile(src):
                raise FileNotFoundError(f"Image '{src}' not found.")
            base, ext = os.path.splitext(os.path.basename(src))
            ext = ext or ".jpg"
            dest_name = f"{idx:05d}_{base}{ext}"
            dest_path = os.path.join(images_dir, dest_name)
            try:
                os.symlink(src, dest_path)
            except (AttributeError, OSError):
                shutil.copy2(src, dest_path)
            mapping[dest_name] = src
        return mapping

    def _run_feature_extractor(self, database_path: str, images_dir: str) -> None:
        cmd = [
            self.colmap_binary,
            "feature_extractor",
            "--database_path",
            database_path,
            "--image_path",
            images_dir,
            "--ImageReader.camera_model",
            self.camera_model,
            "--ImageReader.single_camera",
            "1" if self.single_camera else "0",
            "--SiftExtraction.use_gpu",
            "1" if self.use_gpu else "0",
        ]
        if self.max_image_size:
            cmd += ["--SiftExtraction.max_image_size", str(self.max_image_size)]
        self._run_command(cmd, cwd=os.path.dirname(database_path))

    def _run_matcher(self, database_path: str) -> None:
        matcher_command = (
            "sequential_matcher" if self.matcher == "sequential" else "exhaustive_matcher"
        )
        cmd = [
            self.colmap_binary,
            matcher_command,
            "--database_path",
            database_path,
            "--SiftMatching.use_gpu",
            "1" if self.use_gpu else "0",
        ]
        if self.matcher == "sequential" and self.sequential_overlap is not None:
            cmd += ["--SequentialMatching.overlap", str(self.sequential_overlap)]
        self._run_command(cmd, cwd=os.path.dirname(database_path))

    def _run_mapper(
        self, database_path: str, images_dir: str, sparse_dir: str
    ) -> str:
        cmd = [
            self.colmap_binary,
            "mapper",
            "--database_path",
            database_path,
            "--image_path",
            images_dir,
            "--output_path",
            sparse_dir,
            "--Mapper.ba_refine_focal_length",
            "0",
            "--Mapper.ba_refine_principal_point",
            "0",
            "--Mapper.min_num_matches",
            str(self.min_num_matches),
        ]
        if self.num_threads:
            cmd += ["--Mapper.num_threads", str(self.num_threads)]
        self._run_command(cmd, cwd=os.path.dirname(database_path))
        candidates = [
            os.path.join(sparse_dir, d)
            for d in sorted(os.listdir(sparse_dir))
            if os.path.isdir(os.path.join(sparse_dir, d))
        ]
        if not candidates:
            raise RuntimeError(
                "COLMAP mapper produced no sparse model. Check the logs for details."
            )
        return candidates[0]

    def _parse_model(
        self, model_path: str, mapping: Dict[str, str]
    ) -> Tuple[Dict[str, ColmapView], List[str]]:
        cameras, images_data, _ = read_model(model_path, ext=".bin")
        per_image = {}
        for img_id, img in images_data.items():
            original = mapping.get(img.name)
            if original is None:
                continue
            cam = cameras[img.camera_id]
            intrinsics, _ = get_camera_matrix(cam.params, cam.model)
            world2cam = np.eye(4, dtype=np.float64)
            world2cam[:3, :3] = qvec2rotmat(img.qvec)
            world2cam[:3, 3] = img.tvec
            cam2world = closed_form_pose_inverse(world2cam[None])[0]
            per_image[original] = ColmapView(
                original_path=original,
                intrinsics=intrinsics.astype(np.float32),
                pose=cam2world.astype(np.float32),
                camera_id=cam.id,
                image_id=img_id,
            )
        unregistered = [
            original for original in mapping.values() if original not in per_image
        ]
        return per_image, unregistered

    def _cleanup_workspace(self, chunk_dir: str) -> None:
        try:
            shutil.rmtree(chunk_dir)
        except OSError as exc:
            warnings.warn(f"Failed to remove COLMAP workspace '{chunk_dir}': {exc}")

    def _run_command(self, cmd, cwd: str) -> None:
        self._log("Running: " + " ".join(cmd))
        try:
            subprocess.run(cmd, cwd=cwd, check=True)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"COLMAP command '{' '.join(cmd)}' failed with exit code {exc.returncode}"
            ) from exc
        except FileNotFoundError as exc:
            raise RuntimeError(f"Failed to execute '{cmd[0]}': {exc}") from exc

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message)
