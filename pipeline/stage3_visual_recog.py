#!/usr/bin/env python3
"""Stage A3: Per-Shot Visual Tracking & Adaptive Annotated Keyframes (`stage3_visual_recog.py`).

1. Step 3.1 (Per-Shot TGRP Tracking):
   - Reads canonical `shots` from `shots_and_scenes.json` (where shots >20s are already subdivided).
   - Decodes frames one shot at a time in small 32-frame chunks to a temporary folder and frees the
     decoded frame buffer immediately before tracking begins.
   - Samples 3 seed frames per shot, runs `google/owlv2-large-patch14-ensemble` (with NMS `iou=0.5`
     and capped to top-5 character boxes per seed frame), prompts `SAM 2.1` (`sam2.1_hiera_l.yaml`)
     under `torch.inference_mode()` + `bfloat16` autocast for bidirectional propagation, converts
     mask logits to `[x1, y1, x2, y2]` directly on GPU (zero 2D masks in CPU RAM), and filters
     tracks via tripartite graph matching (>=2 seed sets, IoU >= 0.5).
   - Explicitly calls `inference_state.clear()` and `inference_state = None` BEFORE `gc.collect()`
     and `torch.cuda.empty_cache()` after every shot so `inference_state["images"]` never leaks RAM.
2. Step 3.2 (DINOv2 Track Classification & Global Frame Mapping):
   - Samples up to 5 frames along each track, crops the character region, extracts features with
     the fine-tuned `DINOv2` model from Stage A2, and computes top-3 average cosine similarity
     $s_{\text{vm}}^p$ against each character in the bank.
   - Retains labels with $s_{\text{vm}} > 0.45$ for visual prompting and marks $s_{\text{vm}} > 0.60$
     for Stage A4 voice anchor harvesting.
   - Writes `classified_tracks.json` and `visual_tracks_per_frame.json`.
3. Step 3.3 (Adaptive Per-Shot Sharp Keyframe Extraction with Pre-Drawn Bounding Boxes):
   - Short shots (<2.0s): selects 1 sharpest frame via OpenCV Variance of the Laplacian.
   - Longer shots (>=2.0s): selects 1 sharpest frame per ~2.0s interval.
   - Draws color-coded bounding boxes (`red`, `green`, `blue`, `yellow`, `pink`, `cyan`) for all
     recognized characters ($s_{\text{vm}} > 0.45$) using a consistent per-scene color assignment.
   - Saves annotated keyframes to `workdir/{movie_id}/keyframes/{scene_id}_shot{shot_idx}_{timestamp:06.2f}s.jpg`
     and updates `shots_and_scenes.json` with `characters_present` and `keyframes`.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import ctypes
import gc
import json
import os
import random
import re
import shutil
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, Set, Tuple, Union

if TYPE_CHECKING:
    import numpy as np
    import PIL.Image as Image
    import torch

# Ensure project root and Animated_AD paths are importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline import ensure_animated_ad_paths, patch_owlv2_processor

ensure_animated_ad_paths()

SCENE_COLOR_PALETTE: List[Tuple[str, List[int]]] = [
    ("red", [255, 0, 0]),
    ("green", [0, 255, 0]),
    ("blue", [0, 0, 255]),
    ("yellow", [255, 255, 0]),
    ("pink", [255, 0, 255]),
    ("cyan", [0, 255, 255]),
]

NPZ_NAME_RE = re.compile(r"^(?P<char>.+)_(?P<idx>\d+)\.npz$", re.IGNORECASE)


def load_character_feature_bank(char_features_dir: str) -> Dict[str, np.ndarray]:
    """Load character feature bank `.npz` files into `{char_name: np.ndarray(N, D)}`."""
    import numpy as np

    if not os.path.isdir(char_features_dir):
        raise FileNotFoundError(f"Character feature bank directory does not exist: {char_features_dir}")

    bank: Dict[str, List[np.ndarray]] = defaultdict(list)
    for fname in sorted(os.listdir(char_features_dir)):
        if not fname.lower().endswith(".npz"):
            continue
        fpath = os.path.join(char_features_dir, fname)
        with np.load(fpath, allow_pickle=True) as archive:
            feat = archive["feature"].copy()
            if "character" in archive:
                char_name = str(archive["character"])
            else:
                m = NPZ_NAME_RE.match(fname)
                char_name = m.group("char") if m else os.path.splitext(fname)[0].split("_")[0]
        if feat.ndim != 1:
            feat = feat.reshape(-1)
        bank[char_name].append(feat)

    if not bank:
        raise ValueError(f"No .npz character features found in {char_features_dir}")

    return {char_name: np.stack(feats) for char_name, feats in bank.items()}


def cosine_sim(vec1: np.ndarray, vec2: np.ndarray) -> float:
    import numpy as np

    denom = float(np.linalg.norm(vec1) * np.linalg.norm(vec2))
    if denom == 0.0:
        return 0.0
    return float(np.dot(vec1, vec2) / denom)


def _trim_host_memory(empty_cuda: bool = False) -> None:
    """Release Python GC objects, trim glibc C++ arenas, and optionally clear CUDA cache."""
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass
    if empty_cuda:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


@contextlib.contextmanager
def _local_video_context(video_path: str) -> Iterator[str]:
    """Copy Drive-backed videos to Colab Local SSD once to eliminate FUSE read/seek overhead."""
    if not os.path.isfile(video_path):
        yield video_path
        return

    abs_video = os.path.abspath(video_path)
    if os.path.isdir("/content") and abs_video.startswith("/content/drive"):
        scratch_dir = "/content/tgrp_scratch"
        os.makedirs(scratch_dir, exist_ok=True)
        local_copy = os.path.join(scratch_dir, f"local_{os.path.basename(abs_video)}")
        src_size = os.path.getsize(abs_video)
        if os.path.isfile(local_copy) and os.path.getsize(local_copy) == src_size:
            yield local_copy
            return

        free_bytes = shutil.disk_usage("/content").free
        if free_bytes > (src_size * 2 + 512 * 1024 * 1024):
            print(
                f"[Stage A3] Copying video ({src_size / (1024**2):.1f} MB) from Drive -> Local SSD "
                f"for fast seek/decode: {local_copy}"
            )
            shutil.copy2(abs_video, local_copy)
            try:
                yield local_copy
            finally:
                if os.path.isfile(local_copy):
                    os.remove(local_copy)
            return

    yield video_path


def mask_logits_to_bbox_gpu(mask_logit_tensor: torch.Tensor) -> Optional[List[int]]:
    """Convert a single SAM 2.1 mask logit tensor `(1, H, W)` or `(H, W)` to `[x1, y1, x2, y2]` on GPU.

    Avoids copying full-resolution 2D boolean masks to CPU RAM across hundreds of frames.
    """
    import torch

    mask_2d = mask_logit_tensor.squeeze() > 0.0
    ys, xs = torch.where(mask_2d)
    if ys.numel() == 0:
        return None
    x_min = int(xs.min().item())
    y_min = int(ys.min().item())
    x_max = int(xs.max().item())
    y_max = int(ys.max().item())
    if (x_max - x_min) * (y_max - y_min) <= 0:
        return None
    return [x_min, y_min, x_max, y_max]


def convert_bbox_to_rectangle_mask(
    bbox: List[float],
    canvas_width: int,
    canvas_height: int,
    line_width: int = 6,
) -> np.ndarray:
    """Create binary rectangle outline mask matching Animated_AD's visual prompting style."""
    import numpy as np

    x0, y0, x1, y1 = bbox
    delta_x = (x1 - x0) * 0.05
    delta_y = (y1 - y0) * 0.05

    x0_i = int(max(x0 - delta_x, 0))
    y0_i = int(max(y0 - delta_y, 0))
    x1_i = int(min(x1 + delta_x, canvas_width))
    y1_i = int(min(y1 + delta_y, canvas_height))

    lw = max(2, int(line_width))
    canvas = np.zeros((canvas_height, canvas_width), dtype=np.float32)
    if x1_i <= x0_i or y1_i <= y0_i:
        return canvas

    canvas[y0_i : min(y0_i + lw, canvas_height), x0_i:x1_i] = 1.0
    canvas[max(0, y1_i - lw) : y1_i, x0_i:x1_i] = 1.0
    canvas[y0_i:y1_i, x0_i : min(x0_i + lw, canvas_width)] = 1.0
    canvas[y0_i:y1_i, max(0, x1_i - lw) : x1_i] = 1.0
    return canvas


def overlay_character_boxes_on_frame(
    frame_rgb: np.ndarray,
    char_boxes: List[Tuple[str, List[float], str, List[int]]],
    label_width_ratio: float = 6.0,
    label_alpha: float = 0.85,
) -> np.ndarray:
    """Draw color-coded bounding boxes onto an RGB frame array `(H, W, 3)`."""
    import numpy as np

    if not char_boxes:
        return frame_rgb

    h, w = frame_rgb.shape[:2]
    line_width = max(2, int(round(label_width_ratio / 1000.0 * w)))

    label_masks: Optional[np.ndarray] = None
    total_masks: Optional[np.ndarray] = None

    for _, bbox, _, rgb_color in char_boxes:
        mask = convert_bbox_to_rectangle_mask(bbox, canvas_width=w, canvas_height=h, line_width=line_width)
        color_arr = np.array(rgb_color, dtype=np.float32)[None, None, :]
        if label_masks is None:
            label_masks = mask[:, :, None] * color_arr
            total_masks = mask
        else:
            label_masks = label_masks * (1.0 - mask[:, :, None]) + mask[:, :, None] * color_arr
            total_masks = np.clip(total_masks + mask, 0.0, 1.0)

    if total_masks is None or label_masks is None:
        return frame_rgb

    blended = (
        frame_rgb.astype(np.float32) * (1.0 - total_masks[:, :, None] * label_alpha)
        + total_masks[:, :, None] * label_alpha * label_masks
    )
    return np.clip(blended, 0, 255).astype(np.uint8)


def run_per_shot_tgrp(
    video_path: str,
    movie_id: str,
    shots: List[Dict[str, Any]],
    tgrp_jsonl_path: str,
    sam2_checkpoint: str,
    max_boxes_per_seed: int = 5,
    min_box_area_ratio: float = 0.005,
    restart: bool = False,
) -> str:
    """Run OWLv2 + SAM 2.1 bidirectional tracking + tripartite matching per shot with JSONL resume."""
    import cv2
    import numpy as np
    import PIL.Image as Image
    import torch
    from sam2.build_sam import build_sam2_video_predictor
    from tgrp import (
        box_area,
        load_completed_shots,
        merge_tracks,
        non_max_suppression,
    )
    from transformers import Owlv2ForObjectDetection, Owlv2Processor

    try:
        import psutil

        proc = psutil.Process(os.getpid())
    except Exception:
        proc = None

    @contextlib.contextmanager
    def _silence_sam2_tqdm() -> Iterator[None]:
        prev_tqdm_disable = os.environ.get("TQDM_DISABLE")
        os.environ["TQDM_DISABLE"] = "1"
        patched: List[Tuple[Any, str, Any]] = []
        noop_tqdm = lambda iterable=None, *a, **kw: iterable if iterable is not None else []
        for mod_name in ("sam2.utils.misc", "sam2.sam2_video_predictor"):
            mod = sys.modules.get(mod_name)
            if mod is not None and hasattr(mod, "tqdm"):
                patched.append((mod, "tqdm", getattr(mod, "tqdm")))
                setattr(mod, "tqdm", noop_tqdm)
        try:
            yield
        finally:
            for mod, attr, orig in patched:
                setattr(mod, attr, orig)
            if prev_tqdm_disable is None:
                os.environ.pop("TQDM_DISABLE", None)
            else:
                os.environ["TQDM_DISABLE"] = prev_tqdm_disable

    ensure_animated_ad_paths()
    patch_owlv2_processor()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(os.path.dirname(os.path.abspath(tgrp_jsonl_path)), exist_ok=True)

    if restart and os.path.exists(tgrp_jsonl_path):
        os.remove(tgrp_jsonl_path)

    completed_map = load_completed_shots(tgrp_jsonl_path)
    done_shot_ids: Set[int] = completed_map.get(movie_id, set())

    pending_shots = [s for s in shots if int(s["shot_idx"]) not in done_shot_ids]
    if not pending_shots:
        print(f"[Stage A3.1] All {len(shots)} shots already present in {tgrp_jsonl_path}; skipping TGRP.")
        return tgrp_jsonl_path

    if done_shot_ids:
        print(f"[Stage A3.1] Resuming TGRP from {tgrp_jsonl_path}: {len(done_shot_ids)}/{len(shots)} shots done.")
    else:
        print(f"[Stage A3.1] Starting TGRP across {len(shots)} canonical shots...")

    owlv2_source = os.environ.get("OWLV2_MODEL_PATH", "google/owlv2-large-patch14-ensemble")
    if not os.path.isdir(owlv2_source):
        owlv2_source = "google/owlv2-large-patch14-ensemble"

    print(f"[Stage A3.1] Initializing SAM 2.1 VideoPredictor & OWLv2 ({owlv2_source})...")
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
    processor = Owlv2Processor.from_pretrained(owlv2_source)
    owlv2_model = Owlv2ForObjectDetection.from_pretrained(owlv2_source).to(device)
    owlv2_model.eval()
    texts = [["a photo of an animated character"]]

    with contextlib.ExitStack() as stack:
        stack.enter_context(_silence_sam2_tqdm())
        active_video_path = stack.enter_context(_local_video_context(video_path))
        cap_probe = cv2.VideoCapture(active_video_path)
        if not cap_probe.isOpened():
            raise RuntimeError(f"Failed to open video for Stage A3.1: {active_video_path}")
        total_frames = int(cap_probe.get(cv2.CAP_PROP_FRAME_COUNT))
        cap_probe.release()
        _trim_host_memory()

        oom_shot_ids: List[int] = []
        completed_count = len(done_shot_ids)

        use_autocast = device == "cuda"
        autocast_dtype = torch.bfloat16 if (use_autocast and torch.cuda.is_bf16_supported()) else torch.float16

        # Use a local disk scratch base outside RAM-backed /tmp if /content exists
        scratch_base = "/content/tgrp_scratch" if os.path.isdir("/content") else None
        if scratch_base:
            os.makedirs(scratch_base, exist_ok=True)

        with open(tgrp_jsonl_path, "a", encoding="utf-8") as out_f:
            for shot in pending_shots:
                shot_id = int(shot["shot_idx"])
                start_frame = max(0, min(int(shot["start_frame"]), total_frames))
                end_frame = max(0, min(int(shot["end_frame"]), total_frames))
                if start_frame >= end_frame:
                    continue

                shot_t0 = time.perf_counter()
                shot_frames_n = 0
                shot_tracks_n = 0
                shot_succeeded = False
                inference_state = None
                with tempfile.TemporaryDirectory(prefix=f"tgrp_shot_{shot_id}_", dir=scratch_base) as shot_dir:
                    try:
                        # Open & release cv2.VideoCapture per shot so FFmpeg buffers never accumulate (~26 GB leak in decord)
                        cap = cv2.VideoCapture(active_video_path)
                        if not cap.isOpened():
                            raise RuntimeError(f"Failed to open video for shot {shot_id}: {active_video_path}")
                        try:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                            for global_f_idx in range(start_frame, end_frame):
                                ok, frame_bgr = cap.read()
                                if not ok:
                                    break
                                frame_path = os.path.join(shot_dir, f"{global_f_idx}.jpg")
                                cv2.imwrite(frame_path, frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                        finally:
                            cap.release()
                            _trim_host_memory()

                        frame_names = sorted(
                            [p for p in os.listdir(shot_dir) if p.lower().endswith(".jpg")],
                            key=lambda p: int(os.path.splitext(p)[0]),
                        )
                        n = len(frame_names)
                        shot_frames_n = n
                        if n == 0:
                            continue

                        save_shot_results: Dict[str, Any] = {
                            "video_idx": movie_id,
                            "shot_idx": shot_id,
                            "start_idx": int(start_frame),
                            "end_idx": int(end_frame - 1),
                        }

                        rng = random.Random(f"{movie_id}:{shot_id}")
                        if n < 3:
                            key_frame_idx = rng.sample(range(n), n)
                        else:
                            division_size = n // 3
                            key_frame_idx = [
                                rng.choice(range(i * division_size, (i + 1) * division_size))
                                for i in range(3)
                            ]

                        key_frame_detection_results: Dict[str, Any] = {}
                        for local_f_idx in key_frame_idx:
                            img_path = os.path.join(shot_dir, frame_names[local_f_idx])
                            image = Image.open(img_path).convert("RGB")
                            img_w, img_h = image.size
                            min_box_area = min_box_area_ratio * float(img_w * img_h)

                            inputs = processor(text=texts, images=image, return_tensors="pt")
                            inputs = {k: v.to(device) for k, v in inputs.items()}
                            with torch.inference_mode():
                                outputs = owlv2_model(**inputs)

                            target_sizes = torch.Tensor([image.size[::-1]])
                            detections = processor.post_process_object_detection(
                                outputs=outputs, threshold=0.15, target_sizes=target_sizes
                            )
                            input_boxes = detections[0]["boxes"].detach().cpu().numpy()
                            bbox_scores = detections[0]["scores"].detach().cpu().numpy()

                            # Fallback to threshold 0.10 only if nothing cleared 0.15
                            if input_boxes.shape[0] == 0:
                                detections = processor.post_process_object_detection(
                                    outputs=outputs, threshold=0.10, target_sizes=target_sizes
                                )
                                input_boxes = detections[0]["boxes"].detach().cpu().numpy()
                                bbox_scores = detections[0]["scores"].detach().cpu().numpy()

                            del inputs, outputs, detections
                            if input_boxes.shape[0] == 0:
                                continue

                            # Filter tiny noise detections before NMS
                            valid_mask = np.array([box_area(b) >= min_box_area for b in input_boxes], dtype=bool)
                            input_boxes = input_boxes[valid_mask]
                            bbox_scores = bbox_scores[valid_mask]
                            if input_boxes.shape[0] == 0:
                                continue

                            selected_bboxes, selected_scores = non_max_suppression(
                                input_boxes, bbox_scores, iou_threshold=0.5
                            )
                            # Cap to top-K highest scoring character boxes per seed frame
                            if len(selected_bboxes) > max_boxes_per_seed:
                                selected_bboxes = selected_bboxes[:max_boxes_per_seed]
                                selected_scores = selected_scores[:max_boxes_per_seed]

                            key_frame_detection_results[str(local_f_idx)] = {
                                "bbox": selected_bboxes,
                                "bbox_scores": selected_scores,
                            }

                        track_results_per_shot: Dict[str, Any] = {}
                        if key_frame_detection_results:
                            with torch.inference_mode(), torch.autocast(
                                device_type="cuda" if use_autocast else "cpu",
                                dtype=autocast_dtype,
                                enabled=use_autocast,
                            ):
                                # offload_video_to_cpu=True + offload_state_to_cpu=False keeps VRAM low,
                                # and we explicitly clear inference_state["images"] + trim heap in finally!
                                inference_state = video_predictor.init_state(
                                    video_path=shot_dir,
                                    offload_video_to_cpu=True,
                                )

                                for frame_idx_str, anns in key_frame_detection_results.items():
                                    ann_frame_idx = int(frame_idx_str)
                                    boxes = anns["bbox"]
                                    if not boxes:
                                        continue

                                    for obj_i, box in enumerate(boxes):
                                        video_predictor.add_new_points_or_box(
                                            inference_state=inference_state,
                                            frame_idx=ann_frame_idx,
                                            obj_id=obj_i,
                                            box=box,
                                        )

                                    # Convert mask logits -> [x1, y1, x2, y2] immediately on GPU!
                                    video_segments: Dict[int, Dict[int, List[int]]] = {}
                                    for out_f_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(
                                        inference_state
                                    ):
                                        frame_boxes: Dict[int, List[int]] = {}
                                        for i, out_obj_id in enumerate(out_obj_ids):
                                            bbox = mask_logits_to_bbox_gpu(out_mask_logits[i])
                                            if bbox is not None:
                                                frame_boxes[int(out_obj_id)] = bbox
                                        video_segments[int(out_f_idx)] = frame_boxes

                                    for out_f_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(
                                        inference_state, reverse=True
                                    ):
                                        if int(out_f_idx) in video_segments and video_segments[int(out_f_idx)]:
                                            continue
                                        frame_boxes = {}
                                        for i, out_obj_id in enumerate(out_obj_ids):
                                            bbox = mask_logits_to_bbox_gpu(out_mask_logits[i])
                                            if bbox is not None:
                                                frame_boxes[int(out_obj_id)] = bbox
                                        video_segments[int(out_f_idx)] = frame_boxes

                                    video_predictor.reset_state(inference_state)
                                    track_results_per_shot[frame_idx_str] = dict(sorted(video_segments.items()))

                            track_results = merge_tracks(track_results_per_shot) if track_results_per_shot else []
                        else:
                            track_results = []

                        save_shot_results["tracks"] = track_results
                        out_f.write(json.dumps(save_shot_results) + "\n")
                        out_f.flush()
                        shot_tracks_n = len(track_results)
                        shot_succeeded = True
                        completed_count += 1

                    except torch.cuda.OutOfMemoryError as exc:
                        oom_shot_ids.append(shot_id)
                        print(f"[Stage A3.1] WARNING: CUDA OOM on shot {shot_id} ({exc}); leaving for retry.")
                    finally:
                        # Crucial: clear inference_state dict (including 'images' tensor) BEFORE gc.collect()!
                        if inference_state is not None:
                            try:
                                video_predictor.reset_state(inference_state)
                            except Exception:
                                pass
                            if isinstance(inference_state, dict):
                                inference_state.clear()
                        inference_state = None
                        _trim_host_memory(empty_cuda=True)

                if shot_succeeded:
                    elapsed = time.perf_counter() - shot_t0
                    ram_str = f"{proc.memory_info().rss / (1024 ** 3):.2f} GB" if proc is not None else "n/a"
                    print(
                        f"[Shot {shot_id:03d} ({completed_count}/{len(shots)})] "
                        f"{shot_frames_n} frames | {shot_tracks_n} tracks | {elapsed:.1f}s | RAM: {ram_str}",
                        flush=True,
                    )

    del video_predictor, owlv2_model, processor
    _trim_host_memory(empty_cuda=True)

    if oom_shot_ids:
        print(f"[Stage A3.1] Skipped {len(oom_shot_ids)} shot(s) due to CUDA OOM: {oom_shot_ids}")
    else:
        print(f"[Stage A3.1] Completed TGRP tracking across all shots -> {tgrp_jsonl_path}")

    return tgrp_jsonl_path


def classify_tracks_and_map_frames(
    video_path: str,
    movie_id: str,
    tgrp_jsonl_path: str,
    ckpt_weights_path: str,
    char_features_dir: str,
    output_dir: str,
    model_size: str = "giant",
    use_mask: bool = False,
    sam2_checkpoint: Optional[str] = None,
    visual_prompt_thresh: float = 0.45,
    voice_harvest_thresh: float = 0.60,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Classify TGRP tracks with fine-tuned DINOv2 and map to global movie frames."""
    import numpy as np
    import PIL.Image as Image
    import torch
    from classification import crop_object, load_jsonl
    from decord import VideoReader, cpu
    from torchvision import transforms
    from tqdm import tqdm

    ensure_animated_ad_paths()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    character_bank = load_character_feature_bank(char_features_dir)
    print(f"[Stage A3.2] Loaded character feature bank with {len(character_bank)} characters: {list(character_bank)}")

    predictor = None
    if use_mask:
        if not sam2_checkpoint or not os.path.isfile(sam2_checkpoint):
            raise FileNotFoundError(f"--sam2_checkpoint required when --mask_classify is enabled: {sam2_checkpoint}")
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        sam2_model = build_sam2("configs/sam2.1/sam2.1_hiera_l.yaml", sam2_checkpoint, device=device)
        predictor = SAM2ImagePredictor(sam2_model)

    model_archs = {
        "small": "vits14",
        "base": "vitb14",
        "large": "vitl14",
        "giant": "vitg14",
    }
    model_name = f"dinov2_{model_archs[model_size]}_reg"
    print(f"[Stage A3.2] Loading fine-tuned {model_name} weights from {ckpt_weights_path}...")
    model = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model=model_name, pretrained=False, verbose=False)
    state_dict = torch.load(ckpt_weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    del state_dict
    model.eval()
    model.to(device)

    transform = transforms.Compose(
        [
            transforms.Resize(256, interpolation=3),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    unclassified_shots = load_jsonl(tgrp_jsonl_path)
    unclassified_shots.sort(key=lambda r: int(r["shot_idx"]))

    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)
    fps = float(vr.get_avg_fps()) or 24.0

    classified_shot_records: List[Dict[str, Any]] = []
    frames_predictions: Dict[str, Dict[str, List[Any]]] = {}

    total_tracks_count = 0
    vp_valid_tracks_count = 0
    voice_valid_tracks_count = 0

    k = 5
    for shot_record in tqdm(unclassified_shots, desc="DINOv2 Track Classification"):
        shot_idx = int(shot_record["shot_idx"])
        shot_start_idx = int(shot_record["start_idx"])
        classified_tracks: List[Dict[str, Any]] = []

        for unclassified_track in shot_record.get("tracks", []):
            frame_bbox_dict: Dict[str, List[float]] = unclassified_track.get("track", {})
            if not frame_bbox_dict:
                continue

            total_tracks_count += 1
            if total_tracks_count > 1 and total_tracks_count % 20 == 0:
                del vr
                _trim_host_memory()
                vr = VideoReader(video_path, ctx=cpu(0))

            frame_idx_ls = list(frame_bbox_dict.keys())
            if k < len(frame_idx_ls):
                rng = random.Random(f"{movie_id}:{shot_idx}:{unclassified_track.get('track_id', '')}")
                selected_local_frames = rng.sample(frame_idx_ls, k)
            else:
                selected_local_frames = frame_idx_ls

            valid_pairs: List[Tuple[str, int]] = []
            for local_f_str in selected_local_frames:
                global_f_idx = shot_start_idx + int(local_f_str)
                if 0 <= global_f_idx < total_frames:
                    valid_pairs.append((local_f_str, global_f_idx))

            if not valid_pairs:
                continue

            decoded_frames = vr.get_batch([gf for _, gf in valid_pairs]).asnumpy()
            crop_tensors = []

            for (local_f_str, _), decoded_rgb in zip(valid_pairs, decoded_frames):
                bbox = frame_bbox_dict[local_f_str]
                pil_img = Image.fromarray(decoded_rgb).convert("RGB")
                mask = None
                if use_mask and predictor is not None:
                    predictor.set_image(pil_img)
                    masks, _, _ = predictor.predict(
                        point_coords=None,
                        point_labels=None,
                        box=bbox,
                        multimask_output=False,
                    )
                    mask = np.squeeze(masks[0])
                cropped_pil = crop_object(np.array(pil_img), bbox, mask)
                crop_tensors.append(transform(cropped_pil))

            del decoded_frames
            batch_tensor = torch.stack(crop_tensors).to(device)
            with torch.inference_mode():
                query_features = model(batch_tensor).clone().detach().cpu().numpy()
                if query_features.ndim == 1:
                    query_features = query_features[None, :]

            char_best_scores: Dict[str, float] = {}
            for q_feat in query_features:
                for char_name, bank_feats in character_bank.items():
                    sims = [cosine_sim(q_feat, b_feat) for b_feat in bank_feats]
                    sims.sort(reverse=True)
                    top_m = min(3, len(sims))
                    score = sum(sims[:top_m]) / float(top_m)
                    if char_name not in char_best_scores or score > char_best_scores[char_name]:
                        char_best_scores[char_name] = score

            best_char = max(char_best_scores.items(), key=lambda item: item[1])[0]
            max_score = round(float(char_best_scores[best_char]), 4)
            is_vp_valid = max_score > visual_prompt_thresh
            is_voice_valid = max_score > voice_harvest_thresh

            if is_vp_valid:
                vp_valid_tracks_count += 1
            if is_voice_valid:
                voice_valid_tracks_count += 1

            classified_track = copy.deepcopy(unclassified_track)
            classified_track["label"] = best_char
            classified_track["score"] = max_score
            classified_track["is_visual_prompt_valid"] = is_vp_valid
            classified_track["is_voice_anchor_valid"] = is_voice_valid
            classified_tracks.append(classified_track)

            for local_f_str, bbox in frame_bbox_dict.items():
                global_f_idx = shot_start_idx + int(local_f_str)
                if 0 <= global_f_idx < total_frames:
                    gf_key = str(global_f_idx)
                    entry = frames_predictions.setdefault(
                        gf_key, {"bbox_ls": [], "labels": [], "scores": []}
                    )
                    entry["bbox_ls"].append([int(round(c)) for c in bbox])
                    entry["labels"].append(best_char)
                    entry["scores"].append(max_score)

        shot_out = copy.deepcopy(shot_record)
        shot_out["movie_title"] = movie_id
        shot_out["clip_idx"] = movie_id
        shot_out["tracks"] = classified_tracks
        classified_shot_records.append(shot_out)

    del model, predictor, vr
    _trim_host_memory(empty_cuda=True)

    os.makedirs(output_dir, exist_ok=True)
    classified_tracks_path = os.path.join(output_dir, "classified_tracks.json")
    with open(classified_tracks_path, "w", encoding="utf-8") as f:
        json.dump(classified_shot_records, f, indent=2)

    visual_tracks_payload: Dict[str, Any] = {
        "movie_id": movie_id,
        "total_frames": total_frames,
        "fps": round(fps, 4),
        "visual_prompt_thresh": visual_prompt_thresh,
        "voice_harvest_thresh": voice_harvest_thresh,
        "stats": {
            "total_tracks": total_tracks_count,
            "visual_prompt_tracks": vp_valid_tracks_count,
            "voice_anchor_tracks": voice_valid_tracks_count,
            "frames_with_detections": len(frames_predictions),
        },
        "frames": frames_predictions,
    }
    visual_tracks_path = os.path.join(output_dir, "visual_tracks_per_frame.json")
    with open(visual_tracks_path, "w", encoding="utf-8") as f:
        json.dump(visual_tracks_payload, f)

    print(
        f"[Stage A3.2] Classified {total_tracks_count} tracks "
        f"({vp_valid_tracks_count} > {visual_prompt_thresh} for VLM prompting, "
        f"{voice_valid_tracks_count} > {voice_harvest_thresh} for voice harvesting)."
    )
    print(f"[Stage A3.2] Saved frame-level visual tracks to: {visual_tracks_path}")

    return classified_shot_records, visual_tracks_payload


def extract_adaptive_annotated_keyframes(
    video_path: str,
    shots_and_scenes: Dict[str, Any],
    frames_predictions: Dict[str, Dict[str, List[Any]]],
    output_dir: str,
    visual_prompt_thresh: float = 0.45,
    short_shot_sec: float = 1.0,
    interval_sec: float = 1.0,
    max_keyframes_per_scene: int = 30,
    min_sharpness_floor: float = 10.0,
    pinned_timestamps: Optional[Dict[str, List[float]]] = None,
) -> Dict[str, Any]:
    """Extract adaptive sharp + motion-aware per-shot keyframes (up to 30/scene) and draw character boxes."""
    import cv2
    import numpy as np
    from decord import VideoReader, cpu
    from tqdm import tqdm

    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)
    fps = float(shots_and_scenes.get("fps") or vr.get_avg_fps() or 24.0)

    keyframes_dir = os.path.join(output_dir, "keyframes")
    os.makedirs(keyframes_dir, exist_ok=True)

    total_keyframes_saved = 0
    pinned_map = pinned_timestamps or {}

    def _find_shot_idx_for_frame(scene_shots: List[Dict[str, Any]], f_idx: int) -> int:
        for sh in scene_shots:
            if int(sh["start_frame"]) <= f_idx < int(sh["end_frame"]):
                return int(sh["shot_idx"])
        return int(scene_shots[-1]["shot_idx"]) if scene_shots else 0

    for scene_counter, scene in enumerate(tqdm(shots_and_scenes.get("scenes", []), desc="Adaptive Annotated Keyframes")):
        if scene_counter > 0 and scene_counter % 5 == 0:
            del vr
            _trim_host_memory()
            vr = VideoReader(video_path, ctx=cpu(0))
        scene_id = scene["scene_id"]
        if scene.get("is_credits", False):
            scene["characters_present"] = []
            scene["keyframes"] = []
            continue

        # Remove any stale keyframe files for this scene before writing fresh ones
        for existing_fname in os.listdir(keyframes_dir):
            if existing_fname.startswith(f"{scene_id}_") and existing_fname.endswith(".jpg"):
                try:
                    os.remove(os.path.join(keyframes_dir, existing_fname))
                except OSError:
                    pass

        scene_start_f = max(0, min(int(scene["start_frame"]), total_frames))
        scene_end_f = max(0, min(int(scene["end_frame"]), total_frames))
        scene_total_frames = max(1, scene_end_f - scene_start_f)

        char_max_score: Dict[str, float] = {}
        char_frame_counts: Dict[str, int] = defaultdict(int)

        for f_idx in range(scene_start_f, scene_end_f):
            pred = frames_predictions.get(str(f_idx))
            if not pred:
                continue
            seen_in_frame: Set[str] = set()
            for label, score in zip(pred["labels"], pred["scores"]):
                if float(score) <= visual_prompt_thresh:
                    continue
                if label not in char_max_score or float(score) > char_max_score[label]:
                    char_max_score[label] = float(score)
                seen_in_frame.add(label)
            for label in seen_in_frame:
                char_frame_counts[label] += 1

        sorted_chars = sorted(
            char_max_score.keys(),
            key=lambda c: (char_frame_counts[c], char_max_score[c]),
            reverse=True,
        )

        char_to_color: Dict[str, Tuple[str, List[int]]] = {}
        characters_present_list: List[Dict[str, Any]] = []

        for idx, char_name in enumerate(sorted_chars):
            color_name, rgb_val = SCENE_COLOR_PALETTE[idx % len(SCENE_COLOR_PALETTE)]
            char_to_color[char_name] = (color_name, rgb_val)
            characters_present_list.append(
                {
                    "name": char_name,
                    "color_prompt": color_name,
                    "max_score": round(char_max_score[char_name], 4),
                    "screen_time_ratio": round(char_frame_counts[char_name] / float(scene_total_frames), 4),
                }
            )

        scene["characters_present"] = characters_present_list
        raw_candidates: List[Dict[str, Any]] = []
        scene_shots = scene.get("shot_ranges", [])

        for shot in scene_shots:
            shot_idx = int(shot["shot_idx"])
            s_frame = max(0, min(int(shot["start_frame"]), total_frames))
            e_frame = max(0, min(int(shot["end_frame"]), total_frames))
            if s_frame >= e_frame:
                continue

            shot_dur = (e_frame - s_frame) / fps
            if shot_dur < short_shot_sec:
                intervals = [(s_frame, e_frame)]
            else:
                num_intervals = max(1, int(round(shot_dur / interval_sec)))
                step = (e_frame - s_frame) / float(num_intervals)
                intervals = []
                for i in range(num_intervals):
                    int_s = int(round(s_frame + i * step))
                    int_e = int(round(s_frame + (i + 1) * step)) if i < num_intervals - 1 else e_frame
                    if int_s < int_e:
                        intervals.append((int_s, int_e))

            for int_s, int_e in intervals:
                full_range = list(range(int_s, int_e))
                if not full_range:
                    continue

                # Evenly sample up to 12 frames across the full interval PLUS up to 4 frames with detections
                if len(full_range) > 12:
                    even_idxs = np.linspace(0, len(full_range) - 1, 12, dtype=int)
                    sampled_set = {full_range[i] for i in even_idxs}
                else:
                    sampled_set = set(full_range)

                frames_with_chars = [
                    f_i
                    for f_i in full_range
                    if any(
                        float(sc) > visual_prompt_thresh
                        for sc in frames_predictions.get(str(f_i), {}).get("scores", [])
                    )
                ]
                if frames_with_chars:
                    char_idxs = np.linspace(0, len(frames_with_chars) - 1, min(4, len(frames_with_chars)), dtype=int)
                    for ci in char_idxs:
                        sampled_set.add(frames_with_chars[ci])

                candidate_pool = sorted(sampled_set)
                decoded_candidates = vr.get_batch(candidate_pool).asnumpy()

                cand_metrics: List[Dict[str, Any]] = []
                prev_gray: Optional[np.ndarray] = None

                for cand_f_idx, cand_rgb in zip(candidate_pool, decoded_candidates):
                    gray = cv2.cvtColor(cand_rgb, cv2.COLOR_RGB2GRAY)
                    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                    brightness = float(gray.mean())
                    if prev_gray is not None:
                        motion_delta = float(np.mean(np.abs(gray.astype(np.float32) - prev_gray.astype(np.float32))))
                    else:
                        motion_delta = 0.0
                    prev_gray = gray

                    f_scores = [
                        float(sc)
                        for sc in frames_predictions.get(str(cand_f_idx), {}).get("scores", [])
                        if float(sc) > visual_prompt_thresh
                    ]
                    max_char_sc = max(f_scores) if f_scores else 0.0

                    cand_metrics.append(
                        {
                            "frame_idx": int(cand_f_idx),
                            "rgb": cand_rgb,
                            "sharpness": sharpness,
                            "brightness": brightness,
                            "motion_delta": motion_delta,
                            "char_score": max_char_sc,
                        }
                    )

                max_sharp = max(m["sharpness"] for m in cand_metrics)
                max_motion = max((m["motion_delta"] for m in cand_metrics), default=0.0)
                sharp_cutoff = max(min_sharpness_floor, 0.40 * max_sharp)

                # Step A: Filter out blurry or pure-black fade frames first
                sharp_pool = [
                    m for m in cand_metrics
                    if m["sharpness"] >= sharp_cutoff and m["brightness"] >= 3.5
                ]
                if not sharp_pool:
                    sharp_pool = [max(cand_metrics, key=lambda m: m["sharpness"])]

                # Step B: Rank sharp candidates by sharpness + visual-change peak + character presence
                def _score_candidate(m: Dict[str, Any]) -> float:
                    norm_sharp = m["sharpness"] / max(max_sharp, 1.0)
                    norm_motion = (m["motion_delta"] / max(max_motion, 1.0)) if max_motion > 1.5 else 0.0
                    char_bonus = 0.20 if m["char_score"] > visual_prompt_thresh else 0.0
                    return 0.45 * norm_sharp + 0.35 * norm_motion + char_bonus

                best_cand = max(sharp_pool, key=_score_candidate)
                raw_candidates.append(
                    {
                        "shot_idx": shot_idx,
                        "frame_idx": best_cand["frame_idx"],
                        "rgb": best_cand["rgb"].copy(),
                        "sharpness": best_cand["sharpness"],
                        "motion_delta": best_cand["motion_delta"],
                        "pinned": False,
                    }
                )
                del decoded_candidates, cand_metrics, sharp_pool

        # Optional: Force-include human-pinned timestamps for this scene
        scene_pins = pinned_map.get(scene_id, [])
        for pin_sec in scene_pins:
            pin_f = max(scene_start_f, min(int(round(float(pin_sec) * fps)), scene_end_f - 1))
            if any(abs(c["frame_idx"] - pin_f) <= max(1, int(round(0.35 * fps))) for c in raw_candidates):
                for c in raw_candidates:
                    if abs(c["frame_idx"] - pin_f) <= max(1, int(round(0.35 * fps))):
                        c["pinned"] = True
                continue
            pin_rgb = vr.get_batch([pin_f]).asnumpy()[0].copy()
            pin_gray = cv2.cvtColor(pin_rgb, cv2.COLOR_RGB2GRAY)
            pin_sharp = float(cv2.Laplacian(pin_gray, cv2.CV_64F).var())
            raw_candidates.append(
                {
                    "shot_idx": _find_shot_idx_for_frame(scene_shots, pin_f),
                    "frame_idx": pin_f,
                    "rgb": pin_rgb,
                    "sharpness": pin_sharp,
                    "motion_delta": 999.0,
                    "pinned": True,
                }
            )

        raw_candidates.sort(key=lambda c: c["frame_idx"])

        # Cap at max_keyframes_per_scene (default 30) while preserving pinned frames and temporal coverage
        if max_keyframes_per_scene > 0 and len(raw_candidates) > max_keyframes_per_scene:
            pinned_only = [c for c in raw_candidates if c.get("pinned")]
            unpinned = [c for c in raw_candidates if not c.get("pinned")]
            budget = max(0, max_keyframes_per_scene - len(pinned_only))
            if budget > 0 and unpinned:
                sel_idxs = np.linspace(0, len(unpinned) - 1, min(budget, len(unpinned)), dtype=int)
                kept_unpinned = [unpinned[i] for i in sel_idxs]
            else:
                kept_unpinned = []
            selected_candidates = sorted(pinned_only + kept_unpinned, key=lambda c: c["frame_idx"])
        else:
            selected_candidates = raw_candidates

        scene_keyframes: List[Dict[str, Any]] = []
        for cand in selected_candidates:
            best_frame_idx = int(cand["frame_idx"])
            best_frame_rgb = cand["rgb"]
            best_sharpness = float(cand["sharpness"])
            shot_idx = int(cand["shot_idx"])

            pred = frames_predictions.get(str(best_frame_idx), {})
            best_box_per_char: Dict[str, Tuple[float, List[int]]] = {}
            for bbox, label, score in zip(
                pred.get("bbox_ls", []),
                pred.get("labels", []),
                pred.get("scores", []),
            ):
                sc = float(score)
                if sc <= visual_prompt_thresh or label not in char_to_color:
                    continue
                if label not in best_box_per_char or sc > best_box_per_char[label][0]:
                    best_box_per_char[label] = (sc, [int(round(c)) for c in bbox])

            boxes_to_draw: List[Tuple[str, List[float], str, List[int]]] = []
            characters_visible_meta: List[Dict[str, Any]] = []

            for char_name in sorted_chars:
                if char_name not in best_box_per_char:
                    continue
                sc, bbox_int = best_box_per_char[char_name]
                color_name, rgb_val = char_to_color[char_name]
                boxes_to_draw.append((char_name, bbox_int, color_name, rgb_val))
                characters_visible_meta.append(
                    {
                        "name": char_name,
                        "color": color_name,
                        "bbox": bbox_int,
                        "score": round(sc, 4),
                    }
                )

            annotated_rgb = overlay_character_boxes_on_frame(best_frame_rgb, boxes_to_draw)
            timestamp_sec = round(best_frame_idx / fps, 2)
            kf_filename = f"{scene_id}_shot{shot_idx}_{timestamp_sec:06.2f}s.jpg"
            kf_rel_path = f"keyframes/{kf_filename}"
            kf_abs_path = os.path.join(keyframes_dir, kf_filename)

            cv2.imwrite(kf_abs_path, annotated_rgb[:, :, ::-1], [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            total_keyframes_saved += 1

            kf_entry: Dict[str, Any] = {
                "shot_idx": shot_idx,
                "frame_idx": best_frame_idx,
                "timestamp": timestamp_sec,
                "sharpness": round(best_sharpness, 2),
                "path": kf_rel_path,
                "characters_visible": characters_visible_meta,
            }
            if cand.get("pinned"):
                kf_entry["pinned"] = True
            scene_keyframes.append(kf_entry)

        del raw_candidates, selected_candidates
        scene["keyframes"] = scene_keyframes

    del vr
    _trim_host_memory()
    shots_and_scenes["total_keyframes"] = total_keyframes_saved
    shots_path = os.path.join(output_dir, "shots_and_scenes.json")
    with open(shots_path, "w", encoding="utf-8") as f:
        json.dump(shots_and_scenes, f, indent=2)

    print(
        f"[Stage A3.3] Extracted & annotated {total_keyframes_saved} adaptive shot-keyframes "
        f"into {keyframes_dir} and updated {shots_path}."
    )
    return shots_and_scenes


def reextract_keyframes_only(
    video_path: str,
    shots_file: str,
    output_dir: str,
    visual_prompt_thresh: float = 0.45,
    short_shot_sec: float = 1.0,
    interval_sec: float = 1.0,
    max_keyframes_per_scene: int = 30,
    min_sharpness_floor: float = 10.0,
    credits_start_sec: Optional[Union[str, float, int]] = None,
    pinned_timestamps: Optional[Dict[str, List[float]]] = None,
) -> Dict[str, Any]:
    """Re-run only Step 3.3 (adaptive keyframe selection & box drawing) using existing `visual_tracks_per_frame.json`."""
    from pipeline.stage1_scenes import apply_credits_cutoff_to_shots_file

    if credits_start_sec is not None:
        apply_credits_cutoff_to_shots_file(shots_file, credits_start_sec)

    with open(shots_file, "r", encoding="utf-8") as f:
        shots_and_scenes = json.load(f)

    visual_tracks_path = os.path.join(output_dir, "visual_tracks_per_frame.json")
    frames_predictions: Dict[str, Dict[str, List[Any]]] = {}
    if os.path.isfile(visual_tracks_path):
        with open(visual_tracks_path, "r", encoding="utf-8") as f:
            frames_predictions = json.load(f).get("frames", {})

    with _local_video_context(video_path) as active_video_path:
        return extract_adaptive_annotated_keyframes(
            video_path=active_video_path,
            shots_and_scenes=shots_and_scenes,
            frames_predictions=frames_predictions,
            output_dir=output_dir,
            visual_prompt_thresh=visual_prompt_thresh,
            short_shot_sec=short_shot_sec,
            interval_sec=interval_sec,
            max_keyframes_per_scene=max_keyframes_per_scene,
            min_sharpness_floor=min_sharpness_floor,
            pinned_timestamps=pinned_timestamps,
        )


def run_stage3_visual_recog(
    video_path: str,
    shots_file: str,
    ckpt_weights: str,
    char_features_dir: str,
    sam2_checkpoint: str,
    output_dir: str,
    model_size: str = "giant",
    mask_classify: bool = False,
    visual_prompt_thresh: float = 0.45,
    voice_harvest_thresh: float = 0.60,
    max_boxes_per_seed: int = 5,
    short_shot_sec: float = 1.0,
    interval_sec: float = 1.0,
    max_keyframes_per_scene: int = 30,
    min_sharpness_floor: float = 10.0,
    credits_start_sec: Optional[Union[str, float, int]] = None,
    pinned_timestamps: Optional[Dict[str, List[float]]] = None,
    restart: bool = False,
) -> Dict[str, Any]:
    """Run all 3 steps of Stage A3 and return summary statistics."""
    if not os.path.isfile(shots_file):
        raise FileNotFoundError(f"shots_and_scenes.json not found: {shots_file}")

    if credits_start_sec is not None:
        from pipeline.stage1_scenes import apply_credits_cutoff_to_shots_file

        apply_credits_cutoff_to_shots_file(shots_file, credits_start_sec)

    with open(shots_file, "r", encoding="utf-8") as f:
        shots_and_scenes = json.load(f)

    movie_id = shots_and_scenes["movie_id"]
    shots = [s for s in shots_and_scenes["shots"] if not s.get("is_credits", False)]

    tgrp_jsonl_path = os.path.join(output_dir, "tgrp_tracks.jsonl")

    with _local_video_context(video_path) as active_video_path:
        # Step 3.1: Per-shot TGRP
        run_per_shot_tgrp(
            video_path=active_video_path,
            movie_id=movie_id,
            shots=shots,
            tgrp_jsonl_path=tgrp_jsonl_path,
            sam2_checkpoint=sam2_checkpoint,
            max_boxes_per_seed=max_boxes_per_seed,
            restart=restart,
        )

        # Step 3.2: DINOv2 Track Classification & Frame Mapping
        _, visual_tracks_payload = classify_tracks_and_map_frames(
            video_path=active_video_path,
            movie_id=movie_id,
            tgrp_jsonl_path=tgrp_jsonl_path,
            ckpt_weights_path=ckpt_weights,
            char_features_dir=char_features_dir,
            output_dir=output_dir,
            model_size=model_size,
            use_mask=mask_classify,
            sam2_checkpoint=sam2_checkpoint,
            visual_prompt_thresh=visual_prompt_thresh,
            voice_harvest_thresh=voice_harvest_thresh,
        )

        # Step 3.3: Adaptive Per-Shot Sharp Keyframes with Drawn Bounding Boxes
        updated_scenes = extract_adaptive_annotated_keyframes(
            video_path=active_video_path,
            shots_and_scenes=shots_and_scenes,
            frames_predictions=visual_tracks_payload["frames"],
            output_dir=output_dir,
            visual_prompt_thresh=visual_prompt_thresh,
            short_shot_sec=short_shot_sec,
            interval_sec=interval_sec,
            max_keyframes_per_scene=max_keyframes_per_scene,
            min_sharpness_floor=min_sharpness_floor,
            pinned_timestamps=pinned_timestamps,
        )

    return {
        "movie_id": movie_id,
        "num_shots": len(shots),
        "num_scenes": len(updated_scenes.get("scenes", [])),
        "total_keyframes": updated_scenes.get("total_keyframes", 0),
        "track_stats": visual_tracks_payload["stats"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage A3: Per-shot TGRP tracking, DINOv2 classification, and adaptive annotated keyframes."
    )
    parser.add_argument("--video_path", required=True, type=str, help="Path to input movie video (.mp4/.mkv)")
    parser.add_argument("--shots_file", required=True, type=str, help="Path to workdir/{movie_id}/shots_and_scenes.json")
    parser.add_argument(
        "--ckpt_weights",
        required=True,
        type=str,
        help="Path to fine-tuned DINOv2 weights (finetuned_dinov2_weights.pth)",
    )
    parser.add_argument(
        "--char_features_dir",
        required=True,
        type=str,
        help="Path to extracted character features directory (char_features/{movie_id})",
    )
    parser.add_argument("--sam2_checkpoint", required=True, type=str, help="Path to SAM 2.1 checkpoint (.pt)")
    parser.add_argument("--output_dir", required=True, type=str, help="Output workdir/{movie_id}")
    parser.add_argument(
        "--model_size",
        default="giant",
        choices=["small", "base", "large", "giant"],
        help="DINOv2 model size (must match Stage A2)",
    )
    parser.add_argument("--mask_classify", action="store_true", help="Use SAM 2.1 masks when cropping tracks for DINOv2")
    parser.add_argument(
        "--visual_prompt_thresh",
        default=0.45,
        type=float,
        help="Minimum DINOv2 similarity s_vm to draw character bounding box (default: 0.45)",
    )
    parser.add_argument(
        "--voice_harvest_thresh",
        default=0.60,
        type=float,
        help="Minimum DINOv2 similarity s_vm for Stage A4 voice anchor harvesting (default: 0.60)",
    )
    parser.add_argument(
        "--max_boxes_per_seed",
        default=5,
        type=int,
        help="Maximum OWLv2 boxes propagated per seed frame (default: 5)",
    )
    parser.add_argument("--short_shot_sec", default=1.0, type=float, help="Short shot threshold in seconds (default: 1.0)")
    parser.add_argument("--interval_sec", default=1.0, type=float, help="Keyframe sampling interval in seconds (default: 1.0)")
    parser.add_argument("--max_keyframes_per_scene", default=30, type=int, help="Max keyframes per scene (default: 30)")
    parser.add_argument("--min_sharpness_floor", default=10.0, type=float, help="Minimum Laplacian sharpness floor (default: 10.0)")
    parser.add_argument(
        "--credits_start_sec",
        default=None,
        type=str,
        help="Optional timestamp (e.g. '744.0' or '12:24') where closing credits start",
    )
    parser.add_argument("--restart", action="store_true", help="Discard existing tgrp_tracks.jsonl and restart TGRP")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_stage3_visual_recog(
        video_path=args.video_path,
        shots_file=args.shots_file,
        ckpt_weights=args.ckpt_weights,
        char_features_dir=args.char_features_dir,
        sam2_checkpoint=args.sam2_checkpoint,
        output_dir=args.output_dir,
        model_size=args.model_size,
        mask_classify=args.mask_classify,
        visual_prompt_thresh=args.visual_prompt_thresh,
        voice_harvest_thresh=args.voice_harvest_thresh,
        max_boxes_per_seed=args.max_boxes_per_seed,
        short_shot_sec=args.short_shot_sec,
        interval_sec=args.interval_sec,
        max_keyframes_per_scene=args.max_keyframes_per_scene,
        min_sharpness_floor=args.min_sharpness_floor,
        credits_start_sec=args.credits_start_sec,
        restart=args.restart,
    )
