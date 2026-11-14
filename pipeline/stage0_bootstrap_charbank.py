#!/usr/bin/env python3
"""Stage A0: Hybrid Machine-Cluster & Human-in-the-Loop Character Bank Builder (`stage0_bootstrap_charbank.py`).

Workflow:
1. Step 0.1 (`preview_end_of_movie_frames`):
   - Samples frames across the final portion of the movie and builds a timestamped visual grid
     so the user can pick the exact `CREDITS_START_SEC` cutoff in seconds (or `MM:SS`).
2. Step 0.2 (`harvest_and_cluster_character_crops`):
   - Runs or loads Stage A1 (`shots_and_scenes.json`) up to `credits_start_sec`.
   - Samples sharp frames across narrative shots, detects character/creature boxes with `OWLv2`
     (`google/owlv2-large-patch14-ensemble`), and crops high-sharpness regions.
   - Extracts `DINOv2` embeddings and clusters crops into `num_clusters` (e.g. 16-24) visual
     appearance prototypes (naturally separating young vs. older character states, daylight vs.
     dark cave lighting, baby dragon vs. adult dragon, and background rocks/noise).
   - Renders a visual contact sheet (`clusters_contact_sheet.jpg`) showing each `Cluster ID` row
     with its top medoid crops and timestamps.
3. Step 0.3 (`export_labeled_clusters_to_charbank`):
   - Takes the human's `cluster_to_character` dictionary (e.g. `{"Sintel": [0, 4, 11], "Scales": [2], "Adult Dragon": [9]}`)
     and optional `character_descriptions`, exports the top sharp crops from each assigned cluster into
     `char_bank/{movie_id}/raw_images/{Character}/`, and updates `character_glossary.json`.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure project root and Animated_AD paths are importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline import ensure_animated_ad_paths, patch_owlv2_processor
from pipeline.stage1_scenes import (
    apply_credits_cutoff_to_shots_file,
    get_video_metadata,
    parse_timestamp_sec,
    run_stage1_scenes,
)

ensure_animated_ad_paths()


def format_mmss(sec: float) -> str:
    """Format seconds as MM:SS (or HH:MM:SS)."""
    total_s = max(0, int(round(sec)))
    hrs = total_s // 3600
    mins = (total_s % 3600) // 60
    secs = total_s % 60
    if hrs > 0:
        return f"{hrs:02d}:{mins:02d}:{secs:02d}"
    return f"{mins:02d}:{secs:02d}"


def preview_end_of_movie_frames(
    video_path: str,
    output_dir: str,
    start_ratio: float = 0.75,
    num_samples: int = 16,
    start_sec: Optional[float] = None,
    end_sec: Optional[float] = None,
) -> Dict[str, Any]:
    """Sample frames near the end of the movie to visually identify `CREDITS_START_SEC`."""
    import cv2
    import numpy as np
    import PIL.Image as Image
    import PIL.ImageDraw as ImageDraw
    from decord import VideoReader, cpu

    fps, total_frames, duration_sec = get_video_metadata(video_path)
    win_start = float(start_sec) if start_sec is not None else round(duration_sec * start_ratio, 2)
    win_end = float(end_sec) if end_sec is not None else duration_sec
    win_start = max(0.0, min(win_start, duration_sec - 1.0))
    win_end = max(win_start + 1.0, min(win_end, duration_sec))

    s_frame = max(0, min(int(round(win_start * fps)), total_frames - 1))
    e_frame = max(s_frame + 1, min(int(round(win_end * fps)), total_frames))

    sample_frames = np.linspace(s_frame, e_frame - 1, num_samples, dtype=int).tolist()
    vr = VideoReader(video_path, ctx=cpu(0))
    batch_rgb = vr.get_batch(sample_frames).asnumpy()
    del vr

    thumb_w, thumb_h = 320, 180
    cols = 4
    rows = int(math.ceil(len(sample_frames) / float(cols)))
    header_h = 34
    cell_h = thumb_h + header_h
    grid_img = Image.new("RGB", (cols * thumb_w, rows * cell_h), color=(24, 24, 28))
    draw = ImageDraw.Draw(grid_img)

    samples_meta: List[Dict[str, Any]] = []
    for idx, (f_idx, rgb) in enumerate(zip(sample_frames, batch_rgb)):
        t_sec = round(f_idx / fps, 2)
        mmss = format_mmss(t_sec)
        r = idx // cols
        c = idx % cols
        x0 = c * thumb_w
        y0 = r * cell_h

        pil_thumb = Image.fromarray(rgb).resize((thumb_w - 8, thumb_h - 6), Image.BICUBIC)
        grid_img.paste(pil_thumb, (x0 + 4, y0 + header_h))
        label = f"t = {t_sec:.1f}s ({mmss}) | frame {f_idx}"
        draw.text((x0 + 8, y0 + 8), label, fill=(255, 235, 140))
        samples_meta.append({"frame_idx": int(f_idx), "timestamp_sec": t_sec, "mmss": mmss})

    disc_dir = os.path.join(output_dir, "char_discovery")
    os.makedirs(disc_dir, exist_ok=True)
    grid_path = os.path.join(disc_dir, "credits_preview_grid.jpg")
    grid_img.save(grid_path, format="JPEG", quality=92)
    print(f"[Stage A0.1] Saved end-of-movie timestamp preview grid to: {grid_path}")
    return {
        "duration_sec": duration_sec,
        "window_sec": [win_start, win_end],
        "grid_path": grid_path,
        "samples": samples_meta,
    }


def _nms_boxes(boxes: Any, scores: Any, iou_threshold: float = 0.45) -> Tuple[List[List[float]], List[float]]:
    """Run Greedy IoU Non-Maximum Suppression without importing SAM2/tgrp."""
    import numpy as np

    boxes_np = np.asarray(boxes, dtype=np.float32)
    scores_np = np.asarray(scores, dtype=np.float32)
    if len(boxes_np) == 0:
        return [], []

    order = np.argsort(scores_np)[::-1]
    keep: List[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(boxes_np[i, 0], boxes_np[rest, 0])
        yy1 = np.maximum(boxes_np[i, 1], boxes_np[rest, 1])
        xx2 = np.minimum(boxes_np[i, 2], boxes_np[rest, 2])
        yy2 = np.minimum(boxes_np[i, 3], boxes_np[rest, 3])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        area_i = max(0.0, boxes_np[i, 2] - boxes_np[i, 0]) * max(0.0, boxes_np[i, 3] - boxes_np[i, 1])
        area_r = np.maximum(0.0, boxes_np[rest, 2] - boxes_np[rest, 0]) * np.maximum(0.0, boxes_np[rest, 3] - boxes_np[rest, 1])
        union = area_i + area_r - inter
        iou = np.where(union > 0, inter / union, 0.0)
        order = rest[iou <= iou_threshold]

    return boxes_np[keep].tolist(), scores_np[keep].tolist()


def harvest_and_cluster_character_crops(
    video_path: str,
    movie_id: str,
    output_dir: str,
    movie_title: str = "",
    credits_start_sec: Optional[Union[str, float, int]] = None,
    model_size: str = "giant",
    num_clusters: int = 18,
    sample_interval_sec: float = 2.5,
    max_boxes_per_frame: int = 4,
    owlv2_threshold: float = 0.14,
    min_box_area_ratio: float = 0.012,
    min_crop_sharpness: float = 12.0,
    pad_ratio: float = 0.06,
    samples_per_cluster_row: int = 6,
) -> Dict[str, Any]:
    """Harvest character/creature crops across narrative shots and cluster with DINOv2."""
    import cv2
    import numpy as np
    import PIL.Image as Image
    import PIL.ImageDraw as ImageDraw
    import torch
    from decord import VideoReader, cpu
    from sklearn.cluster import AgglomerativeClustering
    from torchvision import transforms
    from transformers import Owlv2ForObjectDetection, Owlv2Processor
    from tqdm import tqdm

    ensure_animated_ad_paths()
    patch_owlv2_processor()

    shots_file = os.path.join(output_dir, "shots_and_scenes.json")
    cutoff_sec = parse_timestamp_sec(credits_start_sec)

    if not os.path.isfile(shots_file):
        shots_data = run_stage1_scenes(
            video_path=video_path,
            movie_id=movie_id,
            output_dir=output_dir,
            movie_title=movie_title,
            credits_start_sec=cutoff_sec,
        )
    elif cutoff_sec is not None:
        shots_data = apply_credits_cutoff_to_shots_file(shots_file, cutoff_sec)
    else:
        with open(shots_file, "r", encoding="utf-8") as f:
            shots_data = json.load(f)

    fps = float(shots_data.get("fps") or 24.0)
    narrative_shots = [s for s in shots_data.get("shots", []) if not s.get("is_credits", False)]

    disc_dir = os.path.join(output_dir, "char_discovery")
    crops_dir = os.path.join(disc_dir, "crops")
    if os.path.isdir(crops_dir):
        shutil.rmtree(crops_dir)
    os.makedirs(crops_dir, exist_ok=True)

    # 1. Select sharp candidate frames across narrative shots
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)
    sampled_frame_indices: List[Tuple[int, int]] = []  # (shot_idx, frame_idx)

    for shot in narrative_shots:
        shot_idx = int(shot["shot_idx"])
        s_f = max(0, min(int(shot["start_frame"]), total_frames - 1))
        e_f = max(s_f + 1, min(int(shot["end_frame"]), total_frames))
        dur = (e_f - s_f) / fps
        num_windows = max(1, int(round(dur / max(0.5, sample_interval_sec))))
        step = (e_f - s_f) / float(num_windows)
        for w_i in range(num_windows):
            w_s = int(round(s_f + w_i * step))
            w_e = int(round(s_f + (w_i + 1) * step)) if w_i < num_windows - 1 else e_f
            if w_s >= w_e:
                continue
            probe_idxs = np.linspace(w_s, w_e - 1, min(5, w_e - w_s), dtype=int).tolist()
            probe_rgb = vr.get_batch(probe_idxs).asnumpy()
            best_f = probe_idxs[0]
            best_sharp = -1.0
            for pf, prgb in zip(probe_idxs, probe_rgb):
                gray = cv2.cvtColor(prgb, cv2.COLOR_RGB2GRAY)
                sh = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                if sh > best_sharp and float(gray.mean()) >= 4.0:
                    best_sharp = sh
                    best_f = int(pf)
            if best_sharp >= min_crop_sharpness * 0.5:
                sampled_frame_indices.append((shot_idx, best_f))

    print(f"[Stage A0.2] Selected {len(sampled_frame_indices)} sharp narrative frames across {len(narrative_shots)} shots.")

    # 2. Run OWLv2 to harvest character/creature crops
    device = "cuda" if torch.cuda.is_available() else "cpu"
    owlv2_source = os.environ.get("OWLV2_MODEL_PATH", "google/owlv2-large-patch14-ensemble")
    if not os.path.isdir(owlv2_source):
        owlv2_source = "google/owlv2-large-patch14-ensemble"

    print(f"[Stage A0.2] Loading OWLv2 ({owlv2_source}) on {device}...")
    processor = Owlv2Processor.from_pretrained(owlv2_source)
    owlv2_model = Owlv2ForObjectDetection.from_pretrained(owlv2_source).to(device).eval()
    owl_texts = [["a photo of an animated character", "a dragon or creature", "a person or warrior"]]

    harvested_crops: List[Dict[str, Any]] = []
    crop_counter = 0

    for shot_idx, f_idx in tqdm(sampled_frame_indices, desc="Harvesting OWLv2 Crops"):
        frame_rgb = vr.get_batch([f_idx]).asnumpy()[0]
        h, w = frame_rgb.shape[:2]
        frame_area = float(max(1, h * w))
        pil_frame = Image.fromarray(frame_rgb)

        inputs = processor(text=owl_texts, images=pil_frame, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        target_sizes = torch.Tensor([pil_frame.size[::-1]]).to(device)
        with torch.inference_mode():
            outputs = owlv2_model(**inputs)
            dets = processor.post_process_object_detection(
                outputs=outputs,
                threshold=owlv2_threshold,
                target_sizes=target_sizes,
            )
        boxes = dets[0]["boxes"].detach().cpu().numpy()
        scores = dets[0]["scores"].detach().cpu().numpy()
        del inputs, outputs, dets

        if len(boxes) == 0:
            continue

        sel_boxes, sel_scores = _nms_boxes(boxes, scores, iou_threshold=0.45)
        for box, sc in zip(sel_boxes[:max_boxes_per_frame], sel_scores[:max_boxes_per_frame]):
            x1, y1, x2, y2 = [float(c) for c in box]
            bw = max(0.0, x2 - x1)
            bh = max(0.0, y2 - y1)
            area_ratio = (bw * bh) / frame_area
            if area_ratio < min_box_area_ratio or bw < 36 or bh < 36:
                continue

            # Add slight context padding
            pad_x = bw * pad_ratio
            pad_y = bh * pad_ratio
            cx1 = max(0, int(round(x1 - pad_x)))
            cy1 = max(0, int(round(y1 - pad_y)))
            cx2 = min(w, int(round(x2 + pad_x)))
            cy2 = min(h, int(round(y2 + pad_y)))
            if cx2 - cx1 < 36 or cy2 - cy1 < 36:
                continue

            crop_rgb = frame_rgb[cy1:cy2, cx1:cx2]
            crop_gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
            crop_sharp = float(cv2.Laplacian(crop_gray, cv2.CV_64F).var())
            crop_bright = float(crop_gray.mean())
            if crop_sharp < min_crop_sharpness or crop_bright < 5.0:
                continue

            t_sec = round(f_idx / fps, 2)
            crop_fname = f"crop_{crop_counter:04d}_shot{shot_idx:03d}_t{t_sec:06.2f}s.png"
            crop_abs_path = os.path.join(crops_dir, crop_fname)
            Image.fromarray(crop_rgb).save(crop_abs_path, format="PNG")

            harvested_crops.append(
                {
                    "crop_id": crop_counter,
                    "shot_idx": shot_idx,
                    "frame_idx": int(f_idx),
                    "timestamp_sec": t_sec,
                    "mmss": format_mmss(t_sec),
                    "bbox": [cx1, cy1, cx2, cy2],
                    "owl_score": round(float(sc), 4),
                    "sharpness": round(crop_sharp, 2),
                    "brightness": round(crop_bright, 2),
                    "area_ratio": round(area_ratio, 4),
                    "crop_path": crop_abs_path,
                }
            )
            crop_counter += 1

    del owlv2_model, processor, vr
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if not harvested_crops:
        raise RuntimeError("No valid character crops were harvested! Try lowering owlv2_threshold or min_crop_sharpness.")

    print(f"[Stage A0.2] Harvested {len(harvested_crops)} sharp character/creature crops. Extracting DINOv2 features...")

    # 3. Extract pretrained DINOv2 embeddings for all crops
    model_archs = {"small": "vits14", "base": "vitb14", "large": "vitl14", "giant": "vitg14"}
    model_name = f"dinov2_{model_archs[model_size]}_reg"
    dino_model = torch.hub.load("facebookresearch/dinov2", model_name, pretrained=True, verbose=False).to(device).eval()

    dino_transform = transforms.Compose(
        [
            transforms.Resize((224, 224), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    feats_list: List[np.ndarray] = []
    batch_tensors: List[torch.Tensor] = []
    batch_size = 32

    for item in tqdm(harvested_crops, desc="DINOv2 Embedding"):
        pil_c = Image.open(item["crop_path"]).convert("RGB")
        batch_tensors.append(dino_transform(pil_c))
        if len(batch_tensors) >= batch_size:
            b_t = torch.stack(batch_tensors).to(device)
            with torch.inference_mode():
                out_f = dino_model(b_t).detach().cpu().numpy()
            for row in out_f:
                feats_list.append(row.reshape(-1))
            batch_tensors = []

    if batch_tensors:
        b_t = torch.stack(batch_tensors).to(device)
        with torch.inference_mode():
            out_f = dino_model(b_t).detach().cpu().numpy()
        for row in out_f:
            feats_list.append(row.reshape(-1))

    del dino_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    feat_matrix = np.stack(feats_list, axis=0)
    norms = np.linalg.norm(feat_matrix, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    feat_normed = feat_matrix / norms

    # 4. Cluster crops into visual appearance clusters
    actual_k = max(2, min(int(num_clusters), len(harvested_crops)))
    clustering = AgglomerativeClustering(
        n_clusters=actual_k,
        metric="cosine",
        linkage="average",
    )
    cluster_labels = clustering.fit_predict(feat_normed)

    clusters_dict: Dict[int, List[int]] = defaultdict(list)
    for idx, cid in enumerate(cluster_labels):
        clusters_dict[int(cid)].append(idx)

    # Sort clusters by size descending so primary recurring characters appear at the top (Cluster 0, 1, 2...)
    sorted_old_cids = sorted(clusters_dict.keys(), key=lambda c: len(clusters_dict[c]), reverse=True)

    clusters_summary: List[Dict[str, Any]] = []
    for new_cid, old_cid in enumerate(sorted_old_cids):
        member_idxs = clusters_dict[old_cid]
        member_vecs = feat_normed[member_idxs]
        centroid = np.mean(member_vecs, axis=0)
        c_norm = float(np.linalg.norm(centroid))
        if c_norm > 0:
            centroid = centroid / c_norm

        ranked_members: List[Dict[str, Any]] = []
        for m_idx in member_idxs:
            sim_to_centroid = float(np.dot(feat_normed[m_idx], centroid))
            crop_meta = dict(harvested_crops[m_idx])
            crop_meta["cluster_id"] = new_cid
            crop_meta["centrality"] = round(sim_to_centroid, 4)
            # Quality score combines centrality to cluster medoid + visual sharpness + detection confidence
            quality_score = sim_to_centroid * (1.0 + 0.15 * math.log1p(crop_meta["sharpness"])) * (0.8 + 0.2 * crop_meta["owl_score"])
            crop_meta["quality_score"] = round(quality_score, 4)
            ranked_members.append(crop_meta)

        ranked_members.sort(key=lambda x: x["quality_score"], reverse=True)
        timestamps = [m["timestamp_sec"] for m in ranked_members]
        clusters_summary.append(
            {
                "cluster_id": new_cid,
                "count": len(ranked_members),
                "min_time_sec": min(timestamps),
                "max_time_sec": max(timestamps),
                "time_span_mmss": f"{format_mmss(min(timestamps))} - {format_mmss(max(timestamps))}",
                "members": ranked_members,
            }
        )

    # 5. Render Visual Contact Sheet for Human Labeling
    thumb_size = 144
    label_col_w = 250
    row_h = thumb_size + 16
    sheet_w = label_col_w + samples_per_cluster_row * (thumb_size + 8) + 12
    sheet_h = len(clusters_summary) * row_h + 16

    sheet_img = Image.new("RGB", (sheet_w, sheet_h), color=(20, 22, 26))
    draw = ImageDraw.Draw(sheet_img)

    for row_i, c_info in enumerate(clusters_summary):
        cid = c_info["cluster_id"]
        y0 = 8 + row_i * row_h
        bg_color = (30, 33, 40) if row_i % 2 == 0 else (24, 26, 32)
        draw.rectangle([4, y0, sheet_w - 4, y0 + row_h - 4], fill=bg_color)

        title_line = f"CLUSTER {cid:02d}  (n={c_info['count']} crops)"
        span_line = f"Span: {c_info['time_span_mmss']}"
        top_t = ", ".join(f"{m['timestamp_sec']:.0f}s" for m in c_info["members"][:4])
        times_line = f"Top t: {top_t}"

        draw.text((14, y0 + 18), title_line, fill=(110, 220, 255))
        draw.text((14, y0 + 48), span_line, fill=(220, 220, 220))
        draw.text((14, y0 + 76), times_line, fill=(170, 175, 185))

        # Select diverse top crops from the cluster (medoid + distinct shots)
        display_members: List[Dict[str, Any]] = []
        seen_shots = set()
        for m in c_info["members"]:
            if m["shot_idx"] not in seen_shots or len(c_info["members"]) <= samples_per_cluster_row:
                display_members.append(m)
                seen_shots.add(m["shot_idx"])
            if len(display_members) >= samples_per_cluster_row:
                break
        if len(display_members) < min(samples_per_cluster_row, len(c_info["members"])):
            for m in c_info["members"]:
                if m not in display_members:
                    display_members.append(m)
                if len(display_members) >= samples_per_cluster_row:
                    break

        for col_i, m in enumerate(display_members):
            x0 = label_col_w + col_i * (thumb_size + 8)
            try:
                c_img = Image.open(m["crop_path"]).convert("RGB")
                c_img.thumbnail((thumb_size, thumb_size - 20), Image.BICUBIC)
                paste_x = x0 + (thumb_size - c_img.width) // 2
                paste_y = y0 + 4 + (thumb_size - 20 - c_img.height) // 2
                sheet_img.paste(c_img, (paste_x, paste_y))
                caption = f"{m['mmss']} ({m['sharpness']:.0f})"
                draw.text((x0 + 8, y0 + thumb_size - 12), caption, fill=(245, 225, 130))
            except Exception:
                pass

    contact_sheet_path = os.path.join(disc_dir, "clusters_contact_sheet.jpg")
    sheet_img.save(contact_sheet_path, format="JPEG", quality=92)

    manifest_path = os.path.join(disc_dir, "clusters_manifest.json")
    manifest_payload = {
        "movie_id": movie_id,
        "video_path": os.path.abspath(video_path),
        "credits_start_sec": cutoff_sec,
        "model_size": model_size,
        "num_harvested_crops": len(harvested_crops),
        "num_clusters": len(clusters_summary),
        "contact_sheet_path": contact_sheet_path,
        "clusters": clusters_summary,
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_payload, f, indent=2)

    print(f"[Stage A0.2] Clustered {len(harvested_crops)} crops into {len(clusters_summary)} visual clusters.")
    print(f"[Stage A0.2] Visual contact sheet saved to: {contact_sheet_path}")
    print(f"[Stage A0.2] Cluster manifest saved to    : {manifest_path}")
    return manifest_payload


def export_labeled_clusters_to_charbank(
    output_dir: str,
    char_bank_dir: str,
    cluster_to_character: Dict[str, List[int]],
    character_descriptions: Optional[Dict[str, str]] = None,
    max_crops_per_cluster: int = 6,
    replace_existing_raw_images: bool = False,
) -> Dict[str, Any]:
    """Export human-labeled clusters into `char_bank/{movie_id}/raw_images/{Character}/` and update glossary."""
    manifest_path = os.path.join(output_dir, "char_discovery", "clusters_manifest.json")
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(
            f"clusters_manifest.json not found at {manifest_path}. Run harvest_and_cluster_character_crops() first."
        )

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    cluster_lookup: Dict[int, Dict[str, Any]] = {
        int(c["cluster_id"]): c for c in manifest.get("clusters", [])
    }

    raw_images_root = os.path.join(char_bank_dir, "raw_images")
    if replace_existing_raw_images and os.path.isdir(raw_images_root):
        shutil.rmtree(raw_images_root)
    os.makedirs(raw_images_root, exist_ok=True)

    exported_counts: Dict[str, int] = {}
    exported_files: Dict[str, List[str]] = {}

    for char_name, cluster_ids in cluster_to_character.items():
        char_name_clean = str(char_name).strip()
        if not char_name_clean or not cluster_ids:
            continue

        char_dir = os.path.join(raw_images_root, char_name_clean)
        if replace_existing_raw_images and os.path.isdir(char_dir):
            shutil.rmtree(char_dir)
        os.makedirs(char_dir, exist_ok=True)

        # Determine starting index if keeping existing manual reference images
        existing_pngs = [
            f for f in sorted(os.listdir(char_dir))
            if os.path.splitext(f)[1].lower() in {".png", ".jpg", ".jpeg", ".webp"}
        ]
        next_idx = len(existing_pngs)
        saved_paths: List[str] = [os.path.join(char_dir, f) for f in existing_pngs]

        for cid in cluster_ids:
            c_info = cluster_lookup.get(int(cid))
            if not c_info:
                print(f"  [WARN] Cluster {cid} not found in manifest; skipping.")
                continue

            # Pick top-quality crops from distinct shots first so multi-angle diversity is maximized
            members = c_info.get("members", [])
            chosen_members: List[Dict[str, Any]] = []
            seen_shots = set()
            for m in members:
                if m["shot_idx"] not in seen_shots:
                    chosen_members.append(m)
                    seen_shots.add(m["shot_idx"])
                if len(chosen_members) >= max_crops_per_cluster:
                    break

            if len(chosen_members) < min(max_crops_per_cluster, len(members)):
                for m in members:
                    if m not in chosen_members:
                        chosen_members.append(m)
                    if len(chosen_members) >= max_crops_per_cluster:
                        break

            for m in chosen_members:
                src_crop = m["crop_path"]
                if not os.path.isfile(src_crop):
                    continue
                dst_fname = f"{char_name_clean}_{next_idx}.png"
                dst_path = os.path.join(char_dir, dst_fname)
                shutil.copy2(src_crop, dst_path)
                saved_paths.append(dst_path)
                next_idx += 1

        exported_counts[char_name_clean] = len(saved_paths)
        exported_files[char_name_clean] = saved_paths

    # Update character_glossary.json in both char_bank_dir and workdir
    glossary_path = os.path.join(char_bank_dir, "character_glossary.json")
    glossary: Dict[str, str] = {}
    if os.path.isfile(glossary_path):
        with open(glossary_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
            if isinstance(loaded, dict):
                glossary = {str(k): str(v) for k, v in loaded.items()}

    if character_descriptions:
        for k, v in character_descriptions.items():
            if str(v).strip():
                glossary[str(k).strip()] = str(v).strip()

    for char_name_clean in exported_counts:
        if char_name_clean not in glossary:
            glossary[char_name_clean] = f"Character {char_name_clean} appearing in {manifest.get('movie_id', 'the movie')}."

    with open(glossary_path, "w", encoding="utf-8") as f:
        json.dump(glossary, f, indent=2)

    workdir_glossary_path = os.path.join(output_dir, "character_glossary.json")
    with open(workdir_glossary_path, "w", encoding="utf-8") as f:
        json.dump(glossary, f, indent=2)

    print(f"[Stage A0.3] Exported labeled character clusters to: {raw_images_root}")
    for c_name, cnt in exported_counts.items():
        print(f"  - {c_name}: {cnt} total reference image(s) (clusters: {cluster_to_character.get(c_name)})")
    print(f"[Stage A0.3] Updated character_glossary.json ({len(glossary)} characters): {list(glossary.keys())}")

    return {
        "char_bank_dir": char_bank_dir,
        "raw_images_root": raw_images_root,
        "exported_counts": exported_counts,
        "glossary": glossary,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage A0: Hybrid Machine-Cluster & Human-Label Character Bank Builder."
    )
    parser.add_argument("--video_path", required=True, type=str, help="Path to input movie video (.mp4)")
    parser.add_argument("--movie_id", required=True, type=str, help="Movie identifier slug (e.g. sintel)")
    parser.add_argument("--movie_title", default="", type=str, help="Human-readable movie title")
    parser.add_argument("--output_dir", required=True, type=str, help="Output workdir/{movie_id}")
    parser.add_argument("--credits_start_sec", default=None, type=str, help="End-credits cutoff (e.g. '744.0' or '12:24')")
    parser.add_argument("--model_size", default="giant", choices=["small", "base", "large", "giant"])
    parser.add_argument("--num_clusters", default=18, type=int, help="Number of visual clusters (default: 18)")
    parser.add_argument("--sample_interval_sec", default=2.5, type=float, help="Sampling interval in seconds (default: 2.5)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    harvest_and_cluster_character_crops(
        video_path=args.video_path,
        movie_id=args.movie_id,
        output_dir=args.output_dir,
        movie_title=args.movie_title,
        credits_start_sec=args.credits_start_sec,
        model_size=args.model_size,
        num_clusters=args.num_clusters,
        sample_interval_sec=args.sample_interval_sec,
    )
