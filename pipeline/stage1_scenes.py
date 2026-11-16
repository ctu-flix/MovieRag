#!/usr/bin/env python3
"""Stage A1: Shot Detection & Hierarchical Scene Windowing (`stage1_scenes.py`).

1. Runs PySceneDetect (`ContentDetector()`) to detect exact camera shot boundaries.
2. Subdivides any single unbroken camera take exceeding `max_shot_sec` (default 20.0s)
   into `sub_shot_sec` (default 15.0s) sub-shots so SAM 2.1 mask propagation in Stage A3
   never exhausts GPU VRAM.
3. Groups contiguous shots into narrative Scene Windows (`target = 15-30s`, with
   `min_scene_sec = 12.0s` and `max_scene_sec = 30.0s`).
4. Writes `shots_and_scenes.json` to `output_dir`, serving as the canonical shot and
   scene index for Stages A3, A4, and A5.
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure project root is on sys.path when invoked directly as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline import ensure_animated_ad_paths

ensure_animated_ad_paths()


def parse_timestamp_sec(val: Optional[Union[str, float, int]]) -> Optional[float]:
    """Parse a timestamp in seconds (float/int) or 'MM:SS' / 'HH:MM:SS' string into float seconds."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        fval = float(val)
        return fval if fval > 0 else None
    s = str(val).strip()
    if not s or s.lower() in {"none", "null", "0", "0.0"}:
        return None
    if ":" in s:
        parts = [float(p.strip()) for p in s.split(":")]
        if len(parts) == 2:
            return parts[0] * 60.0 + parts[1]
        if len(parts) == 3:
            return parts[0] * 3600.0 + parts[1] * 60.0 + parts[2]
    fval = float(s)
    return fval if fval > 0 else None


def get_video_metadata(video_path: str) -> Tuple[float, int, float]:
    """Return (fps, total_frames, duration_sec) for the given video file."""
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    try:
        from decord import VideoReader, cpu

        vr = VideoReader(video_path, ctx=cpu(0))
        total_frames = len(vr)
        fps = float(vr.get_avg_fps())
        del vr
    except Exception:
        import cv2

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video file: {video_path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 24.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

    if fps <= 0:
        fps = 24.0
    if total_frames <= 0:
        raise ValueError(f"Video has 0 decodable frames: {video_path}")

    duration_sec = round(total_frames / fps, 3)
    return fps, total_frames, duration_sec


def detect_raw_shot_boundaries(
    video_path: str,
    total_frames: int,
    threshold: float = 27.0,
) -> List[Tuple[int, int]]:
    """Run PySceneDetect ContentDetector and return [(start_frame, end_frame), ...].

    Note: end_frame is exclusive, clamped to [0, total_frames].
    """
    from scenedetect import ContentDetector, detect

    scene_list = detect(video_path, ContentDetector(threshold=threshold))
    if not scene_list:
        return [(0, total_frames)]

    raw_shots: List[Tuple[int, int]] = []
    for start_tc, end_tc in scene_list:
        s = max(0, min(int(start_tc.get_frames()), total_frames))
        e = max(0, min(int(end_tc.get_frames()), total_frames))
        if s < e:
            raw_shots.append((s, e))

    if not raw_shots:
        return [(0, total_frames)]

    # Ensure full coverage from frame 0 to total_frames
    if raw_shots[0][0] > 0:
        raw_shots.insert(0, (0, raw_shots[0][0]))
    if raw_shots[-1][1] < total_frames:
        raw_shots.append((raw_shots[-1][1], total_frames))

    return raw_shots


def subdivide_long_shots(
    raw_shots: List[Tuple[int, int]],
    fps: float,
    max_shot_sec: float = 20.0,
    sub_shot_sec: float = 15.0,
) -> Tuple[List[Dict[str, Any]], int]:
    """Subdivide any shot longer than `max_shot_sec` into `sub_shot_sec` sub-shots.

    Also records unbroken take lineage (`parent_shot_idx` and `is_subdivided_continuation`)
    so Stage A3 can carry character boxes across subdivided seams without crossing real cuts.

    Returns:
        (shots_list, num_subdivided_parent_shots)
    """
    max_shot_frames = max(1, int(round(max_shot_sec * fps)))
    sub_shot_frames = max(1, int(round(sub_shot_sec * fps)))
    min_remainder_frames = max(1, int(round(2.0 * fps)))

    shots: List[Dict[str, Any]] = []
    subdivided_count = 0

    for raw_idx, (start_frame, end_frame) in enumerate(raw_shots):
        length = end_frame - start_frame
        if length <= 0:
            continue

        if length <= max_shot_frames:
            shot_idx = len(shots)
            shots.append(
                {
                    "shot_idx": shot_idx,
                    "parent_shot_idx": int(raw_idx),
                    "is_subdivided_continuation": False,
                    "start_frame": int(start_frame),
                    "end_frame": int(end_frame),
                    "start_time": round(start_frame / fps, 3),
                    "end_time": round(end_frame / fps, 3),
                }
            )
        else:
            subdivided_count += 1
            cur = start_frame
            sub_ranges: List[Tuple[int, int]] = []
            while cur < end_frame:
                nxt = min(cur + sub_shot_frames, end_frame)
                remainder = end_frame - nxt
                # If the trailing remainder is short (<2s) and keeping it together stays <= max_shot_frames,
                # merge the remainder into the current sub-shot.
                if 0 < remainder <= min_remainder_frames and (end_frame - cur) <= max_shot_frames:
                    nxt = end_frame
                sub_ranges.append((cur, nxt))
                cur = nxt

            for sub_i, (s, e) in enumerate(sub_ranges):
                shot_idx = len(shots)
                shots.append(
                    {
                        "shot_idx": shot_idx,
                        "parent_shot_idx": int(raw_idx),
                        "is_subdivided_continuation": bool(sub_i > 0),
                        "start_frame": int(s),
                        "end_frame": int(e),
                        "start_time": round(s / fps, 3),
                        "end_time": round(e / fps, 3),
                    }
                )

    return shots, subdivided_count


def build_micro_beats(
    scene_id: str,
    shot_ranges: List[Dict[str, Any]],
    fps: float = 24.0,
    min_beat_sec: float = 4.0,
    max_beat_sec: float = 10.0,
) -> List[Dict[str, Any]]:
    """Partition a 15-30s Scene Window into ordered 4-10s micro-beats for dense VLM captioning."""
    if not shot_ranges:
        return []
    if fps <= 0:
        fps = 24.0

    # Step 1: Split any individual shot/sub-shot longer than max_beat_sec into <= max_beat_sec spans
    atomic_segments: List[Dict[str, Any]] = []
    for shot in shot_ranges:
        s_f = int(shot["start_frame"])
        e_f = int(shot["end_frame"])
        sh_idx = int(shot["shot_idx"])
        if e_f <= s_f:
            continue
        dur = (e_f - s_f) / fps
        if dur <= max_beat_sec:
            atomic_segments.append(
                {
                    "start_frame": s_f,
                    "end_frame": e_f,
                    "start_time": round(s_f / fps, 3),
                    "end_time": round(e_f / fps, 3),
                    "shot_indices": [sh_idx],
                }
            )
        else:
            num_parts = max(2, int(math.ceil(dur / max_beat_sec)))
            step = (e_f - s_f) / float(num_parts)
            for p_i in range(num_parts):
                p_s = int(round(s_f + p_i * step))
                p_e = int(round(s_f + (p_i + 1) * step)) if p_i < num_parts - 1 else e_f
                if p_e <= p_s:
                    continue
                atomic_segments.append(
                    {
                        "start_frame": p_s,
                        "end_frame": p_e,
                        "start_time": round(p_s / fps, 3),
                        "end_time": round(p_e / fps, 3),
                        "shot_indices": [sh_idx],
                    }
                )

    if not atomic_segments:
        return []

    # Step 2: Merge contiguous short shots/segments into [min_beat_sec, max_beat_sec] micro-beats
    grouped: List[List[Dict[str, Any]]] = []
    cur: List[Dict[str, Any]] = []
    for seg in atomic_segments:
        if not cur:
            cur.append(seg)
            continue
        cur_dur = cur[-1]["end_time"] - cur[0]["start_time"]
        cand_dur = seg["end_time"] - cur[0]["start_time"]
        seg_dur = seg["end_time"] - seg["start_time"]
        if (cur_dur < min_beat_sec or seg_dur < 2.5) and cand_dur <= max_beat_sec:
            cur.append(seg)
        else:
            grouped.append(cur)
            cur = [seg]

    if cur:
        tail_dur = cur[-1]["end_time"] - cur[0]["start_time"]
        if (
            grouped
            and tail_dur < 2.5
            and (cur[-1]["end_time"] - grouped[-1][0]["start_time"]) <= max_beat_sec + 1.5
        ):
            grouped[-1].extend(cur)
        else:
            grouped.append(cur)

    micro_beats: List[Dict[str, Any]] = []
    for k, seg_group in enumerate(grouped):
        b_sf = int(seg_group[0]["start_frame"])
        b_ef = int(seg_group[-1]["end_frame"])
        b_st = round(float(seg_group[0]["start_time"]), 3)
        b_et = round(float(seg_group[-1]["end_time"]), 3)
        shot_idxs: List[int] = []
        for sg in seg_group:
            for s_i in sg["shot_indices"]:
                if s_i not in shot_idxs:
                    shot_idxs.append(s_i)
        micro_beats.append(
            {
                "beat_id": f"{scene_id}_b{k:02d}",
                "beat_index": k,
                "scene_id": scene_id,
                "start_frame": b_sf,
                "end_frame": b_ef,
                "start_time": b_st,
                "end_time": b_et,
                "duration_sec": round(b_et - b_st, 3),
                "shot_indices": shot_idxs,
                "keyframes": [],
            }
        )
    return micro_beats


def group_shots_into_scenes(
    shots: List[Dict[str, Any]],
    movie_id: str,
    min_scene_sec: float = 12.0,
    max_scene_sec: float = 30.0,
    fps: float = 24.0,
    min_beat_sec: float = 4.0,
    max_beat_sec: float = 10.0,
) -> List[Dict[str, Any]]:
    """Group contiguous camera shots into 15-30s narrative Scene Windows with 4-10s micro-beats."""
    if not shots:
        return []

    grouped_shot_lists: List[List[Dict[str, Any]]] = []
    current_group: List[Dict[str, Any]] = []

    for shot in shots:
        if not current_group:
            current_group.append(shot)
            continue

        scene_start = current_group[0]["start_time"]
        current_dur = current_group[-1]["end_time"] - scene_start
        candidate_dur = shot["end_time"] - scene_start

        if candidate_dur <= max_scene_sec:
            current_group.append(shot)
        elif current_dur < min_scene_sec and candidate_dur <= max_scene_sec + 10.0:
            # Current window hasn't reached min_scene_sec yet; allow slight overflow to avoid tiny scenes
            current_group.append(shot)
            grouped_shot_lists.append(current_group)
            current_group = []
        else:
            grouped_shot_lists.append(current_group)
            current_group = [shot]

    if current_group:
        # Merge a short trailing scene at the very end of the movie into the previous scene if reasonable
        trailing_dur = current_group[-1]["end_time"] - current_group[0]["start_time"]
        if (
            grouped_shot_lists
            and trailing_dur < min_scene_sec
            and (current_group[-1]["end_time"] - grouped_shot_lists[-1][0]["start_time"]) <= max_scene_sec + 12.0
        ):
            grouped_shot_lists[-1].extend(current_group)
        else:
            grouped_shot_lists.append(current_group)

    scenes: List[Dict[str, Any]] = []
    for scene_index, shot_group in enumerate(grouped_shot_lists):
        scene_id = f"{movie_id}_scene_{scene_index:04d}"
        start_frame = int(shot_group[0]["start_frame"])
        end_frame = int(shot_group[-1]["end_frame"])
        start_time = round(float(shot_group[0]["start_time"]), 3)
        end_time = round(float(shot_group[-1]["end_time"]), 3)
        shot_ranges_copy = [dict(s) for s in shot_group]

        scenes.append(
            {
                "scene_id": scene_id,
                "scene_index": scene_index,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "start_time": start_time,
                "end_time": end_time,
                "duration_sec": round(end_time - start_time, 3),
                "shot_ranges": shot_ranges_copy,
                "micro_beats": build_micro_beats(
                    scene_id=scene_id,
                    shot_ranges=shot_ranges_copy,
                    fps=fps,
                    min_beat_sec=min_beat_sec,
                    max_beat_sec=max_beat_sec,
                ),
                "characters_present": [],
                "keyframes": [],
            }
        )

    return scenes


def apply_credits_cutoff_to_shots_file(
    shots_file: str,
    credits_start_sec: Optional[Union[str, float, int]],
) -> Dict[str, Any]:
    """Filter or trim shots and scenes at or after `credits_start_sec` on an existing `shots_and_scenes.json`."""
    cutoff_sec = parse_timestamp_sec(credits_start_sec)
    if not os.path.isfile(shots_file):
        raise FileNotFoundError(f"shots_and_scenes.json not found: {shots_file}")

    with open(shots_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if cutoff_sec is None:
        return data

    fps = float(data.get("fps") or 24.0)
    cutoff_frame = int(round(cutoff_sec * fps))

    orig_shots = data.get("shots", [])
    kept_shots: List[Dict[str, Any]] = []
    for shot in orig_shots:
        s_f = int(shot["start_frame"])
        e_f = int(shot["end_frame"])
        if s_f >= cutoff_frame:
            continue
        if e_f > cutoff_frame:
            shot = dict(shot)
            shot["end_frame"] = cutoff_frame
            shot["end_time"] = round(cutoff_frame / fps, 3)
        kept_shots.append(shot)

    orig_scenes = data.get("scenes", [])
    kept_scenes: List[Dict[str, Any]] = []
    for scene in orig_scenes:
        s_t = float(scene.get("start_time", 0.0))
        if s_t >= cutoff_sec:
            continue
        scene_copy = dict(scene)
        if float(scene_copy.get("end_time", 0.0)) > cutoff_sec:
            scene_copy["end_frame"] = min(int(scene_copy["end_frame"]), cutoff_frame)
            scene_copy["end_time"] = round(min(float(scene_copy["end_time"]), cutoff_sec), 3)
            scene_copy["duration_sec"] = round(scene_copy["end_time"] - s_t, 3)
            trimmed_ranges: List[Dict[str, Any]] = []
            for sr in scene_copy.get("shot_ranges", []):
                if int(sr["start_frame"]) >= cutoff_frame:
                    continue
                sr_copy = dict(sr)
                if int(sr_copy["end_frame"]) > cutoff_frame:
                    sr_copy["end_frame"] = cutoff_frame
                    sr_copy["end_time"] = round(cutoff_frame / fps, 3)
                trimmed_ranges.append(sr_copy)
            scene_copy["shot_ranges"] = trimmed_ranges
            scene_copy["keyframes"] = [
                kf for kf in scene_copy.get("keyframes", [])
                if float(kf.get("timestamp", 0.0)) < cutoff_sec
            ]
            # Rebuild micro_beats for trimmed scene and re-attach remaining keyframes
            rebuilt_beats = build_micro_beats(
                scene_id=scene_copy["scene_id"],
                shot_ranges=trimmed_ranges,
                fps=fps,
            )
            scene_kfs = scene_copy.get("keyframes", [])
            for b_i, mb in enumerate(rebuilt_beats):
                is_last_b = b_i == len(rebuilt_beats) - 1
                mb["keyframes"] = [
                    kf for kf in scene_kfs
                    if (
                        int(mb["start_frame"]) <= int(kf.get("frame_idx", -1)) <= int(mb["end_frame"])
                        if is_last_b
                        else int(mb["start_frame"]) <= int(kf.get("frame_idx", -1)) < int(mb["end_frame"])
                    )
                ]
            scene_copy["micro_beats"] = rebuilt_beats
        kept_scenes.append(scene_copy)

    data["credits_start_sec"] = round(cutoff_sec, 3)
    data["credits_excluded_shots"] = len(orig_shots) - len(kept_shots)
    data["credits_excluded_scenes"] = len(orig_scenes) - len(kept_scenes)
    data["num_shots"] = len(kept_shots)
    data["num_scenes"] = len(kept_scenes)
    data["shots"] = kept_shots
    data["scenes"] = kept_scenes
    data.setdefault("config", {})["credits_start_sec"] = round(cutoff_sec, 3)

    with open(shots_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(
        f"[Stage A1] Applied credits_start_sec={cutoff_sec:.2f}s to {shots_file}: "
        f"kept {len(kept_shots)}/{len(orig_shots)} shots and {len(kept_scenes)}/{len(orig_scenes)} narrative scenes."
    )
    return data


def run_stage1_scenes(
    video_path: str,
    movie_id: str,
    output_dir: str,
    movie_title: str = "",
    min_scene_sec: float = 12.0,
    max_scene_sec: float = 30.0,
    max_shot_sec: float = 20.0,
    sub_shot_sec: float = 15.0,
    detector_threshold: float = 27.0,
    credits_start_sec: Optional[Union[str, float, int]] = None,
) -> Dict[str, Any]:
    """Execute Stage A1 and save `shots_and_scenes.json` to `output_dir`."""
    if not movie_title:
        movie_title = movie_id.replace("_", " ").title()

    cutoff_sec = parse_timestamp_sec(credits_start_sec)

    print(f"[Stage A1] Inspecting video metadata: {video_path}")
    fps, total_frames, duration_sec = get_video_metadata(video_path)
    print(f"  -> FPS: {fps:.3f} | Total Frames: {total_frames} | Duration: {duration_sec:.2f}s")

    print(f"[Stage A1] Running PySceneDetect ContentDetector (threshold={detector_threshold})...")
    raw_shots = detect_raw_shot_boundaries(video_path, total_frames, threshold=detector_threshold)
    total_raw_detected = len(raw_shots)
    print(f"  -> Detected {total_raw_detected} raw camera shots.")

    excluded_raw_shots = 0
    if cutoff_sec is not None and cutoff_sec < duration_sec:
        cutoff_frame = max(1, min(int(round(cutoff_sec * fps)), total_frames))
        filtered_raw: List[Tuple[int, int]] = []
        for s_f, e_f in raw_shots:
            if s_f >= cutoff_frame:
                excluded_raw_shots += 1
                continue
            filtered_raw.append((s_f, min(e_f, cutoff_frame)))
        raw_shots = filtered_raw
        print(
            f"  -> Applied end-credits cutoff at {cutoff_sec:.2f}s (frame {cutoff_frame}): "
            f"kept {len(raw_shots)} narrative raw shots, excluded {excluded_raw_shots} credit shot(s)."
        )

    shots, num_subdivided = subdivide_long_shots(
        raw_shots=raw_shots,
        fps=fps,
        max_shot_sec=max_shot_sec,
        sub_shot_sec=sub_shot_sec,
    )
    print(
        f"  -> Subdivided {num_subdivided} long shot(s) (> {max_shot_sec}s into ~{sub_shot_sec}s sub-shots) "
        f"-> {len(shots)} total canonical shots."
    )

    scenes = group_shots_into_scenes(
        shots=shots,
        movie_id=movie_id,
        min_scene_sec=min_scene_sec,
        max_scene_sec=max_scene_sec,
        fps=fps,
    )
    avg_scene_dur = (
        sum(s["duration_sec"] for s in scenes) / len(scenes) if scenes else 0.0
    )
    print(
        f"  -> Grouped {len(shots)} shots into {len(scenes)} Scene Windows "
        f"(avg duration: {avg_scene_dur:.2f}s, target: [{min_scene_sec}s, {max_scene_sec}s])."
    )

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "shots_and_scenes.json")
    payload: Dict[str, Any] = {
        "movie_id": movie_id,
        "movie_title": movie_title,
        "video_path": os.path.abspath(video_path),
        "fps": round(fps, 4),
        "total_frames": total_frames,
        "duration_sec": duration_sec,
        "credits_start_sec": round(cutoff_sec, 3) if cutoff_sec is not None else None,
        "credits_excluded_raw_shots": excluded_raw_shots,
        "config": {
            "min_scene_sec": min_scene_sec,
            "max_scene_sec": max_scene_sec,
            "max_shot_sec": max_shot_sec,
            "sub_shot_sec": sub_shot_sec,
            "detector_threshold": detector_threshold,
            "credits_start_sec": round(cutoff_sec, 3) if cutoff_sec is not None else None,
        },
        "num_raw_shots": total_raw_detected,
        "num_subdivided_shots": num_subdivided,
        "num_shots": len(shots),
        "num_scenes": len(scenes),
        "shots": shots,
        "scenes": scenes,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"[Stage A1] Saved canonical shot & scene boundaries to: {output_path}")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage A1: Shot boundary detection and 15-30s hierarchical scene windowing."
    )
    parser.add_argument("--video_path", required=True, type=str, help="Path to input movie video (.mp4/.mkv)")
    parser.add_argument("--movie_id", required=True, type=str, help="Movie identifier slug (e.g. sintel)")
    parser.add_argument("--movie_title", default="", type=str, help="Human-readable movie title (e.g. Sintel)")
    parser.add_argument("--output_dir", required=True, type=str, help="Output directory (workdir/{movie_id})")
    parser.add_argument("--min_scene_sec", default=12.0, type=float, help="Minimum scene duration in seconds")
    parser.add_argument("--max_scene_sec", default=30.0, type=float, help="Maximum scene duration in seconds")
    parser.add_argument(
        "--max_shot_sec",
        default=20.0,
        type=float,
        help="Max duration for a single shot before subdividing (prevents SAM 2.1 OOM)",
    )
    parser.add_argument(
        "--sub_shot_sec",
        default=15.0,
        type=float,
        help="Target sub-shot duration when subdividing shots longer than max_shot_sec",
    )
    parser.add_argument(
        "--detector_threshold",
        default=27.0,
        type=float,
        help="PySceneDetect ContentDetector threshold (default: 27.0)",
    )
    parser.add_argument(
        "--credits_start_sec",
        default=None,
        type=str,
        help="Optional timestamp (e.g. '744.0' or '12:24') where closing credits start; excludes credit shots.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_stage1_scenes(
        video_path=args.video_path,
        movie_id=args.movie_id,
        output_dir=args.output_dir,
        movie_title=args.movie_title,
        min_scene_sec=args.min_scene_sec,
        max_scene_sec=args.max_scene_sec,
        max_shot_sec=args.max_shot_sec,
        sub_shot_sec=args.sub_shot_sec,
        detector_threshold=args.detector_threshold,
        credits_start_sec=args.credits_start_sec,
    )
