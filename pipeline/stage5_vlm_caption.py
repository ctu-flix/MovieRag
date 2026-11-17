"""Stage A5: Micro-Beat DeepSeek-Flash Vision API Captioning & Knowledge Bundle Assembly.

Implements Phase A Stage A5 for MovieRAG:
1. Partitions each 15-30s narrative Scene Window into 4-10s micro-beats (via `build_micro_beats`),
   downscales motion-adaptive 2-6 fps keyframes in memory (`max_image_width=960`), and filters
   blurry/black-fade frames (`sharpness < min_sharpness`).
2. Strictly gates `[Character Reference Glossary]` per micro-beat so characters are only injected
   when visually detected, mentioned/speaking in beat subtitles, or inside manual `appearance_windows`.
3. Eliminates `[EXIT_STATE]` generation: continuity across consecutive beats uses 1-2 trailing
   overlap frames labeled `[Context Only — End of Previous Beat]` plus the previous beat caption
   wrapped as `[ALREADY HAPPENED IN PREVIOUS BEAT — CONTEXT ONLY, DO NOT RE-DESCRIBE]`.
4. Calls `deepseek-flash` (`thinking={"type": "disabled"}`) with a soft cap up to 400 words per
   micro-beat and checkpoints immediately to `vlm_beats.jsonl` for crash-resume.
5. Deterministically merges micro-beat captions inside each 15-30s Scene Window ($0 cost) into
   `movie_knowledge_bundle.json` preserving the exact schema expected by Phase B GraphRAG.
"""

import argparse
import base64
import ctypes
import gc
import io
import json
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from pipeline import PROJECT_ROOT
from pipeline.stage1_scenes import (
    apply_credits_cutoff_to_shots_file,
    build_micro_beats,
    parse_timestamp_sec,
)


SINTEL_FALLBACK_PLOT = (
    "Sintel (2010) — Full Plot & Ending Twist Synopsis:\n\n"
    "Act I.A — The Snowstorm and the Gatekeeper Ambush (00:00 - 02:05):\n"
    "Sintel, a lone young woman warrior bundled in a fur-lined cloak and carrying a wooden staff, "
    "wanders across a desolate, snow-swept mountain wilderness. At a mountain pass, she is ambushed "
    "by a masked Gatekeeper warrior wielding a curved spear-blade with a circular hilt ring. After a "
    "tense close-quarters fight in the snow, Sintel disarms and defeats the Gatekeeper, claims his "
    "distinctive curved spear-blade, and collapses in the freezing storm.\n\n"
    "Act I.B — Awakening in the Shaman's Hut (02:05 - 03:35):\n"
    "Sintel awakens inside a warmly lit desert hut tended by an elderly tribal Shaman with long white "
    "dreadlocks and ceremonial robes, who offers her shelter and asks what brings her to the land of "
    "the gatekeepers. Sintel reveals that she is searching the world for a dragon companion who was "
    "taken from her.\n\n"
    "Act II.A — Flashback in Ishtar: Finding and Healing Scales (03:35 - 04:35):\n"
    "In a flashback to Sintel's earlier life in the bustling sunlit city of Ishtar, she scavenges for "
    "food and comes across a wounded, frightened baby red dragon with an injured wing. She brings the "
    "hatchling back to her rooftop shelter, feeds him, wraps a makeshift bandage around his torn wing, "
    "and names him Scales. Over time, Sintel and Scales form a deep affectionate bond as she heals him "
    "and teaches him how to fly again.\n\n"
    "Act II.B — The Kidnapping of Scales Over Ishtar (04:35 - 05:15):\n"
    "As Scales practices flying and soars high above the rooftops of Ishtar, a giant winged Adult Dragon "
    "descends out of the clouds, snatches Scales in its talons, and carries him away into the sky despite "
    "Sintel's desperate chase across the city.\n\n"
    "Act III.A — The Cross-Continental Quest and Snowstorm Collapse (05:15 - 07:15):\n"
    "Vowing to bring Scales back, Sintel sets out on an arduous solitary quest across scorching desert "
    "dunes, misty bamboo forests, and rocky mountain passes—growing visibly older and more weathered over "
    "years of fruitless searching before collapsing exhausted in a blinding snowstorm.\n\n"
    "Act III.B — Return to the Shaman's Hut and Renewed Resolve (07:15 - 08:40):\n"
    "Back inside the Shaman's hut at the end of her flashback, the Shaman warns Sintel that she has reached "
    "the end of her search and that some things cannot be found the way we expect. Hearing wingbeats or "
    "wind outside, Sintel grabs her cloak and the Gatekeeper's spear-blade and rushes back out into the night.\n\n"
    "Act III.C — Climbing the Volcanic Peak and Entering the Dragon Cavern (08:40 - 10:15):\n"
    "Carrying the Gatekeeper's curved spear-blade with its circular hilt ring, Sintel climbs a jagged "
    "volcanic peak at night and enters a massive dark cavern lair. Deep inside the cave, she discovers a "
    "fledgling baby red dragon curled beside a giant sleeping Adult Dragon and creeps forward hoping to "
    "rescue the hatchling quietly.\n\n"
    "Act IV — The Cavern Battle and the Wing-Scar Twist (10:15 - 12:24):\n"
    "Before Sintel can escape with the hatchling, the giant Adult Dragon wakes up and attacks furiously to "
    "protect the cavern. In the brutal clash, the Adult Dragon knocks Sintel's weapon aside, pins her "
    "helplessly to the cavern floor, and prepares a fatal strike, but abruptly pauses as it stares closely "
    "into Sintel's eyes—recognizing her and pulling back instead of killing her. Unaware of why the beast "
    "hesitated, Sintel seizes her spear-blade and drives it straight into the Adult Dragon's chest. As the "
    "giant dragon collapses dying on the stone floor (~11:15), Sintel notices a familiar healed notch and "
    "bandage-wrap scar along its wing—the exact wound she bound on Scales years ago. Realizing too late "
    "that the giant Adult Dragon she has just slain was her grown-up Scales and the little red dragon "
    "cowering nearby is Scales's offspring, a grief-stricken Sintel walks blindly out into the falling snow "
    "while the orphaned baby dragon quietly trails after her."
)


def _trim_host_memory(empty_cuda: bool = False) -> None:
    """Release cyclic references and OS heap arenas."""
    gc.collect()
    try:
        libc = ctypes.CDLL("libc.so.6")
        if hasattr(libc, "malloc_trim"):
            libc.malloc_trim(0)
    except Exception:
        pass


def _format_mmss(seconds: float) -> str:
    """Format seconds as MM:SS.ss (or HH:MM:SS.ss if >= 1 hour)."""
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600)
    rem = seconds - h * 3600
    m = int(rem // 60)
    s = rem - m * 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:05.2f}"
    return f"{m:02d}:{s:05.2f}"


def _parse_mmss_to_sec(mmss: str) -> float:
    """Convert MM:SS or HH:MM:SS string to seconds."""
    parts = [float(p) for p in mmss.strip().split(":")]
    if len(parts) == 3:
        return parts[0] * 3600.0 + parts[1] * 60.0 + parts[2]
    if len(parts) == 2:
        return parts[0] * 60.0 + parts[1]
    return parts[0] if parts else 0.0


def select_scene_keyframes(
    keyframes: List[Dict[str, Any]],
    max_frames: int = 30,
    min_sharpness: float = 5.0,
) -> List[Dict[str, Any]]:
    """Filter out near-zero blurry/fade frames and select <= max_frames keyframes."""
    if not keyframes:
        return []
    ordered = sorted(keyframes, key=lambda kf: int(kf.get("frame_idx", 0)))

    # Filter out blurry or black-fade keyframes (e.g., sharpness < 5.0) unless all frames are below threshold
    if min_sharpness > 0.0:
        sharp_filtered = [
            kf for kf in ordered
            if kf.get("pinned", False) or float(kf.get("sharpness", 100.0)) >= min_sharpness
        ]
        if sharp_filtered:
            ordered = sharp_filtered
        else:
            # Fallback if an entire scene is soft-focus/dark: keep top sharpest frames
            by_sharp = sorted(ordered, key=lambda kf: float(kf.get("sharpness", 0.0)), reverse=True)
            keep_n = min(len(by_sharp), max(2, max_frames // 2))
            ordered = sorted(by_sharp[:keep_n], key=lambda kf: int(kf.get("frame_idx", 0)))

    if len(ordered) <= max_frames or max_frames <= 2:
        return ordered[:max_frames]

    first_kf = ordered[0]
    last_kf = ordered[-1]
    interior = ordered[1:-1]
    n_slots = max_frames - 2

    chosen: List[Dict[str, Any]] = [first_kf]
    for slot_idx in range(n_slots):
        i0 = round(slot_idx * len(interior) / float(n_slots))
        i1 = round((slot_idx + 1) * len(interior) / float(n_slots))
        bucket = interior[i0:i1]
        if not bucket:
            continue
        best_kf = max(
            bucket,
            key=lambda kf: (
                1 if kf.get("pinned", False) else 0,
                1 if len(kf.get("characters_visible", [])) > 0 else 0,
                len(kf.get("characters_visible", [])),
                float(kf.get("sharpness", 0.0)),
            ),
        )
        chosen.append(best_kf)
    chosen.append(last_kf)
    return chosen


def _stage_keyframes_to_local(output_dir: str, movie_id: str) -> Callable[[str], str]:
    """Copy Drive keyframes to local /content SSD when available for fast I/O."""
    drive_kf_root = os.path.join(output_dir, "keyframes")
    if os.path.isdir("/content") and os.access("/content", os.W_OK):
        try:
            free_gb = shutil.disk_usage("/content").free / (1024**3)
            if free_gb >= 2.0 and os.path.isdir(drive_kf_root):
                local_kf_root = os.path.join("/content", "vlm_scratch", movie_id, "keyframes")
                os.makedirs(local_kf_root, exist_ok=True)
                src_files = [f for f in os.listdir(drive_kf_root) if f.lower().endswith(".jpg")]
                dst_files = [f for f in os.listdir(local_kf_root) if f.lower().endswith(".jpg")]
                if len(dst_files) < len(src_files):
                    print(f"[Stage A5] Staging {len(src_files)} keyframes to local SSD ({local_kf_root})...")
                    for fname in src_files:
                        dst = os.path.join(local_kf_root, fname)
                        if not os.path.isfile(dst):
                            shutil.copy2(os.path.join(drive_kf_root, fname), dst)
                return lambda rel_path: os.path.join(local_kf_root, os.path.basename(rel_path))
        except Exception as exc:
            print(f"[Stage A5] Keyframe local staging skipped ({exc}); reading from output_dir.")

    return lambda rel_path: os.path.join(output_dir, rel_path)


def _sanitize_glossary_desc(name: str, desc: str) -> str:
    """Strip known twist-spoiler clauses from stale glossary files cached on Drive."""
    cleaned = str(desc).strip()
    cleaned = re.sub(
        r";\s*later revealed to have grown into the Adult Dragon",
        "",
        cleaned,
        flags=re.IGNORECASE,
    ).strip()
    if cleaned and not cleaned.endswith("."):
        cleaned += "."
    return cleaned


def _load_character_glossary(
    output_dir: str,
    char_bank_dir: Optional[str] = None,
    glossary_file: Optional[str] = None,
) -> Dict[str, str]:
    """Load and merge static character glossary, omitting spoiler clauses and private metadata keys."""
    merged: Dict[str, str] = {}

    # 1. Start with stage2_charbank_summary.json as base if present
    stage2_summary = os.path.join(output_dir, "stage2_charbank_summary.json")
    if os.path.isfile(stage2_summary):
        try:
            with open(stage2_summary, "r", encoding="utf-8") as f:
                s2 = json.load(f)
            if isinstance(s2.get("glossary"), dict):
                for k, v in s2["glossary"].items():
                    if not str(k).startswith("_"):
                        merged[str(k)] = _sanitize_glossary_desc(str(k), str(v))
        except Exception:
            pass

    # 2. Overlay Drive/workdir glossaries, then give repo character_glossary.json priority for spoiler-free edits
    movie_slug = os.path.basename(output_dir.rstrip("/"))
    candidates: List[str] = []
    if char_bank_dir:
        candidates.append(os.path.join(char_bank_dir, "character_glossary.json"))
    candidates.append(os.path.join(output_dir, "character_glossary.json"))
    candidates.append(str(PROJECT_ROOT / "char_bank" / movie_slug / "character_glossary.json"))
    if glossary_file:
        candidates.append(glossary_file)

    for cand in candidates:
        if cand and os.path.isfile(cand):
            try:
                with open(cand, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    for k, v in data.items():
                        if not str(k).startswith("_") and isinstance(v, str):
                            merged[str(k)] = _sanitize_glossary_desc(str(k), v)
            except Exception:
                pass
    return merged


def _load_appearance_windows(
    output_dir: str,
    char_bank_dir: Optional[str] = None,
    glossary_file: Optional[str] = None,
    appearance_windows: Optional[Dict[str, List[List[float]]]] = None,
) -> Dict[str, List[List[float]]]:
    """Load manual {character_name: [[start_sec, end_sec], ...]} appearance gates."""
    merged: Dict[str, List[List[float]]] = {}
    movie_slug = os.path.basename(output_dir.rstrip("/"))

    def _ingest_windows_map(obj: Any) -> None:
        if not isinstance(obj, dict):
            return
        for k, wins in obj.items():
            if not isinstance(wins, list):
                continue
            parsed_wins: List[List[float]] = []
            for pair in wins:
                if isinstance(pair, (list, tuple)) and len(pair) >= 2:
                    parsed_wins.append([float(pair[0]), float(pair[1])])
            merged[str(k)] = parsed_wins

    # Optional embedded `_appearance_windows` inside glossary JSON files
    glossary_cands: List[Optional[str]] = [
        char_bank_dir and os.path.join(char_bank_dir, "character_glossary.json"),
        os.path.join(output_dir, "character_glossary.json"),
        str(PROJECT_ROOT / "char_bank" / movie_slug / "character_glossary.json"),
        glossary_file,
    ]
    for cand in glossary_cands:
        if cand and os.path.isfile(cand):
            try:
                with open(cand, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict) and "_appearance_windows" in data:
                    _ingest_windows_map(data["_appearance_windows"])
            except Exception:
                pass

    # Dedicated appearance_windows.json files (repo -> workdir -> Drive char_bank -> caller override)
    window_cands: List[Optional[str]] = [
        str(PROJECT_ROOT / "char_bank" / movie_slug / "appearance_windows.json"),
        os.path.join(output_dir, "appearance_windows.json"),
        char_bank_dir and os.path.join(char_bank_dir, "appearance_windows.json"),
    ]
    for cand in window_cands:
        if cand and os.path.isfile(cand):
            try:
                with open(cand, "r", encoding="utf-8") as f:
                    _ingest_windows_map(json.load(f))
            except Exception:
                pass

    if appearance_windows is not None:
        _ingest_windows_map(appearance_windows)

    return merged


def _load_external_plot_documents(
    movie_id: str,
    duration_sec: float,
    char_bank_dir: Optional[str] = None,
    external_plot_file: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Load optional Tier-3 full-movie synopsis document for Phase B GraphRAG & Act grounding."""
    candidates: List[str] = []
    if external_plot_file:
        candidates.append(external_plot_file)
    if char_bank_dir:
        candidates.append(os.path.join(char_bank_dir, "external_plot.txt"))
        candidates.append(os.path.join(char_bank_dir, "external_plot.json"))
    candidates.append(str(PROJECT_ROOT / "char_bank" / movie_id / "external_plot.txt"))

    for cand in candidates:
        if not cand or not os.path.isfile(cand):
            continue
        if cand.lower().endswith(".json"):
            with open(cand, "r", encoding="utf-8") as f:
                obj = json.load(f)
            if isinstance(obj, list):
                return obj
            if isinstance(obj, dict) and obj.get("content"):
                return [obj]
        else:
            with open(cand, "r", encoding="utf-8") as f:
                text = f.read().strip()
            if text:
                return [
                    {
                        "doc_id": f"{movie_id}_full_plot_01",
                        "title": "Full Plot & Ending Twist Synopsis",
                        "tier": "full_movie",
                        "start_time": 0.0,
                        "end_time": round(float(duration_sec), 3),
                        "content": text,
                    }
                ]

    if movie_id.lower() == "sintel":
        return [
            {
                "doc_id": "sintel_full_plot_01",
                "title": "Full Plot & Ending Twist Synopsis",
                "tier": "full_movie",
                "start_time": 0.0,
                "end_time": round(float(duration_sec), 3),
                "content": SINTEL_FALLBACK_PLOT,
            }
        ]
    return []


def _extract_time_matched_plot_context(
    external_docs: List[Dict[str, Any]],
    scene_start: float,
    scene_end: float,
    movie_id: str = "",
) -> str:
    """Extract only the timestamped Act/Chapter paragraph(s) overlapping [scene_start, scene_end]."""
    if not external_docs:
        return ""

    full_text = "\n\n".join(str(d.get("content", "")) for d in external_docs if d.get("content"))
    if not full_text.strip():
        return ""

    # Match paragraphs with (MM:SS - MM:SS) headers
    pattern = re.compile(
        r"([^\n]*\(\s*(\d{1,2}:\d{2}(?::\d{2})?)\s*-\s*(\d{1,2}:\d{2}(?::\d{2})?)\s*\)[^\n]*\n[^\n]+)",
        re.MULTILINE,
    )
    matched_blocks: List[str] = []
    for match in pattern.finditer(full_text):
        block_text = match.group(1).strip()
        act_start = _parse_mmss_to_sec(match.group(2))
        act_end = _parse_mmss_to_sec(match.group(3))
        if min(scene_end, act_end) > max(scene_start, act_start):
            matched_blocks.append(block_text)

    return "\n\n".join(matched_blocks)


def resolve_deepseek_api_key(api_key: Optional[str] = None) -> str:
    """Resolve DEEPSEEK_API_KEY from explicit parameter, environment variable, or Colab Secrets."""
    if api_key and api_key.strip():
        return api_key.strip()

    env_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if env_key:
        return env_key

    try:
        from google.colab import userdata  # type: ignore

        colab_key = userdata.get("DEEPSEEK_API_KEY")
        if colab_key and str(colab_key).strip():
            return str(colab_key).strip()
    except Exception:
        pass

    raise ValueError(
        "Missing DEEPSEEK_API_KEY! Please add 'DEEPSEEK_API_KEY' in Google Colab Secrets "
        "(key icon on the left sidebar -> enable Notebook access) or set os.environ['DEEPSEEK_API_KEY']."
    )


def create_deepseek_client(
    api_key: Optional[str] = None,
    api_base_url: str = "https://api.deepseek.com",
):
    """Create an OpenAI-compatible client configured for DeepSeek Vision API."""
    from openai import OpenAI

    resolved_key = resolve_deepseek_api_key(api_key)
    return OpenAI(api_key=resolved_key, base_url=api_base_url)


def _encode_image_data_uri(
    image_path: str,
    max_width: int = 960,
    jpeg_quality: int = 85,
) -> str:
    """Load keyframe JPEG, downscale proportionally to <= max_width in memory, and return data URI."""
    from PIL import Image

    with Image.open(image_path) as im:
        rgb = im.convert("RGB")
        if max_width and rgb.width > max_width:
            new_h = max(1, int(round(rgb.height * (max_width / float(rgb.width)))))
            rgb = rgb.resize((max_width, new_h), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        rgb.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
        b64_data = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64_data}"


def _format_color_legend_and_glossary(
    scene: Dict[str, Any],
    sampled_kfs: List[Dict[str, Any]],
    glossary: Dict[str, str],
    beat_start: Optional[float] = None,
    beat_end: Optional[float] = None,
    beat_subtitles: Optional[List[Dict[str, Any]]] = None,
    appearance_windows: Optional[Dict[str, List[List[float]]]] = None,
    beat_characters_present: Optional[List[Any]] = None,
) -> Tuple[str, str]:
    """Build synthetic box legend and gate Character Reference Glossary strictly to active beat entities."""
    char_colors: Dict[str, str] = {}
    active_chars: set = set()

    for kf in sampled_kfs:
        for cv in kf.get("characters_visible", []):
            name = cv.get("name")
            color = cv.get("color")
            if name:
                active_chars.add(name)
                if color and name not in char_colors:
                    char_colors[name] = color

    # Include explicit beat_characters_present when provided, or fallback to scene when no beat bounds were passed
    present_source = (
        beat_characters_present
        if beat_characters_present is not None
        else (scene.get("characters_present", []) if (beat_start is None and beat_end is None) else [])
    )
    for cp in present_source:
        if isinstance(cp, dict):
            name = cp.get("name")
            color = cp.get("color_prompt") or cp.get("color")
        else:
            name = str(cp)
            color = None
        if name:
            active_chars.add(name)
            if color and name not in char_colors:
                char_colors[name] = color

    # Fallback color lookup from scene if beat_characters_present lacked color
    if active_chars and not char_colors:
        for cp in scene.get("characters_present", []):
            if isinstance(cp, dict) and cp.get("name") in active_chars and cp.get("color_prompt"):
                char_colors[cp["name"]] = cp["color_prompt"]

    if char_colors:
        legend_items = [f"{color} rectangle = {name}" for name, color in char_colors.items()]
        legend_str = ", ".join(legend_items)
    else:
        legend_str = "No synthetic tracking boxes drawn in this beat."

    # Check subtitles in this beat for speaker attribution or spoken mentions
    subs = beat_subtitles if beat_subtitles is not None else scene.get("subtitles", [])
    for utt in subs:
        spk = str(utt.get("speaker", "")).strip()
        txt_lower = str(utt.get("text", "")).lower()
        for name in glossary:
            if spk.lower() == name.lower() or re.search(rf"\b{re.escape(name.lower())}\b", txt_lower):
                active_chars.add(name)

    # Check manual appearance_windows overlap against beat time range
    b_start = float(beat_start if beat_start is not None else scene.get("start_time", 0.0))
    b_end = float(beat_end if beat_end is not None else scene.get("end_time", 0.0))
    b_mid = 0.5 * (b_start + b_end)
    if appearance_windows:
        for name, windows in appearance_windows.items():
            for win in windows:
                if not isinstance(win, (list, tuple)) or len(win) < 2:
                    continue
                w_start, w_end = float(win[0]), float(win[1])
                if (w_start <= b_mid <= w_end) or (min(b_end, w_end) > max(b_start, w_start)):
                    active_chars.add(name)
                    break

    glossary_lines = [f"- {name}: {desc}" for name, desc in glossary.items() if name in active_chars]
    glossary_str = (
        "\n".join(glossary_lines)
        if glossary_lines
        else "- No pre-registered character glossary entries are active in this beat."
    )
    return legend_str, glossary_str


def build_micro_beat_prompt(
    movie_title: str,
    scene: Dict[str, Any],
    beat: Dict[str, Any],
    sampled_kfs: List[Dict[str, Any]],
    prev_overlap_kfs: List[Dict[str, Any]],
    glossary: Dict[str, str],
    prev_beat_caption: str,
    prev_subtitles: List[Dict[str, Any]],
    act_plot_context: str = "",
    appearance_windows: Optional[Dict[str, List[List[float]]]] = None,
) -> str:
    """Build the micro-beat prompt replacing [EXIT_STATE] with visual overlap + prior beat context."""
    b_start = float(beat.get("start_time", scene["start_time"]))
    b_end = float(beat.get("end_time", scene["end_time"]))
    t0_str = _format_mmss(b_start)
    t1_str = _format_mmss(b_end)
    cur_subs = beat.get("subtitles", scene.get("subtitles", []))

    legend_str, glossary_str = _format_color_legend_and_glossary(
        scene=scene,
        sampled_kfs=sampled_kfs,
        glossary=glossary,
        beat_start=b_start,
        beat_end=b_end,
        beat_subtitles=cur_subs,
        appearance_windows=appearance_windows,
        beat_characters_present=beat.get("characters_present"),
    )

    overlap_timeline = (
        ", ".join(
            f"Context Frame {idx + 1} ({_format_mmss(kf.get('timestamp', 0.0))})"
            for idx, kf in enumerate(prev_overlap_kfs)
        )
        if prev_overlap_kfs
        else "None (opening beat)"
    )
    kf_timeline = ", ".join(
        f"Beat Frame {idx + 1} ({_format_mmss(kf.get('timestamp', 0.0))}, shot {kf.get('shot_idx')})"
        for idx, kf in enumerate(sampled_kfs)
    )

    if prev_subtitles:
        prev_subs_str = "\n".join(
            f"- [{_format_mmss(u['start_time'])}] {u.get('speaker', 'Unknown')}: \"{u['text']}\""
            for u in prev_subtitles[-3:]
        )
    else:
        prev_subs_str = "- (No recent dialogue)"

    if cur_subs:
        cur_subs_str = "\n".join(
            f"- [{_format_mmss(u['start_time'])} - {_format_mmss(u['end_time'])}] "
            f"{u.get('speaker', 'Unknown')}: \"{u['text']}\""
            for u in cur_subs
        )
        dialogue_guidance = "Integrate spoken lines naturally alongside the physical actions and reactions."
    else:
        cur_subs_str = "- [No spoken dialogue in this beat]"
        dialogue_guidance = (
            "There is NO spoken dialogue in this beat. Focus on silent visual storytelling: "
            "movements, environment, props, combat/actions, posture, and facial expressions."
        )

    act_block = ""
    if act_plot_context:
        act_block = (
            f"[Time-Matched Act Reference (Narrative Grounding)]\n"
            f"{act_plot_context}\n"
            f"IMPORTANT GROUNDING RULES:\n"
            f"1. Use this Act Reference only to accurately recognize characters, creatures, props, weapons, "
            f"and settings that visibly appear in [{t0_str} - {t1_str}].\n"
            f"2. NEVER invent monsters, locations, or objects that contradict the frames and Act Reference, "
            f"and NEVER pull future plot beats or absent characters into [{t0_str} - {t1_str}].\n"
            f"3. Report every concrete physical event (weapon pick-ups/drops, disarming, injuries, exits/entrances) "
            f"that actually happens within the current beat frames.\n\n"
        )

    return (
        f"You are analyzing Micro-Beat {beat.get('beat_id', scene.get('scene_id', 'beat'))} "
        f"[{t0_str} - {t1_str}] inside Scene {scene.get('scene_index', 0)} from '{movie_title}'.\n"
        f"Context Overlap Frames Attached First (DO NOT RE-DESCRIBE): {len(prev_overlap_kfs)} [{overlap_timeline}]\n"
        f"Current Beat Keyframes Attached Next (DESCRIBE ONLY THESE): {len(sampled_kfs)} [{kf_timeline}]\n\n"
        f"[Character Reference Glossary (Active in This Beat)]\n{glossary_str}\n"
        f"IMPORTANT ANTI-BOILERPLATE & APPEARANCE RULES:\n"
        f"1. Refer to known active characters by their exact glossary names; do NOT introduce characters "
        f"who are not visibly present or mentioned in this beat.\n"
        f"2. DO NOT repeat static character bio introductions from the glossary in every beat. Only mention "
        f"appearance traits when newly introduced, visibly altered (injuries, aging, weapon swaps), or critical to the action.\n\n"
        f"[Synthetic Tracking Box Overlay Key]\n"
        f"Overlay key: {legend_str}\n"
        f"CRITICAL VISUAL RULE: Any colored rectangles drawn on the frames are synthetic computer-vision tracking overlays. "
        f"NEVER mention bounding boxes, rectangles, box colors, or 'Frame 1/2' labels in your output, and NEVER confuse "
        f"a tracking box color with hair color, clothing color, or glowing objects. "
        f"Note: Tracking boxes are automated hints and may occasionally be missing, shifted, or mislabeled—always verify "
        f"a character's identity against what you actually see in the frames, the [Character Reference Glossary], and the "
        f"[Time-Matched Act Reference].\n\n"
        f"{act_block}"
        f"[ALREADY HAPPENED IN PREVIOUS BEAT — CONTEXT ONLY, DO NOT RE-DESCRIBE]\n"
        f"{prev_beat_caption or 'Opening beat of the film.'}\n"
        f"Recent dialogue before this beat:\n{prev_subs_str}\n\n"
        f"[Spoken Dialogue in This Beat]\n{cur_subs_str}\n\n"
        f"[Task]\n"
        f"Write ONE cohesive, chronological present-tense description covering ONLY what happens in the Current Beat "
        f"Keyframes [{t0_str} - {t1_str}] (soft cap: up to 400 words max, scaling naturally with the amount of action—"
        f"stop immediately once the final frame is described, NEVER repeat sentences or clauses, and DO NOT re-describe "
        f"the Context Overlap Frames):\n"
        f"1. Setting & Camera/Staging: Briefly note environment, lighting, and character spatial layout (especially when cutting or moving).\n"
        f"2. Step-by-Step Physical Action & Exchanges: Trace exact movements, combat strikes, weapon/prop acquisitions or drops, "
        f"and cause-and-effect reactions across the frames. {dialogue_guidance}\n"
        f"3. Expressions & Physical State Changes: Note decisive posture shifts, wounds, discoveries, or exits at the end of the beat."
    )


def build_single_pass_prompt(
    movie_title: str,
    scene: Dict[str, Any],
    sampled_kfs: List[Dict[str, Any]],
    glossary: Dict[str, str],
    prev_exit_state: str,
    prev_subtitles: List[Dict[str, Any]],
    act_plot_context: str = "",
    appearance_windows: Optional[Dict[str, List[List[float]]]] = None,
) -> str:
    """Backward-compatible wrapper delegating to `build_micro_beat_prompt`."""
    pseudo_beat = {
        "beat_id": scene.get("scene_id", "scene"),
        "start_time": scene.get("start_time", 0.0),
        "end_time": scene.get("end_time", 0.0),
        "subtitles": scene.get("subtitles", []),
        "characters_present": scene.get("characters_present", []),
    }
    return build_micro_beat_prompt(
        movie_title=movie_title,
        scene=scene,
        beat=pseudo_beat,
        sampled_kfs=sampled_kfs,
        prev_overlap_kfs=[],
        glossary=glossary,
        prev_beat_caption=prev_exit_state,
        prev_subtitles=prev_subtitles,
        act_plot_context=act_plot_context,
        appearance_windows=appearance_windows,
    )


def _parse_beat_caption(raw_text: str) -> str:
    """Clean micro-beat model output into plain prose (no [EXIT_STATE] section)."""
    text = (raw_text or "").strip()
    if not text:
        return ""

    if text.startswith("{") and text.endswith("}"):
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                cap = str(
                    data.get("beat_caption")
                    or data.get("vlm_caption")
                    or data.get("scene_description")
                    or ""
                ).strip()
                if cap:
                    return cap
        except Exception:
            pass

    split_match = re.split(
        r"(?:\n|^)\s*(?:\*\*|#*\s*)?\[?EXIT_STATE\]?(?:\*\*|:)?\s*\n?",
        text,
        maxsplit=1,
        flags=re.IGNORECASE,
    )
    desc_part = split_match[0].strip()
    desc_part = re.sub(
        r"^\s*(?:\*\*|#*\s*)?\[?(?:SCENE_DESCRIPTION|BEAT_DESCRIPTION)\]?(?:\*\*|:)?\s*\n?",
        "",
        desc_part,
        flags=re.IGNORECASE,
    ).strip()
    return desc_part


def _parse_scene_and_exit_state(raw_text: str) -> Tuple[str, str]:
    """Backward-compatible parser returning `(caption, '')` now that [EXIT_STATE] is retired."""
    return _parse_beat_caption(raw_text), ""


def _call_deepseek_vision(
    client,
    model_id: str,
    prompt_text: str,
    image_paths: List[str],
    prev_overlap_image_paths: Optional[List[str]] = None,
    max_image_width: int = 960,
    max_new_tokens: int = 1100,
    max_retries: int = 3,
) -> Tuple[str, Dict[str, int]]:
    """Send optional overlap context frames + beat keyframes + prompt to deepseek-flash with thinking disabled."""
    content_items: List[Dict[str, Any]] = []
    valid_overlap = [p for p in (prev_overlap_image_paths or []) if p and os.path.isfile(p)]
    valid_cur = [p for p in image_paths if p and os.path.isfile(p)]

    if valid_overlap:
        content_items.append(
            {
                "type": "text",
                "text": "[Context Only — End of Previous Beat, DO NOT re-describe these frames]",
            }
        )
        for p in valid_overlap:
            content_items.append(
                {
                    "type": "image_url",
                    "image_url": {"url": _encode_image_data_uri(p, max_width=max_image_width)},
                }
            )
        content_items.append(
            {
                "type": "text",
                "text": "[Current Beat Keyframes — Describe ONLY these frames chronologically]",
            }
        )

    for p in valid_cur:
        data_uri = _encode_image_data_uri(p, max_width=max_image_width)
        content_items.append(
            {
                "type": "image_url",
                "image_url": {"url": data_uri},
            }
        )
    content_items.append({"type": "text", "text": prompt_text})

    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert film analyst for a multimodal video retrieval system. "
                "Follow all formatting constraints strictly, stop immediately after the last frame, "
                "never repeat sentences, never mention synthetic bounding boxes, and never re-describe overlap context frames."
            ),
        },
        {"role": "user", "content": content_items},
    ]

    last_err: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=messages,
                max_tokens=max_new_tokens,
                temperature=0.2,
                frequency_penalty=0.2,
                extra_body={"thinking": {"type": "disabled"}},
            )
            out_text = (response.choices[0].message.content or "").strip()
            usage_obj = getattr(response, "usage", None)
            usage = {
                "prompt_tokens": int(getattr(usage_obj, "prompt_tokens", 0) or 0),
                "completion_tokens": int(getattr(usage_obj, "completion_tokens", 0) or 0),
                "total_tokens": int(getattr(usage_obj, "total_tokens", 0) or 0),
            }
            return out_text, usage
        except Exception as exc:
            last_err = exc
            # Fallback if endpoint rejects frequency_penalty
            if "frequency_penalty" in str(exc) and attempt == 1:
                try:
                    response = client.chat.completions.create(
                        model=model_id,
                        messages=messages,
                        max_tokens=max_new_tokens,
                        temperature=0.2,
                        extra_body={"thinking": {"type": "disabled"}},
                    )
                    out_text = (response.choices[0].message.content or "").strip()
                    usage_obj = getattr(response, "usage", None)
                    usage = {
                        "prompt_tokens": int(getattr(usage_obj, "prompt_tokens", 0) or 0),
                        "completion_tokens": int(getattr(usage_obj, "completion_tokens", 0) or 0),
                        "total_tokens": int(getattr(usage_obj, "total_tokens", 0) or 0),
                    }
                    return out_text, usage
                except Exception as inner_exc:
                    last_err = inner_exc
            if attempt < max_retries:
                wait_s = 2.0 * attempt
                print(f"[Stage A5] API attempt {attempt}/{max_retries} failed ({last_err}); retrying in {wait_s:.0f}s...")
                time.sleep(wait_s)

    raise RuntimeError(f"DeepSeek Vision API call failed after {max_retries} attempts: {last_err}")


def _prepare_scene_micro_beats(scene: Dict[str, Any], fps: float) -> List[Dict[str, Any]]:
    """Ensure a scene has populated 4-10s micro_beats with keyframes, subtitles, and active characters."""
    beats = scene.get("micro_beats")
    if not beats:
        beats = build_micro_beats(
            scene_id=scene["scene_id"],
            shot_ranges=scene.get("shot_ranges", []),
            fps=fps,
        )
        scene["micro_beats"] = beats

    if not beats:
        beats = [
            {
                "beat_id": f"{scene['scene_id']}_b00",
                "beat_index": 0,
                "scene_id": scene["scene_id"],
                "start_frame": int(scene.get("start_frame", 0)),
                "end_frame": int(scene.get("end_frame", 0)),
                "start_time": float(scene.get("start_time", 0.0)),
                "end_time": float(scene.get("end_time", 0.0)),
                "duration_sec": float(scene.get("duration_sec", 0.0)),
                "shot_indices": [int(s["shot_idx"]) for s in scene.get("shot_ranges", [])],
                "keyframes": list(scene.get("keyframes", [])),
            }
        ]
        scene["micro_beats"] = beats

    scene_kfs = sorted(scene.get("keyframes", []), key=lambda k: int(k.get("frame_idx", 0)))
    char_info_map = {
        cp["name"]: cp
        for cp in scene.get("characters_present", [])
        if isinstance(cp, dict) and cp.get("name")
    }

    for b_i, mb in enumerate(beats):
        is_last_b = b_i == len(beats) - 1
        b_sf = int(mb.get("start_frame", scene.get("start_frame", 0)))
        b_ef = int(mb.get("end_frame", scene.get("end_frame", 0)))
        b_st = float(mb.get("start_time", scene.get("start_time", 0.0)))
        b_et = float(mb.get("end_time", scene.get("end_time", 0.0)))

        if not mb.get("keyframes") and scene_kfs:
            mb_kfs: List[Dict[str, Any]] = []
            for kf in scene_kfs:
                if "frame_idx" in kf:
                    fi = int(kf["frame_idx"])
                    in_beat = (b_sf <= fi <= b_ef) if is_last_b else (b_sf <= fi < b_ef)
                else:
                    ts = float(kf.get("timestamp", 0.0))
                    in_beat = (b_st <= ts <= b_et) if is_last_b else (b_st <= ts < b_et)
                if in_beat:
                    mb_kfs.append(kf)
            mb["keyframes"] = mb_kfs

        if "characters_present" not in mb or mb["characters_present"] is None:
            mb_names: List[str] = []
            for kf in mb.get("keyframes", []):
                for cv_item in kf.get("characters_visible", []):
                    cname = cv_item.get("name")
                    if cname and cname not in mb_names:
                        mb_names.append(cname)
            mb["characters_present"] = [char_info_map[c] for c in mb_names if c in char_info_map]

        mb["subtitles"] = []

    for utt in scene.get("subtitles", []):
        mid_t = 0.5 * (float(utt["start_time"]) + float(utt["end_time"]))
        placed = False
        for mb in beats:
            if float(mb["start_time"]) <= mid_t <= float(mb["end_time"]):
                mb["subtitles"].append(utt)
                placed = True
                break
        if not placed:
            nearest = min(beats, key=lambda b: abs(0.5 * (float(b["start_time"]) + float(b["end_time"])) - mid_t))
            nearest["subtitles"].append(utt)

    return beats


def _select_prev_overlap_keyframes(
    prev_beat_kfs: List[Dict[str, Any]],
    max_frames: int = 4,
    window_sec: float = 1.0,
) -> List[Dict[str, Any]]:
    """Select trailing keyframes from the last `window_sec` of the previous beat, capped at `max_frames`."""
    if max_frames <= 0 or not prev_beat_kfs:
        return []
    ordered = sorted(
        prev_beat_kfs,
        key=lambda kf: (int(kf.get("frame_idx", 0)), float(kf.get("timestamp", 0.0))),
    )
    if window_sec <= 0.0:
        return ordered[-max_frames:]
    end_ts = float(ordered[-1].get("timestamp", 0.0))
    in_window = [kf for kf in ordered if float(kf.get("timestamp", 0.0)) >= (end_ts - window_sec - 1e-6)]
    if not in_window:
        return [ordered[-1]]
    if len(in_window) <= max_frames:
        return in_window
    if max_frames == 1:
        return [in_window[-1]]
    idxs = [round(i * (len(in_window) - 1) / float(max_frames - 1)) for i in range(max_frames)]
    return [in_window[i] for i in idxs]


def _caption_one_beat(
    client,
    vlm_model_id: str,
    movie_id: str,
    movie_title: str,
    scene: Dict[str, Any],
    beat: Dict[str, Any],
    resolve_kf_path: Callable[[str], str],
    glossary: Dict[str, str],
    external_docs: List[Dict[str, Any]],
    prev_beat_caption: str,
    prev_overlap_kfs: List[Dict[str, Any]],
    prev_subtitles: List[Dict[str, Any]],
    appearance_windows: Optional[Dict[str, List[List[float]]]] = None,
    max_keyframes_per_beat: int = 60,
    max_image_width: int = 960,
    min_keyframe_sharpness: float = 5.0,
) -> Tuple[str, List[str], List[str], Dict[str, int]]:
    """Run DeepSeek Vision API call for one 4-10s micro-beat."""
    raw_kfs = beat.get("keyframes", [])
    sampled_kfs = select_scene_keyframes(
        raw_kfs,
        max_frames=max_keyframes_per_beat,
        min_sharpness=min_keyframe_sharpness,
    )
    kf_abs_paths = [resolve_kf_path(kf["path"]) for kf in sampled_kfs]
    overlap_abs_paths = [resolve_kf_path(kf["path"]) for kf in prev_overlap_kfs]

    act_context = _extract_time_matched_plot_context(
        external_docs=external_docs,
        scene_start=float(beat["start_time"]),
        scene_end=float(beat["end_time"]),
        movie_id=movie_id,
    )
    prompt_text = build_micro_beat_prompt(
        movie_title=movie_title,
        scene=scene,
        beat=beat,
        sampled_kfs=sampled_kfs,
        prev_overlap_kfs=prev_overlap_kfs,
        glossary=glossary,
        prev_beat_caption=prev_beat_caption,
        prev_subtitles=prev_subtitles,
        act_plot_context=act_context,
        appearance_windows=appearance_windows,
    )
    raw_output, usage = _call_deepseek_vision(
        client=client,
        model_id=vlm_model_id,
        prompt_text=prompt_text,
        image_paths=kf_abs_paths,
        prev_overlap_image_paths=overlap_abs_paths,
        max_image_width=max_image_width,
        max_new_tokens=1100,
    )
    beat_caption = _parse_beat_caption(raw_output)
    return (
        beat_caption,
        [kf["path"] for kf in sampled_kfs],
        [kf["path"] for kf in prev_overlap_kfs],
        usage,
    )


def archive_previous_vlm_outputs(output_dir: str) -> List[str]:
    """Move previous vlm_beats/captions/bundle files into archive_prev_vlm/."""
    archive_dir = os.path.join(output_dir, "archive_prev_vlm")
    ts = time.strftime("%Y%m%d_%H%M%S")
    archived: List[str] = []

    for fname in ("vlm_beats.jsonl", "vlm_captions.jsonl", "movie_knowledge_bundle.json", "stage5_vlm_summary.json"):
        src = os.path.join(output_dir, fname)
        if os.path.isfile(src):
            os.makedirs(archive_dir, exist_ok=True)
            dst = os.path.join(archive_dir, f"{ts}_{fname}")
            shutil.move(src, dst)
            archived.append(dst)

    return archived


def run_single_scene_diagnostic(
    output_dir: str,
    scene_id: Optional[str] = None,
    scene_index: Optional[int] = 4,
    beat_index: Optional[int] = None,
    movie_title: str = "Sintel",
    char_bank_dir: Optional[str] = None,
    glossary_file: Optional[str] = None,
    appearance_windows: Optional[Dict[str, List[List[float]]]] = None,
    vlm_model_id: str = "deepseek-flash",
    api_key: Optional[str] = None,
    api_base_url: str = "https://api.deepseek.com",
    max_keyframes_per_scene: int = 150,
    max_keyframes_per_beat: int = 60,
    prev_overlap_frames: int = 4,
    prev_overlap_sec: float = 1.0,
    max_image_width: int = 960,
    min_keyframe_sharpness: float = 5.0,
) -> Dict[str, Any]:
    """Run a diagnostic across micro-beats (or one beat) of a Scene Window to inspect quality & token usage."""
    shots_file = os.path.join(output_dir, "shots_and_scenes.json")
    with open(shots_file, "r", encoding="utf-8") as f:
        shots_and_scenes = json.load(f)

    movie_id = shots_and_scenes["movie_id"]
    fps = float(shots_and_scenes.get("fps") or 24.0)
    duration_sec = float(shots_and_scenes.get("duration_sec") or 0.0)
    scenes = shots_and_scenes.get("scenes", [])
    if not duration_sec and scenes:
        duration_sec = float(scenes[-1]["end_time"])

    glossary = _load_character_glossary(output_dir, char_bank_dir=char_bank_dir, glossary_file=glossary_file)
    resolved_windows = _load_appearance_windows(
        output_dir,
        char_bank_dir=char_bank_dir,
        glossary_file=glossary_file,
        appearance_windows=appearance_windows,
    )
    external_docs = _load_external_plot_documents(
        movie_id=movie_id,
        duration_sec=duration_sec,
        char_bank_dir=char_bank_dir,
    )
    resolve_kf_path = _stage_keyframes_to_local(output_dir, movie_id)

    target_idx = 0
    for idx, sc in enumerate(scenes):
        if (scene_id and sc.get("scene_id") == scene_id) or (scene_index is not None and sc.get("scene_index") == scene_index):
            target_idx = idx
            break

    scene = scenes[target_idx]
    beats = _prepare_scene_micro_beats(scene, fps=fps)

    # Seed overlap from previous scene's final keyframes / cached caption if available
    prev_beat_caption = ""
    prev_overlap_kfs: List[Dict[str, Any]] = []
    prev_subtitles: List[Dict[str, Any]] = []
    if target_idx > 0:
        prev_scene = scenes[target_idx - 1]
        prev_beats = _prepare_scene_micro_beats(prev_scene, fps=fps)
        prev_subtitles = prev_scene.get("subtitles", [])
        if prev_beats and prev_overlap_frames > 0:
            prev_overlap_kfs = _select_prev_overlap_keyframes(
                prev_beats[-1].get("keyframes", []),
                max_frames=prev_overlap_frames,
                window_sec=prev_overlap_sec,
            )
        elif prev_scene.get("keyframes") and prev_overlap_frames > 0:
            prev_overlap_kfs = _select_prev_overlap_keyframes(
                prev_scene.get("keyframes", []),
                max_frames=prev_overlap_frames,
                window_sec=prev_overlap_sec,
            )
        if prev_scene.get("vlm_caption"):
            prev_beat_caption = str(prev_scene["vlm_caption"])[-600:]

    client = create_deepseek_client(api_key=api_key, api_base_url=api_base_url)
    target_beats = [b for b in beats if beat_index is None or int(b.get("beat_index", -1)) == beat_index]

    beat_outputs: List[Dict[str, Any]] = []
    total_prompt = 0
    total_comp = 0
    t0_all = time.perf_counter()

    for b in target_beats:
        t0 = time.perf_counter()
        cap, used_kfs, overlap_used, usage = _caption_one_beat(
            client=client,
            vlm_model_id=vlm_model_id,
            movie_id=movie_id,
            movie_title=movie_title,
            scene=scene,
            beat=b,
            resolve_kf_path=resolve_kf_path,
            glossary=glossary,
            external_docs=external_docs,
            prev_beat_caption=prev_beat_caption,
            prev_overlap_kfs=prev_overlap_kfs,
            prev_subtitles=prev_subtitles,
            appearance_windows=resolved_windows,
            max_keyframes_per_beat=max_keyframes_per_beat,
            max_image_width=max_image_width,
            min_keyframe_sharpness=min_keyframe_sharpness,
        )
        elapsed_b = time.perf_counter() - t0
        total_prompt += usage["prompt_tokens"]
        total_comp += usage["completion_tokens"]
        beat_outputs.append(
            {
                "beat_id": b["beat_id"],
                "start_time": b["start_time"],
                "end_time": b["end_time"],
                "keyframes_used": len(used_kfs),
                "overlap_used": len(overlap_used),
                "caption": cap,
                "elapsed_sec": round(elapsed_b, 2),
                "token_usage": usage,
            }
        )
        prev_beat_caption = cap
        prev_overlap_kfs = _select_prev_overlap_keyframes(
            b.get("keyframes", []),
            max_frames=prev_overlap_frames,
            window_sec=prev_overlap_sec,
        )
        prev_subtitles = b.get("subtitles", [])

    merged_caption = "\n\n".join(
        f"[{_format_mmss(bo['start_time'])} - {_format_mmss(bo['end_time'])}] {bo['caption']}"
        for bo in beat_outputs
        if bo["caption"]
    )
    elapsed_all = time.perf_counter() - t0_all

    result = {
        "scene_id": scene["scene_id"],
        "time_range": f"{_format_mmss(scene['start_time'])} - {_format_mmss(scene['end_time'])}",
        "num_beats_run": len(beat_outputs),
        "beats": beat_outputs,
        "vlm_caption": merged_caption,
        "elapsed_sec": round(elapsed_all, 2),
        "token_usage": {
            "prompt_tokens": total_prompt,
            "completion_tokens": total_comp,
            "total_tokens": total_prompt + total_comp,
        },
    }
    print(
        f"[Diagnostic {result['scene_id']}] {len(beat_outputs)} beat(s) | "
        f"{elapsed_all:.1f}s | Tokens: {total_prompt} in / {total_comp} out"
    )
    for bo in beat_outputs:
        print(
            f"\n--- {bo['beat_id']} [{_format_mmss(bo['start_time'])} - {_format_mmss(bo['end_time'])}] "
            f"({bo['keyframes_used']} kfs + {bo['overlap_used']} overlap) ---\n{bo['caption']}"
        )
    return result


def run_stage5_vlm_caption(
    output_dir: str,
    movie_title: Optional[str] = None,
    franchise_id: Optional[str] = None,
    char_bank_dir: Optional[str] = None,
    glossary_file: Optional[str] = None,
    external_plot_file: Optional[str] = None,
    appearance_windows: Optional[Dict[str, List[List[float]]]] = None,
    vlm_model_id: str = "deepseek-flash",
    api_key: Optional[str] = None,
    api_base_url: str = "https://api.deepseek.com",
    max_keyframes_per_scene: int = 150,
    max_keyframes_per_beat: int = 60,
    prev_overlap_frames: int = 4,
    prev_overlap_sec: float = 1.0,
    max_image_width: int = 960,
    min_keyframe_sharpness: float = 5.0,
    credits_start_sec: Optional[Union[str, float, int]] = None,
    restart: bool = False,
) -> Dict[str, Any]:
    """Run Micro-Beat DeepSeek-Flash captioning and deterministically assemble 15-30s scenes into movie_knowledge_bundle.json."""
    shots_file = os.path.join(output_dir, "shots_and_scenes.json")
    if not os.path.isfile(shots_file):
        raise FileNotFoundError(f"shots_and_scenes.json not found: {shots_file}")

    if credits_start_sec is not None:
        apply_credits_cutoff_to_shots_file(shots_file, credits_start_sec)

    if restart:
        archived_files = archive_previous_vlm_outputs(output_dir)
        if archived_files:
            print(f"[Stage A5] Archived previous VLM output files -> {[os.path.basename(p) for p in archived_files]}")

    with open(shots_file, "r", encoding="utf-8") as f:
        shots_and_scenes = json.load(f)

    cutoff_sec = parse_timestamp_sec(credits_start_sec) or parse_timestamp_sec(shots_and_scenes.get("credits_start_sec"))
    movie_id = shots_and_scenes["movie_id"]
    resolved_title = movie_title or movie_id.replace("_", " ").title()
    resolved_franchise = franchise_id or movie_id
    fps = float(shots_and_scenes.get("fps") or 24.0)
    duration_sec = float(shots_and_scenes.get("duration_sec") or 0.0)
    all_scenes = shots_and_scenes.get("scenes", [])
    scenes = [
        sc for sc in all_scenes
        if not sc.get("is_credits", False) and (cutoff_sec is None or float(sc.get("start_time", 0.0)) < cutoff_sec)
    ]
    if len(scenes) < len(all_scenes):
        print(
            f"[Stage A5] Excluded {len(all_scenes) - len(scenes)} end-credit scene(s) "
            f"(credits_start_sec={cutoff_sec}s); captioning {len(scenes)} narrative scenes."
        )
    if not duration_sec and scenes:
        duration_sec = float(scenes[-1]["end_time"])

    # Always refresh scene subtitles from attributed_subtitles.json if present
    subtitles_file = os.path.join(output_dir, "attributed_subtitles.json")
    if os.path.isfile(subtitles_file):
        with open(subtitles_file, "r", encoding="utf-8") as f:
            sub_payload = json.load(f)
        for sc in scenes:
            sc["subtitles"] = []
        for utt in sub_payload.get("utterances", []):
            mid_t = 0.5 * (float(utt["start_time"]) + float(utt["end_time"]))
            placed = False
            for sc in scenes:
                if float(sc["start_time"]) <= mid_t <= float(sc["end_time"]):
                    sc["subtitles"].append(utt)
                    placed = True
                    break
            if not placed and scenes:
                nearest = min(scenes, key=lambda s: abs(0.5 * (s["start_time"] + s["end_time"]) - mid_t))
                nearest["subtitles"].append(utt)

    # Prepare 4-10s micro-beats across every narrative scene
    all_beats_ordered: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    for sc in scenes:
        beats = _prepare_scene_micro_beats(sc, fps=fps)
        for b in beats:
            all_beats_ordered.append((sc, b))

    glossary = _load_character_glossary(
        output_dir=output_dir,
        char_bank_dir=char_bank_dir,
        glossary_file=glossary_file,
    )
    resolved_windows = _load_appearance_windows(
        output_dir=output_dir,
        char_bank_dir=char_bank_dir,
        glossary_file=glossary_file,
        appearance_windows=appearance_windows,
    )
    external_docs = _load_external_plot_documents(
        movie_id=movie_id,
        duration_sec=duration_sec,
        char_bank_dir=char_bank_dir,
        external_plot_file=external_plot_file,
    )

    beats_ckpt_jsonl = os.path.join(output_dir, "vlm_beats.jsonl")
    completed_beats_by_id: Dict[str, Dict[str, Any]] = {}
    if os.path.isfile(beats_ckpt_jsonl):
        with open(beats_ckpt_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                    if rec.get("beat_id") and rec.get("caption"):
                        completed_beats_by_id[rec["beat_id"]] = rec
                except json.JSONDecodeError:
                    continue

    pending_beats = [b for _, b in all_beats_ordered if b["beat_id"] not in completed_beats_by_id]
    print(
        f"[Stage A5] Model: {vlm_model_id} | Scenes: {len(scenes)} | Micro-Beats: {len(all_beats_ordered)} | "
        f"Cached: {len(completed_beats_by_id)} | Pending: {len(pending_beats)}"
    )

    total_prompt_tokens = 0
    total_completion_tokens = 0

    if pending_beats:
        resolve_kf_path = _stage_keyframes_to_local(output_dir, movie_id)
        client = create_deepseek_client(api_key=api_key, api_base_url=api_base_url)

        with open(beats_ckpt_jsonl, "a", encoding="utf-8") as out_f:
            prev_beat_caption = ""
            prev_beat_kfs: List[Dict[str, Any]] = []
            prev_subtitles: List[Dict[str, Any]] = []

            for b_pos, (sc, beat) in enumerate(all_beats_ordered):
                bid = beat["beat_id"]
                if bid in completed_beats_by_id:
                    cached = completed_beats_by_id[bid]
                    beat["caption"] = cached["caption"]
                    prev_beat_caption = beat["caption"] or prev_beat_caption
                    prev_beat_kfs = beat.get("keyframes", [])
                    prev_subtitles = beat.get("subtitles", [])
                    continue

                overlap_kfs = _select_prev_overlap_keyframes(
                    prev_beat_kfs,
                    max_frames=prev_overlap_frames,
                    window_sec=prev_overlap_sec,
                )
                t0 = time.perf_counter()
                beat_caption, used_kfs, overlap_used, usage = _caption_one_beat(
                    client=client,
                    vlm_model_id=vlm_model_id,
                    movie_id=movie_id,
                    movie_title=resolved_title,
                    scene=sc,
                    beat=beat,
                    resolve_kf_path=resolve_kf_path,
                    glossary=glossary,
                    external_docs=external_docs,
                    prev_beat_caption=prev_beat_caption,
                    prev_overlap_kfs=overlap_kfs,
                    prev_subtitles=prev_subtitles,
                    appearance_windows=resolved_windows,
                    max_keyframes_per_beat=max_keyframes_per_beat,
                    max_image_width=max_image_width,
                    min_keyframe_sharpness=min_keyframe_sharpness,
                )
                elapsed = time.perf_counter() - t0
                total_prompt_tokens += usage["prompt_tokens"]
                total_completion_tokens += usage["completion_tokens"]

                beat["caption"] = beat_caption
                record = {
                    "beat_id": bid,
                    "scene_id": sc["scene_id"],
                    "beat_index": int(beat.get("beat_index", b_pos)),
                    "start_time": beat["start_time"],
                    "end_time": beat["end_time"],
                    "caption": beat_caption,
                    "sampled_keyframe_paths": used_kfs,
                    "overlap_keyframe_paths": overlap_used,
                    "token_usage": usage,
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_f.flush()
                completed_beats_by_id[bid] = record

                prev_beat_caption = beat_caption
                prev_beat_kfs = beat.get("keyframes", [])
                prev_subtitles = beat.get("subtitles", [])
                _trim_host_memory()

                print(
                    f"[Beat {b_pos + 1:03d}/{len(all_beats_ordered):03d} ({bid})] "
                    f"kfs: {len(used_kfs)} (+{len(overlap_used)} overlap) | "
                    f"subs: {len(beat.get('subtitles', []))} | "
                    f"{elapsed:.1f}s | tokens: {usage['prompt_tokens']} in / {usage['completion_tokens']} out",
                    flush=True,
                )
    else:
        for _, beat in all_beats_ordered:
            cached = completed_beats_by_id.get(beat["beat_id"], {})
            beat["caption"] = cached.get("caption", "")

    # Pure deterministic scene merge ($0): concatenate beat captions inside each 15-30s Scene Window
    bundle_scenes: List[Dict[str, Any]] = []
    captions_jsonl = os.path.join(output_dir, "vlm_captions.jsonl")
    with open(captions_jsonl, "w", encoding="utf-8") as cap_f:
        for idx, sc in enumerate(scenes):
            scene_beats = sc.get("micro_beats", [])
            merged_caption = "\n\n".join(
                f"[{_format_mmss(b['start_time'])} - {_format_mmss(b['end_time'])}] {b['caption']}"
                for b in scene_beats
                if b.get("caption")
            )
            sc["vlm_caption"] = merged_caption
            sc["exit_state"] = ""

            # Aggregate deduplicated, chronologically sorted keyframes across scene + micro-beats
            kf_by_key: Dict[Any, Dict[str, Any]] = {}
            for kf in sc.get("keyframes", []):
                key = int(kf["frame_idx"]) if "frame_idx" in kf else kf.get("path")
                kf_by_key[key] = kf
            for b in scene_beats:
                for kf in b.get("keyframes", []):
                    key = int(kf["frame_idx"]) if "frame_idx" in kf else kf.get("path")
                    if key not in kf_by_key:
                        kf_by_key[key] = kf
            merged_kfs = sorted(kf_by_key.values(), key=lambda k: int(k.get("frame_idx", 0)))
            sc["keyframes"] = merged_kfs

            bundle_scenes.append(
                {
                    "scene_id": sc["scene_id"],
                    "scene_index": int(sc.get("scene_index", idx)),
                    "start_time": round(float(sc["start_time"]), 3),
                    "end_time": round(float(sc["end_time"]), 3),
                    "duration_sec": round(float(sc.get("duration_sec", sc["end_time"] - sc["start_time"])), 3),
                    "shot_ranges": sc.get("shot_ranges", []),
                    "characters_present": sc.get("characters_present", []),
                    "keyframes": merged_kfs,
                    "subtitles": sc.get("subtitles", []),
                    "vlm_caption": merged_caption,
                    "exit_state": "",
                }
            )
            cap_f.write(
                json.dumps(
                    {
                        "scene_id": sc["scene_id"],
                        "scene_index": int(sc.get("scene_index", idx)),
                        "vlm_caption": merged_caption,
                        "exit_state": "",
                        "beat_ids": [b["beat_id"] for b in scene_beats],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    bundle = {
        "movie_id": movie_id,
        "movie_title": resolved_title,
        "franchise_id": resolved_franchise,
        "duration_sec": round(duration_sec, 3),
        "fps": round(fps, 4),
        "character_glossary": glossary,
        "external_plot_documents": external_docs,
        "scenes": bundle_scenes,
    }

    with open(shots_file, "w", encoding="utf-8") as f:
        json.dump(shots_and_scenes, f, indent=2, ensure_ascii=False)

    bundle_path = os.path.join(output_dir, "movie_knowledge_bundle.json")
    with open(bundle_path, "w", encoding="utf-8") as f:
        json.dump(bundle, f, indent=2, ensure_ascii=False)

    summary = {
        "movie_id": movie_id,
        "movie_title": resolved_title,
        "vlm_model_id": vlm_model_id,
        "total_scenes": len(bundle_scenes),
        "scenes_captioned": sum(1 for s in bundle_scenes if s.get("vlm_caption")),
        "total_beats": len(all_beats_ordered),
        "beats_captioned": sum(1 for _, b in all_beats_ordered if b.get("caption")),
        "external_plot_documents": len(external_docs),
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "bundle_path": bundle_path,
        "bundle_size_mb": round(os.path.getsize(bundle_path) / (1024**2), 3),
    }
    summary_path = os.path.join(output_dir, "stage5_vlm_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[Stage A5] Saved final bundle -> {bundle_path} ({summary['bundle_size_mb']} MB)")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage A5: DeepSeek-Flash Vision API Micro-Beat Captioning")
    parser.add_argument("--output_dir", required=True, help="Workdir containing shots_and_scenes.json")
    parser.add_argument("--movie_title", default=None, help="Display title for movie")
    parser.add_argument("--franchise_id", default=None, help="Franchise identifier")
    parser.add_argument("--char_bank_dir", default=None, help="Character bank directory")
    parser.add_argument("--glossary_file", default=None, help="Path to character_glossary.json")
    parser.add_argument("--external_plot_file", default=None, help="Optional Tier-3 plot file")
    parser.add_argument("--vlm_model_id", default="deepseek-flash")
    parser.add_argument("--api_base_url", default="https://api.deepseek.com")
    parser.add_argument("--max_keyframes_per_scene", type=int, default=150)
    parser.add_argument("--max_keyframes_per_beat", type=int, default=60)
    parser.add_argument("--prev_overlap_frames", type=int, default=4)
    parser.add_argument("--prev_overlap_sec", type=float, default=1.0)
    parser.add_argument("--max_image_width", type=int, default=960)
    parser.add_argument("--min_keyframe_sharpness", type=float, default=5.0)
    parser.add_argument(
        "--credits_start_sec",
        default=None,
        type=str,
        help="Optional timestamp (e.g. '744.0' or '12:24') where closing credits start",
    )
    parser.add_argument("--restart", action="store_true", help="Archive and ignore cached vlm_beats/captions")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_stage5_vlm_caption(
        output_dir=args.output_dir,
        movie_title=args.movie_title,
        franchise_id=args.franchise_id,
        char_bank_dir=args.char_bank_dir,
        glossary_file=args.glossary_file,
        external_plot_file=args.external_plot_file,
        vlm_model_id=args.vlm_model_id,
        api_base_url=args.api_base_url,
        max_keyframes_per_scene=args.max_keyframes_per_scene,
        max_keyframes_per_beat=args.max_keyframes_per_beat,
        prev_overlap_frames=args.prev_overlap_frames,
        prev_overlap_sec=args.prev_overlap_sec,
        max_image_width=args.max_image_width,
        min_keyframe_sharpness=args.min_keyframe_sharpness,
        credits_start_sec=args.credits_start_sec,
        restart=args.restart,
    )
