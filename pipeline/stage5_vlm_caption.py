"""Stage A5: Single-Pass DeepSeek-Flash Vision API Scene Captioning & Knowledge Bundle Assembly.

Implements Phase A Stage A5 for MovieRAG:
1. Filters out blurry / fade-to-black keyframes (`sharpness < min_sharpness`) and selects up to
   `max_keyframes_per_scene` (default 30, i.e. 1 frame per 1s across a 15-30s Scene Window)
   preserving start/end anchors, pinned timestamps, and character boxes.
2. Stages keyframes to fast local SSD (/content/vlm_scratch) in Colab and downscales frames
   in memory (default max_image_width=960) before Base64 JPEG encoding to cut image token
   usage roughly in half without losing scene detail.
3. Supports manual `credits_start_sec` cutoff (e.g. `744.0` or `"12:24"`) so end-credit scenes
   are excluded before calling the VLM API or indexing into `movie_knowledge_bundle.json`.
4. Calls `deepseek-flash` via the OpenAI-compatible DeepSeek Vision API (`thinking={"type": "disabled"}`)
   in a single structured pass per scene to generate both:
   - `vlm_caption`: Detailed action-dense chronological narrative (180-350 words) without repetitive
     character bio boilerplate.
   - `exit_state`: Strict 2-3 sentence physical & spatial state ledger (location, character positions
     & conditions, held/dropped weapons & props) for clean continuity handoff.
5. Grounds each scene with the full Character Reference Glossary (`Sintel`, `Scales`, `Adult Dragon`,
   `Shaman`, `Gatekeeper`) and the time-matched Act synopsis from `external_plot.txt`.
6. Checkpoints every completed scene immediately to `vlm_captions.jsonl` and packages
   `movie_knowledge_bundle.json`.
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
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from pipeline import PROJECT_ROOT
from pipeline.stage1_scenes import apply_credits_cutoff_to_shots_file, parse_timestamp_sec


SINTEL_FALLBACK_PLOT = (
    "Sintel (2010) — Full Plot & Ending Twist Synopsis:\n\n"
    "Act I — The Snowstorm and the Shaman's Hut (00:00 - 03:35):\n"
    "Sintel, a lone young woman warrior bundled in a fur-lined cloak and carrying a wooden staff, "
    "wanders across a desolate, snow-swept mountain wilderness searching for someone. She is ambushed "
    "by a masked gatekeeper warrior, whom she defeats after a tense close-quarters fight, claiming his "
    "distinctive spear-blade before collapsing in the freezing storm. She awakens inside a warmly lit "
    "desert hut tended by an elderly tribal Shaman with white dreadlocks, who offers her shelter and "
    "asks what brings her to the land of the gatekeepers. Sintel reveals that she is searching the "
    "world for a dragon companion who was taken from her.\n\n"
    "Act II — Flashback in Ishtar and the Kidnapping of Scales (03:35 - 05:15):\n"
    "In a flashback to Sintel's earlier life in the bustling sunlit city of Ishtar, she scavenges for "
    "food and comes across a wounded, frightened baby red dragon with an injured wing. She brings the "
    "hatchling back to her rooftop shelter, feeds him, wraps a makeshift bandage around his torn wing, "
    "and names him Scales. Over time, Sintel and Scales form a deep affectionate bond as she heals him "
    "and teaches him how to fly again. One day, as Scales soars above the rooftops of Ishtar, a giant "
    "winged Adult Dragon descends out of the clouds, snatches Scales in its talons, and carries him "
    "away into the sky despite Sintel's desperate chase.\n\n"
    "Act III — The Long Quest and the Mountain Lair (05:15 - 10:15):\n"
    "Vowing to bring Scales back, Sintel sets out on an arduous cross-continental quest across icy "
    "mountain ridges, misty bamboo forests, and scorching dunes—growing visibly older and more weathered "
    "over years of fruitless searching before collapsing in the desert and waking up in the Shaman's hut. "
    "Encouraged by the Shaman's warning that some things cannot be found the way we expect, Sintel climbs "
    "a jagged volcanic peak and enters a massive cavern lair. Deep inside the cave, she spots a giant "
    "sleeping Adult Dragon curled beside a small baby red dragon. Hoping to rescue the young dragon "
    "quietly, she creeps toward it.\n\n"
    "Act IV — The Battle and the Wing-Scar Twist (10:15 - 14:48):\n"
    "Before Sintel can escape with the hatchling, the giant Adult Dragon wakes up and attacks furiously "
    "to protect the cavern. In the brutal clash, the Adult Dragon pins Sintel helplessly to the cavern "
    "floor and prepares a fatal strike, but abruptly pauses as it stares closely into Sintel's eyes—"
    "recognizing her and pulling back instead of killing her. Unaware of why the beast hesitated, Sintel "
    "seizes the opening and drives her spear straight into the Adult Dragon's chest. As the giant dragon "
    "collapses dying on the stone floor (~11:15), Sintel notices a familiar healed notch and torn "
    "bandage-wrap scar along its wing—the exact wound she bound on Scales years ago. Realizing too late "
    "that far more years had passed than she knew, the giant Adult Dragon she has just slain was actually "
    "her grown-up Scales, and the little red dragon cowering nearby is Scales's own offspring. Shattered "
    "by grief, Sintel drops her staff and walks blindly out into the falling snow while the orphaned baby "
    "dragon quietly trails after her."
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


def _load_character_glossary(
    output_dir: str,
    char_bank_dir: Optional[str] = None,
    glossary_file: Optional[str] = None,
) -> Dict[str, str]:
    """Load and merge static character glossary so newly added entities (e.g. Adult Dragon, Gatekeeper) are always present."""
    merged: Dict[str, str] = {}

    # 1. Start with stage2_charbank_summary.json as base if present
    stage2_summary = os.path.join(output_dir, "stage2_charbank_summary.json")
    if os.path.isfile(stage2_summary):
        try:
            with open(stage2_summary, "r", encoding="utf-8") as f:
                s2 = json.load(f)
            if isinstance(s2.get("glossary"), dict):
                for k, v in s2["glossary"].items():
                    merged[str(k)] = str(v)
        except Exception:
            pass

    # 2. Overlay repo char_bank, Drive char_bank, workdir glossary, and explicit glossary_file
    movie_slug = os.path.basename(output_dir.rstrip("/"))
    candidates: List[str] = [
        str(PROJECT_ROOT / "char_bank" / movie_slug / "character_glossary.json"),
        os.path.join(output_dir, "character_glossary.json"),
    ]
    if char_bank_dir:
        candidates.append(os.path.join(char_bank_dir, "character_glossary.json"))
    if glossary_file:
        candidates.append(glossary_file)

    for cand in candidates:
        if cand and os.path.isfile(cand):
            try:
                with open(cand, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    for k, v in data.items():
                        merged[str(k)] = str(v)
            except Exception:
                pass
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
    if movie_id.lower() == "sintel" and scene_start >= 744.0:
        return (
            "Closing Credits Sequence (12:24 - 14:48): Rolling end credits and production artwork/props "
            "following the story's conclusion. Describe the visual credits/artwork concisely without "
            "repeating sentences or inventing new plot actions."
        )

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
) -> Tuple[str, str]:
    """Build synthetic box legend and ALWAYS include the full character glossary."""
    char_colors: Dict[str, str] = {}
    for cp in scene.get("characters_present", []):
        name = cp.get("name")
        color = cp.get("color_prompt")
        if name and color:
            char_colors[name] = color

    for kf in sampled_kfs:
        for cv in kf.get("characters_visible", []):
            name = cv.get("name")
            color = cv.get("color")
            if name and color and name not in char_colors:
                char_colors[name] = color

    if char_colors:
        legend_items = [f"{color} rectangle = {name}" for name, color in char_colors.items()]
        legend_str = ", ".join(legend_items)
    else:
        legend_str = "No synthetic tracking boxes drawn in this scene."

    # Always include ALL glossary characters so missed dark-scene detections never hide character traits
    glossary_lines = [f"- {name}: {desc}" for name, desc in glossary.items()]
    glossary_str = "\n".join(glossary_lines) if glossary_lines else "- No glossary entries applicable."
    return legend_str, glossary_str


def build_single_pass_prompt(
    movie_title: str,
    scene: Dict[str, Any],
    sampled_kfs: List[Dict[str, Any]],
    glossary: Dict[str, str],
    prev_exit_state: str,
    prev_subtitles: List[Dict[str, Any]],
    act_plot_context: str = "",
) -> str:
    """Build the single-pass prompt returning both [SCENE_DESCRIPTION] and [EXIT_STATE]."""
    t0_str = _format_mmss(scene["start_time"])
    t1_str = _format_mmss(scene["end_time"])
    legend_str, glossary_str = _format_color_legend_and_glossary(scene, sampled_kfs, glossary)

    kf_timeline = ", ".join(
        f"Frame {idx + 1} ({_format_mmss(kf.get('timestamp', 0.0))}, shot {kf.get('shot_idx')})"
        for idx, kf in enumerate(sampled_kfs)
    )

    if prev_subtitles:
        prev_subs_str = "\n".join(
            f"- [{_format_mmss(u['start_time'])}] {u.get('speaker', 'Unknown')}: \"{u['text']}\""
            for u in prev_subtitles[-3:]
        )
    else:
        prev_subs_str = "- (No recent dialogue)"

    cur_subs = scene.get("subtitles", [])
    if cur_subs:
        cur_subs_str = "\n".join(
            f"- [{_format_mmss(u['start_time'])} - {_format_mmss(u['end_time'])}] "
            f"{u.get('speaker', 'Unknown')}: \"{u['text']}\""
            for u in cur_subs
        )
        dialogue_guidance = "Integrate what is being said naturally alongside the physical actions and reactions."
    else:
        cur_subs_str = "- [No spoken dialogue in this scene]"
        dialogue_guidance = (
            "There is NO spoken dialogue in this scene. Focus on silent visual storytelling: "
            "movements, environment, objects, combat/actions, posture, and facial expressions."
        )

    act_block = ""
    if act_plot_context:
        act_block = (
            f"[Time-Matched Act Reference (Narrative Grounding)]\n"
            f"{act_plot_context}\n"
            f"IMPORTANT GROUNDING RULES:\n"
            f"1. Use this Act Reference to accurately recognize characters, creatures (such as the masked Gatekeeper, "
            f"baby Scales, or the giant Adult Dragon), weapons (wooden staff vs. Gatekeeper's curved spear-blade with a "
            f"circular hilt ring), and the true setting—especially during fast action cuts, travel montages, or dark cavern shots.\n"
            f"2. NEVER invent non-existent monsters (such as wolves), indoor trees, or magical glowing shapes that contradict "
            f"the Act Reference, and NEVER omit critical physical plot beats visible in this window [{t0_str} - {t1_str}] "
            f"(such as disarming the Gatekeeper and claiming his spear-blade, driving the spear into the Adult Dragon's chest "
            f"after it hesitates and discovering its healed wing-scar, or the orphaned baby dragon quietly following Sintel).\n"
            f"3. Describe ONLY the events occurring within [{t0_str} - {t1_str}] shown in the attached keyframes.\n\n"
        )

    return (
        f"You are analyzing Scene {scene.get('scene_index', 0)} [{t0_str} - {t1_str}] from the animated film '{movie_title}'.\n"
        f"Number of chronological keyframes attached: {len(sampled_kfs)}\n"
        f"Keyframe Timeline: {kf_timeline}\n\n"
        f"[Character Reference Glossary]\n{glossary_str}\n"
        f"IMPORTANT ANTI-BOILERPLATE & APPEARANCE RULES:\n"
        f"1. Refer to known characters directly by name (`Sintel`, `Scales`, `Adult Dragon`, `Shaman`, `Gatekeeper`).\n"
        f"2. DO NOT repeat static character bio introductions from the glossary in every scene (e.g., do NOT start scenes "
        f"with boilerplate like 'Sintel, a young woman with short dark-brown hair wearing a brown tunic...'). Only describe "
        f"physical traits when they visibly change or are directly relevant to the action in this scene (such as injuries, "
        f"aging/longer hair in Act IV, the healed wing-scar on the Adult Dragon, or weapon swaps).\n"
        f"3. Respect canonical traits (Sintel has dark-brown hair, NOT red hair; Shaman has white dreadlocks, no mask; "
        f"Gatekeeper wears a full ceremonial mask).\n\n"
        f"[Synthetic Tracking Box Overlay Key]\n"
        f"Overlay key: {legend_str}\n"
        f"CRITICAL VISUAL RULE: Any colored rectangles drawn on the frames are synthetic computer-vision tracking overlays. "
        f"NEVER mention bounding boxes, rectangles, box colors, or 'Frame 1/2' labels in your output, and NEVER confuse "
        f"a red/green/blue tracking box with hair color, clothing color, or glowing objects.\n\n"
        f"{act_block}"
        f"[Previous Scene Continuity (Physical Entry State)]\n"
        f"State entering this scene: {prev_exit_state or 'Opening sequence of the film.'}\n"
        f"Recent lines before this scene:\n{prev_subs_str}\n\n"
        f"[Spoken Dialogue in This Scene]\n{cur_subs_str}\n\n"
        f"[Task]\n"
        f"Respond using EXACTLY the two section headers below:\n\n"
        f"[SCENE_DESCRIPTION]\n"
        f"Write ONE rich, cohesive, action-dense narrative description (180-350 words, scaling naturally with the amount "
        f"of action—stop immediately once the final frame is described and NEVER repeat sentences) in present tense:\n"
        f"1. Setting & Staging: Describe the specific environment, lighting, weather, camera framing, and spatial layout.\n"
        f"2. Chronological Action & Cause-and-Effect: Trace step-by-step what happens across the keyframes, paying close "
        f"attention to combat strikes, weapon/prop acquisitions or drops, character movements, and transitions. {dialogue_guidance}\n"
        f"3. Micro-Expressions & Physical Reactions: Describe eye contact, posture shifts, wounds, and concrete reactions "
        f"without repeating generic character bios.\n\n"
        f"[EXIT_STATE]\n"
        f"Write a strict, concrete 2-3 sentence PHYSICAL & SPATIAL STATE LEDGER at the exact end of this scene:\n"
        f"- Sentence 1 (Location & Characters): Exact ending location and each present character's physical position, "
        f"posture, and condition (e.g., standing, kneeling, collapsed, unconscious, slain, walking away, trailing behind).\n"
        f"- Sentence 2 (Props & Trajectory): Exact weapons or objects currently held, dropped, or acquired by each character "
        f"and their immediate physical trajectory entering the next scene.\n"
        f"DO NOT include abstract emotional/thematic commentary or formulaic phrases like 'The immediate tension lies in...' "
        f"or 'The emotional tone is...'."
    )


def _parse_scene_and_exit_state(raw_text: str) -> Tuple[str, str]:
    """Parse [SCENE_DESCRIPTION] and [EXIT_STATE] sections cleanly from model output."""
    text = (raw_text or "").strip()
    if not text:
        return "", ""

    # Handle optional JSON output if model ever wraps response in JSON
    if text.startswith("{") and text.endswith("}"):
        try:
            data = json.loads(text)
            if isinstance(data, dict) and ("vlm_caption" in data or "scene_description" in data):
                cap = str(data.get("vlm_caption") or data.get("scene_description") or "").strip()
                ex = str(data.get("exit_state") or "").strip()
                return cap, ex
        except Exception:
            pass

    split_match = re.split(
        r"(?:\n|^)\s*(?:\*\*|#*\s*)?\[?EXIT_STATE\]?(?:\*\*|:)?\s*\n?",
        text,
        maxsplit=1,
        flags=re.IGNORECASE,
    )
    if len(split_match) == 2:
        desc_part, exit_part = split_match[0].strip(), split_match[1].strip()
    else:
        desc_part, exit_part = text, ""

    desc_part = re.sub(
        r"^\s*(?:\*\*|#*\s*)?\[?SCENE_DESCRIPTION\]?(?:\*\*|:)?\s*\n?",
        "",
        desc_part,
        flags=re.IGNORECASE,
    ).strip()

    if not exit_part and desc_part:
        sentences = re.split(r"(?<=[.!?])\s+", desc_part)
        exit_part = " ".join(sentences[-2:]).strip() if len(sentences) >= 2 else desc_part

    return desc_part, exit_part


def _call_deepseek_vision(
    client,
    model_id: str,
    prompt_text: str,
    image_paths: List[str],
    max_image_width: int = 960,
    max_new_tokens: int = 1000,
    max_retries: int = 3,
) -> Tuple[str, Dict[str, int]]:
    """Send keyframes + prompt to deepseek-flash with thinking disabled and return text + token usage."""
    content_items: List[Dict[str, Any]] = []
    for p in image_paths:
        if not os.path.isfile(p):
            continue
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
                "Follow all formatting constraints strictly, never repeat sentences, and never "
                "mention synthetic computer-vision bounding boxes."
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


def _caption_one_scene(
    client,
    vlm_model_id: str,
    movie_id: str,
    movie_title: str,
    scene: Dict[str, Any],
    resolve_kf_path: Callable[[str], str],
    glossary: Dict[str, str],
    external_docs: List[Dict[str, Any]],
    prev_exit_state: str,
    prev_subtitles: List[Dict[str, Any]],
    max_keyframes_per_scene: int = 30,
    max_image_width: int = 960,
    min_keyframe_sharpness: float = 5.0,
) -> Tuple[str, str, List[str], Dict[str, int]]:
    """Run single-pass DeepSeek Vision API call for a Scene Window."""
    raw_kfs = scene.get("keyframes", [])
    sampled_kfs = select_scene_keyframes(
        raw_kfs,
        max_frames=max_keyframes_per_scene,
        min_sharpness=min_keyframe_sharpness,
    )
    kf_abs_paths = [resolve_kf_path(kf["path"]) for kf in sampled_kfs]

    act_context = _extract_time_matched_plot_context(
        external_docs=external_docs,
        scene_start=float(scene["start_time"]),
        scene_end=float(scene["end_time"]),
        movie_id=movie_id,
    )
    prompt_text = build_single_pass_prompt(
        movie_title=movie_title,
        scene=scene,
        sampled_kfs=sampled_kfs,
        glossary=glossary,
        prev_exit_state=prev_exit_state,
        prev_subtitles=prev_subtitles,
        act_plot_context=act_context,
    )
    raw_output, usage = _call_deepseek_vision(
        client=client,
        model_id=vlm_model_id,
        prompt_text=prompt_text,
        image_paths=kf_abs_paths,
        max_image_width=max_image_width,
        max_new_tokens=1000,
    )
    vlm_caption, exit_state = _parse_scene_and_exit_state(raw_output)
    return vlm_caption, exit_state, [kf["path"] for kf in sampled_kfs], usage


def archive_previous_vlm_outputs(output_dir: str) -> List[str]:
    """Move previous vlm_captions.jsonl and movie_knowledge_bundle.json into archive_prev_vlm/."""
    archive_dir = os.path.join(output_dir, "archive_prev_vlm")
    ts = time.strftime("%Y%m%d_%H%M%S")
    archived: List[str] = []

    for fname in ("vlm_captions.jsonl", "movie_knowledge_bundle.json", "stage5_vlm_summary.json"):
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
    movie_title: str = "Sintel",
    char_bank_dir: Optional[str] = None,
    vlm_model_id: str = "deepseek-flash",
    api_key: Optional[str] = None,
    api_base_url: str = "https://api.deepseek.com",
    max_keyframes_per_scene: int = 30,
    max_image_width: int = 960,
    min_keyframe_sharpness: float = 5.0,
) -> Dict[str, Any]:
    """Run a single-scene diagnostic call on deepseek-flash to inspect caption quality and token usage."""
    shots_file = os.path.join(output_dir, "shots_and_scenes.json")
    with open(shots_file, "r", encoding="utf-8") as f:
        shots_and_scenes = json.load(f)

    movie_id = shots_and_scenes["movie_id"]
    duration_sec = float(shots_and_scenes.get("duration_sec") or 0.0)
    scenes = shots_and_scenes.get("scenes", [])
    if not duration_sec and scenes:
        duration_sec = float(scenes[-1]["end_time"])

    glossary = _load_character_glossary(output_dir, char_bank_dir=char_bank_dir)
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
    prev_subtitles = scenes[target_idx - 1].get("subtitles", []) if target_idx > 0 else []
    prev_exit_state = (
        scenes[target_idx - 1].get("exit_state", "")
        if target_idx > 0
        else "Opening sequence of the film."
    )

    client = create_deepseek_client(api_key=api_key, api_base_url=api_base_url)
    t0 = time.perf_counter()
    vlm_caption, exit_state, used_kfs, usage = _caption_one_scene(
        client=client,
        vlm_model_id=vlm_model_id,
        movie_id=movie_id,
        movie_title=movie_title,
        scene=scene,
        resolve_kf_path=resolve_kf_path,
        glossary=glossary,
        external_docs=external_docs,
        prev_exit_state=prev_exit_state,
        prev_subtitles=prev_subtitles,
        max_keyframes_per_scene=max_keyframes_per_scene,
        max_image_width=max_image_width,
        min_keyframe_sharpness=min_keyframe_sharpness,
    )
    elapsed = time.perf_counter() - t0

    result = {
        "scene_id": scene["scene_id"],
        "time_range": f"{_format_mmss(scene['start_time'])} - {_format_mmss(scene['end_time'])}",
        "keyframes_used": len(used_kfs),
        "keyframes_total": len(scene.get("keyframes", [])),
        "subtitles": scene.get("subtitles", []),
        "vlm_caption": vlm_caption,
        "exit_state": exit_state,
        "elapsed_sec": round(elapsed, 2),
        "token_usage": usage,
    }
    print(
        f"[Diagnostic {result['scene_id']}] {result['keyframes_used']}/{result['keyframes_total']} kfs | "
        f"{elapsed:.1f}s | Tokens: {usage['prompt_tokens']} in / {usage['completion_tokens']} out"
    )
    print("\n--- [SCENE_DESCRIPTION] (vlm_caption) ---\n" + vlm_caption)
    print("\n--- [EXIT_STATE] (exit_state) ---\n" + exit_state)
    return result


def run_stage5_vlm_caption(
    output_dir: str,
    movie_title: Optional[str] = None,
    franchise_id: Optional[str] = None,
    char_bank_dir: Optional[str] = None,
    glossary_file: Optional[str] = None,
    external_plot_file: Optional[str] = None,
    vlm_model_id: str = "deepseek-flash",
    api_key: Optional[str] = None,
    api_base_url: str = "https://api.deepseek.com",
    max_keyframes_per_scene: int = 30,
    max_image_width: int = 960,
    min_keyframe_sharpness: float = 5.0,
    credits_start_sec: Optional[Union[str, float, int]] = None,
    restart: bool = False,
) -> Dict[str, Any]:
    """Run Single-Pass DeepSeek-Flash captioning across all narrative Scene Windows and assemble the knowledge bundle."""
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

    glossary = _load_character_glossary(
        output_dir=output_dir,
        char_bank_dir=char_bank_dir,
        glossary_file=glossary_file,
    )
    external_docs = _load_external_plot_documents(
        movie_id=movie_id,
        duration_sec=duration_sec,
        char_bank_dir=char_bank_dir,
        external_plot_file=external_plot_file,
    )

    ckpt_jsonl = os.path.join(output_dir, "vlm_captions.jsonl")
    completed_by_id: Dict[str, Dict[str, Any]] = {}
    if os.path.isfile(ckpt_jsonl):
        with open(ckpt_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                    if rec.get("scene_id") and rec.get("vlm_caption"):
                        completed_by_id[rec["scene_id"]] = rec
                except json.JSONDecodeError:
                    continue

    pending_scenes = [sc for sc in scenes if sc["scene_id"] not in completed_by_id]
    print(
        f"[Stage A5] Model: {vlm_model_id} | Total scenes: {len(scenes)} | "
        f"Cached: {len(completed_by_id)} | Pending: {len(pending_scenes)}"
    )

    total_prompt_tokens = 0
    total_completion_tokens = 0

    if pending_scenes:
        resolve_kf_path = _stage_keyframes_to_local(output_dir, movie_id)
        client = create_deepseek_client(api_key=api_key, api_base_url=api_base_url)

        with open(ckpt_jsonl, "a", encoding="utf-8") as out_f:
            prev_exit_state = "Opening sequence of the film."
            prev_subtitles: List[Dict[str, Any]] = []

            for idx, sc in enumerate(scenes):
                sid = sc["scene_id"]
                if sid in completed_by_id:
                    cached = completed_by_id[sid]
                    sc["vlm_caption"] = cached["vlm_caption"]
                    sc["exit_state"] = cached.get("exit_state", "")
                    prev_exit_state = sc["exit_state"] or prev_exit_state
                    prev_subtitles = sc.get("subtitles", [])
                    continue

                t0 = time.perf_counter()
                vlm_caption, exit_state, used_kfs, usage = _caption_one_scene(
                    client=client,
                    vlm_model_id=vlm_model_id,
                    movie_id=movie_id,
                    movie_title=resolved_title,
                    scene=sc,
                    resolve_kf_path=resolve_kf_path,
                    glossary=glossary,
                    external_docs=external_docs,
                    prev_exit_state=prev_exit_state,
                    prev_subtitles=prev_subtitles,
                    max_keyframes_per_scene=max_keyframes_per_scene,
                    max_image_width=max_image_width,
                    min_keyframe_sharpness=min_keyframe_sharpness,
                )
                elapsed = time.perf_counter() - t0
                total_prompt_tokens += usage["prompt_tokens"]
                total_completion_tokens += usage["completion_tokens"]

                sc["vlm_caption"] = vlm_caption
                sc["exit_state"] = exit_state
                record = {
                    "scene_id": sid,
                    "scene_index": sc.get("scene_index", idx),
                    "vlm_caption": vlm_caption,
                    "exit_state": exit_state,
                    "sampled_keyframe_paths": used_kfs,
                    "token_usage": usage,
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_f.flush()
                completed_by_id[sid] = record

                prev_exit_state = exit_state
                prev_subtitles = sc.get("subtitles", [])
                _trim_host_memory()

                print(
                    f"[Scene {idx + 1:03d}/{len(scenes):03d} ({sid})] "
                    f"kfs: {len(used_kfs)}/{len(sc.get('keyframes', []))} | "
                    f"subs: {len(sc.get('subtitles', []))} | "
                    f"{elapsed:.1f}s | tokens: {usage['prompt_tokens']} in / {usage['completion_tokens']} out",
                    flush=True,
                )
    else:
        for sc in scenes:
            cached = completed_by_id.get(sc["scene_id"], {})
            sc["vlm_caption"] = cached.get("vlm_caption", "")
            sc["exit_state"] = cached.get("exit_state", "")

    # Assemble clean normalized scenes list matching MovieKnowledgeBundle schema
    bundle_scenes: List[Dict[str, Any]] = []
    for idx, sc in enumerate(scenes):
        bundle_scenes.append(
            {
                "scene_id": sc["scene_id"],
                "scene_index": int(sc.get("scene_index", idx)),
                "start_time": round(float(sc["start_time"]), 3),
                "end_time": round(float(sc["end_time"]), 3),
                "duration_sec": round(float(sc.get("duration_sec", sc["end_time"] - sc["start_time"])), 3),
                "shot_ranges": sc.get("shot_ranges", []),
                "characters_present": sc.get("characters_present", []),
                "keyframes": sc.get("keyframes", []),
                "subtitles": sc.get("subtitles", []),
                "vlm_caption": sc.get("vlm_caption", ""),
                "exit_state": sc.get("exit_state", ""),
            }
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
    parser = argparse.ArgumentParser(description="Stage A5: DeepSeek-Flash Vision API Scene Captioning")
    parser.add_argument("--output_dir", required=True, help="Workdir containing shots_and_scenes.json")
    parser.add_argument("--movie_title", default=None, help="Display title for movie")
    parser.add_argument("--franchise_id", default=None, help="Franchise identifier")
    parser.add_argument("--char_bank_dir", default=None, help="Character bank directory")
    parser.add_argument("--glossary_file", default=None, help="Path to character_glossary.json")
    parser.add_argument("--external_plot_file", default=None, help="Optional Tier-3 plot file")
    parser.add_argument("--vlm_model_id", default="deepseek-flash")
    parser.add_argument("--api_base_url", default="https://api.deepseek.com")
    parser.add_argument("--max_keyframes_per_scene", type=int, default=30)
    parser.add_argument("--max_image_width", type=int, default=960)
    parser.add_argument("--min_keyframe_sharpness", type=float, default=5.0)
    parser.add_argument(
        "--credits_start_sec",
        default=None,
        type=str,
        help="Optional timestamp (e.g. '744.0' or '12:24') where closing credits start",
    )
    parser.add_argument("--restart", action="store_true", help="Archive and ignore cached vlm_captions.jsonl")
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
        max_image_width=args.max_image_width,
        min_keyframe_sharpness=args.min_keyframe_sharpness,
        credits_start_sec=args.credits_start_sec,
        restart=args.restart,
    )
