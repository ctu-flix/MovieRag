"""Stage A5: Two-Pass Qwen2.5-VL-7B Scene Captioning & Knowledge Bundle Assembly.

Implements Phase A Stage A5 for MovieRAG:
1. Subsamples up to `max_keyframes_per_scene` (default 8) keyframes per Scene Window,
   preserving temporal span (first & last frame) and prioritizing frames with
   recognized character bounding boxes.
2. Stages keyframes to fast local SSD (/content/vlm_scratch) in Colab to eliminate
   Google Drive FUSE stalls.
3. Runs Two-Pass inference with Qwen/Qwen2.5-VL-7B-Instruct (bfloat16 / fp16):
   - Pass 1: Generates a rich 250-450 word character-grounded visual narrative
     (`vlm_caption`) using annotated keyframes, color-to-character legend, static
     glossary, speaker-attributed subtitles, and previous scene exit_state.
   - Pass 2: Extracts a 2-3 sentence continuity state (`exit_state`) from Pass 1's
     narrative and the final keyframe of the scene.
4. Checkpoints every completed scene immediately to `vlm_captions.jsonl`, trims host
   & GPU memory after every scene, and packages `movie_knowledge_bundle.json`.
"""

import argparse
import ctypes
import gc
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from pipeline import PROJECT_ROOT


SINTEL_FALLBACK_PLOT = (
    "Sintel (2010) Full Plot Synopsis: A lone young warrior named Sintel searches a "
    "snowy wasteland for her lost dragon companion, collapses after fighting a gatekeeper, "
    "and awakens in the desert hut of an elderly Shaman. In a flashback to the city of "
    "Ishtar, a younger Sintel rescues and bandages a baby red dragon she names Scales, "
    "teaching him to fly until a giant adult dragon swoops down and snatches Scales away. "
    "Determined to rescue him, Sintel journeys for years across mountains, forests, and "
    "deserts before finding a mountain cavern containing a baby dragon beside a sleeping "
    "giant Adult Dragon. When the Adult Dragon attacks Sintel, it pins her down and "
    "suddenly hesitates upon looking into her eyes; Sintel seizes the moment and fatally "
    "stabs it with her spear. As the giant dragon dies (~11:15), Sintel sees the healed "
    "bandage scar on its wing and realizes in horror that many years had passed during "
    "her quest, the giant Adult Dragon she killed was actually grown-up Scales, and the "
    "small hatchling beside it is Scales's offspring. Grief-stricken, Sintel walks out "
    "into the snow followed by the young dragon."
)


def _trim_host_memory(empty_cuda: bool = False) -> None:
    """Release cyclic references, OS heap arenas, and optional CUDA cached blocks."""
    gc.collect()
    try:
        libc = ctypes.CDLL("libc.so.6")
        if hasattr(libc, "malloc_trim"):
            libc.malloc_trim(0)
    except Exception:
        pass
    if empty_cuda:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
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


def select_scene_keyframes(
    keyframes: List[Dict[str, Any]],
    max_frames: int = 8,
) -> List[Dict[str, Any]]:
    """Select <= max_frames keyframes preserving start/end anchors and character boxes."""
    if not keyframes:
        return []
    ordered = sorted(keyframes, key=lambda kf: int(kf.get("frame_idx", 0)))
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
    """Load static character glossary from explicit path, Stage A2 summary, or char_bank."""
    candidates: List[str] = []
    if glossary_file:
        candidates.append(glossary_file)
    if char_bank_dir:
        candidates.append(os.path.join(char_bank_dir, "character_glossary.json"))

    stage2_summary = os.path.join(output_dir, "stage2_charbank_summary.json")
    if os.path.isfile(stage2_summary):
        try:
            with open(stage2_summary, "r", encoding="utf-8") as f:
                s2 = json.load(f)
            if isinstance(s2.get("glossary"), dict) and s2["glossary"]:
                return {str(k): str(v) for k, v in s2["glossary"].items()}
        except Exception:
            pass

    for cand in candidates:
        if cand and os.path.isfile(cand):
            with open(cand, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return {str(k): str(v) for k, v in data.items()}
    return {}


def _load_external_plot_documents(
    movie_id: str,
    duration_sec: float,
    char_bank_dir: Optional[str] = None,
    external_plot_file: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Load optional Tier-3 full-movie synopsis document for non-spoiler Phase B GraphRAG."""
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


def load_qwen25_vl(
    model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct",
    min_pixels: int = 256 * 28 * 28,
    max_pixels: int = 512 * 28 * 28,
):
    """Load Qwen2.5-VL-7B-Instruct and bounded-resolution AutoProcessor."""
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else (torch.float16 if torch.cuda.is_available() else torch.float32)

    print(f"[Stage A5] Loading VLM '{model_id}' (dtype={dtype}, max_pixels={max_pixels})...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(
        model_id,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    return model, processor


def _generate_vlm_text(
    model,
    processor,
    prompt_text: str,
    image_paths: List[str],
    max_new_tokens: int = 512,
) -> str:
    """Run a single multimodal Qwen2.5-VL generation pass and free image tensors cleanly."""
    import torch
    from PIL import Image

    pil_images: List[Image.Image] = []
    content_items: List[Dict[str, Any]] = []
    try:
        for p in image_paths:
            if not os.path.isfile(p):
                continue
            with Image.open(p) as im:
                pil_images.append(im.convert("RGB"))
            content_items.append({"type": "image"})

        content_items.append({"type": "text", "text": prompt_text})
        messages = [{"role": "user", "content": content_items}]

        chat_prompt = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = processor(
            text=[chat_prompt],
            images=pil_images if pil_images else None,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)

        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]
        decoded = processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return (decoded[0] if decoded else "").strip()
    finally:
        for im in pil_images:
            try:
                im.close()
            except Exception:
                pass
        pil_images.clear()


def _format_color_legend_and_glossary(
    scene: Dict[str, Any],
    sampled_kfs: List[Dict[str, Any]],
    glossary: Dict[str, str],
) -> Tuple[str, str]:
    """Build exact color-box legend and filtered character glossary text for Pass 1."""
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
        legend_items = [f"{name} ({color} box)" for name, color in char_colors.items()]
        legend_str = ", ".join(legend_items)
        glossary_lines = [
            f"- {name}: {glossary[name]}"
            for name in char_colors
            if name in glossary
        ]
    else:
        legend_str = "None detected cleanly by automated visual tracker."
        glossary_lines = [f"- {name}: {desc}" for name, desc in glossary.items()]

    glossary_str = "\n".join(glossary_lines) if glossary_lines else "- No glossary entries applicable."
    return legend_str, glossary_str


def build_pass1_prompt(
    movie_title: str,
    scene: Dict[str, Any],
    sampled_kfs: List[Dict[str, Any]],
    glossary: Dict[str, str],
    prev_exit_state: str,
    prev_subtitles: List[Dict[str, Any]],
) -> str:
    """Build the Pass-1 character-grounded scene captioning prompt."""
    t0_str = _format_mmss(scene["start_time"])
    t1_str = _format_mmss(scene["end_time"])
    legend_str, glossary_str = _format_color_legend_and_glossary(scene, sampled_kfs, glossary)

    kf_timeline = ", ".join(
        f"Image {idx + 1} ({_format_mmss(kf.get('timestamp', 0.0))}, shot {kf.get('shot_idx')})"
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
        dialogue_guidance = "Integrate what is being said cleanly alongside the physical actions and reactions."
    else:
        cur_subs_str = "- [No spoken dialogue in this scene]"
        dialogue_guidance = (
            "There is NO spoken dialogue in this scene. Focus thoroughly on silent visual storytelling: "
            "movements, environment, objects, combat/actions, posture, and emotional facial expressions."
        )

    return (
        f"You are analyzing a sequential scene from the animated film '{movie_title}'.\n"
        f"Scene Window: [{t0_str} - {t1_str}] | Number of ordered keyframes shown: {len(sampled_kfs)}\n"
        f"Frame Order: {kf_timeline}\n\n"
        f"[Visual Prompt Legend]\n"
        f"Recognized characters overlaid with colored bounding boxes: {legend_str}\n"
        f"CRITICAL RULE: Colored bounding boxes are provided strictly as character identity hints for you. "
        f"NEVER mention bounding boxes, box colors, rectangles, overlays, or 'Image 1/2' frame labels in your response.\n\n"
        f"[Character Reference Glossary]\n{glossary_str}\n\n"
        f"[Previous Scene Continuity]\n"
        f"State entering this scene: {prev_exit_state or 'Opening sequence of the film.'}\n"
        f"Recent lines before this scene:\n{prev_subs_str}\n\n"
        f"[Spoken Dialogue in This Scene]\n{cur_subs_str}\n\n"
        f"[Task]\n"
        f"Write ONE rich, cohesive, detailed narrative description (roughly 250-450 words) in present tense "
        f"describing what happens across these chronological frames in strictly accurate temporal order:\n"
        f"1. Setting & Atmosphere: Describe the exact location, lighting, weather, and props visible.\n"
        f"2. Character Identification & Movements: Name recognized characters accurately (using their color hints) "
        f"and describe any unrecognized characters/creatures objectively by appearance (without inventing names).\n"
        f"3. Actions & Chronology: Narrate step-by-step who does what across the scene, focusing on physical "
        f"movements, gestures, and spatial changes. {dialogue_guidance}\n"
        f"4. Interactions & Facial Expressions: Describe eye contact, interpersonal reactions, and emotions.\n"
        f"STRICT SPOILER RULE: Describe ONLY what is directly shown in this scene's frames and dialogue. "
        f"Do NOT extrapolate or reveal future plot events or hidden twists."
    )


def build_pass2_prompt(
    movie_title: str,
    scene: Dict[str, Any],
    vlm_caption: str,
) -> str:
    """Build the Pass-2 exit-state extraction prompt."""
    t0_str = _format_mmss(scene["start_time"])
    t1_str = _format_mmss(scene["end_time"])
    cur_subs = scene.get("subtitles", [])[-3:]
    subs_str = (
        "\n".join(f"- {u.get('speaker', 'Unknown')}: \"{u['text']}\"" for u in cur_subs)
        if cur_subs
        else "- (No spoken dialogue)"
    )
    return (
        f"You have just viewed the final frame of Scene {scene.get('scene_index', 0)} [{t0_str} - {t1_str}] "
        f"of '{movie_title}'.\n\n"
        f"[Scene Narrative]\n{vlm_caption}\n\n"
        f"[Final Spoken Lines]\n{subs_str}\n\n"
        f"[Task]\n"
        f"Write a concise 2-3 sentence EXIT STATE summarizing the exact continuity status at the end of this scene "
        f"so the next scene understands the handoff: where key characters are positioned, what objects they hold, "
        f"their emotional/physical condition, and any immediate unresolved action. Do not mention bounding boxes or "
        f"the phrase 'final frame'."
    )


def _caption_one_scene(
    model,
    processor,
    movie_title: str,
    scene: Dict[str, Any],
    resolve_kf_path: Callable[[str], str],
    glossary: Dict[str, str],
    prev_exit_state: str,
    prev_subtitles: List[Dict[str, Any]],
    max_keyframes_per_scene: int = 8,
) -> Tuple[str, str, List[str]]:
    """Run Pass 1 (vlm_caption) and Pass 2 (exit_state) for a single Scene Window."""
    raw_kfs = scene.get("keyframes", [])
    sampled_kfs = select_scene_keyframes(raw_kfs, max_frames=max_keyframes_per_scene)
    kf_abs_paths = [resolve_kf_path(kf["path"]) for kf in sampled_kfs]

    pass1_prompt = build_pass1_prompt(
        movie_title=movie_title,
        scene=scene,
        sampled_kfs=sampled_kfs,
        glossary=glossary,
        prev_exit_state=prev_exit_state,
        prev_subtitles=prev_subtitles,
    )
    vlm_caption = _generate_vlm_text(
        model=model,
        processor=processor,
        prompt_text=pass1_prompt,
        image_paths=kf_abs_paths,
        max_new_tokens=512,
    )

    last_kf_paths = [kf_abs_paths[-1]] if kf_abs_paths else []
    pass2_prompt = build_pass2_prompt(
        movie_title=movie_title,
        scene=scene,
        vlm_caption=vlm_caption,
    )
    exit_state = _generate_vlm_text(
        model=model,
        processor=processor,
        prompt_text=pass2_prompt,
        image_paths=last_kf_paths,
        max_new_tokens=160,
    )

    return vlm_caption, exit_state, [kf["path"] for kf in sampled_kfs]


def _gpu_vram_str() -> str:
    """Format current CUDA allocated VRAM string cleanly."""
    try:
        import torch

        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / (1024**3)
            res = torch.cuda.memory_reserved() / (1024**3)
            return f"{alloc:.2f} GB (Res {res:.2f})"
    except Exception:
        pass
    return "n/a"


def _ram_str() -> str:
    """Format current process RSS RAM usage if psutil is installed."""
    try:
        import psutil

        rss = psutil.Process(os.getpid()).memory_info().rss / (1024**3)
        return f"{rss:.2f} GB"
    except Exception:
        return "n/a"


def run_single_scene_diagnostic(
    output_dir: str,
    scene_id: Optional[str] = None,
    scene_index: Optional[int] = 4,
    movie_title: str = "Sintel",
    char_bank_dir: Optional[str] = None,
    vlm_model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct",
    max_keyframes_per_scene: int = 8,
    min_pixels: int = 256 * 28 * 28,
    max_pixels: int = 512 * 28 * 28,
) -> Dict[str, Any]:
    """Run Pass 1 + Pass 2 on a single scene to inspect caption quality and memory usage."""
    shots_file = os.path.join(output_dir, "shots_and_scenes.json")
    with open(shots_file, "r", encoding="utf-8") as f:
        shots_and_scenes = json.load(f)

    movie_id = shots_and_scenes["movie_id"]
    scenes = shots_and_scenes.get("scenes", [])
    glossary = _load_character_glossary(output_dir, char_bank_dir=char_bank_dir)
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

    model, processor = load_qwen25_vl(model_id=vlm_model_id, min_pixels=min_pixels, max_pixels=max_pixels)
    t0 = time.perf_counter()
    try:
        vlm_caption, exit_state, used_kfs = _caption_one_scene(
            model=model,
            processor=processor,
            movie_title=movie_title,
            scene=scene,
            resolve_kf_path=resolve_kf_path,
            glossary=glossary,
            prev_exit_state=prev_exit_state,
            prev_subtitles=prev_subtitles,
            max_keyframes_per_scene=max_keyframes_per_scene,
        )
    finally:
        elapsed = time.perf_counter() - t0
        ram_after = _ram_str()
        vram_after = _gpu_vram_str()
        del model, processor
        _trim_host_memory(empty_cuda=True)

    result = {
        "scene_id": scene["scene_id"],
        "time_range": f"{_format_mmss(scene['start_time'])} - {_format_mmss(scene['end_time'])}",
        "keyframes_used": len(used_kfs),
        "keyframes_total": len(scene.get("keyframes", [])),
        "subtitles": scene.get("subtitles", []),
        "vlm_caption": vlm_caption,
        "exit_state": exit_state,
        "elapsed_sec": round(elapsed, 2),
        "ram": ram_after,
        "vram": vram_after,
    }
    print(f"[Diagnostic {result['scene_id']}] {result['keyframes_used']}/{result['keyframes_total']} kfs | {elapsed:.1f}s | RAM: {ram_after} | VRAM: {vram_after}")
    print("\n--- Pass 1: vlm_caption ---\n" + vlm_caption)
    print("\n--- Pass 2: exit_state ---\n" + exit_state)
    return result


def run_stage5_vlm_caption(
    output_dir: str,
    movie_title: Optional[str] = None,
    franchise_id: Optional[str] = None,
    char_bank_dir: Optional[str] = None,
    glossary_file: Optional[str] = None,
    external_plot_file: Optional[str] = None,
    vlm_model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct",
    max_keyframes_per_scene: int = 8,
    min_pixels: int = 256 * 28 * 28,
    max_pixels: int = 512 * 28 * 28,
    restart: bool = False,
) -> Dict[str, Any]:
    """Run Two-Pass Qwen2.5-VL captioning across all Scene Windows and assemble the knowledge bundle."""
    shots_file = os.path.join(output_dir, "shots_and_scenes.json")
    if not os.path.isfile(shots_file):
        raise FileNotFoundError(f"shots_and_scenes.json not found: {shots_file}")

    with open(shots_file, "r", encoding="utf-8") as f:
        shots_and_scenes = json.load(f)

    movie_id = shots_and_scenes["movie_id"]
    resolved_title = movie_title or movie_id.replace("_", " ").title()
    resolved_franchise = franchise_id or movie_id
    fps = float(shots_and_scenes.get("fps") or 24.0)
    duration_sec = float(shots_and_scenes.get("duration_sec") or 0.0)
    scenes = shots_and_scenes.get("scenes", [])
    if not duration_sec and scenes:
        duration_sec = float(scenes[-1]["end_time"])

    # Merge attributed_subtitles.json if present and scenes lack subtitles
    subtitles_file = os.path.join(output_dir, "attributed_subtitles.json")
    if os.path.isfile(subtitles_file) and all("subtitles" not in sc for sc in scenes):
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
    if restart and os.path.isfile(ckpt_jsonl):
        os.remove(ckpt_jsonl)
    elif os.path.isfile(ckpt_jsonl):
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
        f"[Stage A5] Total scenes: {len(scenes)} | Cached: {len(completed_by_id)} | "
        f"Pending: {len(pending_scenes)}"
    )

    if pending_scenes:
        resolve_kf_path = _stage_keyframes_to_local(output_dir, movie_id)
        model, processor = load_qwen25_vl(
            model_id=vlm_model_id,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )

        try:
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
                    vlm_caption, exit_state, used_kfs = _caption_one_scene(
                        model=model,
                        processor=processor,
                        movie_title=resolved_title,
                        scene=sc,
                        resolve_kf_path=resolve_kf_path,
                        glossary=glossary,
                        prev_exit_state=prev_exit_state,
                        prev_subtitles=prev_subtitles,
                        max_keyframes_per_scene=max_keyframes_per_scene,
                    )
                    elapsed = time.perf_counter() - t0

                    sc["vlm_caption"] = vlm_caption
                    sc["exit_state"] = exit_state
                    record = {
                        "scene_id": sid,
                        "scene_index": sc.get("scene_index", idx),
                        "vlm_caption": vlm_caption,
                        "exit_state": exit_state,
                        "sampled_keyframe_paths": used_kfs,
                    }
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    out_f.flush()
                    completed_by_id[sid] = record

                    prev_exit_state = exit_state
                    prev_subtitles = sc.get("subtitles", [])
                    _trim_host_memory(empty_cuda=True)

                    print(
                        f"[Scene {idx + 1:03d}/{len(scenes):03d} ({sid})] "
                        f"kfs: {len(used_kfs)}/{len(sc.get('keyframes', []))} | "
                        f"subs: {len(sc.get('subtitles', []))} | "
                        f"{elapsed:.1f}s | RAM: {_ram_str()} | VRAM: {_gpu_vram_str()}",
                        flush=True,
                    )
        finally:
            del model, processor
            _trim_host_memory(empty_cuda=True)
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
        "bundle_path": bundle_path,
        "bundle_size_mb": round(os.path.getsize(bundle_path) / (1024**2), 3),
    }
    summary_path = os.path.join(output_dir, "stage5_vlm_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[Stage A5] Saved final bundle -> {bundle_path} ({summary['bundle_size_mb']} MB)")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage A5: Two-Pass Qwen2.5-VL Scene Captioning")
    parser.add_argument("--output_dir", required=True, help="Workdir containing shots_and_scenes.json")
    parser.add_argument("--movie_title", default=None, help="Display title for movie")
    parser.add_argument("--franchise_id", default=None, help="Franchise identifier")
    parser.add_argument("--char_bank_dir", default=None, help="Character bank directory")
    parser.add_argument("--glossary_file", default=None, help="Path to character_glossary.json")
    parser.add_argument("--external_plot_file", default=None, help="Optional Tier-3 plot file")
    parser.add_argument("--vlm_model_id", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--max_keyframes_per_scene", type=int, default=8)
    parser.add_argument("--restart", action="store_true", help="Ignore cached vlm_captions.jsonl")
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
        max_keyframes_per_scene=args.max_keyframes_per_scene,
        restart=args.restart,
    )
