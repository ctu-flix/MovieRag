---
Date: 2026-11-18
Top Level:
  - "[[_tech_]]"
Topic:
  - "[[_VideoRAG_]]"
  - "[[_Character_Detection_]]"
  - "[[_Captioning_]]"
---

# Phase A: Animated Character & Micro-Beat DeepSeek-Flash VLM Pipeline

> Part of [[00_MovieRAG_Master_Plan]]
> Covers the offline Phase A extraction stages designed to run on **Google Colab** across **4 environment-isolated notebooks** communicating via Google Drive JSON checkpoints (`shots_and_scenes.json`, `visual_tracks_per_frame.json`, `attributed_subtitles.json`, `vlm_beats.jsonl`, and `movie_knowledge_bundle.json`). Separating Visual, Audio, and API VLM stacks avoids all CUDA/PyTorch conflicts (`SAM 2.1`/`xformers` vs `faster-whisper`/`SpeechBrain`) and lets Stage A5 run on a **free CPU Colab runtime**.

---

## 0. Four-Notebook Colab Environment Isolation (`notebooks/`)

| Colab Notebook | Runtime & Environment Stack | Stages Executed | Inputs (from Drive / Repo) | Outputs (saved to Drive) |
| :--- | :--- | :--- | :--- | :--- |
| **Notebook 0: `PhaseA0_CharBank_Bootstrap.ipynb`** | **T4 GPU**: `PySceneDetect`, `OpenCV`, `OWLv2`, `DINOv2` *(no `sam2` required)* | **Stage A0**: End-of-movie preview, manual `CREDITS_START_SEC`, & hybrid OWLv2+DINOv2 crop clustering for human labeling | • `movie.mp4` | • `clusters_contact_sheet.jpg`<br>• `char_bank/{movie_id}/raw_images/{char}/`<br>• `character_glossary.json` |
| **Notebook 1: `PhaseA1_Visual_Pipeline.ipynb`** | **L4 / T4 GPU** (`Animated_AD` visual stack): `PySceneDetect`, `OpenCV`, `OWLv2`, `SAM 2.1` + `xformers`, `DINOv2` | **Stage A1**: Shot detection, `parent_shot_idx` lineage, 15–30s scenes & 4–10s `micro_beats`<br>**Stage A2**: 75-epoch `DINOv2` tuning + Farthest-Point 16/char bank<br>**Stage A3**: Sharp-seed `TGRP` + seam carry-over + 1-seed rescue + **2–6 fps Motion-Adaptive Keyframes** | • `movie.mp4`<br>• `char_bank/{movie_id}/` | • `shots_and_scenes.json` (with `micro_beats`)<br>• `classified_tracks.json`<br>• `visual_tracks_per_frame.json`<br>• `keyframes/*.jpg` (with colored bboxes drawn) |
| **Notebook 2: `PhaseA2_Audio_Pipeline.ipynb`** | **T4 GPU** (isolated audio stack): `faster-whisper`, `Demucs` (opt), `LWTNet`, `SpeechBrain` (`ECAPA-TDNN`) | **Stage A4**: Subtitle ingestion / `faster-whisper` ASR, in-movie voice bank harvesting, `ECAPA-TDNN` attribution, & manual `SPEAKER_OVERRIDES` (`apply_subtitle_speaker_fixes`) | • `movie.mp4` + opt `.srt`<br>• `shots_and_scenes.json`<br>• `classified_tracks.json` | • `attributed_subtitles.json`<br>• `voice_bank/{movie_id}/*.npz`<br>• Updated `shots_and_scenes.json` |
| **Notebook 3: `PhaseA3_VLM_Captioning.ipynb`** | **Free CPU Runtime**: `openai>=1.50.0`, `pillow`, `DEEPSEEK_API_KEY` | **Stage A5**: Micro-beat `deepseek-flash` Vision API captioning (`thinking: disabled`, `960px`, overlap frames, telemetry/anti-closure guards, time-gated glossary) + `$0` deterministic scene merge | • `shots_and_scenes.json`<br>• `attributed_subtitles.json`<br>• `keyframes/*.jpg`<br>• `character_glossary.json`, `appearance_windows.json`, `external_plot.txt` | • `vlm_beats.jsonl` & `vlm_captions.jsonl`<br>• **`movie_knowledge_bundle.json`** (Final Handoff Artifact) |

---

## 1. Stage A0: Credits Cutoff Preview & Hybrid Character Bank Bootstrap (`stage0_bootstrap_charbank.py`)

1. **End-of-Movie Credits Preview (`preview_end_of_movie_frames`)**:
   - Samples timestamped frames from the tail of the movie (e.g., `start_ratio=0.75`) into a contact grid so the user inspects the exact timestamp where closing credits begin (`CREDITS_START_SEC`, e.g., `"744.0"` or `"12:24"` for *Sintel* — strictly user-driven, never hardcoded).
2. **Hybrid Machine Clustering + Human Labeling (`harvest_and_cluster_character_crops` & `export_labeled_clusters_to_charbank`)**:
   - Runs/loads Stage A1 (up to `credits_start_sec`), samples frames at `sample_interval_sec=2.5`, detects character/creature candidates with `OWLv2` (`_nms_boxes`), extracts `DINOv2` embeddings, and clusters them (`AgglomerativeClustering`, `num_clusters=18`) into distinct visual appearance states (e.g., young vs. older `Sintel`, `Scales`, `Adult Dragon`, `Shaman`, `Gatekeeper`).
   - Renders `clusters_contact_sheet.jpg` so the user fills in `CLUSTER_TO_CHARACTER` and exports sharp multi-state crops into `char_bank/{movie_id}/raw_images/{Character}/`.

---

## 2. Stage A1: Shot Detection, Take Lineage, Scenes & Micro-Beats (`stage1_scenes.py`)

1. **Shot Boundary Detection & Credits Cutoff**:
   - Runs `PySceneDetect` (`ContentDetector(threshold=27.0)`) and trims/drops raw shots at `credits_start_sec` (`parse_timestamp_sec`, `apply_credits_cutoff_to_shots_file`).
   - Subdivides unbroken takes exceeding `max_shot_sec = 20.0s` into `sub_shot_sec = 15.0s` sub-shots (`subdivide_long_shots`) while tagging every shot with:
     - `"parent_shot_idx"`: index of the original raw camera take,
     - `"is_subdivided_continuation"`: `True` only when `sub_i > 0` inside a subdivided long take (`False` on every real camera cut).
2. **Scene Window Merging (`group_shots_into_scenes`)**:
   - Groups contiguous shots into **15–30s Scene Windows** (`min_scene_sec = 12.0s`, `max_scene_sec = 30.0s`).
3. **Micro-Beat Partitioning (`build_micro_beats`)**:
   - Partitions each Scene Window's `shot_ranges` into ordered **4–10s micro-beats** (`min_beat_sec = 4.0s`, `max_beat_sec = 10.0s`):
     - Splits individual shots/sub-shots $> 10.0\text{s}$ into equal $\le 10.0\text{s}$ segments.
     - Merges contiguous short shots while `(cur_dur < 4.0 or seg_dur < 2.5)` and `cand_dur <= 10.0` (folding a $<2.5\text{s}$ trailing tail if total $\le 11.5\text{s}$).
     - Stores `beat_id` (`{scene_id}_b{k:02d}`), `start_frame`, `end_frame`, `start_time`, `end_time`, `shot_indices`, and `keyframes` inside `scene["micro_beats"]`.

---

## 3. Stage A2: Multi-State Visual Character Bank (`stage2_charbank.py`)

1. **Multi-File Asset Layout (`char_bank/{movie_id}/`)**:
   - `raw_images/{Character}/`: curated/clustered character reference images (`6–16+` spanning costume/lighting changes).
   - `character_glossary.json`: visual-first + timeline-gated + negative-constraint descriptions (`VISUAL CONSTRAINT: ...`, spoiler-free so `Scales` never leaks into `Adult Dragon`).
   - `appearance_windows.json`: manual `{Character: [[start_sec, end_sec], ...]}` timeline gates.
   - `external_plot.txt`: time-aligned sub-beat grounding reference (`Prologue.A`, `Act I.A1..A3`, `Act I.B`, `Act II.A..C`, `Act III.A..C`, `Act IV.A..B`, `Epilogue`).
2. **Contrastive `DINOv2` Fine-Tuning & Farthest-Point Retrieval Bank**:
   - Fine-tunes `DINOv2` (`dinov2_vitg14_reg` on L4, `dinov2_vitl14_reg` on T4) for `75 epochs` using InfoNCE loss ($\tau = 0.07$, $\text{lr} = 6\times 10^{-4} \rightarrow 5\times 10^{-6}$).
   - Extracts normalized `.npz` retrieval prototypes with `require_detection=False` (trusting curated/clustered crops) and **Farthest-Point Diversity Selection** (`num_per_character=16`) so multi-state clusters (e.g., masked/unmasked lighting or young/older `Sintel`) are preserved.

---

## 4. Stage A3: Per-Shot TGRP Tracking & Motion-Adaptive 2–6 fps Keyframes (`stage3_visual_recog.py`)

1. **Upgraded TGRP (`run_per_shot_tgrp`)**:
   - **Expanded Prompt & Small-Box Recall**: Prompts `OWLv2` with `[["a photo of an animated character", "a dragon or creature", "a person or warrior"]]` and lowers `min_box_area_ratio` from `0.005` $\rightarrow$ `0.0015` so distant figures in wide shots aren't pruned.
   - **3 Sharpest Seed Frames**: Replaces random seeds with `_select_sharpest_seed_frames` (picks the sharpest non-black frame from 5 evenly spaced candidates per third of the shot via Laplacian variance + brightness $\ge 4.0$).
   - **Strict Seam Carry-Over (`carry_0`)**: Resets SAM 2.1 (`reset_state` + `inference_state.clear()`) after every shot. Only when `is_subdivided_continuation == True` and `parent_shot_idx` matches does it extract ending bounding boxes from the last 1–2 frames of `shot_idx - 1` and seed them onto `frame_idx=0` of the continuation sub-shot.
   - **1-Seed High-Confidence Rescue (`merge_tracks_with_rescue`)**: Keeps valid single-seed tracks when `owl_score >= 0.20` (`min_frames >= 4`, non-overlapping `IoU < 0.5` with multi-seed tracks).
2. **DINOv2 Track Classification (`classify_tracks_and_map_frames`)**:
   - Standard tracks require $s_{\text{vm}} > 0.45$ (`visual_prompt_thresh`); rescued `single_seed_candidate` tracks must clear stricter `single_seed_dino_thresh = 0.55`.
   - Tracks tagged `carried_from_prev_subshot` preserve identity continuity across subdivided seams (`carried_char_score >= 0.30` and `(max_score - carried_char_score) < 0.20`).
3. **Motion-Adaptive 2–6 fps Annotated Keyframes (`extract_adaptive_annotated_keyframes` & `reextract_keyframes_only`)**:
   - Probes each shot at `max_action_fps = 6.0`, enforces `min_sharpness_floor = 10.0` & `brightness >= 3.5`, and always retains the first and last sharp frame of every shot.
   - Retains interior frames dynamically by motion & shot length:
     - **High motion (`motion_delta >= 12.0`) or rapid cut (`shot_dur < 1.5s`)**: up to **6.0 fps**.
     - **Moderate motion (`motion_delta >= 4.0`)**: **~4.0 fps**.
     - **Static / dialogue (`motion_delta < 4.0`)**: **1.5–2.0 fps** (`base_fps = 2.0`).
   - Supports `PINNED_KEYFRAME_TIMESTAMPS`, caps at `max_keyframes_per_scene = 150`, draws colored character boxes (`red`, `green`, `blue`, `yellow`, `pink`, `cyan`), and partitions retained frames + visible characters into `scene["micro_beats"]`.

---

## 5. Stage A4: In-Movie Voice Bank, Speaker Attribution & Manual Overrides (`stage4_subtitles.py`)

1. **Subtitle Ingestion & Audio Extraction**:
   - Extracts `16 kHz` mono WAV (with optional `Demucs` vocal stem), loads `.srt` / embedded subtitles (preserving SDH `[Speaker]:` tags), or runs `faster-whisper` (`large-v3`, `vad_filter=True`) fallback when no subtitle track exists.
2. **In-Movie Voice Bank + Hybrid Attribution**:
   - Harvests solo speaking clips overlapping high-confidence Stage A3 visual tracks ($s_{\text{vm}} > 0.60$, plus optional `LWTNet` sync scoring $s_{\text{sync}} > 0.30$), encodes `192-d` `ECAPA-TDNN` embeddings, and attributes lines via audio cosine + overlapping visual track evidence.
3. **Pure Manual Speaker Overrides (`apply_subtitle_speaker_fixes`)**:
   - Allows exact manual correction via `SPEAKER_OVERRIDES` (keyed by `utt_id`, `"start_time"`, or exact text) in Notebook 2 Cell 6B or Notebook 3 Cell 4 before VLM captioning (no heuristic auto-rewrites).

---

## 6. Stage A5: Micro-Beat `deepseek-flash` VLM Captioning & `$0` Scene Merge (`stage5_vlm_caption.py`)

Replaces `Qwen2.5-VL-7B` 2-pass (`Pass 1` + `[EXIT_STATE]`) captioning with a **single-pass micro-beat Vision API loop** on a free CPU runtime:

1. **Micro-Beat Inputs & Visual Continuity**:
   - Processes each `4–10s` micro-beat sequentially using up to `max_keyframes_per_beat = 60` sharp frames (`sharpness >= min_keyframe_sharpness`) downscaled in memory to `max_image_width = 960`.
   - Attaches up to `PREV_OVERLAP_FRAMES` (e.g., `2–4` trailing frames within `PREV_OVERLAP_SEC = 1.0s`) labeled `[Context Only — End of Previous Beat, DO NOT re-describe these frames]`, plus the full previous beat caption wrapped as `[ALREADY HAPPENED IN PREVIOUS BEAT — CONTEXT ONLY, DO NOT RE-DESCRIBE]`. (`[EXIT_STATE]` is retired.)
2. **Hallucination-Elimination Safeguards**:
   - **Window-Gated Glossary & Tracker Confidence Floor (`_format_color_legend_and_glossary`)**: Injects a character's glossary entry only if they appear inside their `appearance_windows.json` time span, are spoken/mentioned in the beat's subtitles, or clear `out_of_window_min_score >= 0.60` in the visual tracker (blocking low-score out-of-window misclassifications like `Shaman` at `0.5049` on exterior snow frames).
   - **Time-Aligned Sub-Beat Plot Grounding (`_extract_time_matched_plot_context`)**: Regex-matches `(MM:SS - MM:SS)` sub-blocks from `external_plot.txt` (prioritizing repo `char_bank/{movie_id}/external_plot.txt`) so black opening frames and individual combat phases stay strictly isolated.
   - **Low-Level Telemetry Alerts**: Injects `[TELEMETRY ALERT]` when all beat frames have `sharpness < 1.0` (black/fade screens) and `[TELEMETRY NOTE]` when 0 named characters are tracked.
   - **Duration-Scaled Word Targets & Anti-Premature-Closure**: Sets word targets dynamically (`20–40` words if `max_sharpness < 2.0`, `60–110` words if `beat_dur <= 5.0s`, `100–180` words if `beat_dur > 5.0s`, `max_tokens=450`), uses the forensic system directive (`VISUAL FIDELITY OVER PLOT COMPLETION`), and enforces `ANTI-PREMATURE-CLOSURE RULE` (`thinking: {"type": "disabled"}`).
3. **Checkpointing & `$0` Deterministic Scene Merge**:
   - Checkpoints each completed beat immediately to `vlm_beats.jsonl` for crash-resume.
   - Deterministically concatenates each Scene Window's beats (`[MM:SS.ss - MM:SS.ss] {caption}` separated by `\n\n`), merges deduplicated keyframes, sets `exit_state: ""`, and writes `vlm_captions.jsonl` and `movie_knowledge_bundle.json` (preserving the exact Phase B GraphRAG schema).
