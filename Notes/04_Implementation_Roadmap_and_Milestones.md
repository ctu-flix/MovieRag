---
Date: 2026-11-21
Top Level:
  - "[[_tech_]]"
Topic:
  - "[[_VideoRAG_]]"
  - "[[_Roadmap_]]"
  - "[[_Implementation_]]"
---

# Implementation Roadmap & Validation Milestones

> Part of [[00_MovieRAG_Master_Plan]]
> Step-by-step engineering checklist mapping out exact code changes across `MovieRAG/`, `Animated_AD/`, and `VideoRAG.ipynb`, along with validation milestones on `Sintel.mp4` and `CMD-AM`.

---

## Milestone 1: Refactor `MovieRAG` Core for DeepSeek API, Spoiler-Aware GraphRAG & Agentic Retrieval — **[COMPLETED & VERIFIED]**

**Objective**: Decouple `MovieRAG` from `ImageBind` and runtime `MiniCPM-V` re-captioning, unify on `deepseek-flash` (with selective `thinking: enabled` for chat and `disabled` for indexing/helpers/vision), upgrade `NanoVectorDBStorage` to filter by timestamp/tier metadata, implement `1 Scene = 1 Chunk` + time-sliced Knowledge Graph/`entities_vdb` storage, and build the **6-Block Hybrid Turn-0 GraphRAG + 3-Round Agentic Retrieval Loop** with multilingual support and Source-of-Truth plot grounding.

### Step 1.1: Configure DeepSeek Tool-Calling API, Vision Sub-Tool & Local Embeddings (`MovieRAG/videorag/_llm.py`) — **[COMPLETED]**
- Added `deepseek_config` & `deepseek_bge_config`:
  - Defaults to `model="deepseek-flash"` (configurable via `DEEPSEEK_MODEL`) at `https://api.deepseek.com` with `thinking: {"type": "disabled"}` for fast structured extraction, chapter summaries, and query rewriting.
  - Added `openai_chat_with_tools()` with `thinking: {"type": "enabled"}` by default (configurable via `DEEPSEEK_CHAT_THINKING`), `max_tokens >= 4096`, and `reasoning_content` passthrough for multi-round tool calling.
  - Added `local_or_api_embedding()` supporting local CPU `sentence-transformers` (`BAAI/bge-base-en-v1.5`, 768-dim), Ollama `nomic-embed-text`, SiliconFlow/OpenAI API, and deterministic hash fallback.
- Added `vision_subtool_complete()` (Option A Vision Sub-Tool):
  - Downscales up to 4 annotated shot-keyframes (`960px` width) and queries `deepseek-flash` (`thinking: disabled`, overridable via `VISION_MODEL_ID`), returning a grounded text observation back to the orchestrator agent.

### Step 1.2: Upgrade Schemas & Vector DB Metadata Filtering — **[COMPLETED]**
1. **`MovieRAG/videorag/base.py`**:
   - Added `current_timestamp`, `spoiler_free`, `max_tool_rounds`, `chat_history`, and `wo_reference` to `QueryParam`.
   - Expanded `TextChunkSchema` with `movie_id`, `start_time`, `end_time`, `tier`, and `characters`.
2. **`MovieRAG/videorag/_storage/vdb_nanovectordb.py`**:
   - Updated `NanoVectorDBStorage.query(self, query: str, top_k=5, filter_lambda=None)` to pass `filter_lambda=filter_lambda` into `self._client.query(...)` (matching the real `nano-vectordb` package signature), and respected `llm.query_better_than_threshold`.
   - Lazy-loaded `imagebind` and `torch` inside `NanoVectorDBVideoSegmentStorage` and added pure-Python fallbacks for minimal local environments.
3. **`MovieRAG/videorag/videorag.py`**:
   - Raised `chunk_token_size = 2048` (`1 Scene = 1 Chunk`), set `chapter_window_sec = 300.0`, `enable_visual_index = False`.
   - Configured `chunks_vdb` (`meta_fields={"movie_id", "start_time", "end_time", "tier", "characters", "video_segment_id"}`) and `entities_vdb` (`meta_fields={"entity_name", "movie_id", "start_time", "end_time", "tier"}`).
   - Auto-loads `movie_id`, `movie_title`, `duration_sec`, `character_glossary`, and `external_plot_documents` from `movie_knowledge_bundle.json` and passes `addon_params` by reference so `last_tool_calls_trace` persists on the `VideoRAG` instance.

### Step 1.3: Implement `insert_knowledge_bundle()` & Multi-Resolution Chunking (`MovieRAG/videorag/videorag.py` & `_op.py`) — **[COMPLETED]**
- Implemented `VideoRAG.insert_knowledge_bundle()` / `ainsert_knowledge_bundle()` building Tier 1 (`scene`), Tier 2 (`chapter` via `deepseek-flash`), and Tier 3 (`full_movie` from `external_plot_documents`), with Colab notebook `notebooks/PhaseB1_GraphRAG_Indexing.ipynb`.

### Step 1.4: Implement Time-Sliced Graph Merging & Hybrid Turn-0 + Agentic `videorag_query()` (`MovieRAG/videorag/_op.py` & `prompt.py`) — **[COMPLETED]**
- **Indexing (`extract_entities`)**:
  - Time-sliced `temporal_descriptions` on every node/edge, `first_seen_time` computed from `tier == "scene"` slices first, leading-article & quote normalization, and per-slice `entities_vdb` upserts.
- **Querying (`videorag_query`)**:
  - **Step 0.1 (`_resolve_query_with_chat_history`)**: Resolves follow-up pronouns against `chat_history[-6:]` and translates non-English queries into standalone English for retrieval.
  - **6-Block Turn-0 XML Context**: `<viewer_state>`, `<authoritative_plot_timeline_up_to_T_user>` (completed external plot blocks `end_sec <= T_user`), `<previous_scene_before_T_user>`, `<current_scene_at_T_user>` (micro-beat + subtitle clipped), `<graph_entities_and_relations_up_to_T_user>` (timestamp-filtered glossary, 1st + 5 most recent entity slices, prioritized 1-hop relations), and `<retrieved_scene_and_chapter_chunks_up_to_T_user>` (deduplicated Vector + `tier == "scene"` Graph co-occurrence chunks).
  - **System Prompt (`prompt.py`)**: Enforces Source-of-Truth Hierarchy (external plot overrides conflicting scene captions) and a language-neutral response rule (answer in user's exact language; pass English to tools).
  - **Bounded 3-Round Agentic Tool Loop**: Spoiler-gated `search_scenes`, `get_character_history`, `get_adjacent_scenes`, and `inspect_keyframes` (with optional `target_timestamp`).
- **Verification**: Passed both mock-Sintel unit suite and full real Colab `PhaseB2_GraphRAG_Retrieval_Test.ipynb` benchmark (`02:00` / `05:40` spoiler-free vs. `spoiler_free=False`, `inspect_keyframes` with `target_timestamp=105.0`, and 3-turn timeline-scrubbing chat).

### Step 1.5: Standalone FastAPI Chatbot Service Layer (`api/app.py`) — **[NEXT UP — PHASE C]**
- Create `api/app.py` for the single-movie interactive demo:
  - **Lifespan Startup**: Load the pre-indexed `VideoRAG` instance from `workdir/{movie_id}/` once into memory (CPU-only, zero GPU required).
  - **Endpoints**:
    - `GET /api/v1/movie/info`: Returns `movie_id`, `movie_title`, `duration_sec`, `character_glossary`, and chapter boundaries from `movie_knowledge_bundle.json`.
    - `POST /api/v1/chat`:
      - Request JSON: `{ "query": str, "current_timestamp": float | str, "spoiler_free": bool = True, "chat_history": list[dict] = [] }`
      - Response JSON: `{ "answer": str, "citations": list[str], "tool_calls_trace": list[dict], "effective_timestamp": float, "spoiler_free": bool }`

---

## Milestone 2: Build & Upgrade the Phase A Extraction Pipeline (`pipeline/` & `notebooks/`) — **[COMPLETED]**

**Objective**: Modular, checkpointed CLI scripts in `pipeline/` wrapped by **4 environment-isolated Google Colab notebooks (`notebooks/`)** producing `movie_knowledge_bundle.json` and `keyframes/` for Phase B ingestion.

### Notebook 0 (`notebooks/PhaseA0_CharBank_Bootstrap.ipynb` — T4 GPU)
- **Step 2.0: Credits Preview & Hybrid Cluster Character Bank (`pipeline/stage0_bootstrap_charbank.py`)**:
  - Samples end-of-movie preview frames so the user picks exact `CREDITS_START_SEC` manually.
  - Detects crops with `OWLv2` (`_nms_boxes`), clusters via `DINOv2` + `AgglomerativeClustering` into visual prototypes (`clusters_contact_sheet.jpg`), and exports human-labeled clusters into `char_bank/{movie_id}/raw_images/{Character}/` and `character_glossary.json`.

### Notebook 1 (`notebooks/PhaseA1_Visual_Pipeline.ipynb` — L4/T4 GPU)
- **Step 2.1: Shot, Lineage, Scene & Micro-Beat Grouping (`pipeline/stage1_scenes.py`)**:
  - Runs `PySceneDetect`, applies manual `credits_start_sec`, subdivides unbroken takes $>20\text{s}$ into $15\text{s}$ sub-shots with `parent_shot_idx` & `is_subdivided_continuation`, groups contiguous shots into `15–30s` Scene Windows, and partitions scenes into `4–10s` `micro_beats` (`build_micro_beats`).
- **Step 2.2: Multi-State Character Bank (`pipeline/stage2_charbank.py`)**:
  - Runs 75-epoch `DINOv2` contrastive fine-tuning (`finetune.py`) and builds retrieval `.npz` banks (`require_detection=False`, Farthest-Point Diversity Selection, `num_per_character=16`).
- **Step 2.3: Upgraded Per-Shot TGRP & Motion-Adaptive 2–6 fps Keyframes (`pipeline/stage3_visual_recog.py`)**:
  - Uses expanded OWLv2 prompts (`animated character`, `dragon or creature`, `person or warrior`), `min_box_area_ratio=0.0015`, 3 sharpest non-black seeds per third (`_select_sharpest_seed_frames`), strict seam carry-over (`carry_0` only when `is_subdivided_continuation=True`), 1-seed rescue (`OWLv2 >= 0.20`, `DINOv2 >= 0.55`), and **motion-adaptive 2–6 fps** sharp annotated keyframes (`base_fps=2.0`, `max_action_fps=6.0`, `min_sharpness_floor=10.0`, `max_keyframes_per_scene=150`, `reextract_keyframes_only`, `PINNED_KEYFRAME_TIMESTAMPS`).

### Notebook 2 (`notebooks/PhaseA2_Audio_Pipeline.ipynb` — T4 GPU)
- **Step 2.4: Voice Bank & Speaker Attribution (`pipeline/stage4_subtitles.py`)**:
  - Loads `.srt`/SDH or transcribes with `faster-whisper` (`large-v3`), harvests in-movie `ECAPA-TDNN` voice anchors from $s_{\text{vm}} > 0.60$ tracks (+ optional `LWTNet`), attributes dialogue, and supports pure manual `SPEAKER_OVERRIDES` via `apply_subtitle_speaker_fixes()`.

### Notebook 3 (`notebooks/PhaseA3_VLM_Captioning.ipynb` — Free CPU Runtime)
- **Step 2.5: Micro-Beat `deepseek-flash` VLM Captioner & `$0` Merge (`pipeline/stage5_vlm_caption.py`)**:
  - Runs single-pass `deepseek-flash` (`thinking: {"type": "disabled"}`, `960px` width, `max_tokens=450`) across `4–10s` micro-beats (`vlm_beats.jsonl` resume checkpoint).
  - Applies visual overlap frames + `[ALREADY HAPPENED IN PREVIOUS BEAT]` continuity (retiring `[EXIT_STATE]`), timeline-gated visual-first `character_glossary.json` with `VISUAL CONSTRAINT`s, `appearance_windows.json` filtering (`out_of_window_min_score=0.60`), phase-granular `external_plot.txt` sub-blocks, low-level telemetry alerts (`sharpness < 1.0`, 0 tracked characters), duration-scaled word budgets (`20–40` / `60–110` / `100–180`), and the `ANTI-PREMATURE-CLOSURE RULE`.
  - Deterministically merges beats back into `15–30s` scenes (`$0`) and writes `movie_knowledge_bundle.json`.

---

## Milestone 3: End-to-End `Sintel.mp4` Validation & Phase C Service — **[PHASE B DONE | PHASE C NEXT UP]**

### Step 3.1: Colab Notebooks Setup & Full Sintel Run — **[COMPLETED]**
- Created and verified all 6 notebooks: `PhaseA0_CharBank_Bootstrap.ipynb`, `PhaseA1_Visual_Pipeline.ipynb`, `PhaseA2_Audio_Pipeline.ipynb`, `PhaseA3_VLM_Captioning.ipynb`, `PhaseB1_GraphRAG_Indexing.ipynb`, and `PhaseB2_GraphRAG_Retrieval_Test.ipynb`.
- Re-ran `PhaseA3` + `PhaseB1` + `PhaseB2` on *Sintel* with the disambiguated `character_glossary.json` / `external_plot.txt` and verified zero-hop Turn-0 answers, strict spoiler isolation (`02:00`, `02:20`, `05:40`), full-movie twist accuracy (`spoiler_free=False`), `inspect_keyframes` with `target_timestamp`, and 3-turn timeline-scrubbing `chat_history`.

### Step 3.2: Phase C FastAPI Service & Scaling — **[NEXT UP]**
- Implement **Step 1.5** (`api/app.py`: `GET /api/v1/movie/info` and `POST /api/v1/chat`) serving the pre-indexed `workdir/sintel/` bundle on CPU.
- Optionally scale to a full-length `CMD-AM` animated movie (`Sing` or `Kung Fu Panda`) to benchmark end-to-end ingestion time and retrieval latency.
