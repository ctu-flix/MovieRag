---
Date: 2026-11-11
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

## Milestone 1: Refactor `MovieRAG` Core for DeepSeek API, Spoiler-Aware GraphRAG & Agentic Retrieval

**Objective**: Decouple `MovieRAG` from `ImageBind` and runtime `MiniCPM-V` re-captioning, add `DeepSeek-V3` tool-calling + Vision Sub-Tool API configurations, upgrade `NanoVectorDBStorage` to filter by timestamp/tier metadata, implement `1 Scene = 1 Chunk` + time-sliced Knowledge Graph/`entities_vdb` storage, and build the **Hybrid Turn-0 GraphRAG + 3-Round Agentic Retrieval Loop**.

### Step 1.1: Configure DeepSeek Tool-Calling API, Vision Sub-Tool & Local Embeddings (`MovieRAG/videorag/_llm.py`)
- Add `deepseek_config`:
  - Uses `openai_complete_if_cache` pointed at `base_url="https://api.deepseek.com/v1"`, `model="deepseek-chat"` for both `best_model_func` and `cheap_model_func`.
  - Add `openai_chat_with_tools()` helper supporting multi-turn `messages` + OpenAI `tools` definitions so `deepseek-chat` can execute up to 3 tool-calling rounds in `videorag_query()`.
  - Uses local Ollama `nomic-embed-text` (`embedding_dim=768`) or OpenAI-compatible text embedding API for `embedding_func`.
- Add `vision_subtool_complete()` (Option A Vision Sub-Tool):
  - Accepts `visual_question: str`, `scene_metadata: dict`, and `keyframe_paths: list[str]`.
  - Encodes up to 4 annotated shot-keyframes as base64 data URLs and queries a configurable OpenAI-compatible Vision API (`Qwen/Qwen2.5-VL-72B-Instruct`, `gemini-2.5-flash`, or `gpt-4o-mini`), returning a detailed text observation back to `DeepSeek-V3`.

### Step 1.2: Upgrade Schemas & Vector DB Metadata Filtering
1. **`MovieRAG/videorag/base.py`**:
   - Add `current_timestamp: Optional[float] = None`, `spoiler_free: bool = True`, `max_tool_rounds: int = 3`, `chat_history: list[dict[str, str]] = field(default_factory=list)` (OpenAI-format `[{"role": "user"|"assistant", "content": "..."}]`, capped at the last 6 messages / 3 turns), and `wo_reference: bool = False` to `QueryParam`.
   - Expand `TextChunkSchema` with `movie_id: str`, `start_time: float`, `end_time: float`, `tier: str`, `characters: list[str]`.
2. **`MovieRAG/videorag/_storage/vdb_nanovectordb.py`**:
   - Update `NanoVectorDBStorage.query(self, query: str, top_k=5, filter_lambda=None)` to pass `filter=filter_lambda` into `self._client.query(...)`.
   - Remove the top-level `from imagebind.models import imagebind_model` import (or make it lazy inside `NanoVectorDBVideoSegmentStorage`) so `MovieRAG` runs cleanly without `imagebind` installed.
3. **`MovieRAG/videorag/videorag.py`**:
   - Raise `chunk_token_size: int = 2048` so `1 Scene = 1 Rich Tier-1 Chunk` is never split.
   - Update `self.chunks_vdb` initialization to include:
     ```python
     meta_fields={"movie_id", "start_time", "end_time", "tier", "characters", "video_segment_id"}
     ```
   - Update `self.entities_vdb` initialization to support rich time-sliced entity vectors:
     ```python
     meta_fields={"entity_name", "movie_id", "start_time", "end_time", "tier"}
     ```
   - Remove `self.video_segment_feature_vdb` initialization (or gate behind `enable_visual_index: bool = False`).

### Step 1.3: Implement `insert_knowledge_bundle()` & Multi-Resolution Chunking (`MovieRAG/videorag/videorag.py` & `_op.py`)
- Add `VideoRAG.insert_knowledge_bundle(self, bundle_path: str)`:
  1. Loads `movie_knowledge_bundle.json`.
  2. Populates `self.video_segments` with structured per-scene metadata (`start_time`, `end_time`, `vlm_caption`, `exit_state`, `subtitles`, `keyframes`, `characters`).
  3. Builds **Tier 1 (`scene`)** chunks on a strict **1 Scene = 1 Rich Chunk** basis (`vlm_caption` + formatted speaker-attributed `subtitles`, preserving exact `start_time` and `end_time`).
  4. Calls `deepseek-chat` to generate **Tier 2 (`chapter`)** summaries for every 5–10 minutes of scenes (using only Tier-1 scenes, zero `external_plot`).
  5. Builds **Tier 3 (`full_movie`)** chunks from `external_plot_documents` + chapter synthesis (`end_time = duration_sec`).
  6. Runs `ainsert()` to upsert all chunks into `chunks_vdb`, extract time-sliced entities/relations into `chunk_entity_relation_graph` and time-sliced `entities_vdb`, and persist to `working_dir/{movie_id}/`.

### Step 1.4: Implement Time-Sliced Graph Merging & Hybrid Turn-0 + Agentic `videorag_query()` (`MovieRAG/videorag/_op.py`)
- **Indexing (`extract_entities`)**:
  - Pass `chunk_dp["start_time"]`, `chunk_dp["end_time"]`, and `chunk_dp["tier"]` into `_handle_single_entity_extraction()` and `_handle_single_relationship_extraction()`.
  - Store `temporal_descriptions` (JSON string) and `first_seen_time` (float) on every node and edge in `_merge_nodes_then_upsert()` and `_merge_edges_then_upsert()`.
  - Upsert one vector record per `(entity_name, chunk_id)` temporal slice into `entities_vdb` with `start_time`, `end_time`, and `tier` metadata.
- **Querying (`videorag_query`)**:
  1. **Build Turn-0 Spoiler-Filtered GraphRAG Context (with `chat_history` Coref Resolution)**:
     - Pass `query_param.chat_history[-6:]` into `_refine_entity_retrieval_query()` and Turn-0 vector search query rewriting so follow-up pronouns (*"Why did he say that?"*, *"Where did she get it?"*) resolve against the previous turn before querying `chunks_vdb` and `entities_vdb`.
     - Fetch Active Scene at `T_user` (`start_time <= T_user <= end_time`, clipping subtitles $> T_{\text{user}}$) + unlocked Chapter Table of Contents.
     - Query `chunks_vdb` with straddle-safe `chunk_filter`.
     - Query `entities_vdb` ($\le T_{\text{user}}$) $\rightarrow$ run `_find_most_related_segments_from_entities()` ($\le T_{\text{user}}$) to retrieve both **Top Graph-Ranked Scene Chunks** and **Spoiler-Filtered Entity & Edge Histories** (`temporal_descriptions` $\le T_{\text{user}}$).
     - Never persist previous turns' raw Turn-0 context blocks in `chat_history`—always re-compute Turn-0 context fresh using the current `T_user` and `spoiler_free`.
  2. **Execute Bounded Agentic Tool-Calling Loop (`max_tool_rounds = 3`)**:
     - Bind the 4 spoiler-gated tools (`search_scenes`, `get_character_history`, `get_adjacent_scenes`, `inspect_keyframes`) with `T_user` and `spoiler_free` locked in their closures.
     - Insert `query_param.chat_history[-6:]` into the `DeepSeek-V3` message array between the system prompt (containing the fresh Turn-0 context block) and the current user message.
     - Send the messages to `deepseek-chat`. If it returns a direct answer (0 tool calls), return immediately; if it invokes tools, execute tool calls in parallel for up to 3 rounds, then synthesize the final grounded answer with `[MM:SS - MM:SS]` citations.

### Step 1.5: Standalone FastAPI Chatbot Service Layer (`api/app.py`)
- Create `api/app.py` for the single-movie interactive demo:
  - **Lifespan Startup**: Load the pre-indexed `VideoRAG` instance from `workdir/{movie_id}/` once into memory (CPU-only, zero GPU required).
  - **Endpoints**:
    - `GET /api/v1/movie/info`: Returns `movie_id`, `movie_title`, `duration_sec`, `character_glossary`, and chapter boundaries from `movie_knowledge_bundle.json`.
    - `POST /api/v1/chat`:
      - Request JSON: `{ "query": str, "current_timestamp": float | str, "spoiler_free": bool = True, "chat_history": list[dict] = [] }`
      - Response JSON: `{ "answer": str, "citations": list[str], "tool_calls_trace": list[dict], "effective_timestamp": float, "spoiler_free": bool }`

### Milestone 1 Manual / Script Verification Check
- Run a lightweight verification script using a mock 4-scene `Sintel` bundle containing the ending dragon twist at `t=720s`:
  - Query at `T_user = 200.0, spoiler_free = True`: *"What happens to Scales the baby dragon?"* $\rightarrow$ Must **only** report that Scales was rescued and later kidnapped by an adult dragon; zero mention of Sintel killing Scales at the end.
  - Query at `T_user = 200.0, spoiler_free = False`: *"What happens to Scales the baby dragon?"* $\rightarrow$ Must explain the full tragic ending twist using Tier 3 & full-movie graph edges.
  - Verify that a visual detail question triggers `inspect_keyframes` and returns a grounded visual answer.

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

## Milestone 3: Final Phase A Run on `Sintel.mp4`, Phase B/C Implementation & Validation — **[IN PROGRESS]**

### Step 3.1: Colab Notebooks Setup — **[COMPLETED]**
- Created and configured `PhaseA0_CharBank_Bootstrap.ipynb`, `PhaseA1_Visual_Pipeline.ipynb`, `PhaseA2_Audio_Pipeline.ipynb`, and `PhaseA3_VLM_Captioning.ipynb`.

### Step 3.2: Re-Run Full Stage A5 Bundle & End-to-End Phase B/C Validation — **[NEXT UP]**
- Run `PhaseA3_VLM_Captioning.ipynb` (Cell 6 with `RESTART_STAGE5 = True` and the updated `char_bank/sintel/` + `stage5_vlm_caption.py` patch) to regenerate the clean `movie_knowledge_bundle.json` for *Sintel*.
- Implement **Milestone 1** across `MovieRAG/videorag/` (`base.py`, `_llm.py`, `vdb_nanovectordb.py`, `videorag.py`, `_op.py`) and `api/app.py`, and clean up `VideoRAG.ipynb` (`torch==2.1.2` / `ImageBind` removal).
- Verify Turn-0 zero-hop answers, Rounds 1–3 agent tool calls (`search_scenes`, `get_character_history`, `get_adjacent_scenes`, `inspect_keyframes`), and timestamp/spoiler isolation across early, middle, and late timestamps with `spoiler_free=True` vs. `spoiler_free=False`.
- Scale to a full-length `CMD-AM` animated movie (`Sing` or `Kung Fu Panda`) to benchmark end-to-end ingestion time and retrieval latency.
