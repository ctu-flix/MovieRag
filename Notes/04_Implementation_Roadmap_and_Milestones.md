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

## Milestone 2: Build the 5-Stage `Animated_AD` Extraction Pipeline (`pipeline/` & `notebooks/`)

**Objective**: Create clean, modular, checkpointed CLI scripts in `pipeline/` wrapped by **2 environment-isolated Colab Pro GPU notebooks (`notebooks/`)** that mirror `Animated_AD`'s exact environment split (Visual Stack vs. Audio/VLM Stack) and communicate via Google Drive JSON checkpoints to produce `movie_knowledge_bundle.json` and `keyframes/`.

### Notebook 1 (`notebooks/PhaseA1_Visual_Pipeline.ipynb` — `Animated_AD/conda.yaml` Visual Environment)

#### Step 2.1: Shot & Scene Grouping (`pipeline/stage1_scenes.py`)
- Input: `--video_path`, `--output_dir`, `--min_scene_sec 12`, `--max_scene_sec 30`.
- Runs PySceneDetect (`shot_boundary_detection()`), subdivides unbroken takes $>20\text{s}$ into $15\text{s}$ sub-shots, groups contiguous shots into 15–30s Scene Windows, and writes `shots_and_scenes.json` to Google Drive.

#### Step 2.2: Semi-Automated Character Bank (`pipeline/stage2_charbank.py`)
- Input: `--movie_id`, `--char_bank_dir`, `--output_dir`, `--model_size giant|large`.
- Reads exemplar images from `char_bank/{movie_id}/raw_images/{character}/` and `character_glossary.json` (or triggers optional web crawler / glossary helper if requested).
- Runs 75-epoch `DINOv2` linear adapter contrastive fine-tuning (`finetune.py`) and extracts normalized character feature banks (`feature_extraction.py`), then frees GPU VRAM.

#### Step 2.3: Per-Shot Visual Character Recognition & Adaptive Annotated Keyframes (`pipeline/stage3_visual_recog.py`)
- Input: `--video_path`, `--shots_file`, `--char_features_dir`, `--output_dir`.
- Runs `TGRP` (`OWLv2` + `SAM 2.1` bidirectional mask propagation + tripartite track matching) **per individual camera shot** with JSONL resume support, classifies tracks with the fine-tuned `DINOv2` bank, and maps shot tracks to global frame indices (`visual_tracks_per_frame.json`).
- **Adaptive Per-Shot Keyframe Extraction**:
  - Selects 1 sharpest frame (Laplacian variance) for short shots ($<2\text{s}$) and 1 sharpest frame every ~2s for longer shots ($\ge 2\text{s}$).
  - Draws colored character bounding boxes ($s_{\text{vm}} > 0.45$) onto each selected shot-keyframe and saves to `workdir/{movie_id}/keyframes/` on Google Drive.

---

### Notebook 2 (`notebooks/PhaseA2_Audio_and_VLM.ipynb` — Audio + `Qwen2.5-VL` Environment)

#### Step 2.4: In-Movie Self-Bootstrapped Voice Bank & Speaker Attribution (`pipeline/stage4_subtitles.py`)
- Input: `--video_path`, `--subtitle_path` (optional `.srt`/`.ass`), `--visual_tracks_file`, `--output_file`.
- Parses SDH `[Character]:` tags if available; keeps `.srt` timestamps if standard `.srt` is provided; runs `WhisperX` only if no subtitle file exists.
- **3-Step In-Movie Speaker Attribution**:
  1. Harvests anchor voice clips from shots where a character has high visual confidence ($s_{\text{vm}} > 0.60$) and `LWTNet` confirms active lip-sync ($s_{\text{sync}} > 0.30$).
  2. Extracts `ECAPA-TDNN` embeddings from those anchor clips to construct the per-character In-Movie Voice Bank.
  3. Runs `ECAPA-TDNN` audio matching ($c_a$) + `LWTNet` visual enhancement (for $c_a < 0.35$) + Stage-2 LLM dialogue refinement to output `attributed_subtitles.json` to Google Drive, then flushes GPU VRAM.

#### Step 2.5: Two-Pass Character-Grounded VLM Scene Captioner (`pipeline/stage5_vlm_caption.py`)
- Input: `--scenes_file`, `--keyframes_dir`, `--visual_tracks_file`, `--subtitles_file`, `--glossary_file`, `--output_bundle`.
- For each Scene Window:
  - **Pass 1 (100% Vision Attention)**: Feeds the scene's annotated shot-keyframes + color-to-character legend + static `character_glossary.json` + speaker-attributed subtitles + previous scene's `exit_state` (& last 2–3 subtitle lines) into `Qwen2.5-VL-7B-Instruct` (or `MiniCPM-V-2_6-int4`) to generate a rich `300–600` token `vlm_caption`.
  - **Pass 2 (Exit-State Extraction)**: Feeds Pass 1's output + the final keyframe of the scene + last 2–3 subtitle lines to extract a 2–3 sentence `exit_state` for the next scene.
- Writes the final `movie_knowledge_bundle.json` to Google Drive.

---

## Milestone 3: End-to-End Validation on `Sintel.mp4` & Scale-Up to `CMD-AM`

### Step 3.1: Set Up the 2 Phase-A Colab Notebooks & Clean Up `VideoRAG.ipynb`
- Create `notebooks/PhaseA1_Visual_Pipeline.ipynb` (Stages A1–A3: `PySceneDetect`, `OWLv2`, `SAM 2.1`, `DINOv2`) and `notebooks/PhaseA2_Audio_and_VLM.ipynb` (Stages A4–A5: `WhisperX`, `LWTNet`, `ECAPA-TDNN`, `Qwen2.5-VL-7B`), reading/writing intermediate artifacts to `/content/drive/MyDrive/MovieRAG/{movie_id}/`.
- Clean up `VideoRAG.ipynb` (removing legacy `torch==2.1.2` downgrade and `ImageBind` / `pytorchvideo` cells) so it serves as the **Phase B & C Indexing & Interactive Chat Notebook** (calling `videorag.insert_knowledge_bundle(...)` and testing multi-turn `chat_history` + tool-call traces alongside `api/app.py`).

### Step 3.2: End-to-End Evaluation on `Sintel.mp4` (15 mins) & `CMD-AM` (`Sing` / `Kung Fu Panda`)
- Verify visual grounding accuracy (characters `Sintel`, `Scales`, `Shaman` named accurately in VLM scene captions and speaker-attributed subtitles).
- Verify Turn-0 zero-hop answers vs. Rounds 1–3 multi-hop tool calls (`search_scenes`, `get_character_history`, `get_adjacent_scenes`, `inspect_keyframes`).
- Verify timestamp cutoff precision across early, middle, and late timestamps with `spoiler_free=True` vs. `spoiler_free=False`.
- Scale to a full-length `CMD-AM` animated movie (`Sing` or `Kung Fu Panda`) to benchmark end-to-end ingestion time and retrieval latency.
