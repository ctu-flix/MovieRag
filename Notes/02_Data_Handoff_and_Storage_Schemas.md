---
Date: 2026-11-21
Top Level:
  - "[[_tech_]]"
Topic:
  - "[[_VideoRAG_]]"
  - "[[_Schema_]]"
  - "[[_Database_]]"
---

# Data Handoff & Storage Schemas

> Part of [[00_MovieRAG_Master_Plan]]
> Defines the intermediate JSON contract between Phase A (`Animated_AD` + `DeepSeek-Flash` micro-beat VLM) and Phase B (`MovieRAG`), as well as the upgraded internal schemas for `NanoVectorDB`, `JsonKVStorage`, and `NetworkX`.

---

## 1. Handoff Artifact: `movie_knowledge_bundle.json`

Saved at `workdir/{movie_id}/movie_knowledge_bundle.json`. Once this file and `workdir/{movie_id}/keyframes/` are generated on Colab and synced via `rclone`, **all downstream GraphRAG indexing and chat querying can run locally without touching the raw video file**.

> **Note on Micro-Beats & Schema Compatibility**: Intermediate shot/beat boundaries (`parent_shot_idx`, `is_subdivided_continuation`, and `scene["micro_beats"]`) live in `shots_and_scenes.json` and `vlm_beats.jsonl`. In `movie_knowledge_bundle.json`, Stage A5 merges consecutive 4–10s micro-beats back into `scene["vlm_caption"]` with `[MM:SS.ss - MM:SS.ss]` timestamps and keeps `"exit_state": ""` (retired) so the top-level and scene-level Phase B schema stays 100% compatible.

```json
{
  "movie_id": "sintel",
  "movie_title": "Sintel",
  "franchise_id": "sintel",
  "duration_sec": 888.042,
  "fps": 24.0,
  "character_glossary": {
    "Sintel": "Young woman protagonist. Visuals by timeline: in the Ishtar flashback (02:35 - 05:35), she is a teenager with short dark-brown hair...; in the mountain/desert quest (00:25 - 02:35 and 05:35 - 08:15), she wears a dark fur-lined travel cloak...; in the final volcanic cavern act (08:15 - 12:25), she has weathered shoulder-length hair with gray streaks and facial scars.",
    "Scales": "Small, playful baby light-brown dragon with large amber eyes, leathery light-brown wings, and a long tail. In the Ishtar flashback (02:58 - 05:05), his injured left wing is wrapped in a pale cloth bandage by Sintel. In the volcanic cavern (09:10 - 12:25), a separate, unbandaged fledgling baby light-brown dragon appears—this is a different baby dragon (Scales's own offspring), though Sintel initially mistakes it for young Scales.",
    "Adult Dragon": "Massive, fierce winged dark-brown dragon with horns, spiked spine, and large leathery wings. Over Ishtar (05:00 - 05:30), a giant wild adult dragon swoops down from the clouds and snatches baby Scales away. Inside the volcanic cavern (09:15 - 11:15), a different giant adult dragon lairs beside a fledgling baby dragon and battles Sintel. In the finale (11:15 - 12:35), a healed notch and bandage-scar on this cavern adult dragon's wing reveals it is actually grown-up Scales decades later.",
    "Shaman": "Elderly wearing a dark textured cap/headdress adorned with small curved horn ornaments, white ceremonial face paint across his eyes and nose, patterned robes, and a short weathered greyish-green beard, who shelters and questions Sintel inside a warmly lit desert hut.",
    "Gatekeeper": "Fierce, bald, unmasked tribal warrior with a dark mustache and goatee, wearing heavy robes and tall wooden banner poles strapped to his back (which create a spiky silhouette in the mountain mist), wielding a curved metallic spear-blade with a circular hilt ring (00:35 - 01:26)."
  },
  "external_plot_documents": [
    {
      "doc_id": "sintel_full_plot_01",
      "title": "Time-Aligned Sub-Beat Grounding Reference",
      "tier": "full_movie",
      "start_time": 0.0,
      "end_time": 888.042,
      "content": "Sintel (2010) — Time-Aligned Sub-Beat Grounding Reference:\n\nPrologue.A — Black Screen & Opening Credits Over Empty Snowy Mountains (00:00 - 00:25):\n...\nAct IV.B — The Wing-Scar & Blood-Reflection Twist and Cavern Collapse (11:15 - 12:35):\n..."
    }
  ],
  "scenes": [
    {
      "scene_id": "sintel_scene_0004",
      "scene_index": 4,
      "start_time": 101.25,
      "end_time": 131.04,
      "duration_sec": 29.79,
      "shot_ranges": [
        {
          "shot_idx": 11,
          "parent_shot_idx": 11,
          "is_subdivided_continuation": false,
          "start_frame": 2430,
          "end_frame": 2625,
          "start_time": 101.25,
          "end_time": 109.375
        }
      ],
      "characters_present": [
        {
          "name": "Sintel",
          "color_prompt": "red",
          "max_score": 0.84,
          "screen_time_ratio": 0.95
        },
        {
          "name": "Shaman",
          "color_prompt": "green",
          "max_score": 0.79,
          "screen_time_ratio": 0.88
        }
      ],
      "keyframes": [
        {
          "shot_idx": 11,
          "frame_idx": 2440,
          "timestamp": 101.67,
          "sharpness": 112.4,
          "motion_delta": 5.8,
          "path": "keyframes/sintel_scene_0004_shot11_101.67s.jpg",
          "characters_visible": [{"name": "Shaman", "color": "green", "bbox": [140, 80, 520, 690], "score": 0.79}]
        }
      ],
      "subtitles": [
        {
          "utt_id": "utt_0002",
          "start_time": 104.2,
          "end_time": 107.8,
          "speaker": "Shaman",
          "confidence": 0.82,
          "source": "ecapa_visual_or_manual",
          "text": "This blade has a dark past."
        }
      ],
      "vlm_caption": "[01:41.25 - 01:49.38] Inside a dimly lit, warm-toned hut, the Shaman examines the Gatekeeper's curved spear-blade by the fire while speaking, \"This blade has a dark past.\"\n\n[01:49.38 - 01:58.17] ...",
      "exit_state": ""
    }
  ]
}
```

---

## 2. Upgraded `MovieRAG` Storage Schemas (`workdir/{movie_id}/`)

### 2.1 `kv_store_video_segments.json` (`self.video_segments`)
Stores fine-grained per-scene structured records so `videorag_query()` and the agent tools (`get_adjacent_scenes`, `inspect_keyframes`) can dynamically clip straddling subtitles (`utt.start_time <= T_user`) and locate annotated shot-keyframes without re-parsing text:

```json
{
  "sintel": {
    "sintel_scene_0004": {
      "scene_index": 4,
      "start_time": 92.5,
      "end_time": 118.0,
      "time": "92.5-118.0",
      "vlm_caption": "Inside a dimly lit desert hut, an elderly tribal Shaman...",
      "exit_state": "Sintel remains seated across from the Shaman inside the desert hut...",
      "subtitles": [
        {
          "start_time": 94.2,
          "end_time": 97.8,
          "speaker": "Shaman",
          "text": "What brings you to the land of the gatekeepers?"
        },
        {
          "start_time": 98.5,
          "end_time": 101.4,
          "speaker": "Sintel",
          "text": "I'm looking for a dragon. Someone took him."
        }
      ],
      "characters": ["Sintel", "Shaman"],
      "keyframes": [
        {
          "shot_idx": 11,
          "timestamp": 94.5,
          "path": "keyframes/sintel_scene_0004_shot11_094.50s.jpg",
          "characters_visible": ["Shaman (green)"]
        },
        {
          "shot_idx": 11,
          "timestamp": 98.5,
          "path": "keyframes/sintel_scene_0004_shot11_098.50s.jpg",
          "characters_visible": ["Sintel (red)", "Shaman (green)"]
        }
      ],
      "content": "[SCENE sintel_scene_0004 | 00:01:32.50 - 00:01:58.00 | Characters: Sintel, Shaman]\n[Visual Description]:\nInside a dimly lit desert hut, an elderly tribal Shaman...\n\n[Subtitles / Dialogue]:\n- [00:01:34.20 - 00:01:37.80] Shaman: \"What brings you to the land of the gatekeepers?\"\n- [00:01:38.50 - 00:01:41.40] Sintel: \"I'm looking for a dragon. Someone took him.\"\n"
    }
  }
}
```

---

### 2.2 `kv_store_text_chunks.json` & `vdb_chunks.json` (`TextChunkSchema` — `1 Scene = 1 Chunk`)
In `MovieRAG/videorag/base.py`, `TextChunkSchema` is expanded to support **3-tier multi-resolution indexing** where **each scene is 1 dedicated Tier-1 chunk** (`400–1,000+` tokens, with `chunk_token_size = 2048` so a scene is never artificially packed with other scenes or truncated):

```python
class TextChunkSchema(TypedDict):
    tokens: int
    content: str
    video_segment_id: list[str]                    # e.g. ["sintel_scene_0004"] (1 scene per Tier-1 chunk)
    chunk_order_index: int
    movie_id: str                                  # e.g. "sintel"
    start_time: float                              # float seconds (e.g. 92.5)
    end_time: float                                # float seconds (e.g. 118.0)
    tier: Literal["scene", "chapter", "full_movie"]
    characters: list[str]                          # ["Sintel", "Shaman"]
```

In `MovieRAG/videorag/videorag.py`, `self.chunks_vdb` passes `meta_fields` so `NanoVectorDB` persists these fields alongside `"__id__"` and `"__vector__"`:

```python
self.chunks_vdb = self.vector_db_storage_cls(
    namespace="chunks",
    global_config=asdict(self),
    embedding_func=self.embedding_func,
    meta_fields={"movie_id", "start_time", "end_time", "tier", "characters", "video_segment_id"},
)
```

---

### 2.3 Rich Time-Sliced Entity Vector Database (`vdb_entities.json` / `self.entities_vdb`)
To give `entities_vdb` **full semantic richness** (so descriptive queries like *"the healer in the desert tent"* or *"the baby dragon with an injured wing"* match the right entity) **without leaking future spoilers**, `entities_vdb` stores **time-sliced entity embeddings**:

1. **What gets upserted into `entities_vdb`**:
   - **Base Glossary Slice (`start_time = first_seen_time`)**: `ent-{entity_name}-base` embeds `f"{entity_name}: {glossary_description}"` with `start_time = first_seen_time`, `end_time = first_seen_time`, `tier = "scene"`.
   - **Per-Scene / Per-Chapter Temporal Slices**: For each extracted entity description slice `d` in `temporal_descriptions`, we upsert `compute_mdhash_id(f"{entity_name}_{d['chunk_id']}", prefix="ent-")`:
     ```python
     {
         "content": f"{entity_name}: {d['description']}",
         "entity_name": entity_name,
         "movie_id": movie_id,
         "start_time": d["start_time"],
         "end_time": d["end_time"],
         "tier": d["tier"],
     }
     ```
2. **How `entities_vdb` is initialized in `videorag.py`**:
   ```python
   self.entities_vdb = self.vector_db_storage_cls(
       namespace="entities",
       global_config=asdict(self),
       embedding_func=self.embedding_func,
       meta_fields={"entity_name", "movie_id", "start_time", "end_time", "tier"},
   )
   ```
3. **Why this works**:
   - Every rich description from across the movie is embedded in `entities_vdb`.
   - At query time when `spoiler_free = True`, `entities_vdb.query(query, filter_lambda=...)` filters out any vector slice with `start_time > T_user` or `tier == "full_movie"`, deduplicates the matched `entity_name`s, and looks up their temporal history in the Knowledge Graph!

---

### 2.4 Time-Sliced Knowledge Graph (`graph_chunk_entity_relation.graphml`)
In `MovieRAG/videorag/_op.py`, nodes and edges store a JSON-encoded `temporal_descriptions` list (since GraphML attributes must be strings/scalars) in addition to `first_seen_time`:

#### Node Attributes in `NetworkXStorage`
| Attribute Key | Type | Example Value | Purpose |
| :--- | :--- | :--- | :--- |
| `entity_type` | `str` | `"CHARACTER"` | Standard entity classification |
| `first_seen_time` | `float` | `45.0` | Earliest `start_time` among `tier == "scene"` slices mentioning this entity (falls back to all slices only if no scene slice exists, preventing `00:00` chapter timestamps from leaking late-appearing entities). If `first_seen_time > T_user` when `spoiler_free=True`, the entity is completely hidden. |
| `temporal_descriptions` | `str` (JSON list) | `[{"chunk_id": "chunk-1", "start_time": 45.0, "end_time": 72.0, "tier": "scene", "description": "..."}]` | Filtered at query time so only facts with `start_time <= T_user` (and `tier != "full_movie"`) are assembled into the entity description. |
| `description` | `str` | `"Scales is a baby dragon rescued by Sintel..."` | Full-movie merged summary (used when `spoiler_free=False`). |
| `source_id` | `str` | `"chunk-1<SEP>chunk-9"` | Pointers back to `text_chunks`. |

#### Edge Attributes in `NetworkXStorage`
| Attribute Key | Type | Example Value | Purpose |
| :--- | :--- | :--- | :--- |
| `weight` | `float` | `3.0` | Cumulative edge weight |
| `first_seen_time` | `float` | `92.5` | Earliest `start_time` where this relationship occurs |
| `temporal_descriptions` | `str` (JSON list) | `[{"chunk_id": "chunk-2", "start_time": 92.5, "end_time": 118.0, "tier": "scene", "description": "Shaman questions Sintel about her quest."}]` | Prevents future relationship states (e.g. a betrayal at `01:40:00`) from leaking when `T_user < 01:40:00`. |
| `description` | `str` | `"..."` | Full-movie relationship summary |
| `source_id` | `str` | `"chunk-2"` | Pointers back to `text_chunks` |
