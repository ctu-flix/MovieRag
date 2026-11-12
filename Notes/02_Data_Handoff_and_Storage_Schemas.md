---
Date: 2026-11-11
Top Level:
  - "[[_tech_]]"
Topic:
  - "[[_VideoRAG_]]"
  - "[[_Schema_]]"
  - "[[_Database_]]"
---

# Data Handoff & Storage Schemas

> Part of [[00_MovieRAG_Master_Plan]]
> Defines the intermediate JSON contract between Phase A (`Animated_AD` + VLM) and Phase B (`MovieRAG`), as well as the upgraded internal schemas for `NanoVectorDB`, `JsonKVStorage`, and `NetworkX`.

---

## 1. Handoff Artifact: `movie_knowledge_bundle.json`

Saved at `workdir/{movie_id}/movie_knowledge_bundle.json`. Once this file and `workdir/{movie_id}/keyframes/` are generated on Colab Pro and synced via `rclone`, **all downstream GraphRAG indexing and chat querying can run locally without touching the raw video file**.

```json
{
  "movie_id": "sintel",
  "movie_title": "Sintel",
  "franchise_id": "sintel",
  "duration_sec": 888.0,
  "fps": 24.0,
  "character_glossary": {
    "Sintel": "Young woman searching for her pet dragon Scales; wears a brown cloak and carries a staff.",
    "Scales": "Baby red dragon with a healing wing rescued by Sintel.",
    "Shaman": "Elderly tribal healer with white dreadlocks in a desert hut."
  },
  "external_plot_documents": [
    {
      "doc_id": "sintel_full_plot_01",
      "title": "Full Plot & Ending Twist Analysis",
      "tier": "full_movie",
      "start_time": 0.0,
      "end_time": 888.0,
      "content": "Sintel searches the world for her baby dragon Scales, only to tragically slay an adult dragon in its lair and discover from the wing scar that the adult dragon was Scales grown up..."
    }
  ],
  "scenes": [
    {
      "scene_id": "sintel_scene_0004",
      "scene_index": 4,
      "start_time": 92.5,
      "end_time": 118.0,
      "shot_ranges": [
        {
          "shot_idx": 11,
          "start_frame": 2220,
          "end_frame": 2510,
          "start_time": 92.5,
          "end_time": 104.58
        },
        {
          "shot_idx": 12,
          "start_frame": 2510,
          "end_frame": 2832,
          "start_time": 104.58,
          "end_time": 118.0
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
          "frame_idx": 2268,
          "timestamp": 94.5,
          "path": "keyframes/sintel_scene_0004_shot11_094.50s.jpg",
          "characters_visible": [{"name": "Shaman", "color": "green", "bbox": [140, 80, 520, 690]}]
        },
        {
          "shot_idx": 11,
          "frame_idx": 2364,
          "timestamp": 98.5,
          "path": "keyframes/sintel_scene_0004_shot11_098.50s.jpg",
          "characters_visible": [{"name": "Sintel", "color": "red", "bbox": [210, 95, 610, 700]}, {"name": "Shaman", "color": "green", "bbox": [50, 110, 290, 680]}]
        },
        {
          "shot_idx": 12,
          "frame_idx": 2604,
          "timestamp": 108.5,
          "path": "keyframes/sintel_scene_0004_shot12_108.50s.jpg",
          "characters_visible": [{"name": "Sintel", "color": "red", "bbox": [180, 75, 590, 710]}]
        }
      ],
      "subtitles": [
        {
          "utt_id": "utt_0012",
          "start_time": 94.2,
          "end_time": 97.8,
          "speaker": "Shaman",
          "confidence": 0.82,
          "source": "sdh_or_avsync",
          "text": "What brings you to the land of the gatekeepers?"
        },
        {
          "utt_id": "utt_0013",
          "start_time": 98.5,
          "end_time": 101.4,
          "speaker": "Sintel",
          "confidence": 0.89,
          "source": "sdh_or_avsync",
          "text": "I'm looking for a dragon. Someone took him."
        }
      ],
      "vlm_caption": "Inside a dimly lit desert hut, an elderly tribal Shaman with white dreadlocks sits across a low brazier from Sintel, who is still recovering from exhaustion. In the first shot, Shaman leans forward and gestures toward the entrance while questioning her purpose in the land of the gatekeepers. When the camera cuts to a close-up of Sintel, her eyes narrow with grief and determination as she tightens her grip on her wooden staff and explains that her dragon was abducted...",
      "exit_state": "Sintel remains seated across from the Shaman inside the desert hut clutching her staff, having just revealed that her dragon was kidnapped, while the Shaman listens intently."
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
| `first_seen_time` | `float` | `45.0` | Earliest `start_time` of any chunk mentioning this entity. If `first_seen_time > T_user` when `spoiler_free=True`, the entity is completely hidden. |
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
