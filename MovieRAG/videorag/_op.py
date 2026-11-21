import asyncio
import json
import os
import re
from collections import Counter, defaultdict
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from ._llm import openai_chat_with_tools, vision_subtool_complete
from ._splitter import SeparatorSplitter
from ._utils import (
    clean_str,
    compute_mdhash_id,
    decode_tokens_by_tiktoken,
    encode_string_by_tiktoken,
    get_tiktoken_encoder,
    is_float_regex,
    list_of_list_to_csv,
    logger,
    pack_user_ass_to_openai_messages,
    split_string_by_multi_markers,
    truncate_list_by_token_size,
)
from .base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    CommunitySchema,
    QueryParam,
    SingleCommunitySchema,
    TextChunkSchema,
)
from .prompt import GRAPH_FIELD_SEP, PROMPTS


# ---------------------------------------------------------------------------
# Timestamp & Straddle-Safe Spoiler Filtering Helpers
# ---------------------------------------------------------------------------

def _format_mmss(seconds: float) -> str:
    """Format seconds as MM:SS (or HH:MM:SS if >= 3600s)."""
    seconds = max(0.0, float(seconds))
    total_sec = int(round(seconds))
    h = total_sec // 3600
    rem = total_sec % 3600
    m = rem // 60
    s = rem % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _format_mmss_precise(seconds: float) -> str:
    """Format seconds as MM:SS.ss (or HH:MM:SS.ss if >= 3600s)."""
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600)
    rem = seconds - h * 3600
    m = int(rem // 60)
    s = rem - m * 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:05.2f}"
    return f"{m:02d}:{s:05.2f}"


def _parse_mmss_to_sec(mmss: Union[str, float, int, None]) -> Optional[float]:
    """Parse MM:SS, MM:SS.ss, HH:MM:SS.ss, or numeric seconds into float seconds."""
    if mmss is None:
        return None
    if isinstance(mmss, (int, float)):
        return float(mmss)
    raw = str(mmss).strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        pass
    parts = raw.split(":")
    try:
        if len(parts) == 2:
            return float(parts[0]) * 60.0 + float(parts[1])
        if len(parts) == 3:
            return float(parts[0]) * 3600.0 + float(parts[1]) * 60.0 + float(parts[2])
    except ValueError:
        return None
    return None


def is_chunk_allowed(
    chunk_or_meta: Optional[dict],
    T_user: Optional[float],
    spoiler_free: bool = True,
) -> bool:
    """Straddle-Safe Temporal Cutoff Rule across the 3 resolution tiers:
    - tier == 'scene': allow if start_time <= T_user
    - tier == 'chapter': allow if end_time <= T_user
    - tier == 'full_movie': allow ONLY when spoiler_free == False
    """
    if chunk_or_meta is None:
        return False
    tier = chunk_or_meta.get("tier", "scene")
    if not spoiler_free:
        return True
    if tier == "full_movie":
        return False
    if T_user is None:
        return True
    t_cut = float(T_user)
    start_t = float(chunk_or_meta.get("start_time", 0.0))
    end_t = float(chunk_or_meta.get("end_time", 0.0))
    if tier == "chapter":
        return end_t <= t_cut
    return start_t <= t_cut


def is_temporal_slice_allowed(
    slice_dp: Optional[dict],
    T_user: Optional[float],
    spoiler_free: bool = True,
) -> bool:
    """Check if an entity/edge temporal_descriptions entry or entities_vdb vector slice is spoiler-safe."""
    if slice_dp is None:
        return False
    if not spoiler_free:
        return True
    tier = slice_dp.get("tier", "scene")
    if tier == "full_movie":
        return False
    if T_user is None:
        return True
    t_cut = float(T_user)
    start_t = float(slice_dp.get("start_time", 0.0))
    end_t = float(slice_dp.get("end_time", start_t))
    if tier == "chapter":
        return end_t <= t_cut
    return start_t <= t_cut


def filter_glossary_text_for_timestamp(
    glossary_desc: str,
    T_user: Optional[float],
    spoiler_free: bool = True,
) -> str:
    """Strip future-dated timeline clauses (e.g., '(08:15 - 12:25)') from a glossary bio when spoiler_free=True."""
    if not glossary_desc or not spoiler_free or T_user is None:
        return glossary_desc
    t_cut = float(T_user)
    range_pattern = re.compile(r"(\d{1,2}:\d{2}(?:\.\d+)?)\s*-\s*(\d{1,2}:\d{2}(?:\.\d+)?)")

    sentences = re.split(r"(?<=[.!?])\s+", glossary_desc.strip())
    kept_sentences: List[str] = []
    for sent in sentences:
        if ";" in sent and range_pattern.search(sent):
            prefix = ""
            body = sent
            if ":" in sent:
                first_colon = sent.find(":")
                first_match = range_pattern.search(sent)
                if first_match and first_colon < first_match.start():
                    prefix = sent[: first_colon + 1].strip() + " "
                    body = sent[first_colon + 1 :].strip()
            clauses = [c.strip().rstrip(".") for c in body.split(";") if c.strip()]
            kept_clauses = []
            for cl in clauses:
                matches = list(range_pattern.finditer(cl))
                if not matches:
                    kept_clauses.append(cl)
                else:
                    starts = [_parse_mmss_to_sec(m.group(1)) for m in matches]
                    valid_starts = [s for s in starts if s is not None]
                    if not valid_starts or min(valid_starts) <= t_cut:
                        kept_clauses.append(cl)
            if kept_clauses:
                kept_sentences.append(prefix + "; ".join(kept_clauses) + ".")
        else:
            matches = list(range_pattern.finditer(sent))
            if not matches:
                kept_sentences.append(sent)
            else:
                starts = [_parse_mmss_to_sec(m.group(1)) for m in matches]
                valid_starts = [s for s in starts if s is not None]
                if not valid_starts or min(valid_starts) <= t_cut:
                    kept_sentences.append(sent)
    return " ".join(kept_sentences).strip()


def _clip_vlm_caption_at_timestamp(
    vlm_caption: str,
    T_user: Optional[float],
    spoiler_free: bool = True,
) -> str:
    """When a scene straddles T_user, clip any micro-beat paragraph [MM:SS.ss - MM:SS.ss] that starts after T_user."""
    if not vlm_caption or not spoiler_free or T_user is None:
        return vlm_caption
    t_cut = float(T_user)
    beat_pattern = re.compile(
        r"^\[(\d{1,2}:\d{2}(?::\d{2})?(?:\.\d+)?)\s*-\s*(\d{1,2}:\d{2}(?::\d{2})?(?:\.\d+)?)\]\s*(.*)$",
        re.DOTALL,
    )
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", vlm_caption.strip()) if p.strip()]
    kept_paragraphs: List[str] = []
    any_beat_matched = False
    for para in paragraphs:
        m = beat_pattern.match(para)
        if not m:
            kept_paragraphs.append(para)
            continue
        any_beat_matched = True
        b_start = _parse_mmss_to_sec(m.group(1))
        if b_start is None or b_start <= t_cut:
            kept_paragraphs.append(para)
    if any_beat_matched and not kept_paragraphs and paragraphs:
        # Always keep at least the first beat of a straddling scene where scene.start_time <= T_user
        kept_paragraphs.append(paragraphs[0])
    return "\n\n".join(kept_paragraphs)


def format_scene_segment_for_timestamp(
    scene_id: str,
    seg: dict,
    T_user: Optional[float] = None,
    spoiler_free: bool = True,
    source_label: Optional[str] = None,
) -> str:
    """Format a structured scene record from kv_store_video_segments.json with straddle-safe clipping."""
    start_t = float(seg.get("start_time", 0.0))
    end_t = float(seg.get("end_time", 0.0))
    is_straddling = (
        spoiler_free and T_user is not None and start_t <= float(T_user) < end_t
    )
    chars = seg.get("characters", [])
    chars_str = ", ".join(chars) if chars else "None"
    src_part = f" | Source: {source_label}" if source_label else ""

    header = (
        f"[SCENE {scene_id} | {_format_mmss_precise(start_t)} - {_format_mmss_precise(end_t)} | "
        f"Characters: {chars_str}{src_part}]"
    )
    raw_caption = seg.get("vlm_caption", "").strip()
    if is_straddling:
        visual_desc = _clip_vlm_caption_at_timestamp(raw_caption, T_user, spoiler_free=True)
    else:
        visual_desc = raw_caption

    subtitles = seg.get("subtitles", []) or []
    if is_straddling:
        t_cut = float(T_user)
        subtitles = [
            u for u in subtitles if float(u.get("start_time", 0.0)) <= t_cut
        ]
        sub_header = f"[Subtitles / Dialogue (clipped at {_format_mmss(t_cut)})]:"
    else:
        sub_header = "[Subtitles / Dialogue]:"

    if subtitles:
        sub_lines = []
        for u in subtitles:
            u_st = _format_mmss_precise(float(u.get("start_time", start_t)))
            u_et = _format_mmss_precise(float(u.get("end_time", end_t)))
            speaker = u.get("speaker") or "Unknown"
            text = (u.get("text") or "").strip()
            if text:
                sub_lines.append(f'- [{u_st} - {u_et}] {speaker}: "{text}"')
        sub_block = "\n".join(sub_lines) if sub_lines else "- (No dialogue)"
    else:
        sub_block = "- (No dialogue)"

    return (
        f"{header}\n"
        f"[Visual Description]:\n{visual_desc}\n\n"
        f"{sub_header}\n{sub_block}"
    )


# ---------------------------------------------------------------------------
# Legacy & Multi-Resolution Text Chunking
# ---------------------------------------------------------------------------

def chunking_by_token_size(
    tokens_list: list[list[int]],
    doc_keys,
    tiktoken_model,
    overlap_token_size=128,
    max_token_size=1024,
):
    results = []
    for index, tokens in enumerate(tokens_list):
        chunk_token = []
        lengths = []
        for start in range(0, len(tokens), max_token_size - overlap_token_size):
            chunk_token.append(tokens[start : start + max_token_size])
            lengths.append(min(max_token_size, len(tokens) - start))

        chunk_token = tiktoken_model.decode_batch(chunk_token)
        for i, chunk in enumerate(chunk_token):
            results.append(
                {
                    "tokens": lengths[i],
                    "content": chunk.strip(),
                    "chunk_order_index": i,
                    "full_doc_id": doc_keys[index],
                }
            )
    return results


def chunking_by_video_segments(
    tokens_list: list[list[int]],
    doc_keys,
    tiktoken_model,
    max_token_size=2048,
):
    """1 Scene = 1 Chunk when max_token_size=2048 and each scene is processed individually,
    or legacy segment packing for raw video segments.
    """
    for index in range(len(tokens_list)):
        if len(tokens_list[index]) > max_token_size:
            tokens_list[index] = tokens_list[index][:max_token_size]

    results = []
    chunk_token = []
    chunk_segment_ids = []
    chunk_order_index = 0
    for index, tokens in enumerate(tokens_list):
        if len(chunk_token) + len(tokens) <= max_token_size:
            chunk_token += tokens.copy()
            chunk_segment_ids.append(doc_keys[index])
        else:
            chunk = tiktoken_model.decode(chunk_token)
            results.append(
                {
                    "tokens": len(chunk_token),
                    "content": chunk.strip(),
                    "chunk_order_index": chunk_order_index,
                    "video_segment_id": chunk_segment_ids,
                }
            )
            chunk_token = []
            chunk_segment_ids = []
            chunk_token += tokens.copy()
            chunk_segment_ids.append(doc_keys[index])
            chunk_order_index += 1

    if len(chunk_token) > 0:
        chunk = tiktoken_model.decode(chunk_token)
        results.append(
            {
                "tokens": len(chunk_token),
                "content": chunk.strip(),
                "chunk_order_index": chunk_order_index,
                "video_segment_id": chunk_segment_ids,
            }
        )

    return results


def chunking_by_seperators(
    tokens_list: list[list[int]],
    doc_keys,
    tiktoken_model,
    overlap_token_size=128,
    max_token_size=1024,
):
    splitter = SeparatorSplitter(
        separators=[
            tiktoken_model.encode(s) for s in PROMPTS["default_text_separator"]
        ],
        chunk_size=max_token_size,
        chunk_overlap=overlap_token_size,
    )
    results = []
    for index, tokens in enumerate(tokens_list):
        chunk_token = splitter.split_tokens(tokens)
        lengths = [len(c) for c in chunk_token]

        chunk_token = tiktoken_model.decode_batch(chunk_token)
        for i, chunk in enumerate(chunk_token):
            results.append(
                {
                    "tokens": lengths[i],
                    "content": chunk.strip(),
                    "chunk_order_index": i,
                    "full_doc_id": doc_keys[index],
                }
            )
    return results


def get_chunks(new_videos, chunk_func=chunking_by_video_segments, **chunk_func_params):
    inserting_chunks = {}
    encoder = get_tiktoken_encoder("gpt-4o")

    new_videos_list = list(new_videos.keys())
    for video_name in new_videos_list:
        segment_id_list = list(new_videos[video_name].keys())
        # Check if segments are structured Phase-B scene records with start_time & end_time
        first_seg = new_videos[video_name][segment_id_list[0]] if segment_id_list else {}
        if isinstance(first_seg, dict) and "start_time" in first_seg and "end_time" in first_seg:
            for order_idx, seg_id in enumerate(segment_id_list):
                seg = new_videos[video_name][seg_id]
                content = seg["content"].strip()
                tokens = encoder.encode(content)
                chunk_dict: TextChunkSchema = {
                    "tokens": len(tokens),
                    "content": content,
                    "video_segment_id": [seg_id],
                    "chunk_order_index": int(seg.get("scene_index", order_idx)),
                    "movie_id": video_name,
                    "start_time": float(seg["start_time"]),
                    "end_time": float(seg["end_time"]),
                    "tier": seg.get("tier", "scene"),
                    "characters": list(seg.get("characters", [])),
                }
                inserting_chunks[compute_mdhash_id(content, prefix="chunk-")] = chunk_dict
            continue

        docs = [new_videos[video_name][index]["content"] for index in segment_id_list]
        doc_keys = [f"{video_name}_{index}" for index in segment_id_list]

        tokens = encoder.encode_batch(docs, num_threads=16)
        chunks = chunk_func(
            tokens, doc_keys=doc_keys, tiktoken_model=encoder, **chunk_func_params
        )

        for chunk in chunks:
            inserting_chunks.update(
                {compute_mdhash_id(chunk["content"], prefix="chunk-"): chunk}
            )

    return inserting_chunks


# ---------------------------------------------------------------------------
# Time-Sliced Entity & Relationship Extraction & Graph Merging
# ---------------------------------------------------------------------------

async def _handle_entity_relation_summary(
    entity_or_relation_name: str,
    description: str,
    global_config: dict,
) -> str:
    use_llm_func: callable = global_config["llm"]["cheap_model_func"]
    llm_max_tokens = global_config["llm"]["cheap_model_max_token_size"]
    tiktoken_model_name = global_config["tiktoken_model_name"]
    summary_max_tokens = global_config["entity_summary_to_max_tokens"]

    tokens = encode_string_by_tiktoken(description, model_name=tiktoken_model_name)
    if len(tokens) < summary_max_tokens:
        return description
    prompt_template = PROMPTS["summarize_entity_descriptions"]
    use_description = decode_tokens_by_tiktoken(
        tokens[:llm_max_tokens], model_name=tiktoken_model_name
    )
    context_base = dict(
        entity_name=entity_or_relation_name,
        description_list=use_description.split(GRAPH_FIELD_SEP),
    )
    use_prompt = prompt_template.format(**context_base)
    logger.debug(f"Trigger summary: {entity_or_relation_name}")
    summary = await use_llm_func(use_prompt, max_tokens=summary_max_tokens)
    return summary


def _clean_extracted_field(raw: str) -> str:
    return clean_str(raw).strip().strip('"').strip("'").strip()


def _normalize_entity_name(raw: str) -> str:
    name = _clean_extracted_field(raw).upper()
    for article in ("THE ", "A ", "AN "):
        if name.startswith(article) and len(name) > len(article):
            name = name[len(article):].strip()
            break
    return name


async def _handle_single_entity_extraction(
    record_attributes: list[str],
    chunk_key: str,
    chunk_dp: Optional[dict] = None,
):
    if len(record_attributes) < 4 or _clean_extracted_field(record_attributes[0]).lower() != "entity":
        return None
    entity_name = _normalize_entity_name(record_attributes[1])
    if not entity_name:
        return None
    entity_type = _clean_extracted_field(record_attributes[2]).upper()
    entity_description = _clean_extracted_field(record_attributes[3])
    entity_source_id = chunk_key
    chunk_dp = chunk_dp or {}
    return dict(
        entity_name=entity_name,
        entity_type=entity_type,
        description=entity_description,
        source_id=entity_source_id,
        movie_id=str(chunk_dp.get("movie_id", "")),
        start_time=float(chunk_dp.get("start_time", 0.0)),
        end_time=float(chunk_dp.get("end_time", 0.0)),
        tier=str(chunk_dp.get("tier", "scene")),
    )


async def _handle_single_relationship_extraction(
    record_attributes: list[str],
    chunk_key: str,
    chunk_dp: Optional[dict] = None,
):
    if len(record_attributes) < 5 or _clean_extracted_field(record_attributes[0]).lower() != "relationship":
        return None
    source = _normalize_entity_name(record_attributes[1])
    target = _normalize_entity_name(record_attributes[2])
    if not source or not target:
        return None
    edge_description = _clean_extracted_field(record_attributes[3])
    edge_source_id = chunk_key
    raw_weight = _clean_extracted_field(record_attributes[-1])
    weight = float(raw_weight) if is_float_regex(raw_weight) else 1.0
    chunk_dp = chunk_dp or {}
    return dict(
        src_id=source,
        tgt_id=target,
        weight=weight,
        description=edge_description,
        source_id=edge_source_id,
        movie_id=str(chunk_dp.get("movie_id", "")),
        start_time=float(chunk_dp.get("start_time", 0.0)),
        end_time=float(chunk_dp.get("end_time", 0.0)),
        tier=str(chunk_dp.get("tier", "scene")),
    )


def _merge_temporal_slices(
    existing_json_str: Optional[str],
    new_records: list[dict],
) -> Tuple[List[Dict[str, Any]], float]:
    """Merge existing JSON-encoded temporal_descriptions with new chunk extraction records.

    Returns `(sorted_temporal_descriptions, first_seen_time)`.
    """
    merged_map: Dict[Tuple[str, str], Dict[str, Any]] = {}
    if existing_json_str:
        try:
            existing_list = json.loads(existing_json_str)
            if isinstance(existing_list, list):
                for item in existing_list:
                    if isinstance(item, dict) and item.get("description"):
                        key = (str(item.get("chunk_id", "")), str(item["description"]))
                        merged_map[key] = {
                            "chunk_id": str(item.get("chunk_id", "")),
                            "start_time": float(item.get("start_time", 0.0)),
                            "end_time": float(item.get("end_time", 0.0)),
                            "tier": str(item.get("tier", "scene")),
                            "description": str(item["description"]),
                        }
        except Exception:
            pass

    for dp in new_records:
        desc = str(dp.get("description", "")).strip()
        if not desc:
            continue
        cid = str(dp.get("source_id", dp.get("chunk_id", "")))
        key = (cid, desc)
        merged_map[key] = {
            "chunk_id": cid,
            "start_time": float(dp.get("start_time", 0.0)),
            "end_time": float(dp.get("end_time", 0.0)),
            "tier": str(dp.get("tier", "scene")),
            "description": desc,
        }

    temporal_list = sorted(
        merged_map.values(),
        key=lambda d: (
            0 if d["tier"] == "scene" else (1 if d["tier"] == "chapter" else 2),
            float(d["start_time"]),
            float(d["end_time"]),
        ),
    )

    scene_starts = [
        float(d["start_time"])
        for d in temporal_list
        if d.get("tier") == "scene"
    ]
    chapter_starts = [
        float(d["start_time"])
        for d in temporal_list
        if d.get("tier") == "chapter"
    ]
    if scene_starts:
        first_seen_time = min(scene_starts)
    elif chapter_starts:
        first_seen_time = min(chapter_starts)
    elif temporal_list:
        # Entity appears ONLY in full_movie plot document -> set first_seen_time to movie end
        first_seen_time = max(float(d.get("end_time", 0.0)) for d in temporal_list)
    else:
        first_seen_time = 0.0

    return temporal_list, float(first_seen_time)


async def _merge_nodes_then_upsert(
    entity_name: str,
    nodes_data: list[dict],
    knowledge_graph_inst: BaseGraphStorage,
    global_config: dict,
):
    already_entity_types = []
    already_source_ids = []
    already_description = []
    existing_temporal_json = None

    already_node = await knowledge_graph_inst.get_node(entity_name)
    if already_node is not None:
        already_entity_types.append(already_node["entity_type"])
        already_source_ids.extend(
            split_string_by_multi_markers(already_node["source_id"], [GRAPH_FIELD_SEP])
        )
        already_description.append(already_node["description"])
        existing_temporal_json = already_node.get("temporal_descriptions")

    entity_type = sorted(
        Counter([dp["entity_type"] for dp in nodes_data] + already_entity_types).items(),
        key=lambda x: x[1],
        reverse=True,
    )[0][0]
    description = GRAPH_FIELD_SEP.join(
        sorted(set([dp["description"] for dp in nodes_data] + already_description))
    )
    source_id = GRAPH_FIELD_SEP.join(
        set([dp["source_id"] for dp in nodes_data] + already_source_ids)
    )
    temporal_list, first_seen_time = _merge_temporal_slices(
        existing_temporal_json, nodes_data
    )
    if already_node is not None and "first_seen_time" in already_node:
        try:
            prev_fst = float(already_node["first_seen_time"])
            if prev_fst > 0.0 and prev_fst < first_seen_time:
                first_seen_time = prev_fst
        except (TypeError, ValueError):
            pass

    description = await _handle_entity_relation_summary(
        entity_name, description, global_config
    )
    temporal_json = json.dumps(temporal_list, ensure_ascii=False)
    node_data = dict(
        entity_type=entity_type,
        description=description,
        source_id=source_id,
        first_seen_time=float(first_seen_time),
        temporal_descriptions=temporal_json,
    )
    await knowledge_graph_inst.upsert_node(
        entity_name,
        node_data=node_data,
    )
    movie_id = next((dp.get("movie_id") for dp in nodes_data if dp.get("movie_id")), "")
    node_data["entity_name"] = entity_name
    node_data["movie_id"] = movie_id
    node_data["temporal_descriptions_list"] = temporal_list
    return node_data


async def _merge_edges_then_upsert(
    src_id: str,
    tgt_id: str,
    edges_data: list[dict],
    knowledge_graph_inst: BaseGraphStorage,
    global_config: dict,
):
    already_weights = []
    already_source_ids = []
    already_description = []
    already_order = []
    existing_temporal_json = None

    if await knowledge_graph_inst.has_edge(src_id, tgt_id):
        already_edge = await knowledge_graph_inst.get_edge(src_id, tgt_id)
        already_weights.append(float(already_edge.get("weight", 1.0)))
        already_source_ids.extend(
            split_string_by_multi_markers(already_edge["source_id"], [GRAPH_FIELD_SEP])
        )
        already_description.append(already_edge["description"])
        already_order.append(already_edge.get("order", 1))
        existing_temporal_json = already_edge.get("temporal_descriptions")

    order = min([dp.get("order", 1) for dp in edges_data] + already_order)
    weight = sum([dp["weight"] for dp in edges_data] + already_weights)
    description = GRAPH_FIELD_SEP.join(
        sorted(set([dp["description"] for dp in edges_data] + already_description))
    )
    source_id = GRAPH_FIELD_SEP.join(
        set([dp["source_id"] for dp in edges_data] + already_source_ids)
    )
    temporal_list, first_seen_time = _merge_temporal_slices(
        existing_temporal_json, edges_data
    )
    temporal_json = json.dumps(temporal_list, ensure_ascii=False)

    for need_insert_id in [src_id, tgt_id]:
        if not (await knowledge_graph_inst.has_node(need_insert_id)):
            await knowledge_graph_inst.upsert_node(
                need_insert_id,
                node_data={
                    "source_id": source_id,
                    "description": description,
                    "entity_type": '"UNKNOWN"',
                    "first_seen_time": float(first_seen_time),
                    "temporal_descriptions": temporal_json,
                },
            )
    description = await _handle_entity_relation_summary(
        (src_id, tgt_id), description, global_config
    )
    await knowledge_graph_inst.upsert_edge(
        src_id,
        tgt_id,
        edge_data=dict(
            weight=weight,
            description=description,
            source_id=source_id,
            order=order,
            first_seen_time=float(first_seen_time),
            temporal_descriptions=temporal_json,
        ),
    )
    return_edge_data = dict(
        src_tgt=(src_id, tgt_id),
        description=description,
        weight=weight,
        first_seen_time=float(first_seen_time),
        temporal_descriptions=temporal_json,
    )
    return return_edge_data


async def extract_entities(
    chunks: dict[str, TextChunkSchema],
    knowledge_graph_inst: BaseGraphStorage,
    entity_vdb: BaseVectorStorage,
    global_config: dict,
) -> Union[Tuple[BaseGraphStorage, list, list], Tuple[None, list, list]]:
    use_llm_func: callable = global_config["llm"]["best_model_func"]
    entity_extract_max_gleaning = global_config["entity_extract_max_gleaning"]

    ordered_chunks = list(chunks.items())

    entity_extract_prompt = PROMPTS["entity_extraction"]
    context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        entity_types=",".join(PROMPTS["DEFAULT_ENTITY_TYPES"]),
    )
    continue_prompt = PROMPTS["entiti_continue_extraction"]
    if_loop_prompt = PROMPTS["entiti_if_loop_extraction"]

    already_processed = 0
    already_entities = 0
    already_relations = 0

    async def _process_single_content(chunk_key_dp: tuple[str, TextChunkSchema]):
        nonlocal already_processed, already_entities, already_relations
        chunk_key = chunk_key_dp[0]
        chunk_dp = chunk_key_dp[1]
        content = chunk_dp["content"]
        hint_prompt = entity_extract_prompt.format(**context_base, input_text=content)
        final_result = await use_llm_func(hint_prompt)

        history = pack_user_ass_to_openai_messages(hint_prompt, final_result)
        for now_glean_index in range(entity_extract_max_gleaning):
            glean_result = await use_llm_func(continue_prompt, history_messages=history)
            history += pack_user_ass_to_openai_messages(continue_prompt, glean_result)
            final_result += glean_result
            if now_glean_index == entity_extract_max_gleaning - 1:
                break

            if_loop_result: str = await use_llm_func(
                if_loop_prompt, history_messages=history
            )
            if_loop_result = if_loop_result.strip().strip('"').strip("'").lower()
            if if_loop_result != "yes":
                break

        records = split_string_by_multi_markers(
            final_result,
            [context_base["record_delimiter"], context_base["completion_delimiter"]],
        )

        maybe_nodes = defaultdict(list)
        maybe_edges = defaultdict(list)
        for record in records:
            record = re.search(r"\((.*)\)", record)
            if record is None:
                continue
            record = record.group(1)
            record_attributes = split_string_by_multi_markers(
                record, [context_base["tuple_delimiter"]]
            )
            if_entities = await _handle_single_entity_extraction(
                record_attributes, chunk_key, chunk_dp=chunk_dp
            )
            if if_entities is not None:
                maybe_nodes[if_entities["entity_name"]].append(if_entities)
                continue

            if_relation = await _handle_single_relationship_extraction(
                record_attributes, chunk_key, chunk_dp=chunk_dp
            )
            if if_relation is not None:
                maybe_edges[(if_relation["src_id"], if_relation["tgt_id"])].append(
                    if_relation
                )

        already_processed += 1
        already_entities += len(maybe_nodes)
        already_relations += len(maybe_edges)
        now_ticks = PROMPTS["process_tickers"][
            already_processed % len(PROMPTS["process_tickers"])
        ]
        print(
            f"{now_ticks} Processed {already_processed}/{len(ordered_chunks)} chunks, "
            f"{already_entities} entities(duplicated), {already_relations} relations(duplicated)\r",
            end="",
            flush=True,
        )
        return dict(maybe_nodes), dict(maybe_edges)

    results = await asyncio.gather(
        *[_process_single_content(c) for c in ordered_chunks]
    )
    print()
    maybe_nodes = defaultdict(list)
    maybe_edges = defaultdict(list)
    for m_nodes, m_edges in results:
        for k, v in m_nodes.items():
            maybe_nodes[k].extend(v)
        for k, v in m_edges.items():
            maybe_edges[tuple(sorted(k))].extend(v)

    all_entities_data = await asyncio.gather(
        *[
            _merge_nodes_then_upsert(k, v, knowledge_graph_inst, global_config)
            for k, v in maybe_nodes.items()
        ]
    )
    all_edges_data = await asyncio.gather(
        *[
            _merge_edges_then_upsert(k[0], k[1], v, knowledge_graph_inst, global_config)
            for k, v in maybe_edges.items()
        ]
    )
    if not len(all_entities_data):
        logger.warning("Didn't extract any entities, maybe your LLM is not working")
        return None, [], []

    if entity_vdb is not None:
        addon_params = global_config.get("addon_params") or {}
        default_movie_id = addon_params.get("movie_id", "")
        character_glossary: Dict[str, str] = addon_params.get("character_glossary") or {}
        glossary_upper = {k.upper().strip(): v for k, v in character_glossary.items()}

        data_for_vdb: Dict[str, Dict[str, Any]] = {}
        for dp in all_entities_data:
            ent_name = dp["entity_name"]
            movie_id = dp.get("movie_id") or default_movie_id
            first_seen = float(dp.get("first_seen_time", 0.0))
            t_slices = dp.get("temporal_descriptions_list") or []

            # 1. Base Glossary Slice (if entity is in character_glossary)
            if ent_name in glossary_upper:
                base_id = compute_mdhash_id(f"{ent_name}_base", prefix="ent-")
                data_for_vdb[base_id] = {
                    "content": f"{ent_name}: {glossary_upper[ent_name]}",
                    "entity_name": ent_name,
                    "movie_id": movie_id,
                    "start_time": first_seen,
                    "end_time": first_seen,
                    "tier": "scene",
                }

            # 2. Per-Scene / Per-Chapter / Full-Movie Temporal Slices
            if t_slices:
                for d in t_slices:
                    slice_id = compute_mdhash_id(
                        f"{ent_name}_{d['chunk_id']}_{d['description'][:32]}",
                        prefix="ent-",
                    )
                    data_for_vdb[slice_id] = {
                        "content": f"{ent_name}: {d['description']}",
                        "entity_name": ent_name,
                        "movie_id": movie_id,
                        "start_time": float(d.get("start_time", first_seen)),
                        "end_time": float(d.get("end_time", first_seen)),
                        "tier": str(d.get("tier", "scene")),
                    }
            else:
                fallback_id = compute_mdhash_id(ent_name, prefix="ent-")
                data_for_vdb[fallback_id] = {
                    "content": f"{ent_name}: {dp['description']}",
                    "entity_name": ent_name,
                    "movie_id": movie_id,
                    "start_time": first_seen,
                    "end_time": first_seen,
                    "tier": "scene",
                }

        await entity_vdb.upsert(data_for_vdb)

    return knowledge_graph_inst, all_entities_data, all_edges_data


# ---------------------------------------------------------------------------
# Spoiler-Filtered Graph Traversal & Turn-0 Context Construction
# ---------------------------------------------------------------------------

async def _find_most_related_segments_from_entities(
    topk_chunks: int,
    node_datas: list[dict],
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    knowledge_graph_inst: BaseGraphStorage,
    T_user: Optional[float] = None,
    spoiler_free: bool = True,
) -> Set[str]:
    """Rank 1-hop co-occurring chunks from matched entities while strictly filtering out future chunks."""
    if not node_datas:
        return set()

    text_units = [
        split_string_by_multi_markers(dp.get("source_id", ""), [GRAPH_FIELD_SEP])
        for dp in node_datas
    ]
    edges = await asyncio.gather(
        *[knowledge_graph_inst.get_node_edges(dp["entity_name"]) for dp in node_datas]
    )
    all_one_hop_nodes = set()
    for this_edges in edges:
        if not this_edges:
            continue
        all_one_hop_nodes.update([e[1] for e in this_edges])
    all_one_hop_nodes = list(all_one_hop_nodes)
    all_one_hop_nodes_data = await asyncio.gather(
        *[knowledge_graph_inst.get_node(e) for e in all_one_hop_nodes]
    )

    # Filter 1-hop neighbors by first_seen_time <= T_user when spoiler_free=True
    all_one_hop_text_units_lookup = {}
    for k, v in zip(all_one_hop_nodes, all_one_hop_nodes_data):
        if v is None:
            continue
        if spoiler_free and T_user is not None:
            if float(v.get("first_seen_time", 0.0)) > float(T_user):
                continue
        all_one_hop_text_units_lookup[k] = set(
            split_string_by_multi_markers(v.get("source_id", ""), [GRAPH_FIELD_SEP])
        )

    all_text_units_lookup = {}
    for index, (this_text_units, this_edges) in enumerate(zip(text_units, edges)):
        for c_id in this_text_units:
            if c_id in all_text_units_lookup:
                continue
            chunk_dp = await text_chunks_db.get_by_id(c_id)
            if (
                chunk_dp is None
                or chunk_dp.get("tier", "scene") != "scene"
                or not is_chunk_allowed(chunk_dp, T_user, spoiler_free)
            ):
                continue
            relation_counts = 0
            if this_edges:
                for e in this_edges:
                    if (
                        e[1] in all_one_hop_text_units_lookup
                        and c_id in all_one_hop_text_units_lookup[e[1]]
                    ):
                        relation_counts += 1
            all_text_units_lookup[c_id] = {
                "data": chunk_dp,
                "order": index,
                "relation_counts": relation_counts,
            }

    all_text_units = [
        {"id": k, **v} for k, v in all_text_units_lookup.items() if v is not None and v.get("data") is not None
    ]
    sorted_text_units = sorted(
        all_text_units, key=lambda x: (-x["relation_counts"], x["order"])
    )[:topk_chunks]

    chunk_related_segments = set()
    for _chunk_data in sorted_text_units:
        seg_ids = _chunk_data["data"].get("video_segment_id", [])
        if isinstance(seg_ids, str):
            seg_ids = [seg_ids]
        for s_id in seg_ids:
            chunk_related_segments.add(s_id)

    return chunk_related_segments


async def _reconstruct_graph_entities_and_relations_context(
    node_datas: list[dict],
    knowledge_graph_inst: BaseGraphStorage,
    T_user: Optional[float],
    spoiler_free: bool,
    character_glossary: Optional[Dict[str, str]] = None,
    max_entities: int = 8,
    max_relations: int = 10,
    max_slices_per_entity: Optional[int] = 6,
) -> str:
    """Reconstruct spoiler-filtered Entity & 1-Hop Relationship Histories from temporal_descriptions."""
    if not node_datas:
        return "  (No matching graph entities up to current timestamp)"

    glossary_upper = {
        k.upper().strip(): v for k, v in (character_glossary or {}).items()
    }
    lines: List[str] = []
    seen_entities: Set[str] = set()
    kept_nodes: List[dict] = []

    for node in node_datas:
        ent_name = node.get("entity_name", "")
        if not ent_name or ent_name in seen_entities:
            continue
        fst = float(node.get("first_seen_time", 0.0))
        if spoiler_free and T_user is not None and fst > float(T_user):
            continue
        seen_entities.add(ent_name)
        kept_nodes.append(node)
        if len(kept_nodes) >= max_entities:
            break

    if not kept_nodes:
        return "  (No matching graph entities introduced up to current timestamp)"

    kept_node_names: Set[str] = {n["entity_name"] for n in kept_nodes}

    for node in kept_nodes:
        ent_name = node["entity_name"]
        fst = float(node.get("first_seen_time", 0.0))
        ent_lines: List[str] = []

        if ent_name in glossary_upper:
            safe_g = filter_glossary_text_for_timestamp(
                glossary_upper[ent_name], T_user, spoiler_free
            )
            if safe_g:
                ent_lines.append(f"    - [Glossary]: {safe_g}")

        raw_temporal = node.get("temporal_descriptions")
        t_slices: List[dict] = []
        if raw_temporal:
            try:
                t_slices = json.loads(raw_temporal)
            except Exception:
                t_slices = []

        allowed_slices = [
            d for d in t_slices if is_temporal_slice_allowed(d, T_user, spoiler_free)
        ]
        if allowed_slices:
            if (
                max_slices_per_entity is not None
                and max_slices_per_entity > 0
                and len(allowed_slices) > max_slices_per_entity
            ):
                if max_slices_per_entity == 1:
                    selected_slices = [allowed_slices[0]]
                else:
                    selected_slices = [allowed_slices[0]] + allowed_slices[
                        -(max_slices_per_entity - 1) :
                    ]
            else:
                selected_slices = allowed_slices
            for d in selected_slices:
                st_str = _format_mmss(float(d.get("start_time", 0.0)))
                et_str = _format_mmss(float(d.get("end_time", 0.0)))
                tier_tag = f" ({d['tier']})" if d.get("tier") in ("chapter", "full_movie") else ""
                ent_lines.append(f"    - [{st_str} - {et_str}{tier_tag}]: {d['description']}")
        elif not spoiler_free and node.get("description"):
            ent_lines.append(f"    - [Summary]: {node['description']}")

        if ent_lines:
            lines.append(f"  Entity [{ent_name}] (first seen {_format_mmss(fst)}):")
            lines.extend(ent_lines)

    # 1-hop edges connected to kept_nodes: collect all spoiler-safe candidates, then prioritize & sort
    seen_edges: Set[Tuple[str, str]] = set()
    candidate_edges: List[Dict[str, Any]] = []
    for node in kept_nodes:
        ent_name = node["entity_name"]
        node_edges = await knowledge_graph_inst.get_node_edges(ent_name)
        if not node_edges:
            continue
        for u, v in node_edges:
            edge_key = tuple(sorted((u, v)))
            if edge_key in seen_edges:
                continue
            seen_edges.add(edge_key)

            other = v if u == ent_name else u
            other_node = await knowledge_graph_inst.get_node(other)
            if other_node is not None and spoiler_free and T_user is not None:
                if float(other_node.get("first_seen_time", 0.0)) > float(T_user):
                    continue

            edge_data = await knowledge_graph_inst.get_edge(u, v)
            if edge_data is None:
                continue
            edge_fst = float(edge_data.get("first_seen_time", 0.0))
            if spoiler_free and T_user is not None and edge_fst > float(T_user):
                continue

            raw_edge_temp = edge_data.get("temporal_descriptions")
            edge_slices: List[dict] = []
            if raw_edge_temp:
                try:
                    edge_slices = json.loads(raw_edge_temp)
                except Exception:
                    edge_slices = []
            allowed_edge_slices = [
                d for d in edge_slices if is_temporal_slice_allowed(d, T_user, spoiler_free)
            ]
            if not allowed_edge_slices and (spoiler_free or not edge_data.get("description")):
                continue

            both_in_kept = (
                1 if (edge_key[0] in kept_node_names and edge_key[1] in kept_node_names) else 0
            )
            edge_weight = float(edge_data.get("weight", 1.0))
            most_recent_end = max(
                (float(d.get("end_time", 0.0)) for d in allowed_edge_slices),
                default=edge_fst,
            )
            candidate_edges.append(
                {
                    "edge_key": edge_key,
                    "both_in_kept": both_in_kept,
                    "weight": edge_weight,
                    "most_recent_end": most_recent_end,
                    "allowed_slices": allowed_edge_slices,
                    "description": edge_data.get("description", ""),
                }
            )

    candidate_edges.sort(
        key=lambda e: (-e["both_in_kept"], -e["weight"], -e["most_recent_end"])
    )

    for cand in candidate_edges[:max_relations]:
        edge_key = cand["edge_key"]
        allowed_edge_slices = cand["allowed_slices"]
        lines.append(f"  Relation [{edge_key[0]} <-> {edge_key[1]}]:")
        if allowed_edge_slices:
            for d in allowed_edge_slices[:4]:
                st_str = _format_mmss(float(d.get("start_time", 0.0)))
                et_str = _format_mmss(float(d.get("end_time", 0.0)))
                lines.append(f"    - [{st_str} - {et_str}]: {d['description']}")
        else:
            lines.append(f"    - [Summary]: {cand['description']}")

    return "\n".join(lines) if lines else "  (No spoiler-safe entity history available up to current timestamp)"


async def _resolve_query_with_chat_history(
    query: str,
    query_param: QueryParam,
    global_config: dict,
) -> str:
    """Resolve pronouns and deictics in follow-up questions against the last 6 messages (3 turns) of chat_history."""
    history = (query_param.chat_history or [])[-6:]
    if not history:
        return query
    use_llm_func: callable = global_config["llm"]["cheap_model_func"]
    history_lines = []
    for msg in history:
        role = (msg.get("role") or "user").upper()
        content = (msg.get("content") or "").strip()
        if content:
            history_lines.append(f"{role}: {content}")
    if not history_lines:
        return query

    prompt = PROMPTS["query_coref_resolution"].format(
        chat_history_text="\n".join(history_lines),
        input_text=query,
    )
    try:
        rewritten = await use_llm_func(prompt)
        rewritten = (rewritten or "").strip().strip('"')
        return rewritten if rewritten else query
    except Exception as exc:
        logger.warning(f"Coreference resolution fallback to raw query ({exc})")
        return query


async def _refine_entity_retrieval_query(
    query: str,
    query_param: QueryParam,
    global_config: dict,
) -> str:
    use_llm_func: callable = global_config["llm"]["cheap_model_func"]
    query_rewrite_prompt = PROMPTS["query_rewrite_for_entity_retrieval"].format(
        input_text=query
    )
    try:
        final_result = await use_llm_func(query_rewrite_prompt)
        return (final_result or query).strip()
    except Exception:
        return query


async def _refine_visual_retrieval_query(
    query,
    query_param: QueryParam,
    global_config: dict,
):
    use_llm_func: callable = global_config["llm"]["cheap_model_func"]
    query_rewrite_prompt = PROMPTS["query_rewrite_for_visual_retrieval"]
    query_rewrite_prompt = query_rewrite_prompt.format(input_text=query)
    final_result = await use_llm_func(query_rewrite_prompt)
    return final_result


async def _extract_keywords_query(
    query,
    query_param: QueryParam,
    global_config: dict,
):
    use_llm_func: callable = global_config["llm"]["cheap_model_func"]
    keywords_prompt = PROMPTS["keywords_extraction"]
    keywords_prompt = keywords_prompt.format(input_text=query)
    final_result = await use_llm_func(keywords_prompt)
    return final_result


def _get_all_scenes_flat(video_segments: BaseKVStorage) -> Tuple[str, Dict[str, dict], List[Tuple[str, dict]]]:
    """Return `(movie_id, scene_map, chronological_scenes_list)` from `video_segments._data`."""
    data = getattr(video_segments, "_data", {}) or {}
    if not data:
        return "unknown_movie", {}, []
    movie_id = next(iter(data.keys()))
    raw_scenes = data.get(movie_id, {}) or {}
    scene_map: Dict[str, dict] = {}
    for s_id, seg in raw_scenes.items():
        if isinstance(seg, dict):
            scene_map[s_id] = seg
    ordered = sorted(
        scene_map.items(),
        key=lambda kv: (float(kv[1].get("start_time", 0.0)), int(kv[1].get("scene_index", 0))),
    )
    return movie_id, scene_map, ordered


def _load_bundle_meta_from_workdir(global_config: dict) -> Dict[str, Any]:
    """Load movie_title, duration_sec, character_glossary from addon_params or working_dir/movie_knowledge_bundle.json."""
    addon = global_config.get("addon_params") or {}
    if addon.get("character_glossary") or addon.get("movie_title"):
        return addon
    wdir = global_config.get("working_dir", "")
    bundle_file = os.path.join(wdir, "movie_knowledge_bundle.json")
    if os.path.isfile(bundle_file):
        try:
            with open(bundle_file, "r", encoding="utf-8") as f:
                b = json.load(f)
            return {
                "movie_id": b.get("movie_id", ""),
                "movie_title": b.get("movie_title", ""),
                "duration_sec": b.get("duration_sec", 0.0),
                "character_glossary": b.get("character_glossary", {}),
            }
        except Exception:
            pass
    return addon


async def _build_turn0_context(
    query: str,
    resolved_query: str,
    entities_vdb: Optional[BaseVectorStorage],
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    chunks_vdb: Optional[BaseVectorStorage],
    video_segments: BaseKVStorage,
    knowledge_graph_inst: BaseGraphStorage,
    query_param: QueryParam,
    global_config: dict,
) -> Tuple[str, Dict[str, Any]]:
    """Construct the fresh Turn-0 Spoiler-Filtered GraphRAG XML context block."""
    T_user = query_param.current_timestamp
    spoiler_free = query_param.spoiler_free
    bundle_meta = _load_bundle_meta_from_workdir(global_config)
    movie_id, scene_map, ordered_scenes = _get_all_scenes_flat(video_segments)
    movie_title = bundle_meta.get("movie_title") or movie_id.replace("_", " ").title()
    character_glossary = bundle_meta.get("character_glossary") or {}

    # 1. Viewer State & Unlocked Chapters Table of Contents
    t_str = (
        f"{_format_mmss(T_user)} ({float(T_user):.1f}s)"
        if T_user is not None
        else "Full Timeline (No Cutoff)"
    )
    sf_str = "ON" if spoiler_free else "OFF"

    all_chunk_keys = await text_chunks_db.all_keys()
    all_chunks = await text_chunks_db.get_by_ids(all_chunk_keys)
    chapter_chunks = [
        c
        for c in all_chunks
        if c is not None
        and c.get("tier") == "chapter"
        and is_chunk_allowed(c, T_user, spoiler_free)
    ]
    chapter_chunks.sort(key=lambda c: float(c.get("start_time", 0.0)))

    toc_lines = []
    for idx, ch in enumerate(chapter_chunks, start=1):
        first_line = (ch.get("content") or "").splitlines()[0].strip()
        st_m = _format_mmss(float(ch.get("start_time", 0.0)))
        et_m = _format_mmss(float(ch.get("end_time", 0.0)))
        if first_line.lower().startswith("chapter"):
            toc_lines.append(f"  - {first_line}")
        else:
            toc_lines.append(f"  - Chapter {idx} [{st_m} - {et_m}]: {first_line[:100]}")
    toc_block = "\n".join(toc_lines) if toc_lines else "  - (No completed chapters unlocked yet at this timestamp)"

    viewer_state_xml = (
        f"<viewer_state>\n"
        f"  Movie: {movie_title} | Current Watch Timestamp (T_user): {t_str} | Spoiler-Free Mode: {sf_str}\n"
        f"  Unlocked Chapters up to T_user:\n"
        f"{toc_block}\n"
        f"</viewer_state>"
    )

    # 2. Active Scene at T_user & Immediately Preceding Scene
    active_scene_id: Optional[str] = None
    active_scene_seg: Optional[dict] = None
    active_idx: Optional[int] = None
    if T_user is not None and ordered_scenes:
        t_val = float(T_user)
        for idx, (s_id, seg) in enumerate(ordered_scenes):
            st = float(seg.get("start_time", 0.0))
            et = float(seg.get("end_time", 0.0))
            if st <= t_val <= et:
                active_scene_id = s_id
                active_scene_seg = seg
                active_idx = idx
                break
        if active_scene_id is None:
            # Pick the most recent scene that started at or before T_user
            prior = [
                (idx, s_id, seg)
                for idx, (s_id, seg) in enumerate(ordered_scenes)
                if float(seg.get("start_time", 0.0)) <= t_val
            ]
            if prior:
                active_idx, active_scene_id, active_scene_seg = prior[-1]

    prev_scene_id: Optional[str] = None
    prev_scene_xml: Optional[str] = None
    if active_idx is not None and active_idx > 0:
        prev_scene_id, prev_scene_seg = ordered_scenes[active_idx - 1]
        prev_scene_text = format_scene_segment_for_timestamp(
            prev_scene_id, prev_scene_seg, T_user=T_user, spoiler_free=spoiler_free
        )
        prev_scene_xml = (
            f"<previous_scene_before_T_user>\n"
            f"{prev_scene_text}\n"
            f"</previous_scene_before_T_user>"
        )

    if active_scene_id and active_scene_seg:
        active_scene_text = format_scene_segment_for_timestamp(
            active_scene_id, active_scene_seg, T_user=T_user, spoiler_free=spoiler_free
        )
    else:
        active_scene_text = "  (No active scene at current timestamp)"

    current_scene_xml = (
        f"<current_scene_at_T_user>\n"
        f"{active_scene_text}\n"
        f"</current_scene_at_T_user>"
    )

    # 3. Direct Vector Search (chunks_vdb) with straddle-safe filter_lambda
    vector_chunk_ids: List[str] = []
    if chunks_vdb is not None:
        vec_results = await chunks_vdb.query(
            resolved_query,
            top_k=query_param.top_k,
            filter_lambda=lambda dp: is_chunk_allowed(dp, T_user, spoiler_free),
        )
        vector_chunk_ids = [r["id"] for r in vec_results]

    # 4. Time-Sliced Entity Search -> 1-Hop Graph Walk & Node/Edge History Reconstruction
    graph_segment_ids: Set[str] = set()
    graph_context_body = "  (Knowledge graph retrieval disabled or empty)"
    if entities_vdb is not None:
        entity_query = await _refine_entity_retrieval_query(
            resolved_query, query_param, global_config
        )
        raw_entity_hits = await entities_vdb.query(
            entity_query,
            top_k=query_param.top_k * 2,
            filter_lambda=lambda dp: is_temporal_slice_allowed(dp, T_user, spoiler_free),
        )
        # Deduplicate entity_name preserving direct query/active-scene matches + vector rank order
        dedup_entity_names: List[str] = []
        seen_names: Set[str] = set()
        q_upper = f"{query} {resolved_query}".upper()
        for cname in character_glossary.keys():
            c_up = cname.upper().strip()
            if c_up and c_up in q_upper and c_up not in seen_names:
                seen_names.add(c_up)
                dedup_entity_names.append(c_up)
        if active_scene_seg:
            for cname in active_scene_seg.get("characters", []):
                c_up = str(cname).upper().strip()
                if c_up and c_up not in seen_names:
                    seen_names.add(c_up)
                    dedup_entity_names.append(c_up)
        for r in raw_entity_hits:
            ename = r.get("entity_name")
            if ename and ename not in seen_names:
                seen_names.add(ename)
                dedup_entity_names.append(ename)

        if dedup_entity_names:
            fetched_nodes = await asyncio.gather(
                *[knowledge_graph_inst.get_node(ename) for ename in dedup_entity_names]
            )
            node_degrees = await asyncio.gather(
                *[knowledge_graph_inst.node_degree(ename) for ename in dedup_entity_names]
            )
            node_datas = []
            for ename, n, d in zip(dedup_entity_names, fetched_nodes, node_degrees):
                if n is None:
                    continue
                if spoiler_free and T_user is not None:
                    if float(n.get("first_seen_time", 0.0)) > float(T_user):
                        continue
                node_datas.append({**n, "entity_name": ename, "rank": d})

            if node_datas:
                graph_segment_ids = await _find_most_related_segments_from_entities(
                    global_config.get("retrieval_topk_chunks", 5),
                    node_datas,
                    text_chunks_db,
                    knowledge_graph_inst,
                    T_user=T_user,
                    spoiler_free=spoiler_free,
                )
                graph_context_body = await _reconstruct_graph_entities_and_relations_context(
                    node_datas,
                    knowledge_graph_inst,
                    T_user=T_user,
                    spoiler_free=spoiler_free,
                    character_glossary=character_glossary,
                )

    graph_xml = (
        f"<graph_entities_and_relations_up_to_T_user>\n"
        f"{graph_context_body}\n"
        f"</graph_entities_and_relations_up_to_T_user>"
    )

    # 5. Union of Direct Vector Chunks + 1-Hop Graph Co-Occurrence Scenes
    formatted_retrieved_blocks: List[str] = []
    included_scene_ids: Set[str] = set()
    if prev_scene_id:
        included_scene_ids.add(prev_scene_id)
    vector_chunks = await text_chunks_db.get_by_ids(vector_chunk_ids)

    for c_id, c_dp in zip(vector_chunk_ids, vector_chunks):
        if c_dp is None or not is_chunk_allowed(c_dp, T_user, spoiler_free):
            continue
        tier = c_dp.get("tier", "scene")
        seg_ids = c_dp.get("video_segment_id", [])
        if isinstance(seg_ids, str):
            seg_ids = [seg_ids]

        if tier == "scene" and len(seg_ids) == 1 and seg_ids[0] in scene_map:
            s_id = seg_ids[0]
            if s_id in included_scene_ids:
                continue
            included_scene_ids.add(s_id)
            src_label = "Graph+Vector" if s_id in graph_segment_ids else "Vector"
            formatted_retrieved_blocks.append(
                format_scene_segment_for_timestamp(
                    s_id,
                    scene_map[s_id],
                    T_user=T_user,
                    spoiler_free=spoiler_free,
                    source_label=src_label,
                )
            )
        else:
            st_m = _format_mmss(float(c_dp.get("start_time", 0.0)))
            et_m = _format_mmss(float(c_dp.get("end_time", 0.0)))
            tier_upper = tier.upper()
            formatted_retrieved_blocks.append(
                f"[{tier_upper} {c_id} | {st_m} - {et_m} | Source: Vector]:\n{c_dp['content']}"
            )

    # Add remaining Graph-retrieved scene segments not already in vector chunks
    for s_id in sorted(graph_segment_ids):
        if s_id in included_scene_ids or s_id not in scene_map:
            continue
        seg = scene_map[s_id]
        if not is_chunk_allowed(
            {"tier": "scene", "start_time": seg.get("start_time", 0.0), "end_time": seg.get("end_time", 0.0)},
            T_user,
            spoiler_free,
        ):
            continue
        included_scene_ids.add(s_id)
        formatted_retrieved_blocks.append(
            format_scene_segment_for_timestamp(
                s_id,
                seg,
                T_user=T_user,
                spoiler_free=spoiler_free,
                source_label="Graph",
            )
        )

    # Token-budget truncation on retrieved blocks
    truncated_blocks = truncate_list_by_token_size(
        formatted_retrieved_blocks,
        key=lambda x: x,
        max_token_size=query_param.naive_max_token_for_text_unit,
    )
    retrieved_body = (
        "\n\n".join(truncated_blocks)
        if truncated_blocks
        else "  (No scene or chapter chunks matched up to current timestamp)"
    )
    retrieved_xml = (
        f"<retrieved_scene_and_chapter_chunks_up_to_T_user>\n"
        f"{retrieved_body}\n"
        f"</retrieved_scene_and_chapter_chunks_up_to_T_user>"
    )

    turn0_blocks = [viewer_state_xml]
    if prev_scene_xml:
        turn0_blocks.append(prev_scene_xml)
    turn0_blocks.extend([current_scene_xml, graph_xml, retrieved_xml])
    turn0_context = "\n\n".join(turn0_blocks)
    meta_info = {
        "movie_id": movie_id,
        "movie_title": movie_title,
        "character_glossary": character_glossary,
        "prev_scene_id": prev_scene_id,
        "active_scene_id": active_scene_id,
        "included_scene_ids": list(included_scene_ids),
    }
    return turn0_context, meta_info


# ---------------------------------------------------------------------------
# 4 Spoiler-Gated Agent Tools (Bound with Server-Side T_user & spoiler_free)
# ---------------------------------------------------------------------------

AGENT_TOOL_SCHEMAS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_scenes",
            "description": (
                "Semantic search over Tier-1 scene and Tier-2 chapter chunks within an optional "
                "[start_time, end_time] window (in seconds)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural-language search query describing the scene, dialogue, or event.",
                    },
                    "start_time": {
                        "type": "number",
                        "description": "Optional lower bound timestamp in seconds (default 0.0).",
                    },
                    "end_time": {
                        "type": "number",
                        "description": "Optional upper bound timestamp in seconds.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_character_history",
            "description": (
                "Fetch a character or entity's glossary entry, chronological actions, and 1-hop "
                "relationships from the Knowledge Graph up to the viewer's current timestamp."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "character_name": {
                        "type": "string",
                        "description": "Name of the character or entity (e.g. 'Sintel', 'Scales', 'Shaman').",
                    }
                },
                "required": ["character_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_adjacent_scenes",
            "description": (
                "Fetch the scenes immediately preceding and succeeding a given scene_id along the movie timeline."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "scene_id": {
                        "type": "string",
                        "description": "Target scene identifier (e.g. 'sintel_scene_0004').",
                    },
                    "before": {
                        "type": "integer",
                        "description": "Number of preceding scenes to retrieve (default 2).",
                    },
                    "after": {
                        "type": "integer",
                        "description": "Number of succeeding scenes to retrieve (default 1).",
                    },
                },
                "required": ["scene_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "inspect_keyframes",
            "description": (
                "Option A Vision Sub-Tool: Inspect the high-resolution annotated shot keyframes of a specific "
                "scene_id using the Vision API to answer a fine-grained visual question."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "scene_id": {
                        "type": "string",
                        "description": "Scene identifier whose keyframes should be inspected (e.g. 'sintel_scene_0004').",
                    },
                    "visual_question": {
                        "type": "string",
                        "description": "Specific visual question about clothing, weapons, posture, expressions, or background details.",
                    },
                    "target_timestamp": {
                        "type": "number",
                        "description": (
                            "Optional target timestamp in seconds within the scene (e.g. 105.0 for 01:45) "
                            "to focus keyframe selection on the 4 frames closest to that moment."
                        ),
                    },
                },
                "required": ["scene_id", "visual_question"],
            },
        },
    },
]


def _build_spoiler_gated_tool_executors(
    entities_vdb: Optional[BaseVectorStorage],
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    chunks_vdb: Optional[BaseVectorStorage],
    video_segments: BaseKVStorage,
    knowledge_graph_inst: BaseGraphStorage,
    T_user: Optional[float],
    spoiler_free: bool,
    global_config: dict,
    character_glossary: Optional[Dict[str, str]] = None,
) -> Dict[str, Callable[..., Any]]:
    """Create the 4 tool callables with T_user and spoiler_free hard-locked inside their Python closures."""
    _, scene_map, ordered_scenes = _get_all_scenes_flat(video_segments)
    ordered_ids = [s_id for s_id, _ in ordered_scenes]
    working_dir = global_config.get("working_dir", "")

    async def search_scenes(
        query: str,
        start_time: float = 0.0,
        end_time: Optional[float] = None,
    ) -> str:
        if chunks_vdb is None:
            return "chunks_vdb is not enabled."
        low_t = max(0.0, float(start_time or 0.0))
        if spoiler_free and T_user is not None:
            effective_end = (
                min(float(end_time), float(T_user))
                if end_time is not None
                else float(T_user)
            )
        else:
            effective_end = float(end_time) if end_time is not None else None

        def _filter(dp: dict) -> bool:
            if not is_chunk_allowed(dp, T_user, spoiler_free):
                return False
            st = float(dp.get("start_time", 0.0))
            et = float(dp.get("end_time", 0.0))
            if et < low_t:
                return False
            if effective_end is not None:
                if dp.get("tier") == "chapter" and et > effective_end:
                    return False
                if st > effective_end:
                    return False
            return True

        hits = await chunks_vdb.query(query, top_k=5, filter_lambda=_filter)
        if not hits:
            return f"No scenes found matching '{query}' within the allowed timeline."
        chunk_ids = [h["id"] for h in hits]
        chunks = await text_chunks_db.get_by_ids(chunk_ids)
        blocks = []
        for c_id, c_dp in zip(chunk_ids, chunks):
            if c_dp is None:
                continue
            seg_ids = c_dp.get("video_segment_id", [])
            if isinstance(seg_ids, str):
                seg_ids = [seg_ids]
            if c_dp.get("tier") == "scene" and len(seg_ids) == 1 and seg_ids[0] in scene_map:
                s_id = seg_ids[0]
                blocks.append(
                    format_scene_segment_for_timestamp(
                        s_id,
                        scene_map[s_id],
                        T_user=T_user,
                        spoiler_free=spoiler_free,
                        source_label="search_scenes",
                    )
                )
            else:
                st_m = _format_mmss(float(c_dp.get("start_time", 0.0)))
                et_m = _format_mmss(float(c_dp.get("end_time", 0.0)))
                blocks.append(
                    f"[{c_dp.get('tier', 'chunk').upper()} {c_id} | {st_m} - {et_m}]:\n{c_dp['content']}"
                )
        return "\n\n".join(blocks)

    async def get_character_history(character_name: str) -> str:
        target = _normalize_entity_name(character_name or "")
        if not target:
            return "Please provide a non-empty character_name."

        matched_nodes: List[dict] = []
        direct_node = await knowledge_graph_inst.get_node(target)
        if direct_node is not None:
            matched_nodes.append({**direct_node, "entity_name": target})
        elif entities_vdb is not None:
            v_hits = await entities_vdb.query(
                character_name,
                top_k=5,
                filter_lambda=lambda dp: is_temporal_slice_allowed(dp, T_user, spoiler_free),
            )
            seen = set()
            for h in v_hits:
                ename = h.get("entity_name")
                if ename and ename not in seen:
                    seen.add(ename)
                    n = await knowledge_graph_inst.get_node(ename)
                    if n is not None:
                        matched_nodes.append({**n, "entity_name": ename})

        if not matched_nodes:
            return f"No character or entity matching '{character_name}' found up to the current timestamp."

        return await _reconstruct_graph_entities_and_relations_context(
            matched_nodes,
            knowledge_graph_inst,
            T_user=T_user,
            spoiler_free=spoiler_free,
            character_glossary=character_glossary,
            max_entities=3,
            max_relations=8,
            max_slices_per_entity=None,
        )

    async def get_adjacent_scenes(
        scene_id: str,
        before: int = 2,
        after: int = 1,
    ) -> str:
        if scene_id not in scene_map:
            return f"Scene '{scene_id}' not found. Available scenes: {', '.join(ordered_ids[:10])}..."
        idx = ordered_ids.index(scene_id)
        start_idx = max(0, idx - max(0, int(before)))
        end_idx = min(len(ordered_ids) - 1, idx + max(0, int(after)))

        blocks = []
        for i in range(start_idx, end_idx + 1):
            s_id = ordered_ids[i]
            seg = scene_map[s_id]
            if spoiler_free and T_user is not None and float(seg.get("start_time", 0.0)) > float(T_user):
                blocks.append(
                    f"[SCENE {s_id}]: Blocked by Spoiler-Free Mode (starts after T_user={_format_mmss(T_user)})."
                )
                continue
            blocks.append(
                format_scene_segment_for_timestamp(
                    s_id,
                    seg,
                    T_user=T_user,
                    spoiler_free=spoiler_free,
                    source_label="adjacent_scene",
                )
            )
        return "\n\n".join(blocks)

    async def inspect_keyframes(
        scene_id: str,
        visual_question: str,
        target_timestamp: Optional[float] = None,
    ) -> str:
        if scene_id not in scene_map:
            return f"Scene '{scene_id}' not found in video_segments."
        seg = scene_map[scene_id]
        if spoiler_free and T_user is not None and float(seg.get("start_time", 0.0)) > float(T_user):
            return (
                f"Scene '{scene_id}' starts at {_format_mmss(seg['start_time'])}, which is after "
                f"the viewer's current timestamp T_user={_format_mmss(T_user)}. Inspection blocked."
            )

        raw_kfs = seg.get("keyframes", []) or []
        if spoiler_free and T_user is not None:
            allowed_kfs = [
                kf
                for kf in raw_kfs
                if float(kf.get("timestamp", seg.get("start_time", 0.0))) <= float(T_user)
            ]
        else:
            allowed_kfs = list(raw_kfs)

        if not allowed_kfs and raw_kfs:
            allowed_kfs = [raw_kfs[0]]

        # Select up to 4 keyframes: either closest to target_timestamp (clamped <= T_user) or evenly spread
        if len(allowed_kfs) > 4:
            if target_timestamp is not None:
                t_target = float(target_timestamp)
                if spoiler_free and T_user is not None:
                    t_target = min(t_target, float(T_user))
                closest_four = sorted(
                    allowed_kfs,
                    key=lambda kf: abs(
                        float(kf.get("timestamp", seg.get("start_time", 0.0))) - t_target
                    ),
                )[:4]
                selected_kfs = sorted(
                    closest_four,
                    key=lambda kf: float(kf.get("timestamp", seg.get("start_time", 0.0))),
                )
            else:
                indices = [
                    int(round(i * (len(allowed_kfs) - 1) / 3.0)) for i in range(4)
                ]
                selected_kfs = [allowed_kfs[i] for i in indices]
        else:
            selected_kfs = allowed_kfs

        resolved_paths: List[str] = []
        for kf in selected_kfs:
            rel_or_abs = kf.get("path", "")
            if not rel_or_abs:
                continue
            if os.path.isabs(rel_or_abs) and os.path.isfile(rel_or_abs):
                resolved_paths.append(rel_or_abs)
            else:
                cand = os.path.join(working_dir, rel_or_abs)
                if os.path.isfile(cand):
                    resolved_paths.append(cand)
                elif os.path.isfile(rel_or_abs):
                    resolved_paths.append(rel_or_abs)

        scene_meta = {**seg, "scene_id": scene_id, "keyframes": selected_kfs}
        safe_glossary = {
            k: filter_glossary_text_for_timestamp(v, T_user, spoiler_free)
            for k, v in (character_glossary or {}).items()
        }
        return await vision_subtool_complete(
            visual_question=visual_question,
            scene_metadata=scene_meta,
            keyframe_paths=resolved_paths,
            character_glossary=safe_glossary,
        )

    return {
        "search_scenes": search_scenes,
        "get_character_history": get_character_history,
        "get_adjacent_scenes": get_adjacent_scenes,
        "inspect_keyframes": inspect_keyframes,
    }


# ---------------------------------------------------------------------------
# Main Spoiler-Aware Hybrid Turn-0 + 3-Round Agentic Query Entrypoint
# ---------------------------------------------------------------------------

async def videorag_query(
    query: str,
    entities_vdb: Optional[BaseVectorStorage],
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    chunks_vdb: Optional[BaseVectorStorage],
    video_path_db: BaseKVStorage,
    video_segments: BaseKVStorage,
    video_segment_feature_vdb: Optional[BaseVectorStorage],
    knowledge_graph_inst: BaseGraphStorage,
    caption_model: Any,
    caption_tokenizer: Any,
    query_param: QueryParam,
    global_config: dict,
) -> str:
    """Execute Hybrid Turn-0 Spoiler-Filtered GraphRAG Pre-Retrieval + Bounded 3-Round Agentic Tool Loop."""
    T_user = query_param.current_timestamp
    spoiler_free = query_param.spoiler_free
    max_rounds = max(0, int(getattr(query_param, "max_tool_rounds", 3)))

    # Step 1: Resolve follow-up pronouns against stateless chat_history[-6:]
    resolved_query = await _resolve_query_with_chat_history(
        query, query_param, global_config
    )

    # Step 2: Build fresh Turn-0 Spoiler-Filtered GraphRAG Context
    turn0_context, meta_info = await _build_turn0_context(
        query=query,
        resolved_query=resolved_query,
        entities_vdb=entities_vdb,
        text_chunks_db=text_chunks_db,
        chunks_vdb=chunks_vdb,
        video_segments=video_segments,
        knowledge_graph_inst=knowledge_graph_inst,
        query_param=query_param,
        global_config=global_config,
    )

    if query_param.only_need_context:
        return turn0_context

    # Step 3: Build Spoiler Guardrail & System Prompt
    if spoiler_free and T_user is not None:
        t_fmt = _format_mmss(T_user)
        spoiler_guardrail = (
            f"CRITICAL SPOILER-FREE DIRECTIVE (ACTIVE): The viewer is currently paused at {t_fmt} "
            f"({float(T_user):.1f}s) and has NOT watched beyond this point. Answer strictly using facts, "
            f"scenes, and dialogue up to {t_fmt}. NEVER reveal, hint at, or foreshadow any events, twists, "
            f"or character fates that occur after {t_fmt}."
        )
    elif spoiler_free:
        spoiler_guardrail = (
            "SPOILER-FREE DIRECTIVE: Spoiler protection is ON. Rely strictly on the retrieved scenes "
            "and character histories without revealing full-movie ending spoilers."
        )
    else:
        spoiler_guardrail = (
            "FULL-MOVIE ANALYSIS MODE (Spoiler Protection OFF): You have unrestricted access to the entire "
            "movie timeline, including Tier-3 full-movie plot documents, character arcs, foreshadowing, "
            "and ending outcomes."
        )

    if getattr(query_param, "wo_reference", False):
        citation_instruction = "Do not include a separate Reference section at the end."
    else:
        citation_instruction = (
            "Cite relevant scenes inline using `[MM:SS - MM:SS]` timestamps (and scene IDs where helpful)."
        )

    sys_prompt = PROMPTS["movierag_agent_system_prompt"].format(
        movie_title=meta_info["movie_title"],
        spoiler_guardrail=spoiler_guardrail,
        max_tool_rounds=max_rounds,
        citation_instruction=citation_instruction,
        response_type=query_param.response_type,
        turn0_context=turn0_context,
    )

    # Step 4: Bind spoiler-gated tool closures & construct messages array
    tool_executors = _build_spoiler_gated_tool_executors(
        entities_vdb=entities_vdb,
        text_chunks_db=text_chunks_db,
        chunks_vdb=chunks_vdb,
        video_segments=video_segments,
        knowledge_graph_inst=knowledge_graph_inst,
        T_user=T_user,
        spoiler_free=spoiler_free,
        global_config=global_config,
        character_glossary=meta_info["character_glossary"],
    )

    history_msgs = [
        {"role": m["role"], "content": m["content"]}
        for m in (query_param.chat_history or [])[-6:]
        if isinstance(m, dict) and m.get("role") in ("user", "assistant") and m.get("content")
    ]
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": sys_prompt},
        *history_msgs,
        {"role": "user", "content": query},
    ]

    tool_calls_trace: List[Dict[str, Any]] = []
    custom_tool_chat_fn = (global_config.get("addon_params") or {}).get("chat_with_tools_func")
    chat_with_tools_fn = custom_tool_chat_fn or openai_chat_with_tools
    best_model_name = global_config["llm"].get("best_model_name", "deepseek-chat")

    # Step 5: Execute Bounded Agentic Tool-Calling Loop (Rounds 1..max_rounds)
    for round_idx in range(max_rounds):
        try:
            assistant_msg = await chat_with_tools_fn(
                messages=messages,
                tools=AGENT_TOOL_SCHEMAS,
                model=best_model_name,
            )
        except Exception as exc:
            logger.warning(
                f"Tool-calling chat failed on round {round_idx + 1} ({exc}); falling back to best_model_func."
            )
            use_model_func = global_config["llm"]["best_model_func"]
            fallback_ans = await use_model_func(
                query,
                system_prompt=sys_prompt,
                history_messages=history_msgs,
            )
            if isinstance(global_config.get("addon_params"), dict):
                global_config["addon_params"]["last_tool_calls_trace"] = tool_calls_trace
            return fallback_ans

        tool_calls = assistant_msg.get("tool_calls") or []
        if not tool_calls:
            if isinstance(global_config.get("addon_params"), dict):
                global_config["addon_params"]["last_tool_calls_trace"] = tool_calls_trace
            return (assistant_msg.get("content") or "").strip()

        messages.append(assistant_msg)

        async def _run_one_tool(tc: dict) -> Tuple[dict, Dict[str, Any]]:
            tc_id = tc.get("id", f"call_{round_idx}")
            fn_obj = tc.get("function") or {}
            fn_name = fn_obj.get("name", "")
            raw_args = fn_obj.get("arguments", "{}")
            try:
                parsed_args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
            except Exception:
                parsed_args = {}

            executor = tool_executors.get(fn_name)
            if executor is None:
                tool_out = f"Error: Unknown tool '{fn_name}'."
            else:
                try:
                    tool_out = await executor(**parsed_args)
                except Exception as tool_exc:
                    tool_out = f"Error executing {fn_name}: {tool_exc}"

            tool_msg = {
                "role": "tool",
                "tool_call_id": tc_id,
                "name": fn_name,
                "content": str(tool_out),
            }
            trace_entry = {
                "round": round_idx + 1,
                "tool": fn_name,
                "arguments": parsed_args,
                "output_preview": str(tool_out)[:300],
            }
            return tool_msg, trace_entry

        tool_outputs = await asyncio.gather(*[_run_one_tool(tc) for tc in tool_calls])
        for tool_msg, trace_entry in tool_outputs:
            messages.append(tool_msg)
            tool_calls_trace.append(trace_entry)

    # Step 6: If max_tool_rounds reached and last step was a tool call, force final grounded answer (tools=None)
    messages.append(
        {
            "role": "user",
            "content": (
                "Tool-calling budget reached. Please synthesize the final grounded answer to the user's "
                "question now using all Turn-0 and tool-retrieved evidence gathered above, including "
                "`[MM:SS - MM:SS]` citations."
            ),
        }
    )
    final_msg = await chat_with_tools_fn(
        messages=messages,
        tools=None,
        model=best_model_name,
    )
    if isinstance(global_config.get("addon_params"), dict):
        global_config["addon_params"]["last_tool_calls_trace"] = tool_calls_trace
    return (final_msg.get("content") or "").strip()


async def videorag_query_multiple_choice(
    query: str,
    entities_vdb: Optional[BaseVectorStorage],
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    chunks_vdb: Optional[BaseVectorStorage],
    video_path_db: BaseKVStorage,
    video_segments: BaseKVStorage,
    video_segment_feature_vdb: Optional[BaseVectorStorage],
    knowledge_graph_inst: BaseGraphStorage,
    caption_model: Any,
    caption_tokenizer: Any,
    query_param: QueryParam,
    global_config: dict,
) -> dict:
    """Spoiler-aware multiple-choice query handler returning `{'Answer': ..., 'Explanation': ...}`."""
    use_model_func = global_config["llm"]["best_model_func"]
    resolved_query = await _resolve_query_with_chat_history(
        query, query_param, global_config
    )
    turn0_context, _ = await _build_turn0_context(
        query=query,
        resolved_query=resolved_query,
        entities_vdb=entities_vdb,
        text_chunks_db=text_chunks_db,
        chunks_vdb=chunks_vdb,
        video_segments=video_segments,
        knowledge_graph_inst=knowledge_graph_inst,
        query_param=query_param,
        global_config=global_config,
    )
    sys_prompt = PROMPTS["videorag_response_for_multiple_choice_question"].format(
        video_data=turn0_context,
        chunk_data="",
        response_type=query_param.response_type,
    )
    for _ in range(3):
        response = await use_model_func(
            query,
            system_prompt=sys_prompt,
            use_cache=False,
        )
        try:
            json_body = re.search(r"\{.*\}", response or "", re.DOTALL)
            json_response = json.loads(json_body.group(0) if json_body else response)
            if "Answer" in json_response and "Explanation" in json_response:
                return json_response
        except Exception as e:
            logger.info(f"Response is not valid JSON for query {query}. Found {e}. Retrying...")
    return {"Answer": "", "Explanation": response or ""}
