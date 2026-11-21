import asyncio
import json
import os
import queue
import shutil
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union, cast

from ._llm import (
    LLMConfig,
    azure_openai_config,
    deepseek_bge_config,
    deepseek_config,
    ollama_config,
    openai_4o_mini_config,
    openai_config,
)
from ._op import (
    _format_mmss,
    chunking_by_video_segments,
    extract_entities,
    format_scene_segment_for_timestamp,
    get_chunks,
    videorag_query,
    videorag_query_multiple_choice,
)
from ._storage import (
    JsonKVStorage,
    NanoVectorDBStorage,
    NanoVectorDBVideoSegmentStorage,
    NetworkXStorage,
)
from ._utils import (
    EmbeddingFunc,
    always_get_an_event_loop,
    compute_mdhash_id,
    convert_response_to_json,
    encode_string_by_tiktoken,
    limit_async_func_call,
    logger,
    wrap_embedding_func_with_attrs,
)
from .base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    QueryParam,
    StorageNameSpace,
    TextChunkSchema,
)
from .prompt import PROMPTS


@dataclass
class VideoRAG:
    working_dir: str = field(
        default_factory=lambda: f"./videorag_cache_{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"
    )

    # video
    threads_for_split: int = 10
    video_segment_length: int = 30  # seconds
    rough_num_frames_per_segment: int = 5  # frames
    fine_num_frames_per_segment: int = 15  # frames
    video_output_format: str = "mp4"
    audio_output_format: str = "mp3"
    video_embedding_batch_num: int = 2
    segment_retrieval_top_k: int = 4
    video_embedding_dim: int = 1024
    enable_visual_index: bool = False

    # query
    retrieval_topk_chunks: int = 5
    query_better_than_threshold: float = 0.2

    # graph mode
    enable_local: bool = True
    enable_naive_rag: bool = True

    # text chunking (1 Scene = 1 Rich Chunk)
    chunk_func: Callable[..., List[Dict[str, Union[str, int]]]] = chunking_by_video_segments
    chunk_token_size: int = 2048
    chapter_window_sec: float = 300.0
    tiktoken_model_name: str = "gpt-4o"

    # entity extraction
    entity_extract_max_gleaning: int = 0
    entity_summary_to_max_tokens: int = 500

    # LLM provider (defaults to deepseek_config)
    llm: LLMConfig = field(default_factory=lambda: deepseek_config)

    # entity extraction function
    entity_extraction_func: callable = extract_entities

    # storage
    key_string_value_json_storage_cls: Type[BaseKVStorage] = JsonKVStorage
    vector_db_storage_cls: Type[BaseVectorStorage] = NanoVectorDBStorage
    vs_vector_db_storage_cls: Type[BaseVectorStorage] = NanoVectorDBVideoSegmentStorage
    vector_db_storage_cls_kwargs: dict = field(default_factory=dict)
    graph_storage_cls: Type[BaseGraphStorage] = NetworkXStorage
    enable_llm_cache: bool = True

    # extension
    always_create_working_dir: bool = True
    addon_params: dict = field(default_factory=dict)
    convert_response_to_json_func: callable = convert_response_to_json

    def load_caption_model(self, debug=False):
        if not debug:
            from transformers import AutoModel, AutoTokenizer

            self.caption_model = AutoModel.from_pretrained(
                "./MiniCPM-V-2_6-int4", trust_remote_code=True
            )
            self.caption_tokenizer = AutoTokenizer.from_pretrained(
                "./MiniCPM-V-2_6-int4", trust_remote_code=True
            )
            self.caption_model.eval()
        else:
            self.caption_model = None
            self.caption_tokenizer = None

    def __post_init__(self):
        self.caption_model = None
        self.caption_tokenizer = None

        _print_config = ",\n  ".join([f"{k} = {v}" for k, v in asdict(self).items()])
        logger.debug(f"VideoRAG init with param:\n\n  {_print_config}\n")

        if not os.path.exists(self.working_dir) and self.always_create_working_dir:
            logger.info(f"Creating working directory {self.working_dir}")
            os.makedirs(self.working_dir, exist_ok=True)

        # Auto-load bundle metadata if movie_knowledge_bundle.json already exists in working_dir
        bundle_in_workdir = os.path.join(self.working_dir, "movie_knowledge_bundle.json")
        if os.path.isfile(bundle_in_workdir):
            try:
                with open(bundle_in_workdir, "r", encoding="utf-8") as f:
                    b_meta = json.load(f)
                self.addon_params.setdefault("movie_id", b_meta.get("movie_id", ""))
                self.addon_params.setdefault("movie_title", b_meta.get("movie_title", ""))
                self.addon_params.setdefault("duration_sec", b_meta.get("duration_sec", 0.0))
                self.addon_params.setdefault(
                    "character_glossary", b_meta.get("character_glossary", {})
                )
                self.addon_params.setdefault(
                    "external_plot_documents", b_meta.get("external_plot_documents", [])
                )
            except Exception as exc:
                logger.warning(f"Failed to preload bundle metadata from {bundle_in_workdir}: {exc}")

        self.video_path_db = self.key_string_value_json_storage_cls(
            namespace="video_path", global_config=asdict(self)
        )

        self.video_segments = self.key_string_value_json_storage_cls(
            namespace="video_segments", global_config=asdict(self)
        )

        self.text_chunks = self.key_string_value_json_storage_cls(
            namespace="text_chunks", global_config=asdict(self)
        )

        self.llm_response_cache = (
            self.key_string_value_json_storage_cls(
                namespace="llm_response_cache", global_config=asdict(self)
            )
            if self.enable_llm_cache
            else None
        )

        self.chunk_entity_relation_graph = self.graph_storage_cls(
            namespace="chunk_entity_relation", global_config=asdict(self)
        )

        self.embedding_func = limit_async_func_call(self.llm.embedding_func_max_async)(
            wrap_embedding_func_with_attrs(
                embedding_dim=self.llm.embedding_dim,
                max_token_size=self.llm.embedding_max_token_size,
                model_name=self.llm.embedding_model_name,
            )(self.llm.embedding_func)
        )
        self.entities_vdb = (
            self.vector_db_storage_cls(
                namespace="entities",
                global_config=asdict(self),
                embedding_func=self.embedding_func,
                meta_fields={"entity_name", "movie_id", "start_time", "end_time", "tier"},
            )
            if self.enable_local
            else None
        )
        self.chunks_vdb = (
            self.vector_db_storage_cls(
                namespace="chunks",
                global_config=asdict(self),
                embedding_func=self.embedding_func,
                meta_fields={
                    "movie_id",
                    "start_time",
                    "end_time",
                    "tier",
                    "characters",
                    "video_segment_id",
                },
            )
            if self.enable_naive_rag
            else None
        )

        self.video_segment_feature_vdb = (
            self.vs_vector_db_storage_cls(
                namespace="video_segment_feature",
                global_config=asdict(self),
                embedding_func=None,
            )
            if self.enable_visual_index
            else None
        )

        self.llm.best_model_func = limit_async_func_call(self.llm.best_model_max_async)(
            partial(self.llm.best_model_func, hashing_kv=self.llm_response_cache)
        )
        self.llm.cheap_model_func = limit_async_func_call(self.llm.cheap_model_max_async)(
            partial(self.llm.cheap_model_func, hashing_kv=self.llm_response_cache)
        )

    def insert_knowledge_bundle(self, bundle_path: str):
        """Synchronous entrypoint to ingest Phase-A `movie_knowledge_bundle.json` into 3-tier GraphRAG."""
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.ainsert_knowledge_bundle(bundle_path))

    async def ainsert_knowledge_bundle(self, bundle_path: str) -> Dict[str, Any]:
        """Ingest `movie_knowledge_bundle.json` and build Tier 1 (`scene`), Tier 2 (`chapter`), and Tier 3 (`full_movie`) chunks."""
        if not os.path.isfile(bundle_path):
            raise FileNotFoundError(f"movie_knowledge_bundle.json not found at: {bundle_path}")

        with open(bundle_path, "r", encoding="utf-8") as f:
            bundle = json.load(f)

        # Persist a copy of movie_knowledge_bundle.json in working_dir if loaded from another path
        target_bundle_path = os.path.join(self.working_dir, "movie_knowledge_bundle.json")
        if os.path.abspath(bundle_path) != os.path.abspath(target_bundle_path):
            shutil.copyfile(bundle_path, target_bundle_path)

        movie_id = str(bundle.get("movie_id") or "movie")
        movie_title = str(bundle.get("movie_title") or movie_id.replace("_", " ").title())
        raw_scenes = bundle.get("scenes", []) or []
        duration_sec = float(
            bundle.get("duration_sec")
            or (max((float(s.get("end_time", 0.0)) for s in raw_scenes), default=0.0))
        )
        character_glossary: Dict[str, str] = bundle.get("character_glossary") or {}
        external_plot_docs: List[Dict[str, Any]] = bundle.get("external_plot_documents") or []

        self.addon_params["movie_id"] = movie_id
        self.addon_params["movie_title"] = movie_title
        self.addon_params["duration_sec"] = duration_sec
        self.addon_params["character_glossary"] = character_glossary
        self.addon_params["external_plot_documents"] = external_plot_docs

        # 1. Populate self.video_segments and build Tier 1 (1 Scene = 1 Rich Chunk)
        segments_info: Dict[str, Dict[str, Any]] = {}
        tier1_chunks: Dict[str, TextChunkSchema] = {}
        ordered_scene_records: List[Tuple[str, Dict[str, Any]]] = []

        sorted_raw_scenes = sorted(
            raw_scenes,
            key=lambda s: (float(s.get("start_time", 0.0)), int(s.get("scene_index", 0))),
        )
        for idx, sc in enumerate(sorted_raw_scenes):
            scene_id = str(sc.get("scene_id") or f"{movie_id}_scene_{idx:04d}")
            scene_index = int(sc.get("scene_index", idx))
            start_t = float(sc.get("start_time", 0.0))
            end_t = float(sc.get("end_time", start_t))

            # Normalize character list
            chars: List[str] = []
            for cp in sc.get("characters_present", []) or []:
                cname = cp.get("name") if isinstance(cp, dict) else str(cp)
                if cname and cname not in chars:
                    chars.append(cname)
            for utt in sc.get("subtitles", []) or []:
                spk = utt.get("speaker")
                if spk and spk != "Unknown" and spk not in chars:
                    chars.append(spk)

            seg_record: Dict[str, Any] = {
                "scene_index": scene_index,
                "start_time": start_t,
                "end_time": end_t,
                "time": f"{start_t}-{end_t}",
                "vlm_caption": str(sc.get("vlm_caption") or ""),
                "exit_state": str(sc.get("exit_state") or ""),
                "subtitles": list(sc.get("subtitles") or []),
                "characters": chars,
                "keyframes": list(sc.get("keyframes") or []),
            }
            content_str = format_scene_segment_for_timestamp(
                scene_id=scene_id,
                seg=seg_record,
                T_user=None,
                spoiler_free=False,
            )
            seg_record["content"] = content_str
            segments_info[scene_id] = seg_record
            ordered_scene_records.append((scene_id, seg_record))

            tokens = encode_string_by_tiktoken(
                content_str, model_name=self.tiktoken_model_name
            )
            if len(tokens) > self.chunk_token_size:
                tokens = tokens[: self.chunk_token_size]

            chunk_key = compute_mdhash_id(content_str, prefix="chunk-")
            tier1_chunks[chunk_key] = {
                "tokens": len(tokens),
                "content": content_str,
                "video_segment_id": [scene_id],
                "chunk_order_index": scene_index,
                "movie_id": movie_id,
                "start_time": start_t,
                "end_time": end_t,
                "tier": "scene",
                "characters": chars,
            }

        await self.video_segments.upsert({movie_id: segments_info})
        await self._save_video_segments()

        # 2. Build Tier 2 ('chapter') progressive summaries (using ONLY Tier-1 scenes, zero external_plot)
        tier2_chunks: Dict[str, TextChunkSchema] = {}
        chapter_window = max(60.0, float(self.chapter_window_sec or 300.0))
        chapter_groups: List[List[Tuple[str, Dict[str, Any]]]] = []
        if ordered_scene_records:
            current_group: List[Tuple[str, Dict[str, Any]]] = []
            group_start_t = float(ordered_scene_records[0][1]["start_time"])
            for s_id, seg in ordered_scene_records:
                if (
                    current_group
                    and (float(seg["end_time"]) - group_start_t) > chapter_window
                ):
                    chapter_groups.append(current_group)
                    current_group = [(s_id, seg)]
                    group_start_t = float(seg["start_time"])
                else:
                    current_group.append((s_id, seg))
            if current_group:
                chapter_groups.append(current_group)

        async def _summarize_one_chapter(
            chap_idx: int, group: List[Tuple[str, Dict[str, Any]]]
        ) -> Tuple[str, TextChunkSchema]:
            c_start = float(group[0][1]["start_time"])
            c_end = float(group[-1][1]["end_time"])
            start_mmss = _format_mmss(c_start)
            end_mmss = _format_mmss(c_end)
            scenes_text = "\n\n".join(seg["content"] for _, seg in group)
            prompt = PROMPTS["chapter_summary_generation"].format(
                chapter_num=chap_idx + 1,
                start_mmss=start_mmss,
                end_mmss=end_mmss,
                movie_title=movie_title,
                scenes_text=scenes_text,
            )
            try:
                summary_text = await self.llm.cheap_model_func(prompt)
                summary_text = (summary_text or "").strip()
            except Exception as exc:
                logger.warning(f"Chapter {chap_idx + 1} LLM summary fallback ({exc})")
                summary_text = ""

            if not summary_text:
                summary_text = (
                    f"Chapter {chap_idx + 1} [{start_mmss} - {end_mmss}]: Progressive Summary\n"
                    + "\n".join(
                        f"- [{_format_mmss(seg['start_time'])} - {_format_mmss(seg['end_time'])}] "
                        f"{(seg['vlm_caption'] or '')[:200]}"
                        for _, seg in group
                    )
                )
            elif not summary_text.lower().startswith("chapter"):
                summary_text = (
                    f"Chapter {chap_idx + 1} [{start_mmss} - {end_mmss}]:\n{summary_text}"
                )

            chap_chars: List[str] = []
            for _, seg in group:
                for c in seg.get("characters", []):
                    if c not in chap_chars:
                        chap_chars.append(c)

            tokens = encode_string_by_tiktoken(
                summary_text, model_name=self.tiktoken_model_name
            )
            c_key = compute_mdhash_id(summary_text, prefix="chunk-")
            c_schema: TextChunkSchema = {
                "tokens": len(tokens),
                "content": summary_text,
                "video_segment_id": [s_id for s_id, _ in group],
                "chunk_order_index": 1000 + chap_idx,
                "movie_id": movie_id,
                "start_time": c_start,
                "end_time": c_end,
                "tier": "chapter",
                "characters": chap_chars,
            }
            return c_key, c_schema

        if chapter_groups:
            chapter_results = await asyncio.gather(
                *[
                    _summarize_one_chapter(idx, grp)
                    for idx, grp in enumerate(chapter_groups)
                ]
            )
            for c_key, c_schema in chapter_results:
                tier2_chunks[c_key] = c_schema

        # 3. Build Tier 3 ('full_movie') chunks from external_plot_documents
        tier3_chunks: Dict[str, TextChunkSchema] = {}
        all_known_chars = list(character_glossary.keys())
        for doc_idx, doc in enumerate(external_plot_docs):
            doc_content = (doc.get("content") or "").strip()
            if not doc_content:
                continue
            doc_title = doc.get("title") or f"{movie_title} Full Plot Synopsis"
            formatted_full = f"[FULL MOVIE PLOT — {doc_title}]\n{doc_content}"
            tokens = encode_string_by_tiktoken(
                formatted_full, model_name=self.tiktoken_model_name
            )
            c_key = compute_mdhash_id(formatted_full, prefix="chunk-")
            tier3_chunks[c_key] = {
                "tokens": len(tokens),
                "content": formatted_full,
                "video_segment_id": [s_id for s_id, _ in ordered_scene_records],
                "chunk_order_index": 2000 + doc_idx,
                "movie_id": movie_id,
                "start_time": float(doc.get("start_time", 0.0)),
                "end_time": float(doc.get("end_time") or duration_sec),
                "tier": "full_movie",
                "characters": all_known_chars,
            }

        all_inserting_chunks: Dict[str, TextChunkSchema] = {
            **tier1_chunks,
            **tier2_chunks,
            **tier3_chunks,
        }
        await self.ainsert_chunks(all_inserting_chunks)

        return {
            "movie_id": movie_id,
            "movie_title": movie_title,
            "duration_sec": duration_sec,
            "tier1_scene_chunks": len(tier1_chunks),
            "tier2_chapter_chunks": len(tier2_chunks),
            "tier3_full_movie_chunks": len(tier3_chunks),
            "total_chunks": len(all_inserting_chunks),
        }

    async def ainsert_chunks(self, inserting_chunks: Dict[str, TextChunkSchema]):
        """Upsert pre-constructed multi-tier chunks into chunks_vdb, extract time-sliced entities/relations, and persist."""
        await self._insert_start()
        try:
            _add_chunk_keys = await self.text_chunks.filter_keys(
                list(inserting_chunks.keys())
            )
            new_chunks = {
                k: v for k, v in inserting_chunks.items() if k in _add_chunk_keys
            }
            if not len(new_chunks):
                logger.warning("All chunks are already in the storage")
                return
            logger.info(f"[New Chunks] inserting {len(new_chunks)} chunks")
            if self.enable_naive_rag and self.chunks_vdb is not None:
                logger.info("Insert chunks for naive RAG")
                await self.chunks_vdb.upsert(new_chunks)

            logger.info("[Entity Extraction]...")
            maybe_new_kg, _, _ = await self.entity_extraction_func(
                new_chunks,
                knowledge_graph_inst=self.chunk_entity_relation_graph,
                entity_vdb=self.entities_vdb,
                global_config=asdict(self),
            )
            if maybe_new_kg is None:
                logger.warning("No new entities found")
            else:
                self.chunk_entity_relation_graph = maybe_new_kg

            await self.text_chunks.upsert(new_chunks)
        finally:
            await self._insert_done()

    def insert_video(self, video_path_list=None):
        from ._videoutil import (
            merge_segment_information,
            saving_video_segments,
            segment_caption,
            speech_to_text,
            split_video,
        )

        loop = always_get_an_event_loop()
        for video_path in video_path_list:
            video_name = os.path.basename(video_path).split(".")[0]
            if video_name in self.video_segments._data:
                logger.info(
                    f"Find the video named {os.path.basename(video_path)} in storage and skip it."
                )
                continue
            loop.run_until_complete(
                self.video_path_db.upsert({video_name: video_path})
            )

            segment_index2name, segment_times_info = split_video(
                video_path,
                self.working_dir,
                self.video_segment_length,
                self.rough_num_frames_per_segment,
                self.audio_output_format,
            )

            transcripts = speech_to_text(
                video_name,
                self.working_dir,
                segment_index2name,
                self.audio_output_format,
            )

            captions = {}
            error_queue = queue.Queue()

            try:
                saving_video_segments(
                    video_name,
                    video_path,
                    self.working_dir,
                    segment_index2name,
                    segment_times_info,
                    error_queue,
                    self.video_output_format,
                )
                if not error_queue.empty():
                    raise RuntimeError(error_queue.get())

                segment_caption(
                    video_name,
                    video_path,
                    segment_index2name,
                    transcripts,
                    segment_times_info,
                    captions,
                    error_queue,
                )

                while not error_queue.empty():
                    error_message = error_queue.get()
                    with open("error_log_videorag.txt", "a", encoding="utf-8") as log_file:
                        log_file.write(
                            f"Video Name:{video_name} Error processing:\n{error_message}\n\n"
                        )
                    raise RuntimeError(error_message)

            except Exception as e:
                logger.error(f"Failed to process video {video_name}: {str(e)}")
                raise

            segments_information = merge_segment_information(
                segment_index2name,
                segment_times_info,
                transcripts,
                captions,
            )

            loop.run_until_complete(
                self.video_segments.upsert({video_name: segments_information})
            )

            video_segment_cache_path = os.path.join(
                self.working_dir, "_cache", video_name
            )
            if os.path.exists(video_segment_cache_path):
                shutil.rmtree(video_segment_cache_path)

            loop.run_until_complete(self._save_video_segments())

        loop.run_until_complete(self.ainsert(self.video_segments._data))

    def query(self, query: str, param: QueryParam = QueryParam()):
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.aquery(query, param))

    async def aquery(self, query: str, param: QueryParam = QueryParam()):
        cfg = asdict(self)
        cfg["addon_params"] = self.addon_params
        if param.mode in ("videorag", "local", "global", "naive"):
            response = await videorag_query(
                query,
                self.entities_vdb,
                self.text_chunks,
                self.chunks_vdb,
                self.video_path_db,
                self.video_segments,
                self.video_segment_feature_vdb,
                self.chunk_entity_relation_graph,
                self.caption_model,
                self.caption_tokenizer,
                param,
                cfg,
            )
        elif param.mode == "videorag_multiple_choice":
            response = await videorag_query_multiple_choice(
                query,
                self.entities_vdb,
                self.text_chunks,
                self.chunks_vdb,
                self.video_path_db,
                self.video_segments,
                self.video_segment_feature_vdb,
                self.chunk_entity_relation_graph,
                self.caption_model,
                self.caption_tokenizer,
                param,
                cfg,
            )
        else:
            raise ValueError(f"Unknown mode {param.mode}")
        await self._query_done()
        return response

    async def ainsert(self, new_video_segment):
        inserting_chunks = get_chunks(
            new_videos=new_video_segment,
            chunk_func=self.chunk_func,
            max_token_size=self.chunk_token_size,
        )
        await self.ainsert_chunks(inserting_chunks)

    async def _insert_start(self):
        tasks = []
        for storage_inst in [
            self.chunk_entity_relation_graph,
        ]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_start_callback())
        await asyncio.gather(*tasks)

    async def _save_video_segments(self):
        tasks = []
        for storage_inst in [
            self.video_segment_feature_vdb,
            self.video_segments,
            self.video_path_db,
        ]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)

    async def _insert_done(self):
        tasks = []
        for storage_inst in [
            self.text_chunks,
            self.llm_response_cache,
            self.entities_vdb,
            self.chunks_vdb,
            self.chunk_entity_relation_graph,
            self.video_segment_feature_vdb,
            self.video_segments,
            self.video_path_db,
        ]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)

    async def _query_done(self):
        tasks = []
        for storage_inst in [self.llm_response_cache]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)
