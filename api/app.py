"""Phase C — Standalone FastAPI Chatbot Service for MovieRAG (`api/app.py`).

Loads a pre-indexed `VideoRAG` knowledge base from `workdir/{movie_id}/` on CPU
and exposes:
- `GET /health`
- `POST /api/v1/chat`
"""

from contextlib import asynccontextmanager
import logging
import os
from pathlib import Path
import sys
from typing import Any, Awaitable, Callable, Dict, List, Literal, Optional, Tuple, Union

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

# Ensure MovieRAG/ is importable regardless of current working directory
REPO_ROOT = Path(__file__).resolve().parent.parent
MOVIERAG_PKG_DIR = REPO_ROOT / "MovieRAG"
for _p in (str(MOVIERAG_PKG_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from api.guardrail import check_movie_relevance  # noqa: E402
from videorag import QueryParam, VideoRAG, deepseek_config  # noqa: E402
from videorag._op import _parse_mmss_to_sec  # noqa: E402

logger = logging.getLogger("movierag.api")

REQUIRED_INDEX_FILES: tuple[str, ...] = (
    "movie_knowledge_bundle.json",
    "kv_store_video_segments.json",
    "kv_store_text_chunks.json",
    "vdb_chunks.json",
    "vdb_entities.json",
    "graph_chunk_entity_relation.graphml",
)


# ---------------------------------------------------------------------------
# Pydantic Request & Response Models
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1)

    @field_validator("content")
    @classmethod
    def _strip_content(cls, v: str) -> str:
        cleaned = (v or "").strip()
        if not cleaned:
            raise ValueError("Message content must not be empty.")
        return cleaned


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User question about the movie.")
    current_timestamp: Optional[Union[float, int, str]] = Field(
        default=None,
        description="Current playback position in float seconds or 'MM:SS' / 'HH:MM:SS' string.",
    )
    spoiler_free: bool = Field(
        default=True,
        description="When True, strictly blocks plot beats, scenes, and entity twists after current_timestamp.",
    )
    chat_history: List[ChatMessage] = Field(
        default_factory=list,
        description="Stateless multi-turn dialogue history (capped to the last 6 messages / 3 turns).",
    )
    max_tool_rounds: int = Field(
        default=3,
        ge=0,
        le=5,
        description="Maximum agentic tool-calling rounds (0..5).",
    )

    @field_validator("query")
    @classmethod
    def _validate_query(cls, v: str) -> str:
        cleaned = (v or "").strip()
        if not cleaned:
            raise ValueError("query must not be empty.")
        return cleaned

    @field_validator("current_timestamp")
    @classmethod
    def _validate_timestamp(cls, v: Optional[Union[float, int, str]]) -> Optional[float]:
        if v is None:
            return None
        if isinstance(v, str) and not v.strip():
            return None
        parsed = _parse_mmss_to_sec(v)
        if parsed is None:
            raise ValueError(
                f"Invalid current_timestamp '{v}'. Expected non-negative float seconds or 'MM:SS' / 'HH:MM:SS'."
            )
        if parsed < 0.0:
            raise ValueError("current_timestamp must be >= 0.0 seconds.")
        return float(parsed)


class ToolCallTraceItem(BaseModel):
    round: int
    tool: str
    arguments: Dict[str, Any] = Field(default_factory=dict)
    output_preview: str = ""


class ChatResponse(BaseModel):
    answer: str
    tool_calls_trace: List[ToolCallTraceItem] = Field(default_factory=list)
    effective_timestamp: Optional[float] = None
    spoiler_free: bool = True


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    ready: bool
    movie_id: str
    movie_title: str
    workdir: str
    api_key_configured: bool
    missing_files: List[str] = Field(default_factory=list)
    index_counts: Dict[str, int] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Service State & Lifespan Loader
# ---------------------------------------------------------------------------


class MovieRAGServiceState:
    """Holds the pre-indexed CPU VideoRAG singleton and workdir diagnostics."""

    def __init__(
        self,
        movie_id: Optional[str] = None,
        workdir: Optional[str] = None,
        rag: Optional[VideoRAG] = None,
    ) -> None:
        self.movie_id: str = (
            movie_id or os.environ.get("MOVIERAG_MOVIE_ID", "sintel")
        ).strip() or "sintel"
        default_workdir = REPO_ROOT / "workdir" / self.movie_id
        raw_workdir = workdir or os.environ.get("MOVIERAG_WORKDIR") or str(default_workdir)
        self.workdir: str = os.path.abspath(raw_workdir)
        self.rag: Optional[VideoRAG] = rag
        self.load_error: Optional[str] = None

    def check_missing_files(self) -> List[str]:
        missing: List[str] = []
        for fname in REQUIRED_INDEX_FILES:
            fpath = os.path.join(self.workdir, fname)
            if not os.path.isfile(fpath):
                missing.append(fname)
        return missing

    @staticmethod
    def is_api_key_configured() -> bool:
        key = (
            os.environ.get("DEEPSEEK_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or ""
        ).strip()
        if not key:
            return False
        placeholder_prefixes = ("sk-placeholder", "sk-***", "your_api_key")
        return not any(key.lower().startswith(p) for p in placeholder_prefixes)

    def try_load(self) -> bool:
        """Load VideoRAG from workdir if all required index files exist."""
        if self.rag is not None:
            return True
        missing = self.check_missing_files()
        if missing:
            self.load_error = (
                f"Missing required GraphRAG index files in '{self.workdir}': {', '.join(missing)}"
            )
            logger.warning(self.load_error)
            return False

        try:
            self.rag = VideoRAG(
                working_dir=self.workdir,
                llm=deepseek_config,
                always_create_working_dir=False,
            )
            loaded_id = self.rag.addon_params.get("movie_id")
            if loaded_id:
                self.movie_id = str(loaded_id)
            self.load_error = None
            logger.info(
                f"Loaded pre-indexed VideoRAG for movie_id='{self.movie_id}' from {self.workdir}"
            )
            return True
        except Exception as exc:
            self.load_error = f"Failed to initialize VideoRAG from '{self.workdir}': {exc}"
            logger.error(self.load_error)
            self.rag = None
            return False

    def get_index_counts(self) -> Dict[str, int]:
        if self.rag is None:
            return {
                "scenes": 0,
                "text_chunks": 0,
                "tier1_scene_chunks": 0,
                "tier2_chapter_chunks": 0,
                "tier3_full_movie_chunks": 0,
                "graph_nodes": 0,
                "graph_edges": 0,
            }

        seg_data = getattr(self.rag.video_segments, "_data", {}) or {}
        scene_count = sum(
            len(v) for v in seg_data.values() if isinstance(v, dict)
        )

        chunk_data = getattr(self.rag.text_chunks, "_data", {}) or {}
        t1 = sum(
            1
            for c in chunk_data.values()
            if isinstance(c, dict) and c.get("tier", "scene") == "scene"
        )
        t2 = sum(
            1
            for c in chunk_data.values()
            if isinstance(c, dict) and c.get("tier") == "chapter"
        )
        t3 = sum(
            1
            for c in chunk_data.values()
            if isinstance(c, dict) and c.get("tier") == "full_movie"
        )

        nx_graph = getattr(self.rag.chunk_entity_relation_graph, "_graph", None)
        n_nodes = nx_graph.number_of_nodes() if nx_graph is not None else 0
        n_edges = nx_graph.number_of_edges() if nx_graph is not None else 0

        return {
            "scenes": scene_count,
            "text_chunks": len(chunk_data),
            "tier1_scene_chunks": t1,
            "tier2_chapter_chunks": t2,
            "tier3_full_movie_chunks": t3,
            "graph_nodes": n_nodes,
            "graph_edges": n_edges,
        }


def _is_guardrail_enabled() -> bool:
    raw = os.environ.get("MOVIERAG_GUARDRAIL_ENABLED", "true").strip().lower()
    return raw not in ("0", "false", "off", "no", "disabled")


def create_app(
    movie_id: Optional[str] = None,
    workdir: Optional[str] = None,
    rag: Optional[VideoRAG] = None,
    guardrail_func: Optional[
        Callable[[str, str, Optional[List[Dict[str, str]]]], Awaitable[Tuple[bool, str]]]
    ] = None,
) -> FastAPI:
    """Factory to create the MovieRAG FastAPI service instance."""
    state = MovieRAGServiceState(movie_id=movie_id, workdir=workdir, rag=rag)
    resolved_guardrail = guardrail_func or check_movie_relevance

    @asynccontextmanager
    async def lifespan(app_instance: FastAPI):
        app_instance.state.movierag = state
        state.try_load()
        yield

    app_instance = FastAPI(
        title="MovieRAG Spoiler-Aware Chatbot API",
        description=(
            "Phase C standalone FastAPI service for animated movie multimodal GraphRAG "
            "with straddle-safe spoiler filtering and bounded 3-round agentic tool calling."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )
    app_instance.state.movierag = state

    raw_origins = os.environ.get("MOVIERAG_CORS_ORIGINS", "*").strip()
    allow_origins = (
        ["*"]
        if raw_origins == "*"
        else [o.strip() for o in raw_origins.split(",") if o.strip()]
    )
    app_instance.add_middleware(
        CORSMiddleware,
        allow_origins=allow_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app_instance.get("/health", response_model=HealthResponse, tags=["system"])
    async def get_health(request: Request) -> HealthResponse:
        svc: MovieRAGServiceState = request.app.state.movierag
        if svc.rag is None:
            svc.try_load()
        missing = svc.check_missing_files() if svc.rag is None else []
        ready = svc.rag is not None and len(missing) == 0
        movie_title = (
            str(svc.rag.addon_params.get("movie_title") or svc.movie_id.title())
            if svc.rag is not None
            else svc.movie_id.title()
        )
        return HealthResponse(
            status="ok" if ready else "degraded",
            ready=ready,
            movie_id=svc.movie_id,
            movie_title=movie_title,
            workdir=svc.workdir,
            api_key_configured=svc.is_api_key_configured(),
            missing_files=missing,
            index_counts=svc.get_index_counts(),
        )

    @app_instance.post("/api/v1/chat", response_model=ChatResponse, tags=["chat"])
    async def post_chat(payload: ChatRequest, request: Request) -> ChatResponse:
        svc: MovieRAGServiceState = request.app.state.movierag
        if svc.rag is None and not svc.try_load():
            missing = svc.check_missing_files()
            detail_msg = (
                svc.load_error
                or f"MovieRAG index is not loaded from '{svc.workdir}'. Missing: {missing}"
            )
            raise HTTPException(status_code=503, detail=detail_msg)

        assert svc.rag is not None
        has_custom_chat_fn = bool(svc.rag.addon_params.get("chat_with_tools_func"))
        if not has_custom_chat_fn and not svc.is_api_key_configured():
            raise HTTPException(
                status_code=503,
                detail=(
                    "DEEPSEEK_API_KEY is not configured. Export DEEPSEEK_API_KEY in the "
                    "server environment before calling /api/v1/chat."
                ),
            )

        # Cap stateless chat_history to the last 6 messages (3 turns)
        capped_history = [
            {"role": m.role, "content": m.content}
            for m in (payload.chat_history or [])[-6:]
        ]

        # Pre-retrieval movie-relevance guardrail (isolated LLM call, fail-closed)
        if _is_guardrail_enabled():
            movie_title = str(
                svc.rag.addon_params.get("movie_title") or svc.movie_id.title()
            )
            is_related, refusal_msg = await resolved_guardrail(
                payload.query,
                movie_title,
                capped_history,
            )
            if not is_related:
                return ChatResponse(
                    answer=refusal_msg,
                    tool_calls_trace=[],
                    effective_timestamp=payload.current_timestamp,
                    spoiler_free=payload.spoiler_free,
                )

        query_param = QueryParam(
            mode="videorag",
            current_timestamp=payload.current_timestamp,
            spoiler_free=payload.spoiler_free,
            max_tool_rounds=payload.max_tool_rounds,
            chat_history=capped_history,
        )

        try:
            answer, raw_trace = await svc.rag.aquery_with_trace(
                payload.query, param=query_param
            )
        except Exception as exc:
            logger.exception(f"Error executing MovieRAG query: {exc}")
            raise HTTPException(
                status_code=500,
                detail=f"MovieRAG query execution failed: {exc}",
            ) from exc

        trace_items = [
            ToolCallTraceItem(
                round=int(t.get("round", 1)),
                tool=str(t.get("tool", "")),
                arguments=dict(t.get("arguments") or {}),
                output_preview=str(t.get("output_preview", "")),
            )
            for t in (raw_trace or [])
            if isinstance(t, dict)
        ]

        return ChatResponse(
            answer=str(answer or ""),
            tool_calls_trace=trace_items,
            effective_timestamp=payload.current_timestamp,
            spoiler_free=payload.spoiler_free,
        )

    return app_instance


app = create_app()
