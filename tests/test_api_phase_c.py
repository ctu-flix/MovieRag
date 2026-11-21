"""Smoke & unit tests for Phase C — Standalone FastAPI Chatbot Service (`api/app.py`).

Covers:
1. Degraded startup & 503 handling when workdir index files are missing.
2. `/health` readiness & 3-tier index counts on a pre-indexed Sintel workdir.
3. Spoiler-free (`spoiler_free=True`, `T_user="02:00"`) vs. Full-movie (`spoiler_free=False`) `/api/v1/chat`.
4. Timestamp parsing (`float`, `"MM:SS"`, `"HH:MM:SS"`, `None`, and invalid -> 422).
5. `chat_history` capping to the last 6 messages (3 turns) and passthrough.
6. Concurrency safety of `tool_calls_trace` across parallel requests (`asyncio.gather`).
"""

import asyncio
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, List

import httpx
import pytest

pytest_plugins = ("anyio",)

REPO_ROOT = Path(__file__).resolve().parent.parent
MOVIERAG_PKG_DIR = REPO_ROOT / "MovieRAG"
for _p in (str(MOVIERAG_PKG_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from api.app import create_app  # noqa: E402
from videorag import VideoRAG  # noqa: E402
from videorag._llm import LLMConfig, _deterministic_ngram_hash_embedding  # noqa: E402


async def _mock_hash_embedding(model_name: str, texts: list[str]):
    return _deterministic_ngram_hash_embedding(texts, dim=768)


async def _mock_llm_complete(
    model_name: str,
    prompt: str,
    system_prompt: str = None,
    history_messages: list = [],
    **kwargs,
) -> str:
    p_low = (prompt or "").lower()
    # Entity extraction stub during fixture indexing
    if "entity_types" in p_low or ("entity" in p_low and "relationship" in p_low and "tuple_delimiter" in p_low):
        if "adult dragon is actually grown-up scales" in p_low or "full movie plot" in p_low:
            return (
                '("entity"<|>"ADULT DRAGON"<|>"CHARACTER"<|>"The adult dragon in the volcanic cavern is revealed by his wing bandage scar to be grown-up Scales.")##\n'
                '("relationship"<|>"SINTEL"<|>"ADULT DRAGON"<|>"Sintel fatally strikes the adult dragon before realizing from his wing scar that he is Scales."<|>9)'
            )
        if "baby dragon" in p_low or "scales" in p_low:
            return (
                '("entity"<|>"SCALES"<|>"CHARACTER"<|>"Scales is a playful baby dragon rescued by Sintel in Ishtar at 03:00.")##\n'
                '("relationship"<|>"SINTEL"<|>"SCALES"<|>"Sintel bandages Scales injured wing in Ishtar."<|>8)'
            )
        return (
            '("entity"<|>"SINTEL"<|>"CHARACTER"<|>"Sintel is a young traveler trekking across snowy mountains and sheltered by the Shaman.")##\n'
            '("entity"<|>"SHAMAN"<|>"CHARACTER"<|>"The Shaman is an elder in a firelit hut examining Sintel curved spear-blade at 01:45.")##\n'
            '("relationship"<|>"SHAMAN"<|>"SINTEL"<|>"The Shaman shelters Sintel in his hut and warns her that the blade has a dark past."<|>7)'
        )
    # Chapter summary stub
    if "chapter" in p_low and "progressive" in p_low:
        return "Chapter 1 [00:00 - 05:20]: Sintel fights the Gatekeeper, speaks with the Shaman, and recalls finding baby Scales in Ishtar."
    # Coreference rewrite stub
    if "rewrite" in p_low or "conversation history" in p_low:
        return prompt.splitlines()[-1].strip()
    return "Grounded fallback response [01:41 - 02:11]."


def _build_mock_llm_config() -> LLMConfig:
    return LLMConfig(
        embedding_func_raw=_mock_hash_embedding,
        embedding_model_name="hash-768",
        embedding_dim=768,
        embedding_max_token_size=8192,
        embedding_batch_num=16,
        embedding_func_max_async=8,
        query_better_than_threshold=-1.0,
        best_model_func_raw=_mock_llm_complete,
        best_model_name="deepseek-flash",
        best_model_max_token_size=32768,
        best_model_max_async=8,
        cheap_model_func_raw=_mock_llm_complete,
        cheap_model_name="deepseek-flash",
        cheap_model_max_token_size=32768,
        cheap_model_max_async=8,
    )


@pytest.fixture()
def indexed_sintel_workdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Build a real 3-tier indexed Sintel workdir on CPU using deterministic hash embeddings."""
    monkeypatch.setenv("MOVIERAG_EMBEDDING_PROVIDER", "hash")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-valid-key-12345")

    workdir = tmp_path / "workdir" / "sintel"
    workdir.mkdir(parents=True, exist_ok=True)
    kf_dir = workdir / "keyframes"
    kf_dir.mkdir(parents=True, exist_ok=True)
    dummy_kf = kf_dir / "sintel_scene_0001_shot1_105.00s.jpg"
    dummy_kf.write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xd9")

    bundle = {
        "movie_id": "sintel",
        "movie_title": "Sintel",
        "duration_sec": 755.0,
        "character_glossary": {
            "Sintel": "Young woman protagonist. In the mountain quest (00:25 - 02:35), she wears a travel cloak; in the volcanic cavern (08:15 - 12:25), she has gray streaks.",
            "Shaman": "Elderly healer in a firelit hut who examines the curved spear-blade (01:41 - 02:35).",
            "Scales": "Small baby dragon rescued by Sintel in Ishtar (02:58 - 05:05).",
            "Adult Dragon": "Massive dragon. In the finale (11:15 - 12:35), a wing bandage-scar reveals it is actually grown-up Scales.",
        },
        "external_plot_documents": [
            {
                "doc_id": "sintel_plot_01",
                "title": "Sintel Time-Aligned Grounding Reference",
                "tier": "full_movie",
                "start_time": 0.0,
                "end_time": 755.0,
                "content": (
                    "Act I.A — Mountain Clash & Shaman Hut (00:18 - 02:11):\n"
                    "Sintel survives a fight with the Gatekeeper and wakes in the Shaman's firelit hut.\n\n"
                    "Act II.A — Rescuing Baby Scales in Ishtar (02:35 - 05:20):\n"
                    "In a flashback, Sintel finds an injured baby dragon named Scales and bandages his wing.\n\n"
                    "Act IV.B — The Wing-Scar Twist in the Volcanic Cavern (11:15 - 12:35):\n"
                    "Sintel slays the Adult Dragon and discovers from the bandage scar on its wing that the Adult Dragon is actually grown-up Scales."
                ),
            }
        ],
        "scenes": [
            {
                "scene_id": "sintel_scene_0000",
                "scene_index": 0,
                "start_time": 18.0,
                "end_time": 101.0,
                "characters_present": [{"name": "Sintel"}],
                "keyframes": [],
                "subtitles": [],
                "vlm_caption": "[00:18.00 - 01:41.00] Sintel treks across a snowy pass and fights the Gatekeeper.",
                "exit_state": "",
            },
            {
                "scene_id": "sintel_scene_0001",
                "scene_index": 1,
                "start_time": 101.0,
                "end_time": 131.0,
                "characters_present": [{"name": "Sintel"}, {"name": "Shaman"}],
                "keyframes": [
                    {
                        "shot_idx": 1,
                        "timestamp": 105.0,
                        "path": "keyframes/sintel_scene_0001_shot1_105.00s.jpg",
                        "characters_visible": [{"name": "Shaman", "color": "green"}],
                    }
                ],
                "subtitles": [
                    {
                        "start_time": 107.0,
                        "end_time": 109.0,
                        "speaker": "Shaman",
                        "text": "This blade has a dark past.",
                    },
                    {
                        "start_time": 128.0,
                        "end_time": 130.5,
                        "speaker": "Sintel",
                        "text": "I am searching for a dragon.",
                    },
                ],
                "vlm_caption": (
                    "[01:41.00 - 01:55.00] Inside the firelit hut, the Shaman examines the curved spear-blade.\n\n"
                    "[01:55.00 - 02:11.00] Sintel tells the Shaman she is searching for a dragon."
                ),
                "exit_state": "",
            },
            {
                "scene_id": "sintel_scene_0002",
                "scene_index": 2,
                "start_time": 175.0,
                "end_time": 320.0,
                "characters_present": [{"name": "Sintel"}, {"name": "Scales"}],
                "keyframes": [],
                "subtitles": [
                    {
                        "start_time": 210.0,
                        "end_time": 215.0,
                        "speaker": "Sintel",
                        "text": "Good night, Scales.",
                    }
                ],
                "vlm_caption": "[02:55.00 - 05:20.00] Sintel finds baby Scales on a rooftop in Ishtar and bandages his wing.",
                "exit_state": "",
            },
        ],
    }
    bundle_path = workdir / "movie_knowledge_bundle.json"
    bundle_path.write_text(json.dumps(bundle, indent=2), encoding="utf-8")

    rag = VideoRAG(
        working_dir=str(workdir),
        llm=_build_mock_llm_config(),
        chapter_window_sec=400.0,
    )
    rag.insert_knowledge_bundle(str(bundle_path))
    return workdir


@pytest.mark.anyio
async def test_health_degraded_when_workdir_missing(tmp_path: Path):
    missing_dir = tmp_path / "nonexistent_workdir"
    app = create_app(movie_id="sintel", workdir=str(missing_dir))
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "degraded"
        assert data["ready"] is False
        assert "movie_knowledge_bundle.json" in data["missing_files"]
        # Ensure degraded startup did NOT create empty dummy files in missing_dir
        assert not missing_dir.exists()

        chat_resp = await client.post(
            "/api/v1/chat",
            json={"query": "Who is Sintel?", "current_timestamp": "02:00"},
        )
        assert chat_resp.status_code == 503


@pytest.mark.anyio
async def test_health_loaded(indexed_sintel_workdir: Path):
    app = create_app(movie_id="sintel", workdir=str(indexed_sintel_workdir))
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["ready"] is True
        assert data["movie_id"] == "sintel"
        assert data["movie_title"] == "Sintel"
        assert data["missing_files"] == []
        counts = data["index_counts"]
        assert counts["scenes"] == 3
        assert counts["tier1_scene_chunks"] == 3
        assert counts["tier2_chapter_chunks"] >= 1
        assert counts["tier3_full_movie_chunks"] == 1
        assert counts["graph_nodes"] >= 2


@pytest.mark.anyio
async def test_chat_spoiler_free_vs_full_movie(indexed_sintel_workdir: Path):
    captured_system_prompts: List[str] = []

    async def _fake_chat_with_tools(messages: list[dict], tools=None, model=None, **kwargs) -> dict:
        sys_msg = next((m["content"] for m in messages if m.get("role") == "system"), "")
        captured_system_prompts.append(sys_msg)
        if "FULL-MOVIE ANALYSIS MODE" in sys_msg:
            return {
                "role": "assistant",
                "content": "In the finale, the Adult Dragon is revealed to be grown-up Scales [11:15 - 12:35].",
            }
        return {
            "role": "assistant",
            "content": "At 01:50, Sintel is in the Shaman's hut [01:41 - 01:55]. Scales has not appeared yet.",
        }

    rag = VideoRAG(
        working_dir=str(indexed_sintel_workdir),
        llm=_build_mock_llm_config(),
        always_create_working_dir=False,
    )
    rag.addon_params["chat_with_tools_func"] = _fake_chat_with_tools
    app = create_app(movie_id="sintel", workdir=str(indexed_sintel_workdir), rag=rag)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. Spoiler-Free at 01:50 (110.0s)
        sf_resp = await client.post(
            "/api/v1/chat",
            json={
                "query": "What happens to Scales and the Adult Dragon?",
                "current_timestamp": "01:50",
                "spoiler_free": True,
            },
        )
        assert sf_resp.status_code == 200
        sf_data = sf_resp.json()
        assert sf_data["effective_timestamp"] == 110.0
        assert sf_data["spoiler_free"] is True
        assert "01:41 - 01:55" in sf_data["answer"]
        assert sf_data["tool_calls_trace"] == []

        # Verify Turn-0 context at 01:50 excluded future scenes/entities/plot beats & clipped subtitle at 128s
        sf_sys = captured_system_prompts[-1]
        assert "This blade has a dark past." in sf_sys
        assert "I am searching for a dragon." not in sf_sys  # spoken at 128.0s > 110.0s
        assert "grown-up Scales" not in sf_sys  # Tier-3 plot & future glossary clause stripped

        # 2. Full-Movie Mode (spoiler_free=False)
        fm_resp = await client.post(
            "/api/v1/chat",
            json={
                "query": "What happens to Scales and the Adult Dragon?",
                "current_timestamp": "01:50",
                "spoiler_free": False,
            },
        )
        assert fm_resp.status_code == 200
        fm_data = fm_resp.json()
        assert fm_data["spoiler_free"] is False
        assert "grown-up Scales" in fm_data["answer"]
        fm_sys = captured_system_prompts[-1]
        assert "grown-up Scales" in fm_sys


@pytest.mark.anyio
async def test_chat_timestamp_parsing_and_validation(indexed_sintel_workdir: Path):
    async def _noop_chat(messages: list[dict], tools=None, model=None, **kwargs) -> dict:
        return {"role": "assistant", "content": "OK"}

    rag = VideoRAG(
        working_dir=str(indexed_sintel_workdir),
        llm=_build_mock_llm_config(),
        always_create_working_dir=False,
    )
    rag.addon_params["chat_with_tools_func"] = _noop_chat
    app = create_app(movie_id="sintel", workdir=str(indexed_sintel_workdir), rag=rag)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        for raw_ts, expected_sec in [
            (95.5, 95.5),
            ("02:05", 125.0),
            ("00:02:05.5", 125.5),
            (None, None),
        ]:
            r = await client.post(
                "/api/v1/chat",
                json={"query": "Test timestamp", "current_timestamp": raw_ts},
            )
            assert r.status_code == 200
            assert r.json()["effective_timestamp"] == expected_sec

        # Invalid timestamp string -> 422
        bad_r = await client.post(
            "/api/v1/chat",
            json={"query": "Test bad timestamp", "current_timestamp": "not-a-timestamp"},
        )
        assert bad_r.status_code == 422

        # Negative timestamp -> 422
        neg_r = await client.post(
            "/api/v1/chat",
            json={"query": "Test negative timestamp", "current_timestamp": -10.0},
        )
        assert neg_r.status_code == 422


@pytest.mark.anyio
async def test_chat_history_capping_and_passthrough(indexed_sintel_workdir: Path):
    observed_messages: List[Dict[str, Any]] = []

    async def _capture_chat(messages: list[dict], tools=None, model=None, **kwargs) -> dict:
        observed_messages.clear()
        observed_messages.extend(messages)
        return {"role": "assistant", "content": "Follow-up answered."}

    rag = VideoRAG(
        working_dir=str(indexed_sintel_workdir),
        llm=_build_mock_llm_config(),
        always_create_working_dir=False,
    )
    rag.addon_params["chat_with_tools_func"] = _capture_chat
    app = create_app(movie_id="sintel", workdir=str(indexed_sintel_workdir), rag=rag)

    # Build 10 messages (5 turns); service must cap to the last 6 messages (turns 3..5)
    ten_history_msgs = []
    for i in range(5):
        ten_history_msgs.append({"role": "user", "content": f"User question turn {i + 1}"})
        ten_history_msgs.append({"role": "assistant", "content": f"Assistant answer turn {i + 1}"})

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/api/v1/chat",
            json={
                "query": "Why did he say that?",
                "current_timestamp": "02:00",
                "spoiler_free": True,
                "chat_history": ten_history_msgs,
            },
        )
        assert resp.status_code == 200
        # System message + 6 history messages + 1 current user message = 8 messages total
        assert len(observed_messages) == 8
        history_slice = observed_messages[1:-1]
        assert len(history_slice) == 6
        assert history_slice[0]["content"] == "User question turn 3"
        assert history_slice[-1]["content"] == "Assistant answer turn 5"


@pytest.mark.anyio
async def test_concurrent_chat_tool_calls_trace_isolation(indexed_sintel_workdir: Path):
    """Verify two concurrent /api/v1/chat requests do not clobber each other's tool_calls_trace."""

    async def _concurrent_mock_chat(messages: list[dict], tools=None, model=None, **kwargs) -> dict:
        user_queries = [m.get("content", "") for m in messages if m.get("role") == "user"]
        has_tool_response = any(m.get("role") == "tool" for m in messages)
        first_user_q = user_queries[0] if user_queries else ""

        if "USE_TOOL" in first_user_q and not has_tool_response:
            await asyncio.sleep(0.05)
            return {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_adj_1",
                        "type": "function",
                        "function": {
                            "name": "get_adjacent_scenes",
                            "arguments": json.dumps({"scene_id": "sintel_scene_0001", "before": 1, "after": 0}),
                        },
                    }
                ],
            }
        if "NO_TOOL" in first_user_q:
            await asyncio.sleep(0.02)
            return {"role": "assistant", "content": "Direct zero-hop answer [01:41 - 02:11]."}

        return {"role": "assistant", "content": "Answer after tool call [00:18 - 02:11]."}

    rag = VideoRAG(
        working_dir=str(indexed_sintel_workdir),
        llm=_build_mock_llm_config(),
        always_create_working_dir=False,
    )
    rag.addon_params["chat_with_tools_func"] = _concurrent_mock_chat
    app = create_app(movie_id="sintel", workdir=str(indexed_sintel_workdir), rag=rag)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp_tool, resp_direct = await asyncio.gather(
            client.post(
                "/api/v1/chat",
                json={"query": "USE_TOOL: Check adjacent scenes", "current_timestamp": "02:10"},
            ),
            client.post(
                "/api/v1/chat",
                json={"query": "NO_TOOL: Quick question", "current_timestamp": "02:10"},
            ),
        )

        assert resp_tool.status_code == 200
        assert resp_direct.status_code == 200
        data_tool = resp_tool.json()
        data_direct = resp_direct.json()

        assert len(data_tool["tool_calls_trace"]) == 1
        assert data_tool["tool_calls_trace"][0]["tool"] == "get_adjacent_scenes"
        assert data_direct["tool_calls_trace"] == []
