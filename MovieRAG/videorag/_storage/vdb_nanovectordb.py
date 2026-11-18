import asyncio
import base64
import json
import os
from dataclasses import dataclass
from typing import Callable, Optional
import numpy as np
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

try:
    from nano_vectordb import NanoVectorDB
except ImportError:

    class NanoVectorDB:
        """Compatible fallback implementation of NanoVectorDB when nano_vectordb package is not installed."""

        def __init__(self, embedding_dim: int, metric: str = "cosine", storage_file: Optional[str] = None):
            self.embedding_dim = embedding_dim
            self.metric = metric
            self.storage_file = storage_file
            self._data: list[dict] = []
            self._matrix: np.ndarray = np.zeros((0, embedding_dim), dtype=np.float32)
            if storage_file and os.path.exists(storage_file):
                self._load()

        def _load(self):
            with open(self.storage_file, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self.embedding_dim = int(payload.get("embedding_dim", self.embedding_dim))
            self._data = list(payload.get("data", []))
            matrix_b64 = payload.get("matrix", "")
            if matrix_b64 and len(self._data) > 0:
                raw = base64.b64decode(matrix_b64)
                arr = np.frombuffer(raw, dtype=np.float32)
                self._matrix = arr.reshape((len(self._data), self.embedding_dim)).copy()
            else:
                self._matrix = np.zeros((0, self.embedding_dim), dtype=np.float32)

        def save(self):
            if not self.storage_file:
                return
            matrix_b64 = base64.b64encode(self._matrix.astype(np.float32).tobytes()).decode("utf-8")
            payload = {
                "embedding_dim": self.embedding_dim,
                "data": self._data,
                "matrix": matrix_b64,
            }
            with open(self.storage_file, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)

        def upsert(self, datas: list[dict]):
            index_map = {d["__id__"]: i for i, d in enumerate(self._data)}
            update_ids, insert_ids = [], []
            for item in datas:
                item_copy = dict(item)
                vec = np.asarray(item_copy.pop("__vector__"), dtype=np.float32).reshape(1, self.embedding_dim)
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
                doc_id = item_copy["__id__"]
                if doc_id in index_map:
                    idx = index_map[doc_id]
                    self._data[idx] = item_copy
                    self._matrix[idx] = vec[0]
                    update_ids.append(doc_id)
                else:
                    index_map[doc_id] = len(self._data)
                    self._data.append(item_copy)
                    self._matrix = (
                        np.vstack([self._matrix, vec]) if len(self._matrix) > 0 else vec.copy()
                    )
                    insert_ids.append(doc_id)
            return {"update": update_ids, "insert": insert_ids}

        def query(
            self,
            query: np.ndarray,
            top_k: int = 10,
            better_than_threshold: Optional[float] = None,
            filter: Optional[Callable[[dict], bool]] = None,
        ) -> list[dict]:
            if len(self._data) == 0:
                return []
            q = np.asarray(query, dtype=np.float32).reshape(-1)
            q_norm = np.linalg.norm(q)
            if q_norm > 0:
                q = q / q_norm

            if filter is not None:
                valid_indices = [i for i, d in enumerate(self._data) if filter(d)]
            else:
                valid_indices = list(range(len(self._data)))
            if not valid_indices:
                return []

            sub_matrix = self._matrix[valid_indices]
            row_norms = np.linalg.norm(sub_matrix, axis=1, keepdims=True)
            row_norms = np.where(row_norms > 0, row_norms, 1.0)
            scores = np.dot(sub_matrix / row_norms, q)

            ranked = np.argsort(-scores)
            results = []
            for r_idx in ranked:
                score = float(scores[r_idx])
                if better_than_threshold is not None and score < better_than_threshold:
                    continue
                orig_idx = valid_indices[int(r_idx)]
                results.append({**self._data[orig_idx], "__metrics__": np.float32(score)})
                if len(results) >= top_k:
                    break
            return results


from .._utils import logger
from ..base import BaseVectorStorage


@dataclass
class NanoVectorDBStorage(BaseVectorStorage):
    cosine_better_than_threshold: float = 0.2

    def __post_init__(self):
        self._client_file_name = os.path.join(
            self.global_config["working_dir"], f"vdb_{self.namespace}.json"
        )
        self._max_batch_size = self.global_config["llm"]["embedding_batch_num"]
        self._client = NanoVectorDB(
            self.embedding_func.embedding_dim, storage_file=self._client_file_name
        )
        llm_threshold = (self.global_config.get("llm") or {}).get("query_better_than_threshold")
        self.cosine_better_than_threshold = (
            llm_threshold
            if llm_threshold is not None
            else self.global_config.get(
                "query_better_than_threshold", self.cosine_better_than_threshold
            )
        )

    async def upsert(self, data: dict[str, dict]):
        logger.info(f"Inserting {len(data)} vectors to {self.namespace}")
        if not len(data):
            logger.warning("You insert an empty data to vector DB")
            return []
        list_data = [
            {
                "__id__": k,
                **{k1: v1 for k1, v1 in v.items() if k1 in self.meta_fields},
            }
            for k, v in data.items()
        ]
        contents = [v["content"] for v in data.values()]
        batches = [
            contents[i : i + self._max_batch_size]
            for i in range(0, len(contents), self._max_batch_size)
        ]
        embeddings_list = await asyncio.gather(
            *[self.embedding_func(batch) for batch in batches]
        )
        embeddings = np.concatenate(embeddings_list)
        for i, d in enumerate(list_data):
            d["__vector__"] = embeddings[i]
        results = self._client.upsert(datas=list_data)
        return results

    async def query(
        self,
        query: str,
        top_k: int = 5,
        filter_lambda: Optional[Callable[[dict], bool]] = None,
    ) -> list[dict]:
        embedding = await self.embedding_func([query])
        embedding = embedding[0]
        results = self._client.query(
            query=embedding,
            top_k=top_k,
            better_than_threshold=self.cosine_better_than_threshold,
            filter=filter_lambda,
        )
        results = [
            {**dp, "id": dp["__id__"], "distance": dp["__metrics__"]} for dp in results
        ]
        return results

    async def index_done_callback(self):
        self._client.save()


@dataclass
class NanoVectorDBVideoSegmentStorage(BaseVectorStorage):
    embedding_func = None
    segment_retrieval_top_k: float = 2

    def __post_init__(self):
        self._client_file_name = os.path.join(
            self.global_config["working_dir"], f"vdb_{self.namespace}.json"
        )
        self._max_batch_size = self.global_config["video_embedding_batch_num"]
        self._client = NanoVectorDB(
            self.global_config["video_embedding_dim"], storage_file=self._client_file_name
        )
        self.top_k = self.global_config.get(
            "segment_retrieval_top_k", self.segment_retrieval_top_k
        )

    async def upsert(self, video_name, segment_index2name, video_output_format):
        import torch
        from imagebind.models import imagebind_model
        from .._videoutil import encode_video_segments

        embedder = imagebind_model.imagebind_huge(pretrained=True).cuda()
        embedder.eval()

        logger.info(f"Inserting {len(segment_index2name)} segments to {self.namespace}")
        if not len(segment_index2name):
            logger.warning("You insert an empty data to vector DB")
            return []
        list_data, video_paths = [], []
        cache_path = os.path.join(self.global_config["working_dir"], "_cache", video_name)
        index_list = list(segment_index2name.keys())
        for index in index_list:
            list_data.append(
                {
                    "__id__": f"{video_name}_{index}",
                    "__video_name__": video_name,
                    "__index__": index,
                }
            )
            segment_name = segment_index2name[index]
            video_file = os.path.join(cache_path, f"{segment_name}.{video_output_format}")
            video_paths.append(video_file)
        batches = [
            video_paths[i : i + self._max_batch_size]
            for i in range(0, len(video_paths), self._max_batch_size)
        ]
        embeddings = []
        for _batch in tqdm(batches, desc=f"Encoding Video Segments {video_name}"):
            batch_embeddings = encode_video_segments(_batch, embedder)
            embeddings.append(batch_embeddings)
        embeddings = torch.concat(embeddings, dim=0)
        embeddings = embeddings.numpy()
        for i, d in enumerate(list_data):
            d["__vector__"] = embeddings[i]
        results = self._client.upsert(datas=list_data)
        return results

    async def query(self, query: str):
        from imagebind.models import imagebind_model
        from .._videoutil import encode_string_query

        embedder = imagebind_model.imagebind_huge(pretrained=True).cuda()
        embedder.eval()

        embedding = encode_string_query(query, embedder)
        embedding = embedding[0]
        results = self._client.query(
            query=embedding,
            top_k=self.top_k,
            better_than_threshold=-1,
        )
        results = [
            {**dp, "id": dp["__id__"], "distance": dp["__metrics__"]} for dp in results
        ]
        return results

    async def index_done_callback(self):
        self._client.save()
