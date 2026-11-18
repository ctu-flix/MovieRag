import html
import json
import os
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Union, cast
import numpy as np

try:
    import networkx as nx
except ImportError:

    class _FallbackNXGraph:
        """Pure-Python fallback for nx.Graph when networkx is not installed in system Python."""

        def __init__(self):
            self.nodes: dict[str, dict] = {}
            self._adj: dict[str, dict[str, dict]] = defaultdict(dict)

        def is_directed(self) -> bool:
            return False

        def number_of_nodes(self) -> int:
            return len(self.nodes)

        def number_of_edges(self) -> int:
            seen = set()
            for u, nbrs in self._adj.items():
                for v in nbrs:
                    seen.add(tuple(sorted((u, v))))
            return len(seen)

        def has_node(self, node_id: str) -> bool:
            return node_id in self.nodes

        def has_edge(self, u: str, v: str) -> bool:
            return u in self._adj and v in self._adj[u]

        def add_node(self, node_id: str, **attr):
            if node_id not in self.nodes:
                self.nodes[node_id] = {}
            self.nodes[node_id].update(attr)

        def add_edge(self, u: str, v: str, **attr):
            self.add_node(u)
            self.add_node(v)
            shared = self._adj[u].get(v, {})
            shared.update(attr)
            self._adj[u][v] = shared
            self._adj[v][u] = shared

        def degree(self, node_id: str) -> int:
            return len(self._adj.get(node_id, {}))

        def edges(self, nbunch=None, data: bool = False):
            if nbunch is not None:
                return [(nbunch, v) for v in self._adj.get(nbunch, {})]
            out = []
            seen = set()
            for u, nbrs in self._adj.items():
                for v, d in nbrs.items():
                    key = tuple(sorted((u, v)))
                    if key in seen:
                        continue
                    seen.add(key)
                    out.append((u, v, d) if data else (u, v))
            return out

        @property
        def _edges_view(self):
            class _EdgeLookup:
                def __init__(self, adj):
                    self._adj = adj

                def get(self, pair, default=None):
                    u, v = pair
                    return self._adj.get(u, {}).get(v, default)

            return _EdgeLookup(self._adj)

    class _FallbackNXModule:
        Graph = _FallbackNXGraph

        @staticmethod
        def write_graphml(graph: _FallbackNXGraph, file_name: str):
            root = ET.Element("graphml")
            g_el = ET.SubElement(root, "graph", id="G", edgedefault="undirected")
            for n_id, attrs in graph.nodes.items():
                n_el = ET.SubElement(g_el, "node", id=str(n_id))
                for k, v in attrs.items():
                    d_el = ET.SubElement(n_el, "data", key=str(k))
                    d_el.text = json.dumps({"t": type(v).__name__, "v": v}, ensure_ascii=False)
            for u, v, attrs in graph.edges(data=True):
                e_el = ET.SubElement(g_el, "edge", source=str(u), target=str(v))
                for k, val in attrs.items():
                    d_el = ET.SubElement(e_el, "data", key=str(k))
                    d_el.text = json.dumps({"t": type(val).__name__, "v": val}, ensure_ascii=False)
            tree = ET.ElementTree(root)
            tree.write(file_name, encoding="utf-8", xml_declaration=True)

        @staticmethod
        def read_graphml(file_name: str) -> _FallbackNXGraph:
            g = _FallbackNXGraph()
            tree = ET.parse(file_name)
            root = tree.getroot()
            for g_el in root.findall(".//graph"):
                for n_el in g_el.findall("node"):
                    n_id = n_el.attrib["id"]
                    attrs = {}
                    for d_el in n_el.findall("data"):
                        k = d_el.attrib["key"]
                        raw = d_el.text or ""
                        try:
                            parsed = json.loads(raw)
                            attrs[k] = parsed["v"] if isinstance(parsed, dict) and "v" in parsed else parsed
                        except Exception:
                            attrs[k] = raw
                    g.add_node(n_id, **attrs)
                for e_el in g_el.findall("edge"):
                    u = e_el.attrib["source"]
                    v = e_el.attrib["target"]
                    attrs = {}
                    for d_el in e_el.findall("data"):
                        k = d_el.attrib["key"]
                        raw = d_el.text or ""
                        try:
                            parsed = json.loads(raw)
                            attrs[k] = parsed["v"] if isinstance(parsed, dict) and "v" in parsed else parsed
                        except Exception:
                            attrs[k] = raw
                    g.add_edge(u, v, **attrs)
            return g

    nx = _FallbackNXModule()


from .._utils import logger
from ..base import (
    BaseGraphStorage,
    SingleCommunitySchema,
)
from ..prompt import GRAPH_FIELD_SEP


@dataclass
class NetworkXStorage(BaseGraphStorage):
    @staticmethod
    def load_nx_graph(file_name) -> nx.Graph:
        if os.path.exists(file_name):
            return nx.read_graphml(file_name)
        return None

    @staticmethod
    def write_nx_graph(graph: nx.Graph, file_name):
        logger.info(
            f"Writing graph with {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges"
        )
        nx.write_graphml(graph, file_name)

    @staticmethod
    def stable_largest_connected_component(graph: nx.Graph) -> nx.Graph:
        """Refer to https://github.com/microsoft/graphrag/index/graph/utils/stable_lcc.py
        Return the largest connected component of the graph, with nodes and edges sorted in a stable way.
        """
        from graspologic.utils import largest_connected_component

        graph = graph.copy()
        graph = cast(nx.Graph, largest_connected_component(graph))
        node_mapping = {node: html.unescape(node.upper().strip()) for node in graph.nodes()}  # type: ignore
        graph = nx.relabel_nodes(graph, node_mapping)
        return NetworkXStorage._stabilize_graph(graph)

    @staticmethod
    def _stabilize_graph(graph: nx.Graph) -> nx.Graph:
        """Refer to https://github.com/microsoft/graphrag/index/graph/utils/stable_lcc.py
        Ensure an undirected graph with the same relationships will always be read the same way.
        """
        fixed_graph = nx.DiGraph() if graph.is_directed() else nx.Graph()

        sorted_nodes = graph.nodes(data=True)
        sorted_nodes = sorted(sorted_nodes, key=lambda x: x[0])

        fixed_graph.add_nodes_from(sorted_nodes)
        edges = list(graph.edges(data=True))

        if not graph.is_directed():

            def _sort_source_target(edge):
                source, target, edge_data = edge
                if source > target:
                    temp = source
                    source = target
                    target = temp
                return source, target, edge_data

            edges = [_sort_source_target(edge) for edge in edges]

        def _get_edge_key(source: Any, target: Any) -> str:
            return f"{source} -> {target}"

        edges = sorted(edges, key=lambda x: _get_edge_key(x[0], x[1]))

        fixed_graph.add_edges_from(edges)
        return fixed_graph

    def __post_init__(self):
        self._graphml_xml_file = os.path.join(
            self.global_config["working_dir"], f"graph_{self.namespace}.graphml"
        )
        preloaded_graph = NetworkXStorage.load_nx_graph(self._graphml_xml_file)
        if preloaded_graph is not None:
            logger.info(
                f"Loaded graph from {self._graphml_xml_file} with {preloaded_graph.number_of_nodes()} nodes, {preloaded_graph.number_of_edges()} edges"
            )
        self._graph = preloaded_graph or nx.Graph()
        self._clustering_algorithms = {
            "leiden": self._leiden_clustering,
        }
        self._node_embed_algorithms = {
            "node2vec": self._node2vec_embed,
        }

    async def index_done_callback(self):
        NetworkXStorage.write_nx_graph(self._graph, self._graphml_xml_file)

    async def has_node(self, node_id: str) -> bool:
        return self._graph.has_node(node_id)

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        return self._graph.has_edge(source_node_id, target_node_id)

    async def get_node(self, node_id: str) -> Union[dict, None]:
        return self._graph.nodes.get(node_id)

    async def node_degree(self, node_id: str) -> int:
        # [numberchiffre]: node_id not part of graph returns `DegreeView({})` instead of 0
        return self._graph.degree(node_id) if self._graph.has_node(node_id) else 0

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        return (self._graph.degree(src_id) if self._graph.has_node(src_id) else 0) + (
            self._graph.degree(tgt_id) if self._graph.has_node(tgt_id) else 0
        )

    async def get_edge(
        self, source_node_id: str, target_node_id: str
    ) -> Union[dict, None]:
        edges_obj = getattr(self._graph, "_edges_view", None) or self._graph.edges
        return edges_obj.get((source_node_id, target_node_id))

    async def get_node_edges(self, source_node_id: str):
        if self._graph.has_node(source_node_id):
            return list(self._graph.edges(source_node_id))
        return None

    async def upsert_node(self, node_id: str, node_data: dict[str, str]):
        self._graph.add_node(node_id, **node_data)

    async def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]
    ):
        self._graph.add_edge(source_node_id, target_node_id, **edge_data)

    async def clustering(self, algorithm: str):
        if algorithm not in self._clustering_algorithms:
            raise ValueError(f"Clustering algorithm {algorithm} not supported")
        await self._clustering_algorithms[algorithm]()

    async def community_schema(self) -> dict[str, SingleCommunitySchema]:
        results = defaultdict(
            lambda: dict(
                level=None,
                title=None,
                edges=set(),
                nodes=set(),
                chunk_ids=set(),
                occurrence=0.0,
                sub_communities=[],
            )
        )
        max_num_ids = 0
        levels = defaultdict(set)
        for node_id, node_data in self._graph.nodes(data=True):
            if "clusters" not in node_data:
                continue
            clusters = json.loads(node_data["clusters"])
            this_node_edges = self._graph.edges(node_id)

            for cluster in clusters:
                level = cluster["level"]
                cluster_key = str(cluster["cluster"])
                levels[level].add(cluster_key)
                results[cluster_key]["level"] = level
                results[cluster_key]["title"] = f"Cluster {cluster_key}"
                results[cluster_key]["nodes"].add(node_id)
                results[cluster_key]["edges"].update(
                    [tuple(sorted(e)) for e in this_node_edges]
                )
                results[cluster_key]["chunk_ids"].update(
                    node_data["source_id"].split(GRAPH_FIELD_SEP)
                )
                max_num_ids = max(max_num_ids, len(results[cluster_key]["chunk_ids"]))

        ordered_levels = sorted(levels.keys())
        for i, curr_level in enumerate(ordered_levels[:-1]):
            next_level = ordered_levels[i + 1]
            this_level_comms = levels[curr_level]
            next_level_comms = levels[next_level]
            # compute the sub-communities by nodes intersection
            for comm in this_level_comms:
                results[comm]["sub_communities"] = [
                    c
                    for c in next_level_comms
                    if results[c]["nodes"].issubset(results[comm]["nodes"])
                ]

        for k, v in results.items():
            v["edges"] = list(v["edges"])
            v["edges"] = [list(e) for e in v["edges"]]
            v["nodes"] = list(v["nodes"])
            v["chunk_ids"] = list(v["chunk_ids"])
            v["occurrence"] = len(v["chunk_ids"]) / max_num_ids
        return dict(results)

    def _cluster_data_to_subgraphs(self, cluster_data: dict[str, list[dict[str, str]]]):
        for node_id, clusters in cluster_data.items():
            self._graph.nodes[node_id]["clusters"] = json.dumps(clusters)

    async def _leiden_clustering(self):
        from graspologic.partition import hierarchical_leiden

        graph = NetworkXStorage.stable_largest_connected_component(self._graph)
        community_mapping = hierarchical_leiden(
            graph,
            max_cluster_size=self.global_config["max_graph_cluster_size"],
            random_seed=self.global_config["graph_cluster_seed"],
        )

        node_communities: dict[str, list[dict[str, str]]] = defaultdict(list)
        __levels = defaultdict(set)
        for partition in community_mapping:
            level_key = partition.level
            cluster_id = partition.cluster
            node_communities[partition.node].append(
                {"level": level_key, "cluster": cluster_id}
            )
            __levels[level_key].add(cluster_id)
        node_communities = dict(node_communities)
        __levels = {k: len(v) for k, v in __levels.items()}
        logger.info(f"Each level has communities: {dict(__levels)}")
        self._cluster_data_to_subgraphs(node_communities)

    async def embed_nodes(self, algorithm: str) -> tuple[np.ndarray, list[str]]:
        if algorithm not in self._node_embed_algorithms:
            raise ValueError(f"Node embedding algorithm {algorithm} not supported")
        return await self._node_embed_algorithms[algorithm]()

    async def _node2vec_embed(self):
        from graspologic import embed

        embeddings, nodes = embed.node2vec_embed(
            self._graph,
            **self.global_config["node2vec_params"],
        )

        nodes_ids = [self._graph.nodes[node_id]["id"] for node_id in nodes]
        return embeddings, nodes_ids
