"""Neo4j-backed knowledge graph. Entities are :Entity nodes, relations are typed
edges (created via APOC), and chunks link to the entities they mention.

Every node is scoped by `corpus` — the tenant boundary. Uniqueness is enforced on
(corpus, key) / (corpus, id), never on the bare key: a bare-key constraint makes
MERGE match another tenant's node and silently re-tag it, which both corrupts
that tenant's graph and leaks it into this one.
"""

from __future__ import annotations

import json

from graphrag.core.types import Chunk, Entity, Relation, RetrievedChunk
from graphrag.storage.graph.base import GraphStore

# Full-text indexes are database-wide, so the corpus filter must run *after* the
# index call — which means the index's own LIMIT must over-fetch, or another
# tenant's matches consume the budget and this tenant sees nothing.
_OVERFETCH = 4


def _meta(raw) -> dict:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except (TypeError, ValueError):
        return {}


class Neo4jGraphStore(GraphStore):
    def __init__(self, driver, database: str, corpus: str) -> None:
        self._driver = driver
        self._db = database
        self._corpus = corpus

    def _run(self, query: str, **params):
        with self._driver.session(database=self._db) as session:
            return list(session.run(query, corpus=self._corpus, **params))

    # -- schema ---------------------------------------------------------------
    def setup(self) -> None:
        stmts = [
            # Legacy single-property constraints enforced *global* uniqueness,
            # which is exactly the cross-tenant merge bug. Drop before creating
            # the composite ones or the old rule keeps winning.
            "DROP CONSTRAINT entity_key IF EXISTS",
            "DROP CONSTRAINT chunk_id IF EXISTS",
            "CREATE CONSTRAINT entity_corpus_key IF NOT EXISTS "
            "FOR (e:Entity) REQUIRE (e.corpus, e.key) IS UNIQUE",
            "CREATE CONSTRAINT chunk_corpus_id IF NOT EXISTS "
            "FOR (c:Chunk) REQUIRE (c.corpus, c.id) IS UNIQUE",
            # delete_document matches on (corpus, source); without this it's a
            # full label scan on every delete.
            "CREATE INDEX chunk_corpus_source IF NOT EXISTS "
            "FOR (c:Chunk) ON (c.corpus, c.source)",
            "CREATE INDEX entity_corpus IF NOT EXISTS FOR (e:Entity) ON (e.corpus)",
            "CREATE INDEX community_corpus IF NOT EXISTS FOR (co:Community) ON (co.corpus)",
            "CREATE FULLTEXT INDEX chunk_fulltext IF NOT EXISTS "
            "FOR (c:Chunk) ON EACH [c.text]",
            "CREATE FULLTEXT INDEX entity_fulltext IF NOT EXISTS "
            "FOR (e:Entity) ON EACH [e.name, e.description]",
        ]
        with self._driver.session(database=self._db) as session:
            for stmt in stmts:
                session.run(stmt)

    # -- writes ---------------------------------------------------------------
    def add_entities(self, entities: list[Entity]) -> None:
        rows = [
            {"key": e.key, "name": e.name, "type": e.type, "description": e.description}
            for e in entities
        ]
        self._run(
            """
            UNWIND $rows AS row
            MERGE (e:Entity {corpus: $corpus, key: row.key})
            SET e.name = row.name, e.type = row.type,
                e.description = CASE WHEN size(row.description) > size(coalesce(e.description,''))
                                     THEN row.description ELSE e.description END
            """,
            rows=rows,
        )

    def add_relations(self, relations: list[Relation]) -> None:
        rows = [
            {
                "s": r.source.strip().lower(),
                "t": r.target.strip().lower(),
                "type": r.type,
                "desc": r.description,
            }
            for r in relations
        ]
        try:
            self._run(
                # Two MATCHes with a WITH between them, not `MATCH (a), (b)`:
                # both are index seeks on the unique (corpus, key), but the comma
                # form reads as a disconnected pattern and Neo4j logs a cartesian
                # product warning on every batch. Same plan, no noise.
                """
                UNWIND $rows AS row
                MATCH (a:Entity {corpus: $corpus, key: row.s})
                WITH row, a
                MATCH (b:Entity {corpus: $corpus, key: row.t})
                CALL apoc.merge.relationship(a, row.type, {}, {description: row.desc}, b, {})
                YIELD rel RETURN count(rel)
                """,
                rows=rows,
            )
        except Exception as exc:  # APOC missing -> generic typed edge fallback
            self._run(
                """
                UNWIND $rows AS row
                MATCH (a:Entity {corpus: $corpus, key: row.s})
                WITH row, a
                MATCH (b:Entity {corpus: $corpus, key: row.t})
                MERGE (a)-[r:RELATED_TO]->(b)
                SET r.type = row.type, r.description = row.desc
                """,
                rows=rows,
            )
            _ = exc

    def upsert_chunks(self, chunks: list[Chunk]) -> None:
        """Create/refresh chunk nodes without embeddings — used when vectors
        live in an external store but fulltext + MENTIONS still need the node."""
        rows = [
            {
                "id": c.id,
                "doc_id": c.doc_id,
                "text": c.text,
                "source": c.source,
                "metadata": json.dumps(c.metadata or {}),
            }
            for c in chunks
        ]
        self._run(
            """
            UNWIND $rows AS row
            MERGE (c:Chunk {corpus: $corpus, id: row.id})
            SET c.text = row.text, c.source = row.source, c.doc_id = row.doc_id,
                c.metadata = row.metadata
            """,
            rows=rows,
        )

    def link_chunk_sequence(self, chunks: list[Chunk]) -> None:
        """Stamp the document position and chain the chunks with :NEXT."""
        if not chunks:
            return
        ordered = sorted(chunks, key=lambda c: c.index)
        self._run(
            """
            UNWIND $rows AS row
            MATCH (c:Chunk {corpus: $corpus, id: row.id})
            SET c.index = row.index
            """,
            rows=[{"id": c.id, "index": c.index} for c in ordered],
        )
        pairs = [
            {"prev": a.id, "next": b.id}
            for a, b in zip(ordered, ordered[1:], strict=False)
        ]
        if not pairs:
            return
        self._run(
            # MERGE, not CREATE: re-ingesting a document that produced the same
            # chunk boundaries would otherwise stack duplicate edges and make
            # the window query return each neighbour several times.
            """
            UNWIND $pairs AS pair
            MATCH (a:Chunk {corpus: $corpus, id: pair.prev})
            MATCH (b:Chunk {corpus: $corpus, id: pair.next})
            MERGE (a)-[:NEXT]->(b)
            """,
            pairs=pairs,
        )

    # Bound on how far `chunk_window` will walk. Variable-length patterns must
    # carry a literal upper bound in Cypher, so this is interpolated — hence
    # `int()` and the clamp, never a raw caller value.
    _MAX_WINDOW = 5

    def chunk_window(
        self, chunk_ids: list[str], before: int = 1, after: int = 1
    ) -> list[RetrievedChunk]:
        ids = [i for i in dict.fromkeys(chunk_ids) if i]
        before = max(0, min(int(before), self._MAX_WINDOW))
        after = max(0, min(int(after), self._MAX_WINDOW))
        if not ids or (before == 0 and after == 0):
            return []

        # One query per direction, unioned here. A single query with two
        # OPTIONAL MATCHes would multiply the predecessors by the successors
        # before deduplicating, and the subquery forms that avoid that are
        # newer than the Neo4j this deploys against.
        found: dict[str, RetrievedChunk] = {}
        order: dict[str, tuple[str, int]] = {}
        directions = []
        if before:
            directions.append(f"(n:Chunk {{corpus: $corpus}})-[:NEXT*1..{before}]->(c)")
        if after:
            directions.append(f"(c)-[:NEXT*1..{after}]->(n:Chunk {{corpus: $corpus}})")

        for pattern in directions:
            for r in self._run(
                f"""
                MATCH (c:Chunk {{corpus: $corpus}}) WHERE c.id IN $ids
                MATCH {pattern}
                WHERE NOT n.id IN $ids
                RETURN DISTINCT n.id AS id, n.text AS text, n.source AS source,
                       n.metadata AS metadata, n.doc_id AS doc_id,
                       coalesce(n.index, 0) AS index
                """,
                ids=ids,
            ):
                found[r["id"]] = RetrievedChunk(
                    chunk_id=r["id"], text=r["text"], source=r["source"],
                    # Neighbours were reached structurally, not scored by
                    # retrieval. Inventing a relevance score would let them
                    # outrank genuinely retrieved chunks in fusion.
                    score=0.0, retriever="window", metadata=_meta(r["metadata"]),
                )
                order[r["id"]] = (r["doc_id"] or "", int(r["index"] or 0))

        # Reading order, so the caller can splice a window back together.
        return [found[k] for k in sorted(found, key=lambda k: order[k])]

    def link_chunk_entities(self, chunk_id: str, entity_keys: list[str]) -> None:
        if not entity_keys:
            return
        self._run(
            """
            MATCH (c:Chunk {corpus: $corpus, id: $chunk_id})
            UNWIND $keys AS key
            MATCH (e:Entity {corpus: $corpus, key: key})
            MERGE (c)-[:MENTIONS]->(e)
            """,
            chunk_id=chunk_id,
            keys=entity_keys,
        )

    def delete_document(self, source: str) -> int:
        rows = self._run(
            """
            MATCH (c:Chunk {corpus: $corpus, source: $source})
            DETACH DELETE c
            RETURN count(c) AS removed
            """,
            source=source,
        )
        removed = rows[0]["removed"] if rows else 0
        # Entities are only ever evidence from a chunk; once nothing in *this
        # corpus* mentions them they'd linger in graph expansion as dead weight.
        self._run(
            """
            MATCH (e:Entity {corpus: $corpus})
            WHERE NOT EXISTS { (:Chunk {corpus: $corpus})-[:MENTIONS]->(e) }
            DETACH DELETE e
            """
        )
        # Community summaries may quote the deleted document; drop them rather
        # than keep serving its content. The next ingest rebuilds them.
        if removed:
            self._run("MATCH (co:Community {corpus: $corpus}) DETACH DELETE co")
        return int(removed)

    # -- reads (agent tools) --------------------------------------------------
    def neighbors(self, entity_name: str, hops: int = 2) -> str:
        hops = max(1, min(int(hops), 4))  # bound + validate before interpolation
        rows = self._run(
            f"""
            MATCH path = (e:Entity {{corpus: $corpus, key: $key}})-[*1..{hops}]-(:Entity)
            WHERE all(n IN nodes(path) WHERE n.corpus = $corpus)
            WITH relationships(path) AS rels
            UNWIND rels AS r
            RETURN DISTINCT startNode(r).name AS source, type(r) AS type,
                   endNode(r).name AS target LIMIT 100
            """,
            key=entity_name.strip().lower(),
        )
        if not rows:
            return f"No graph connections found for '{entity_name}'."
        lines = [f"- {r['source']} —[{r['type']}]→ {r['target']}" for r in rows]
        return f"Relations around '{entity_name}':\n" + "\n".join(lines)

    def get_entity(self, name: str) -> dict:
        rows = self._run(
            """
            MATCH (e:Entity {corpus: $corpus, key: $key})
            OPTIONAL MATCH (e)-[r]-(n:Entity {corpus: $corpus})
            RETURN e.name AS name, e.type AS type, e.description AS description,
                   collect(DISTINCT n.name)[0..25] AS connected
            """,
            key=name.strip().lower(),
        )
        if not rows:
            return {}
        row = rows[0]
        return {
            "name": row["name"],
            "type": row["type"],
            "description": row["description"],
            "connected": row["connected"],
        }

    def fulltext_entities(self, query: str, k: int = 5) -> list[str]:
        rows = self._run(
            """
            CALL db.index.fulltext.queryNodes('entity_fulltext', $q, {limit: $fetch})
            YIELD node WHERE node.corpus = $corpus
            RETURN node.name AS name LIMIT $k
            """,
            q=_escape(query),
            k=k,
            fetch=k * _OVERFETCH,
        )
        return [r["name"] for r in rows]

    def fulltext_chunks(self, query: str, k: int = 8) -> list[RetrievedChunk]:
        rows = self._run(
            """
            CALL db.index.fulltext.queryNodes('chunk_fulltext', $q, {limit: $fetch})
            YIELD node, score WHERE node.corpus = $corpus
            RETURN node.id AS id, node.text AS text, node.source AS source,
                   node.metadata AS metadata, score
            LIMIT $k
            """,
            q=_escape(query),
            k=k,
            fetch=k * _OVERFETCH,
        )
        return [
            RetrievedChunk(
                chunk_id=r["id"], text=r["text"], source=r["source"],
                score=float(r["score"]), retriever="fulltext", metadata=_meta(r["metadata"]),
            )
            for r in rows
        ]

    def chunks_for_entities(self, entity_names: list[str], limit: int = 12) -> list[RetrievedChunk]:
        keys = [n.strip().lower() for n in entity_names]
        rows = self._run(
            """
            MATCH (c:Chunk)-[:MENTIONS]->(e:Entity {corpus: $corpus})
            WHERE e.key IN $keys AND c.corpus = $corpus
            WITH c, count(DISTINCT e) AS hits
            RETURN c.id AS id, c.text AS text, c.source AS source,
                   c.metadata AS metadata, hits
            ORDER BY hits DESC LIMIT $limit
            """,
            keys=keys,
            limit=limit,
        )
        return [
            RetrievedChunk(
                chunk_id=r["id"], text=r["text"], source=r["source"],
                score=float(r["hits"]), retriever="graph", metadata=_meta(r["metadata"]),
            )
            for r in rows
        ]

    def expand_chunks(
        self, entity_names: list[str], hops: int = 2, limit: int = 12
    ) -> list[RetrievedChunk]:
        """Chunks mentioning the seed entities *or their graph neighborhood*,
        scored by how close (in hops) their entities sit to a seed. This is the
        traversal that makes graph-augmented retrieval follow relationships
        instead of just re-finding the seeds."""
        hops = max(1, min(int(hops), 4))
        keys = [n.strip().lower() for n in entity_names]
        rows = self._run(
            f"""
            UNWIND $keys AS k
            MATCH (s:Entity {{corpus: $corpus, key: k}})
            MATCH path = (s)-[*0..{hops}]-(e:Entity)
            WHERE all(n IN nodes(path) WHERE n.corpus = $corpus)
            WITH e, min(length(path)) AS dist
            LIMIT 500
            MATCH (c:Chunk {{corpus: $corpus}})-[:MENTIONS]->(e)
            WITH c, sum(1.0 / (1 + dist)) AS score
            RETURN c.id AS id, c.text AS text, c.source AS source,
                   c.metadata AS metadata, score
            ORDER BY score DESC LIMIT $limit
            """,
            keys=keys,
            limit=limit,
        )
        return [
            RetrievedChunk(
                chunk_id=r["id"], text=r["text"], source=r["source"],
                score=float(r["score"]), retriever="graph", metadata=_meta(r["metadata"]),
            )
            for r in rows
        ]

    # -- entity resolution ----------------------------------------------------
    def all_entities(self, limit: int = 5000) -> list[dict]:
        rows = self._run(
            """
            MATCH (e:Entity {corpus: $corpus})
            RETURN e.key AS key, e.name AS name, e.type AS type
            LIMIT $limit
            """,
            limit=limit,
        )
        return [dict(r) for r in rows]

    def merge_entities(self, winner_key: str, loser_keys: list[str]) -> None:
        """Fold duplicate entities into one node, preserving edges and recording
        the losers' names as aliases. APOC does it losslessly; the fallback keeps
        MENTIONS and collapses typed edges into RELATED_TO."""
        losers = [k for k in loser_keys if k and k != winner_key]
        if not losers:
            return
        self._run(
            """
            MATCH (w:Entity {corpus: $corpus, key: $winner})
            UNWIND $losers AS lk
            MATCH (l:Entity {corpus: $corpus, key: lk})
            SET w.aliases = CASE WHEN l.name IN coalesce(w.aliases, [])
                                 THEN w.aliases ELSE coalesce(w.aliases, []) + l.name END
            """,
            winner=winner_key,
            losers=losers,
        )
        try:
            self._run(
                """
                MATCH (w:Entity {corpus: $corpus, key: $winner})
                UNWIND $losers AS lk
                MATCH (l:Entity {corpus: $corpus, key: lk})
                WITH w, collect(l) AS ls
                CALL apoc.refactor.mergeNodes([w] + ls,
                     {properties: "discard", mergeRels: true})
                YIELD node RETURN count(node)
                """,
                winner=winner_key,
                losers=losers,
            )
        except Exception:  # no APOC: redirect mentions, then generic edges
            self._run(
                """
                MATCH (w:Entity {corpus: $corpus, key: $winner})
                UNWIND $losers AS lk
                MATCH (l:Entity {corpus: $corpus, key: lk})
                OPTIONAL MATCH (c:Chunk {corpus: $corpus})-[:MENTIONS]->(l)
                FOREACH (_ IN CASE WHEN c IS NULL THEN [] ELSE [1] END |
                    MERGE (c)-[:MENTIONS]->(w))
                WITH DISTINCT w, l
                OPTIONAL MATCH (l)-[r]-(o:Entity {corpus: $corpus})
                WHERE o <> w
                FOREACH (_ IN CASE WHEN o IS NULL THEN [] ELSE [1] END |
                    MERGE (w)-[nr:RELATED_TO]->(o)
                    SET nr.type = type(r), nr.description = coalesce(r.description, ''))
                WITH DISTINCT l
                DETACH DELETE l
                """,
                winner=winner_key,
                losers=losers,
            )

    # -- communities (global search) ------------------------------------------
    def entity_edges(self, limit: int = 20000) -> list[tuple[str, str]]:
        rows = self._run(
            """
            MATCH (a:Entity {corpus: $corpus})-[]-(b:Entity {corpus: $corpus})
            WHERE a.key < b.key
            RETURN DISTINCT a.key AS a, b.key AS b LIMIT $limit
            """,
            limit=limit,
        )
        return [(r["a"], r["b"]) for r in rows]

    def replace_communities(self, communities: list[dict]) -> None:
        """Overwrite this corpus's community summaries (id, summary, entities,
        size, embedding)."""
        self._run("MATCH (co:Community {corpus: $corpus}) DETACH DELETE co")
        if not communities:
            return
        self._run(
            """
            UNWIND $rows AS row
            CREATE (co:Community {corpus: $corpus, id: row.id})
            SET co.summary = row.summary, co.entities = row.entities,
                co.size = row.size, co.embedding = row.embedding
            """,
            rows=communities,
        )

    def communities(self) -> list[dict]:
        rows = self._run(
            """
            MATCH (co:Community {corpus: $corpus})
            RETURN co.id AS id, co.summary AS summary, co.entities AS entities,
                   co.size AS size, co.embedding AS embedding
            ORDER BY co.size DESC
            """
        )
        return [dict(r) for r in rows]

    # -- inspection (admin) ---------------------------------------------------
    def stats(self) -> dict[str, int]:
        """Corpus-level counts for the admin panel.

        One query with subqueries rather than four round trips — this is called
        per user on a page that lists many of them.
        """
        rows = self._run(
            """
            CALL { MATCH (c:Chunk {corpus: $corpus}) RETURN count(c) AS chunks }
            CALL { MATCH (e:Entity {corpus: $corpus}) RETURN count(e) AS entities }
            CALL { MATCH (co:Community {corpus: $corpus}) RETURN count(co) AS communities }
            CALL {
                MATCH (a:Entity {corpus: $corpus})-[r]-(b:Entity {corpus: $corpus})
                WHERE a.key < b.key
                RETURN count(r) AS relations
            }
            CALL {
                MATCH (c:Chunk {corpus: $corpus})
                RETURN count(DISTINCT c.source) AS documents
            }
            RETURN chunks, entities, communities, relations, documents
            """
        )
        if not rows:
            return {
                "chunks": 0, "entities": 0, "communities": 0,
                "relations": 0, "documents": 0,
            }
        return {k: int(v or 0) for k, v in dict(rows[0]).items()}

    def sample_subgraph(self, limit: int = 100) -> dict:
        """A slice of the entity graph for visualization.

        Ordered by degree so the sample shows the corpus's hubs — an arbitrary
        `LIMIT` would usually return disconnected leaves and draw as dust.
        """
        limit = max(1, min(int(limit), 500))
        nodes = self._run(
            """
            MATCH (e:Entity {corpus: $corpus})
            OPTIONAL MATCH (e)-[r]-(:Entity {corpus: $corpus})
            WITH e, count(r) AS degree
            ORDER BY degree DESC
            LIMIT $limit
            RETURN e.key AS key, e.name AS name, e.type AS type, degree
            """,
            limit=limit,
        )
        keys = [r["key"] for r in nodes]
        if not keys:
            return {"nodes": [], "edges": []}
        edges = self._run(
            """
            MATCH (a:Entity {corpus: $corpus})-[r]-(b:Entity {corpus: $corpus})
            WHERE a.key IN $keys AND b.key IN $keys AND a.key < b.key
            RETURN DISTINCT a.key AS source, b.key AS target, type(r) AS type
            LIMIT 1000
            """,
            keys=keys,
        )
        return {
            "nodes": [dict(r) for r in nodes],
            "edges": [dict(r) for r in edges],
        }

    def purge_corpus(self) -> int:
        """Delete everything belonging to this corpus. Used when an admin
        removes a user; the tenant's storage should not outlive the account."""
        rows = self._run(
            """
            MATCH (n {corpus: $corpus})
            WITH n LIMIT 100000
            DETACH DELETE n
            RETURN count(n) AS removed
            """
        )
        return int(rows[0]["removed"]) if rows else 0


def _escape(query: str) -> str:
    """Escape Lucene special chars so a raw user query can't break the syntax."""
    specials = r'+-&|!(){}[]^"~*?:\/'
    out = []
    for ch in query:
        if ch in specials:
            out.append("\\" + ch)
        else:
            out.append(ch)
    escaped = "".join(out).strip()
    return escaped or "*"
