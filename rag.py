import math
import re
import requests
from typing import List, Dict, Any, Tuple

class HybridRAG:
    """
    Hybrid RAG pipeline combining keyword-based and semantic retrieval.
    - Expects a processor object compatible with `EbookProcessor` from `input.py`:
      processor.collection (Chroma collection) and processor.embedding_model (SentenceTransformer).
    - Uses Ollama (gemma3:12B) for answer generation.
    """

    def __init__(
        self,
        processor: Any,
        ollama_url: str = "http://localhost:11434",
        model: str = "gemma3:12B",
    ):
        self.processor = processor
        self.collection = processor.collection
        self.embedder = processor.embedding_model
        self.ollama_url = ollama_url.rstrip("/")
        self.model = model

    def _tokenize(self, text: str) -> List[str]:
        tokens = re.findall(r"[A-Za-z0-9]+", (text or "").lower())
        return [t for t in tokens if len(t) > 1]

    def _bm25_scores(
        self,
        query_terms: List[str],
        docs_tokens: List[List[str]],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> List[float]:
        if not docs_tokens or not query_terms:
            return [0.0] * len(docs_tokens)

        N = len(docs_tokens)
        avgdl = sum(len(d) for d in docs_tokens) / max(N, 1)

        # Document frequencies for query terms
        dfs: Dict[str, int] = {}
        for t in set(query_terms):
            df = sum(1 for dt in docs_tokens if t in dt)
            dfs[t] = df

        # Precompute IDF
        idf: Dict[str, float] = {}
        for t, df in dfs.items():
            # Okapi BM25 idf
            idf[t] = math.log((N - df + 0.5) / (df + 0.5) + 1.0)

        scores: List[float] = []
        for dtokens in docs_tokens:
            score = 0.0
            dl = len(dtokens) or 1
            tf_counts: Dict[str, int] = {}
            for t in query_terms:
                # term frequency
                if t not in tf_counts:
                    tf_counts[t] = dtokens.count(t)

            for t in query_terms:
                tf = tf_counts.get(t, 0)
                if tf == 0:
                    continue
                denom = tf + k1 * (1 - b + b * (dl / avgdl))
                score += idf.get(t, 0.0) * ((tf * (k1 + 1)) / denom)
            scores.append(score)
        return scores

    def retrieve_semantic(self, query: str, n_results: int = 10) -> List[Dict[str, Any]]:
        q_emb = self.embedder.encode([query]).tolist()
        results = self.collection.query(query_embeddings=q_emb, n_results=n_results)
        docs = results.get("documents", [[]])[0] or []
        metas = results.get("metadatas", [[]])[0] or []
        ids = results.get("ids", [[]])[0] or []
        dists = results.get("distances", [[]])[0] or []

        items: List[Dict[str, Any]] = []
        for i in range(len(docs)):
            # Higher score = more relevant. Convert distance to similarity if available.
            if dists and i < len(dists):
                sim = 1.0 / (1.0 + float(dists[i]))
            else:
                sim = 1.0 - (i / max(1, len(docs)))  # rank-based fallback
            items.append(
                {
                    "id": ids[i] if i < len(ids) else None,
                    "text": docs[i],
                    "metadata": metas[i] if i < len(metas) else {},
                    "score": float(sim),
                    "source": "semantic",
                }
            )
        return items

    def retrieve_keyword(
        self,
        query: str,
        n_results: int = 10,
        max_docs: int = 3000,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        q_terms = self._tokenize(query)
        if not q_terms:
            return []

        total = self.collection.count()
        limit = min(max_docs, total - offset if total > offset else 0)
        if limit <= 0:
            return []

        got = self.collection.get(
            limit=limit,
            offset=offset,
            include=["documents", "metadatas"],
        )
        docs = got.get("documents", []) or []
        metas = got.get("metadatas", []) or []
        ids = got.get("ids", []) or []

        docs_tokens = [self._tokenize(d or "") for d in docs]
        scores = self._bm25_scores(q_terms, docs_tokens)

        ranked: List[Tuple[int, float]] = sorted(
            [(i, s) for i, s in enumerate(scores)], key=lambda x: x[1], reverse=True
        )

        out: List[Dict[str, Any]] = []
        for i, s in ranked[:n_results]:
            out.append(
                {
                    "id": ids[i] if i < len(ids) else None,
                    "text": docs[i],
                    "metadata": metas[i] if i < len(metas) else {},
                    "score": float(s),
                    "source": "keyword",
                }
            )
        return out

    def _min_max_norm(self, vals: List[float]) -> List[float]:
        if not vals:
            return []
        vmin, vmax = min(vals), max(vals)
        if math.isclose(vmin, vmax):
            return [0.5] * len(vals)
        return [(v - vmin) / (vmax - vmin) for v in vals]

    def hybrid_retrieve(
        self,
        query: str,
        n_semantic: int = 12,
        n_keyword: int = 12,
        alpha: float = 0.5,
        top_k: int = 10,
        keyword_max_docs: int = 3000,
    ) -> List[Dict[str, Any]]:
        sem = self.retrieve_semantic(query, n_results=n_semantic)
        kw = self.retrieve_keyword(query, n_results=n_keyword, max_docs=keyword_max_docs)

        # Normalize scores within each modality
        sem_scores = self._min_max_norm([x["score"] for x in sem]) if sem else []
        kw_scores = self._min_max_norm([x["score"] for x in kw]) if kw else []
        for i, s in enumerate(sem):
            s["norm_score"] = sem_scores[i]
        for i, s in enumerate(kw):
            s["norm_score"] = kw_scores[i]

        # Merge by id or text fallback
        merged: Dict[str, Dict[str, Any]] = {}

        def _key(item: Dict[str, Any]) -> str:
            if item.get("id"):
                return f"id::{item['id']}"
            # fallback on text hash if id missing
            return f"tx::{hash(item.get('text',''))}"

        for item in sem:
            merged[_key(item)] = {**item, "sem_score": item["norm_score"], "kw_score": 0.0}

        for item in kw:
            k = _key(item)
            if k in merged:
                merged[k]["kw_score"] = item["norm_score"]
                # Keep the longer text/metadata if different variants
                if len(item.get("text", "")) > len(merged[k].get("text", "")):
                    merged[k]["text"] = item.get("text", "")
                    merged[k]["metadata"] = item.get("metadata", {})
            else:
                merged[k] = {**item, "sem_score": 0.0, "kw_score": item["norm_score"]}

        # Final hybrid score
        out = []
        for v in merged.values():
            final_score = alpha * v.get("sem_score", 0.0) + (1 - alpha) * v.get("kw_score", 0.0)
            v["score"] = float(final_score)
            out.append(v)

        out.sort(key=lambda x: x["score"], reverse=True)
        return out[:top_k]

    def _build_context(self, items: List[Dict[str, Any]], max_chars: int = 4000) -> str:
        parts: List[str] = []
        total = 0
        for i, it in enumerate(items, 1):
            meta = it.get("metadata", {}) or {}
            src = meta.get("filename", "unknown")
            idx = meta.get("chunk_index", -1)
            header = f"[{i}] Source: {src} | Chunk: {idx}"
            body = it.get("text", "")
            chunk = f"{header}\n{body}\n"
            if total + len(chunk) > max_chars and parts:
                break
            parts.append(chunk)
            total += len(chunk)
        return "\n---\n".join(parts)

    def generate_answer(
        self,
        query: str,
        contexts: List[Dict[str, Any]],
        temperature: float = 0.2,
        max_context_chars: int = 4000,
    ) -> Dict[str, Any]:
        context_text = self._build_context(contexts, max_chars=max_context_chars)

        prompt = (
            "You are an audiobook reader and an expert in literature. You are given a question and a context of an e-book. You need to answer the question based on the context.\n"
            "If the answer is not present in the context, say you don't know.\n\n"
            f"Context:\n{context_text}\n\n"
            f"Question: {query}\n\n"
            "Answer:"
        )

        resp = requests.post(
            f"{self.ollama_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": temperature},
            },
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        answer = data.get("response", "")

        return {
            "answer": answer,
            "model": self.model,
            "contexts": contexts,
        }

    def answer(
        self,
        query: str,
        n_semantic: int = 12,
        n_keyword: int = 12,
        alpha: float = 0.6,
        top_k: int = 8,
        keyword_max_docs: int = 3000,
        temperature: float = 0.2,
        max_context_chars: int = 4000,
    ) -> Dict[str, Any]:
        retrieved = self.hybrid_retrieve(
            query=query,
            n_semantic=n_semantic,
            n_keyword=n_keyword,
            alpha=alpha,
            top_k=top_k,
            keyword_max_docs=keyword_max_docs,
        )
        reranked = self.rerank(query, retrieved, top_k=top_k)
        return self.generate_answer(
            query=query,
            contexts=reranked,
            temperature=temperature,
            max_context_chars=max_context_chars,
        )

    def rerank(self, query: str, items: List[Dict[str, Any]], top_k: int = 8) -> List[Dict[str, Any]]:
        """
        Second-stage reranking with a cross-encoder. Improves precision of the final top_k.
        Lazily loads 'cross-encoder/ms-marco-MiniLM-L-6-v2'. Falls back to original order if unavailable.
        """
        if not items:
            return []
        try:
            from sentence_transformers import CrossEncoder
        except Exception:
            # If CrossEncoder isn't available, return the initial ranking truncated to top_k
            return items[:top_k]

        if not hasattr(self, "_cross_encoder") or getattr(self, "_cross_encoder") is None:
            self._cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

        pairs = [(query, it.get("text", "")) for it in items]
        scores = self._cross_encoder.predict(pairs)
        for i, s in enumerate(scores):
            items[i]["rerank_score"] = float(s)

        items.sort(key=lambda x: x.get("rerank_score", x.get("score", 0.0)), reverse=True)
        return items[:top_k]
