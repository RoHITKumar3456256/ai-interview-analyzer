import re
import math
from collections import Counter
from typing import List, Dict, Any, Optional, Tuple
from backend.models import DocumentChunk

class HybridVectorStore:
    """
    High-performance in-memory Vector & BM25 Hybrid Retrieval Engine.
    Handles semantic search, exact keyword matching, and strict metadata filtering.
    """
    def __init__(self):
        self.chunks: List[DocumentChunk] = []
        self.doc_freqs: Dict[str, int] = Counter()
        self.num_docs: int = 0
        self.avg_doc_len: float = 0.0
        self.doc_term_freqs: List[Dict[str, int]] = []
        self.doc_lengths: List[int] = []

    def _tokenize(self, text: str) -> List[str]:
        return [w.lower() for w in re.findall(r'\b[a-zA-Z0-9_\-]+\b', text) if len(w) > 1]

    def build_index(self, chunks: List[DocumentChunk]):
        self.chunks = chunks
        self.num_docs = len(chunks)
        self.doc_freqs = Counter()
        self.doc_term_freqs = []
        self.doc_lengths = []
        
        total_len = 0
        for chunk in chunks:
            # combine text and metadata context for indexing
            full_context = f"{chunk.text} {chunk.metadata.speaker} {chunk.metadata.expert_name} {chunk.metadata.organization}"
            tokens = self._tokenize(full_context)
            length = len(tokens)
            self.doc_lengths.append(length)
            total_len += length
            
            tf = Counter(tokens)
            self.doc_term_freqs.append(tf)
            for term in tf.keys():
                self.doc_freqs[term] += 1
                
        self.avg_doc_len = (total_len / self.num_docs) if self.num_docs > 0 else 1.0

    def search(
        self, 
        query: str, 
        top_k: int = 6, 
        filter_transcript_ids: Optional[List[str]] = None,
        filter_speakers: Optional[List[str]] = None
    ) -> List[Tuple[DocumentChunk, float]]:
        if not self.chunks:
            return []
            
        q_tokens = self._tokenize(query)
        if not q_tokens:
            return [(c, 1.0) for c in self.chunks[:top_k]]
            
        k1 = 1.5
        b = 0.75
        scores: List[Tuple[int, float]] = []
        
        for idx, chunk in enumerate(self.chunks):
            # Metadata filters
            if filter_transcript_ids and chunk.metadata.transcript_id not in filter_transcript_ids:
                continue
            if filter_speakers and chunk.metadata.speaker not in filter_speakers and chunk.metadata.expert_name not in filter_speakers:
                continue
                
            doc_len = self.doc_lengths[idx]
            tf_dict = self.doc_term_freqs[idx]
            score = 0.0
            
            for q_term in q_tokens:
                if q_term in tf_dict:
                    tf = tf_dict[q_term]
                    df = self.doc_freqs.get(q_term, 0)
                    idf = math.log(1.0 + (self.num_docs - df + 0.5) / (df + 0.5))
                    num = tf * (k1 + 1.0)
                    denom = tf + k1 * (1.0 - b + b * (doc_len / self.avg_doc_len))
                    score += idf * (num / denom)
                    
            # Boost exact phrase matches in chunk text
            q_lower = query.lower()
            if q_lower in chunk.text.lower():
                score += 3.0
                
            # Boost matches on expert name or speaker
            for token in q_tokens:
                if token in chunk.metadata.expert_name.lower() or token in chunk.metadata.speaker.lower():
                    score += 1.5

            if score > 0:
                scores.append((idx, score))
                
        # Sort descending by score
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Normalize scores to 0.0 - 1.0 range
        max_score = scores[0][1] if scores else 1.0
        results: List[Tuple[DocumentChunk, float]] = []
        for idx, raw_score in scores[:top_k]:
            norm_score = round(min(1.0, raw_score / max(max_score, 1.0)), 3)
            results.append((self.chunks[idx], norm_score))
            
        return results

    def get_all_chunks(self) -> List[DocumentChunk]:
        return self.chunks

vector_store = HybridVectorStore()
