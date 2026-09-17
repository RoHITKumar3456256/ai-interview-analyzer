from typing import List, Dict, Any
from backend.models import Transcript, DocumentChunk, ChunkMetadata

class MetadataChunker:
    """
    Processes transcripts into structured chunks with rich metadata injection 
    for accurate evidence retrieval, exact quote extraction, and zero-hallucination citations.
    """
    def __init__(self, max_chunk_words: int = 150):
        self.max_chunk_words = max_chunk_words

    def chunk_transcript(self, transcript: Transcript) -> List[DocumentChunk]:
        chunks: List[DocumentChunk] = []
        chunk_idx = 0
        
        for u_idx, utterance in enumerate(transcript.utterances):
            # Clean text
            clean_text = utterance.text.strip()
            if not clean_text:
                continue
                
            speaker = utterance.speaker
            start_t = utterance.start_time
            end_t = utterance.end_time
            timestamp_disp = f"[{start_t} - {end_t}]"
            
            # If utterance is of reasonable length, create 1 chunk directly
            words = clean_text.split()
            if len(words) <= self.max_chunk_words:
                meta = ChunkMetadata(
                    transcript_id=transcript.id,
                    expert_name=transcript.expert_name,
                    expert_title=transcript.expert_title,
                    organization=transcript.organization,
                    speaker=speaker,
                    start_time=start_t,
                    end_time=end_t,
                    timestamp_display=timestamp_disp,
                    chunk_index=chunk_idx,
                    topic_hint=transcript.title
                )
                chunk_id = f"{transcript.id}_u{u_idx}_c0"
                chunks.append(DocumentChunk(chunk_id=chunk_id, text=clean_text, metadata=meta))
                chunk_idx += 1
            else:
                # Segment long utterance into overlapping sub-chunks while maintaining metadata
                step = self.max_chunk_words - 25
                sub_idx = 0
                for start in range(0, len(words), step):
                    sub_words = words[start:start + self.max_chunk_words]
                    sub_text = " ".join(sub_words)
                    
                    meta = ChunkMetadata(
                        transcript_id=transcript.id,
                        expert_name=transcript.expert_name,
                        expert_title=transcript.expert_title,
                        organization=transcript.organization,
                        speaker=speaker,
                        start_time=start_t,
                        end_time=end_t,
                        timestamp_display=timestamp_disp,
                        chunk_index=chunk_idx,
                        topic_hint=transcript.title
                    )
                    chunk_id = f"{transcript.id}_u{u_idx}_c{sub_idx}"
                    chunks.append(DocumentChunk(chunk_id=chunk_id, text=sub_text, metadata=meta))
                    chunk_idx += 1
                    sub_idx += 1
                    if start + self.max_chunk_words >= len(words):
                        break
                        
        return chunks

    def chunk_all(self, transcripts: List[Transcript]) -> List[DocumentChunk]:
        all_chunks: List[DocumentChunk] = []
        for t in transcripts:
            all_chunks.extend(self.chunk_transcript(t))
        return all_chunks

chunker = MetadataChunker()
