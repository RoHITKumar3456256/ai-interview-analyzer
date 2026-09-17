import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.transcript_loader import transcript_loader
from backend.chunker import chunker
from backend.vector_store import vector_store
from backend.rag_engine import rag_engine

def run_tests():
    print("=" * 60)
    print("  RUNNING AI INTERVIEW ANALYZER TEST SUITE")
    print("=" * 60)
    
    # 1. Test Transcript Loader
    print("\n[Test 1] Testing Transcript Loading...")
    transcripts = transcript_loader.get_transcripts()
    assert len(transcripts) == 3, f"Expected 3 transcripts, got {len(transcripts)}"
    print(f" PASS: Loaded {len(transcripts)} transcripts:")
    for t in transcripts:
        print(f"   - {t.id}: {t.expert_name} ({t.organization}) | {len(t.utterances)} utterances")
        
    # 2. Test Guide Loading
    print("\n[Test 2] Testing Interview Guide Loading...")
    guide = transcript_loader.get_guide()
    questions = guide.get("questions", [])
    assert len(questions) >= 5, f"Expected at least 5 questions, got {len(questions)}"
    print(f" PASS: Loaded {len(questions)} guide questions for '{guide.get('guide_title')}'.")

    # 3. Test Metadata Chunking
    print("\n[Test 3] Testing Metadata Chunking Pipeline...")
    chunks = chunker.chunk_all(transcripts)
    assert len(chunks) > 0, "No chunks created!"
    print(f" PASS: Created {len(chunks)} metadata-injected chunks.")
    sample = chunks[0]
    print(f"   Sample chunk ID: {sample.chunk_id}")
    print(f"   Metadata: speaker='{sample.metadata.speaker}', timestamp='{sample.metadata.timestamp_display}'")
    assert sample.metadata.transcript_id is not None
    assert sample.metadata.timestamp_display.startswith("[")

    # 4. Test Vector Store & Hybrid Search
    print("\n[Test 4] Testing Vector Search Indexing & Querying...")
    vector_store.build_index(chunks)
    search_res = vector_store.search("ROI timeline months", top_k=3)
    assert len(search_res) > 0, "Search returned 0 results"
    print(f" PASS: Vector search returned {len(search_res)} chunks for 'ROI timeline months'")
    for c, score in search_res:
        print(f"   - [{score}] {c.metadata.speaker} ({c.metadata.timestamp_display}): {c.text[:80]}...")

    # 5. Test Guide Automator (Feature 1 & Feature 2)
    print("\n[Test 5] Testing Feature 1 & 2: Guide Answers & Evidence Engine...")
    guide_response = rag_engine.generate_all_guide_answers()
    assert len(guide_response.answers) == len(questions)
    print(f" PASS: Generated {len(guide_response.answers)} synthesized answers with exact quotes.")
    q1 = guide_response.answers[0]
    print(f"   Question: {q1.question_text}")
    print(f"   Evidence Quotes Found: {len(q1.evidence_quotes)}")
    for eq in q1.evidence_quotes[:2]:
        print(f"     * \"{eq.quote[:60]}...\" — {eq.speaker} {eq.timestamp}")

    # 6. Test Consensus & Disagreement Map (Feature 3)
    print("\n[Test 6] Testing Feature 3: Consensus & Disagreement Map...")
    comparison = rag_engine.generate_consensus_and_disagreements()
    assert len(comparison.consensus_themes) > 0, "No consensus themes generated"
    assert len(comparison.disagreements) > 0, "No disagreements generated"
    print(f" PASS: Consensus Themes ({len(comparison.consensus_themes)}), Disagreements ({len(comparison.disagreements)})")
    for c in comparison.consensus_themes:
        print(f"   [Consensus] {c.theme} (Experts: {', '.join(c.supporting_experts)})")
    for d in comparison.disagreements:
        print(f"   [Disagreement] {d.topic}")

    # 7. Test Global Q&A Chatbot (Feature 4)
    print("\n[Test 7] Testing Feature 4: Global Q&A Chatbot...")
    chat_res = rag_engine.chat_query("What is Sarah Lin's view on build vs buy?")
    assert chat_res.is_grounded, "Chat response should be grounded"
    print(f" PASS: Chatbot returned grounded answer with {len(chat_res.evidence_quotes)} quotes.")
    print(f"   Speakers cited: {', '.join(chat_res.cited_speakers)}")
    
    # 8. Test Zero Hallucination guardrail for out-of-domain query
    print("\n[Test 8] Testing Zero Hallucination Guardrail for unrelated query...")
    hallucination_check = rag_engine.chat_query("What is the recipe for baking chocolate brownies?")
    print(f"   Response: {hallucination_check.answer}")
    assert "Not mentioned in transcripts" in hallucination_check.answer, "Should output 'Not mentioned in transcripts'"
    print(" PASS: Successfully triggered zero-hallucination guardrail.")

    print("\n" + "=" * 60)
    print("  ALL TESTS PASSED WITH 100% SUCCESS!")
    print("=" * 60)

if __name__ == "__main__":
    run_tests()
