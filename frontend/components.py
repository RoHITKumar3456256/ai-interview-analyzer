import streamlit as st
from typing import List, Dict, Any
from backend.models import EvidenceQuote, ExpertPerspective

def render_badge(text: str, badge_type: str = "primary") -> str:
    type_map = {
        "primary": "badge-primary",
        "emerald": "badge-emerald",
        "amber": "badge-amber",
        "rose": "badge-rose",
        "timestamp": "badge-timestamp"
    }
    css_cls = type_map.get(badge_type, "badge-primary")
    return f'<span class="badge {css_cls}">{text}</span>'

def render_evidence_quote_card(quote: EvidenceQuote):
    org_str = f" | {quote.organization}" if quote.organization else ""
    st.markdown(
        f"""
        <div class="evidence-quote-box">
            <div style="font-size: 0.95rem; line-height: 1.5; color: #e2e8f0;">
                "{quote.quote}"
            </div>
            <div class="evidence-quote-meta">
                <span class="badge badge-primary">🎙️ {quote.speaker}</span>
                <span class="badge badge-timestamp">⏱️ {quote.timestamp}</span>
                <span style="font-size: 0.75rem; color: #94a3b8;">📄 {quote.transcript_id}{org_str}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

def render_expert_perspective_card(perspective: ExpertPerspective):
    with st.container():
        st.markdown(
            f"""
            <div class="perspective-box">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                    <span style="font-weight: 700; color: #818cf8; font-size: 0.95rem;">👤 {perspective.expert_name}</span>
                    <span class="badge badge-primary">{perspective.organization}</span>
                </div>
                <div style="color: #94a3b8; font-size: 0.8rem; margin-bottom: 8px;">{perspective.expert_title}</div>
                <div style="font-size: 0.9rem; color: #f1f5f9; line-height: 1.45;">{perspective.stance_summary}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

def render_grounding_indicator(score: float = 1.0, is_zero_hallucination: bool = True):
    if is_zero_hallucination:
        st.markdown(
            """
            <div style="display: inline-flex; align-items: center; gap: 6px; background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 6px; padding: 4px 10px; font-size: 0.75rem; color: #34d399; font-weight: 600;">
                🛡️ Zero Hallucination Verified (100% Transcript Grounded)
            </div>
            """,
            unsafe_allow_html=True
        )
