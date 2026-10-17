"""Gradio interface for the RAG engine."""

from __future__ import annotations

import dataclasses
import html
from collections.abc import Iterator
from pathlib import Path

import gradio as gr

from .config import Settings
from .engine import RAGEngine, SearchResult


LOADING_HTML = """
<div class="answer-loader" role="status" aria-live="polite">
  <span class="answer-loader__spinner" aria-hidden="true"></span>
  <span>Searching documents and generating an answer…</span>
</div>
"""

APP_THEME = gr.themes.Soft(
    primary_hue=gr.themes.colors.indigo,
    secondary_hue=gr.themes.colors.slate,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "-apple-system", "sans-serif"],
)

APP_CSS = """
/* Container & Global Layout */
.gradio-container {
    max-width: 1240px !important;
    margin: 0 auto !important;
}

/* Header Banner */
.rag-header {
    background: linear-gradient(135deg, rgba(79, 70, 229, 0.08) 0%, rgba(147, 51, 234, 0.04) 100%);
    border: 1px solid var(--border-color-primary, rgba(99, 102, 241, 0.25));
    border-radius: 14px;
    padding: 1.25rem 1.75rem;
    margin-bottom: 1rem;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
}

.rag-header-title {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    font-size: 1.6rem;
    font-weight: 700;
    color: var(--body-text-color, #1e293b);
    margin: 0 0 0.35rem 0;
    line-height: 1.2;
}

.rag-header-icon {
    font-size: 1.6rem;
}

.rag-header-subtitle {
    font-size: 0.92rem;
    color: var(--body-text-color-subdued, #64748b);
    margin: 0 0 0.85rem 0;
    line-height: 1.45;
}

/* Badges */
.rag-badge-group {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    align-items: center;
}

.rag-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.28rem 0.7rem;
    border-radius: 9999px;
    font-size: 0.78rem;
    font-weight: 600;
    line-height: 1.1;
    letter-spacing: 0.01em;
}

.rag-badge-indigo {
    background: rgba(99, 102, 241, 0.12);
    color: #4f46e5;
    border: 1px solid rgba(99, 102, 241, 0.25);
}

.rag-badge-purple {
    background: rgba(168, 85, 247, 0.12);
    color: #9333ea;
    border: 1px solid rgba(168, 85, 247, 0.25);
}

.rag-badge-emerald {
    background: rgba(16, 185, 129, 0.12);
    color: #059669;
    border: 1px solid rgba(16, 185, 129, 0.25);
}

.rag-badge-amber {
    background: rgba(245, 158, 11, 0.12);
    color: #d97706;
    border: 1px solid rgba(245, 158, 11, 0.25);
}

.rag-badge-slate {
    background: rgba(100, 116, 139, 0.12);
    color: var(--body-text-color, #475569);
    border: 1px solid var(--border-color-primary, rgba(100, 116, 139, 0.25));
}

/* Sidebar & Quick Stats Card */
.sidebar-card {
    background: var(--background-fill-secondary, #f8fafc);
    border: 1px solid var(--border-color-primary, #e2e8f0);
    border-radius: 12px;
    padding: 1rem;
    margin-bottom: 1rem;
}

.sidebar-card-title {
    font-size: 0.82rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--body-text-color-subdued, #64748b);
    margin-bottom: 0.75rem;
}

.sidebar-stats-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 0.5rem;
    margin-bottom: 0.75rem;
}

.sidebar-stat-box {
    background: var(--block-background-fill, #ffffff);
    border: 1px solid var(--border-color-primary, #e2e8f0);
    border-radius: 8px;
    padding: 0.6rem 0.75rem;
}

.sidebar-stat-val {
    font-size: 1.25rem;
    font-weight: 700;
    color: var(--body-text-color, #1e293b);
    line-height: 1.1;
}

.sidebar-stat-lbl {
    font-size: 0.72rem;
    color: var(--body-text-color-subdued, #64748b);
    margin-top: 0.2rem;
}

/* Source Citation Cards */
.sources-wrapper {
    margin-top: 0.5rem;
    display: flex;
    flex-direction: column;
    gap: 0.6rem;
}

.sources-summary-bar {
    font-size: 0.84rem;
    color: var(--body-text-color-subdued, #64748b);
    margin-bottom: 0.2rem;
}

.source-card {
    background: var(--block-background-fill, #ffffff);
    border: 1px solid var(--border-color-primary, #e2e8f0);
    border-left: 4px solid #6366f1;
    border-radius: 8px;
    padding: 0.75rem 0.95rem;
    transition: all 0.15s ease-in-out;
}

.source-card:hover {
    border-color: #6366f1;
    box-shadow: 0 2px 6px rgba(99, 102, 241, 0.08);
}

.source-card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.45rem;
    gap: 0.5rem;
    flex-wrap: wrap;
}

.source-title {
    display: flex;
    align-items: center;
    gap: 0.45rem;
    font-size: 0.88rem;
    color: var(--body-text-color, #1e293b);
}

.source-chunk-tag {
    background: var(--background-fill-secondary, #f1f5f9);
    color: var(--body-text-color-subdued, #475569);
    padding: 0.15rem 0.45rem;
    border-radius: 4px;
    font-size: 0.74rem;
    font-weight: 500;
}

.source-score-pill {
    font-size: 0.76rem;
    font-weight: 700;
    padding: 0.18rem 0.55rem;
    border-radius: 9999px;
    line-height: 1.1;
}

.source-content {
    font-size: 0.84rem;
    line-height: 1.5;
    color: var(--body-text-color-subdued, #334155);
    background: var(--background-fill-secondary, #f8fafc);
    border-radius: 6px;
    padding: 0.6rem 0.75rem;
    max-height: 140px;
    overflow-y: auto;
    white-space: pre-wrap;
    word-break: break-word;
}

.sources-empty-state {
    padding: 1.25rem;
    text-align: center;
    color: var(--body-text-color-subdued, #64748b);
    font-size: 0.88rem;
    background: var(--background-fill-secondary, #f8fafc);
    border: 1px dashed var(--border-color-primary, #cbd5e1);
    border-radius: 8px;
}

.sources-loading {
    padding: 1rem;
    font-size: 0.88rem;
    color: var(--primary-500, #4f46e5);
}

/* Search Explorer */
.search-explorer-container {
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
    margin-top: 0.5rem;
}

.search-summary-header {
    font-size: 0.9rem;
    color: var(--body-text-color, #1e293b);
    margin-bottom: 0.25rem;
}

.search-result-card {
    background: var(--block-background-fill, #ffffff);
    border: 1px solid var(--border-color-primary, #e2e8f0);
    border-left: 4px solid #3b82f6;
    border-radius: 8px;
    padding: 0.85rem 1rem;
    margin-bottom: 0.5rem;
}

.search-result-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.5rem;
    gap: 0.5rem;
    flex-wrap: wrap;
}

.search-result-title {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.9rem;
}

.rank-badge {
    background: #2563eb;
    color: #ffffff;
    font-size: 0.74rem;
    font-weight: 700;
    padding: 0.15rem 0.45rem;
    border-radius: 4px;
}

/* Document Meta Card */
.doc-meta-card {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(110px, 1fr));
    gap: 0.75rem;
    background: var(--background-fill-secondary, #f8fafc);
    border: 1px solid var(--border-color-primary, #e2e8f0);
    border-radius: 8px;
    padding: 0.75rem 1rem;
    margin-bottom: 0.75rem;
}

.doc-meta-item {
    display: flex;
    flex-direction: column;
}

.doc-meta-item span {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--body-text-color-subdued, #64748b);
}

.doc-meta-item strong {
    font-size: 1.05rem;
    color: var(--body-text-color, #1e293b);
}

/* Original Spinner Compatibility */
.answer-loader {
    align-items: center;
    background: var(--block-background-fill);
    border: 1px solid var(--border-color-primary);
    border-radius: 10px;
    display: flex;
    gap: 0.75rem;
    margin: 0.5rem 0;
    padding: 0.85rem 1rem;
}

.answer-loader__spinner {
    animation: answer-loader-spin 0.8s linear infinite;
    border: 3px solid var(--border-color-primary);
    border-radius: 50%;
    border-top-color: var(--primary-500, #2563eb);
    flex: 0 0 auto;
    height: 1.25rem;
    width: 1.25rem;
}

@keyframes answer-loader-spin {
    to { transform: rotate(360deg); }
}
"""


def stream_answer(engine: RAGEngine, question: str) -> Iterator[tuple[str, str]]:
    """Emit a loading state before running the blocking answer request."""
    if not question.strip():
        yield "", "Please enter a question."
        return

    yield LOADING_HTML, ""
    answer = engine.answer_question(question)
    if engine.semantic_error and engine.api_key:
        answer += "\n\n> Semantic search was unavailable; keyword fallback was used."
    yield "", answer


def _generate_header_html(rag: RAGEngine) -> str:
    docs_count = len(rag.documents)
    chunks_count = len(rag.chunks)
    key_status = "API Key Connected" if rag.api_key else "Missing API Key (Keyword Fallback)"
    key_cls = "rag-badge-emerald" if rag.api_key else "rag-badge-amber"
    mode = rag.settings.search_mode.capitalize()
    model = rag.model

    return f"""
<div class="rag-header">
  <div class="rag-header-main">
    <div class="rag-header-title">
      <span class="rag-header-icon">⚡</span>
      <span>Document Intelligence & RAG Studio</span>
    </div>
    <p class="rag-header-subtitle">
      Ground queries across your local PDFs and text files with hybrid retrieval and cited answers.
    </p>
  </div>
  <div class="rag-badge-group">
    <span class="rag-badge {key_cls}">● {key_status}</span>
    <span class="rag-badge rag-badge-indigo">📄 {docs_count} Documents ({chunks_count} Chunks)</span>
    <span class="rag-badge rag-badge-purple">🔍 {mode} Mode</span>
    <span class="rag-badge rag-badge-slate">🤖 {html.escape(model)}</span>
  </div>
</div>
"""


def _generate_sidebar_stats(rag: RAGEngine) -> str:
    docs_count = len(rag.documents)
    chunks_count = len(rag.chunks)
    mode = rag.settings.search_mode.capitalize()
    top_k = rag.settings.top_k
    doc_list = "\n".join(f"- `{doc.path.name}`" for doc in rag.documents) or "_No documents loaded_"

    return f"""
<div class="sidebar-card">
  <div class="sidebar-card-title">Knowledge Base Status</div>
  <div class="sidebar-stats-grid">
    <div class="sidebar-stat-box">
      <div class="sidebar-stat-val">{docs_count}</div>
      <div class="sidebar-stat-lbl">Documents</div>
    </div>
    <div class="sidebar-stat-box">
      <div class="sidebar-stat-val">{chunks_count}</div>
      <div class="sidebar-stat-lbl">Index Chunks</div>
    </div>
    <div class="sidebar-stat-box">
      <div class="sidebar-stat-val">{mode}</div>
      <div class="sidebar-stat-lbl">Search Mode</div>
    </div>
    <div class="sidebar-stat-box">
      <div class="sidebar-stat-val">{top_k}</div>
      <div class="sidebar-stat-lbl">Top-K Chunks</div>
    </div>
  </div>
  <div style="font-size: 0.8rem; color: var(--body-text-color-subdued, #64748b);">
    <strong>Loaded Files:</strong><br/>
    {doc_list}
  </div>
</div>
"""


def _format_sources_html(results: list[SearchResult]) -> str:
    if not results:
        return (
            "<div class='sources-empty-state'>"
            "<span>ℹ️ No matching source passages were found for this query.</span>"
            "</div>"
        )

    cards: list[str] = []
    for res in results:
        score_pct = int(round(res.score * 100))
        if res.score >= 0.75:
            badge_color = "#059669"
            badge_bg = "rgba(16, 185, 129, 0.12)"
            badge_border = "rgba(16, 185, 129, 0.25)"
        elif res.score >= 0.45:
            badge_color = "#2563eb"
            badge_bg = "rgba(37, 99, 235, 0.12)"
            badge_border = "rgba(37, 99, 235, 0.25)"
        else:
            badge_color = "#d97706"
            badge_bg = "rgba(245, 158, 11, 0.12)"
            badge_border = "rgba(245, 158, 11, 0.25)"

        escaped_source = html.escape(res.chunk.source.name)
        escaped_content = html.escape(res.chunk.content)
        cards.append(f"""
        <div class="source-card">
          <div class="source-card-header">
            <div class="source-title">
              <span>📄</span>
              <strong>{escaped_source}</strong>
              <span class="source-chunk-tag">Chunk #{res.chunk.index + 1}</span>
            </div>
            <span class="source-score-pill" style="color: {badge_color}; background: {badge_bg}; border: 1px solid {badge_border};">
              {score_pct}% Match
            </span>
          </div>
          <div class="source-content">{escaped_content}</div>
        </div>
        """)

    return f"""
    <div class="sources-wrapper">
      <div class="sources-summary-bar">
        <span>Showing <strong>{len(results)}</strong> retrieved context chunks used for answer grounding:</span>
      </div>
      {''.join(cards)}
    </div>
    """


def _format_search_explorer_results(results: list[SearchResult], query: str, mode: str) -> str:
    if not query.strip():
        return "<div class='sources-empty-state'>Enter a query above and click 'Search Chunks' to inspect retrieval results.</div>"
    if not results:
        return f"<div class='sources-empty-state'>No chunks found matching '<strong>{html.escape(query)}</strong>' under {mode} mode. Try adjusting your query or threshold.</div>"

    cards: list[str] = []
    for rank, res in enumerate(results, start=1):
        score_pct = int(round(res.score * 100))
        if res.score >= 0.75:
            badge_color = "#059669"
            badge_bg = "rgba(16, 185, 129, 0.12)"
            badge_border = "rgba(16, 185, 129, 0.25)"
        elif res.score >= 0.45:
            badge_color = "#2563eb"
            badge_bg = "rgba(37, 99, 235, 0.12)"
            badge_border = "rgba(37, 99, 235, 0.25)"
        else:
            badge_color = "#d97706"
            badge_bg = "rgba(245, 158, 11, 0.12)"
            badge_border = "rgba(245, 158, 11, 0.25)"

        escaped_source = html.escape(res.chunk.source.name)
        escaped_content = html.escape(res.chunk.content)
        cards.append(f"""
        <div class="search-result-card">
          <div class="search-result-header">
            <div class="search-result-title">
              <span class="rank-badge">#{rank}</span>
              <strong>{escaped_source}</strong>
              <span class="source-chunk-tag">Chunk #{res.chunk.index + 1}</span>
            </div>
            <span class="source-score-pill" style="color: {badge_color}; background: {badge_bg}; border: 1px solid {badge_border};">
              Score: {res.score:.3f} ({score_pct}%)
            </span>
          </div>
          <div class="source-content" style="max-height: none;">{escaped_content}</div>
        </div>
        """)

    return f"""
    <div class="search-explorer-container">
      <div class="search-summary-header">
        <span>Retrieved <strong>{len(results)} chunks</strong> matching <em>"{html.escape(query)}"</em> using <strong>{mode.title()}</strong> search.</span>
      </div>
      {''.join(cards)}
    </div>
    """


def _get_document_metadata(doc_name: str | None, rag: RAGEngine) -> tuple[str, str]:
    if not doc_name:
        return "<p style='color: #64748b;'>Select a document from the dropdown above to view its metadata and content.</p>", ""
    doc = next((d for d in rag.documents if d.path.name == doc_name), None)
    if not doc:
        return f"<p style='color: #ef4444;'>Document '{doc_name}' was not found in the index.</p>", ""

    try:
        size_kb = f"{doc.path.stat().st_size / 1024:.1f} KB"
    except OSError:
        size_kb = "Unknown"

    chunk_count = sum(1 for c in rag.chunks if c.source.name == doc_name)
    char_count = len(doc.content)
    word_count = len(doc.content.split())

    meta_html = f"""
    <div class="doc-meta-card">
      <div class="doc-meta-item"><span>Format</span><strong>{doc.kind.upper()}</strong></div>
      <div class="doc-meta-item"><span>File Size</span><strong>{size_kb}</strong></div>
      <div class="doc-meta-item"><span>Chunks</span><strong>{chunk_count}</strong></div>
      <div class="doc-meta-item"><span>Characters</span><strong>{char_count:,}</strong></div>
      <div class="doc-meta-item"><span>Words</span><strong>{word_count:,}</strong></div>
    </div>
    """
    return meta_html, doc.content


def build_app(engine: RAGEngine | None = None) -> gr.Blocks:
    rag = engine or RAGEngine()

    with gr.Blocks(title="Document Intelligence & RAG Studio") as app:
        header_display = gr.HTML(_generate_header_html(rag))

        with gr.Tabs():
            # Tab 1: Conversational Q&A Assistant
            with gr.Tab("💬 Q&A Assistant", id="tab_qa"):
                with gr.Row():
                    # Sidebar
                    with gr.Column(scale=1, min_width=280):
                        sidebar_stats = gr.HTML(_generate_sidebar_stats(rag))

                        with gr.Accordion("⚙️ Quick Retrieval Tuning", open=False):
                            quick_mode = gr.Dropdown(
                                choices=["hybrid", "semantic", "keyword"],
                                value=rag.settings.search_mode,
                                label="Search Mode",
                            )
                            quick_top_k = gr.Slider(
                                minimum=1,
                                maximum=10,
                                value=rag.settings.top_k,
                                step=1,
                                label="Top-K Chunks",
                            )
                            update_quick_btn = gr.Button("Update Mode & Top-K", size="sm")

                        clear_chat_btn = gr.Button("🗑️ Clear Conversation", variant="secondary")

                    # Main Chat Area
                    with gr.Column(scale=3):
                        chatbot = gr.Chatbot(
                            label="Document Conversation",
                            height=480,
                            buttons=["copy"],
                            layout="bubble",
                        )

                        with gr.Row():
                            question_input = gr.Textbox(
                                label="Question",
                                placeholder="Ask a question about the loaded documents... (e.g., What is the MVP?)",
                                lines=2,
                                max_lines=4,
                                scale=5,
                            )
                            ask_button = gr.Button("Ask Question", variant="primary", scale=1)

                        gr.Examples(
                            examples=[
                                ["What is the MVP?"],
                                ["What city is analyzed and why?"],
                                ["Summarize the key financial projections."],
                                ["What are the unit economics and margins?"],
                                ["What are the main risks or assumptions?"],
                            ],
                            inputs=question_input,
                            label="Sample Questions (Click to populate)",
                        )

                        with gr.Accordion("📚 Retrieved Sources & Citation Chunks", open=True):
                            sources_display = gr.HTML(
                                "<div class='sources-empty-state'>Ask a question above to see the retrieved source passages and relevance scores here.</div>"
                            )

            # Tab 2: Search & Retrieval Explorer
            with gr.Tab("🔍 Retrieval Explorer", id="tab_search"):
                gr.Markdown(
                    "### Direct Document & Chunk Retrieval\n"
                    "Test and evaluate retrieval quality directly without calling the generation model. "
                    "Inspect relevance scores and matched text passages."
                )
                with gr.Row():
                    explorer_query = gr.Textbox(
                        label="Search Query",
                        placeholder="e.g. Pune market growth, financial breakeven, customer acquisition...",
                        lines=1,
                        scale=4,
                    )
                    explorer_mode = gr.Dropdown(
                        choices=["hybrid", "semantic", "keyword"],
                        value=rag.settings.search_mode,
                        label="Search Strategy",
                        scale=1,
                    )
                    explorer_top_k = gr.Slider(
                        minimum=1,
                        maximum=10,
                        value=rag.settings.top_k,
                        step=1,
                        label="Chunks (Top-K)",
                        scale=1,
                    )
                    search_btn = gr.Button("Search Chunks", variant="primary", scale=1)

                search_results_display = gr.HTML(
                    "<div class='sources-empty-state'>Enter a query above and click 'Search Chunks' to view ranked passages.</div>"
                )

            # Tab 3: Knowledge Base Manager
            with gr.Tab("📁 Knowledge Base", id="tab_docs"):
                gr.Markdown(
                    "### Document Management & Inspection\n"
                    "Upload new PDF or TXT files, inspect document extracts, or rescan the documents directory."
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("#### Add Documents")
                        file_uploader = gr.File(
                            label="Upload PDF or TXT Documents",
                            file_count="multiple",
                            file_types=[".pdf", ".txt"],
                            type="filepath",
                        )
                        upload_btn = gr.Button("Add to Knowledge Base", variant="primary")
                        upload_alert = gr.Markdown()
                        rescan_btn = gr.Button("🔄 Rescan data/documents Directory", variant="secondary")

                    with gr.Column(scale=2):
                        gr.Markdown("#### Inspect Loaded Documents")
                        doc_selector = gr.Dropdown(
                            choices=[doc.path.name for doc in rag.documents],
                            value=rag.documents[0].path.name if rag.documents else None,
                            label="Select Document to View",
                        )
                        doc_meta_display = gr.HTML()
                        doc_content_display = gr.Textbox(
                            label="Extracted Text Content",
                            lines=12,
                            max_lines=20,
                            interactive=False,
                        )

            # Tab 4: Settings & Diagnostics
            with gr.Tab("⚙️ Settings & Diagnostics", id="tab_settings"):
                gr.Markdown(
                    "### Engine Configuration & Diagnostics\n"
                    "Adjust retrieval thresholds, switch LLM models, and view system status."
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("#### Retrieval Parameters")
                        cfg_mode = gr.Radio(
                            choices=["hybrid", "semantic", "keyword"],
                            value=rag.settings.search_mode,
                            label="Search Mode",
                            info="Hybrid balances semantic embeddings and keyword matching.",
                        )
                        cfg_top_k = gr.Slider(
                            minimum=1,
                            maximum=10,
                            value=rag.settings.top_k,
                            step=1,
                            label="Default Top-K Chunks",
                        )
                        cfg_weight = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=rag.settings.semantic_weight,
                            step=0.05,
                            label="Semantic Weight (Hybrid mode)",
                            info="Weight given to semantic similarity vs keyword frequency.",
                        )
                        cfg_threshold = gr.Slider(
                            minimum=0.0,
                            maximum=0.5,
                            value=rag.settings.semantic_threshold,
                            step=0.05,
                            label="Relevance Score Threshold",
                            info="Minimum score required for a chunk to be considered.",
                        )
                    with gr.Column(scale=1):
                        gr.Markdown("#### Generation Model & API")
                        cfg_model = gr.Textbox(
                            value=rag.model,
                            label="OpenRouter Model",
                            placeholder="e.g. openrouter/auto, anthropic/claude-3.5-sonnet",
                        )
                        cfg_embedding_model = gr.Textbox(
                            value=rag.settings.embedding_model,
                            label="Embedding Model (Read-only)",
                            interactive=False,
                        )
                        cfg_documents_dir = gr.Textbox(
                            value=str(rag.settings.documents_dir),
                            label="Documents Directory",
                            interactive=False,
                        )
                        cfg_cache_path = gr.Textbox(
                            value=str(rag.settings.embedding_cache),
                            label="Vector Embedding Cache",
                            interactive=False,
                        )
                        apply_settings_btn = gr.Button("Save & Apply Settings", variant="primary")
                        settings_alert = gr.Markdown()

        # ======================================================================
        # Event Handlers
        # ======================================================================

        def chat_stream(
            question: str, history: list[dict] | None
        ) -> Iterator[tuple[list[dict], str, str]]:
            hist = list(history or [])
            if not question.strip():
                yield hist, "", "<div class='sources-empty-state'>Please enter a question to view citations.</div>"
                return

            hist.append({"role": "user", "content": question})
            hist.append({
                "role": "assistant",
                "content": "🔍 *Searching documents & synthesizing grounded answer...*",
            })
            yield hist, "", "<div class='sources-loading'><em>Searching knowledge base for matching passages...</em></div>"

            # Retrieve sources using configured settings
            results = rag.search(question, top_k=rag.settings.top_k)
            sources_html = _format_sources_html(results)

            # Generate LLM answer
            answer = rag.answer_question(question)
            if rag.semantic_error and rag.api_key:
                answer += "\n\n> ⚠️ *Semantic search was unavailable; keyword fallback was used.*"

            hist[-1]["content"] = answer
            yield hist, "", sources_html

        def clear_chat() -> tuple[list[dict], str, str]:
            return [], "", "<div class='sources-empty-state'>Ask a question above to see the retrieved source passages and relevance scores here.</div>"

        def handle_quick_tuning(mode: str, top_k: int) -> tuple[str, str, str]:
            rag.settings = dataclasses.replace(
                rag.settings,
                search_mode=mode,
                top_k=int(top_k),
            )
            return (
                _generate_header_html(rag),
                _generate_sidebar_stats(rag),
                f"Updated search mode to **{mode.title()}** and top-k to **{top_k}**.",
            )

        def handle_search_explorer(query: str, mode: str, top_k: int) -> str:
            if not query.strip():
                return "<div class='sources-empty-state'>Please enter a search query.</div>"
            old_mode = rag.settings.search_mode
            rag.settings = dataclasses.replace(rag.settings, search_mode=mode)
            try:
                results = rag.search(query, top_k=int(top_k))
                return _format_search_explorer_results(results, query, mode)
            finally:
                rag.settings = dataclasses.replace(rag.settings, search_mode=old_mode)

        def handle_upload(files: list[str] | str | None) -> tuple[str, str, str, gr.Dropdown, str, str]:
            if not files:
                empty_meta, empty_content = _get_document_metadata(None, rag)
                return (
                    "⚠️ Choose at least one PDF or TXT file to upload.",
                    _generate_header_html(rag),
                    _generate_sidebar_stats(rag),
                    gr.Dropdown(choices=[d.path.name for d in rag.documents]),
                    empty_meta,
                    empty_content,
                )
            paths = [files] if isinstance(files, str) else files
            try:
                added = rag.add_documents(paths)
                msg = f"✅ Successfully added **{added}** new document{'s' if added != 1 else ''} to the index."
            except ValueError as exc:
                msg = f"⚠️ {exc}"

            choices = [d.path.name for d in rag.documents]
            new_val = choices[-1] if choices else None
            meta_html, content_str = _get_document_metadata(new_val, rag)
            return (
                msg,
                _generate_header_html(rag),
                _generate_sidebar_stats(rag),
                gr.Dropdown(choices=choices, value=new_val),
                meta_html,
                content_str,
            )

        def handle_rescan() -> tuple[str, str, str, gr.Dropdown, str, str]:
            paths = rag.discover_documents()
            rag.documents.clear()
            rag.chunks.clear()
            rag._chunk_embeddings = None
            added = rag.add_documents(paths)
            choices = [d.path.name for d in rag.documents]
            new_val = choices[0] if choices else None
            meta_html, content_str = _get_document_metadata(new_val, rag)
            msg = f"🔄 Reloaded **{added}** documents ({len(rag.chunks)} chunks) from `{rag.settings.documents_dir.name}`."
            return (
                msg,
                _generate_header_html(rag),
                _generate_sidebar_stats(rag),
                gr.Dropdown(choices=choices, value=new_val),
                meta_html,
                content_str,
            )

        def handle_doc_selection(doc_name: str | None) -> tuple[str, str]:
            return _get_document_metadata(doc_name, rag)

        def handle_apply_settings(
            mode: str,
            top_k: int,
            weight: float,
            threshold: float,
            model: str,
        ) -> tuple[str, str, str, str]:
            clean_model = model.strip() or "openrouter/auto"
            rag.settings = dataclasses.replace(
                rag.settings,
                search_mode=mode,
                top_k=int(top_k),
                semantic_weight=float(weight),
                semantic_threshold=float(threshold),
                model=clean_model,
            )
            status_msg = f"✅ Settings applied: Mode=**{mode.title()}**, Model=**{clean_model}**, Top-K=**{top_k}**, Semantic Weight=**{weight}**."
            return (
                status_msg,
                _generate_header_html(rag),
                _generate_sidebar_stats(rag),
                status_msg,
            )

        # Wire Chat Events
        ask_button.click(
            chat_stream,
            inputs=[question_input, chatbot],
            outputs=[chatbot, question_input, sources_display],
            show_progress="hidden",
        )
        question_input.submit(
            chat_stream,
            inputs=[question_input, chatbot],
            outputs=[chatbot, question_input, sources_display],
            show_progress="hidden",
        )
        clear_chat_btn.click(
            clear_chat,
            outputs=[chatbot, question_input, sources_display],
        )

        # Wire Quick Tuning
        update_quick_btn.click(
            handle_quick_tuning,
            inputs=[quick_mode, quick_top_k],
            outputs=[header_display, sidebar_stats, question_input],
        )

        # Wire Search Explorer
        search_btn.click(
            handle_search_explorer,
            inputs=[explorer_query, explorer_mode, explorer_top_k],
            outputs=[search_results_display],
        )
        explorer_query.submit(
            handle_search_explorer,
            inputs=[explorer_query, explorer_mode, explorer_top_k],
            outputs=[search_results_display],
        )

        # Wire Knowledge Base Tab
        upload_btn.click(
            handle_upload,
            inputs=[file_uploader],
            outputs=[
                upload_alert,
                header_display,
                sidebar_stats,
                doc_selector,
                doc_meta_display,
                doc_content_display,
            ],
        )
        rescan_btn.click(
            handle_rescan,
            outputs=[
                upload_alert,
                header_display,
                sidebar_stats,
                doc_selector,
                doc_meta_display,
                doc_content_display,
            ],
        )
        doc_selector.change(
            handle_doc_selection,
            inputs=[doc_selector],
            outputs=[doc_meta_display, doc_content_display],
        )

        # Initialize document preview on load
        initial_doc = rag.documents[0].path.name if rag.documents else None
        initial_meta, initial_content = _get_document_metadata(initial_doc, rag)
        doc_meta_display.value = initial_meta
        doc_content_display.value = initial_content

        # Wire Settings Tab
        apply_settings_btn.click(
            handle_apply_settings,
            inputs=[cfg_mode, cfg_top_k, cfg_weight, cfg_threshold, cfg_model],
            outputs=[settings_alert, header_display, sidebar_stats, upload_alert],
        )

        app.queue(default_concurrency_limit=4)

    return app
