import hashlib
import time
from collections import OrderedDict

import streamlit as st
from dotenv import load_dotenv
from runtime_config import public_deployment
from ai_client import ai_provider, default_chat_model
from ollama import ResponseError
from pypdf.errors import PdfReadError
import os

from rag import answer_question, build_vector_store
from assistants import transcribe_audio, meeting_notes, research_report

load_dotenv()
try:
    # Root-level Streamlit Secrets become server environment values, never browser fields.
    secret_values = st.secrets.to_dict()
    for name in ("AI_PROVIDER", "PUBLIC_DEPLOYMENT", "GEMINI_API_KEY", "GEMINI_MODEL", "GEMINI_DAILY_REQUEST_LIMIT", "OLLAMA_BASE_URL", "CHAT_MODEL", "EMBEDDING_MODEL"):
        if name in secret_values:
            os.environ[name] = str(secret_values[name])
except FileNotFoundError:
    pass
FREE_CLOUD = ai_provider() == "gemini"
PUBLIC_MODE = public_deployment() or FREE_CLOUD
CLOUD_READY = not FREE_CLOUD or bool(os.getenv("GEMINI_API_KEY", "").strip())
st.set_page_config(page_title="ClarityDesk AI | Documents, Meetings & Research", page_icon=":material/description:", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
:root {--accent: #7357dc; --ink: #24243e; --muted: #7b7c92;}
.stApp {background: #f7f8fc; color: var(--ink);}
[data-testid="stHeader"] {background: #f7f8fcdd;}
[data-testid="stMainBlockContainer"] {max-width: 1180px; padding: 2rem 2.4rem 6rem;}
[data-testid="stSidebar"] {background: #fff; border-right: 1px solid #e9e8f1;}
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {gap: .85rem;}
[data-testid="stSidebar"] [data-testid="stText"] {color: #4b4b65;}
[data-testid="stSidebar"] [data-testid="stFileUploader"] section {background: #f8f7fd; border: 1px dashed #c9bfee; border-radius: 14px;}
[data-testid="stSidebar"] [data-testid="stFileUploader"] p {color: #66647e;}
h1, h2, h3 {letter-spacing: -.035em; color: var(--ink);}
h1 {font-size: 2rem !important;}
footer {visibility: hidden;}
.brand {display: flex; align-items: center; gap: 12px; margin: 5px 0 14px;}
.brand-icon {background: linear-gradient(140deg,#8c6aee,#6047c6); color: white; padding: 10px 12px; border-radius: 13px; font-size: 18px; font-weight: 750; box-shadow: 0 5px 14px #7357dc28;}
.brand-title {font-size: 17px; font-weight: 750; line-height: 1.3;}
.brand-sub {color: #9692ab; font-size: 11px; letter-spacing: .07em; margin-top: 3px;}
.workspace-top {display: flex; justify-content: space-between; align-items: center; margin-bottom: 24px; color: #8d8ba2; font-size: 12px;}
.private-pill {background: #eef7f1; color: #438563; padding: 6px 12px; border-radius: 20px; font-size: 11px;}
.hero {position: relative; overflow: hidden; border: 1px solid #e6dff8; background: linear-gradient(110deg,#f0eaff,#f4f2ff 60%,#eaf2fc); border-radius: 22px; padding: 30px 32px; margin-bottom: 22px;}
.hero:after {content: ""; position: absolute; width: 200px; height: 200px; border: 30px solid #b8a3ed26; border-radius: 50%; right: -40px; top: -65px; pointer-events: none;}
.hero-label {font-size: 10px; font-weight: 750; letter-spacing: .16em; color: #8261c3; margin-bottom: 10px;}
.hero h1 {font-size: 30px !important; margin: 0 0 9px; padding: 0; color: #34304f;}
.hero p {color: #7b7396; font-size: 14px; margin: 0; max-width: 650px;}
[data-testid="stMetric"] {background: white; border: 1px solid #eeedf5; border-radius: 15px; padding: 16px 20px; box-shadow: 0 3px 10px #29243d03;}
[data-testid="stMetricLabel"] {color: #9995ab; font-size: 10px; letter-spacing: .06em;}
[data-testid="stMetricValue"] {color: #514675; font-size: 25px; font-weight: 650;}
[data-testid="stButton"] button, [data-testid="stDownloadButton"] button {border-radius: 10px; min-height: 42px; font-size: 13px; border-color: #e7e3f0;}
[data-testid="stButton"] button[kind="primary"], [data-testid="stFormSubmitButton"] button[kind="primary"] {background: #7357dc; border-color: #7357dc; color: white;}
[data-testid="stChatMessage"] {background: #fff; border: 1px solid #ece9f4; border-radius: 18px; padding: 20px; margin-bottom: 14px; box-shadow: 0 3px 14px #35245504;}
[data-testid="stChatMessageAvatarAssistant"] {background: #eee7ff; color: #7357dc;}
[data-testid="stChatInput"] {background: white; border: 1px solid #ded7ed; border-radius: 16px; box-shadow: 0 5px 22px #6047c60b;}
[data-testid="stBottom"] > div {background: #f7f8fc;}
[data-testid="stExpander"] {border-color: #ece9f4; border-radius: 12px; background: #ffffff;}
.empty-state {text-align: center; padding: 35px 20px 25px;}
.empty-icon {width: 58px; height: 58px; border-radius: 18px; background: #eee8fc; color: #8161cf; display: grid; place-items: center; margin: 0 auto 16px; font-size: 25px;}
.empty-state h2 {font-size: 21px; margin: 0 0 8px;}
.empty-state p {color: #9290a5; font-size: 13px; margin: 0;}
.section-label {color: #504764; font-size: 14px; font-weight: 650; margin: 22px 0 10px;}
.prompt-desc {color: #9a94a9; font-size: 12px; margin-top: 5px; line-height: 1.6;}
@media(max-width: 800px) {
[data-testid="stMainBlockContainer"] {padding: 1rem 1rem 6rem;}
.hero {padding: 24px 20px;}
.hero h1 {font-size: 25px !important;}
.workspace-top {margin-bottom: 16px;}
}
</style>
""", unsafe_allow_html=True)

for key, default in (
    ("vector_store", None), ("active_key", None), ("document_names", []),
    ("history", []), ("index_cache", OrderedDict()), ("index_elapsed", 0),
    ("answer_cache", OrderedDict()), ("transcripts", OrderedDict()),
    ("meeting_reports", OrderedDict()), ("research_reports", OrderedDict()),
):
    st.session_state.setdefault(key, default)


def show_error(error):
    if isinstance(error, ResponseError) and error.status_code == 404:
        st.error("The selected Ollama model is not installed. Check model settings and install it with ollama pull <model-name>.")
    elif isinstance(error, (ConnectionError, TimeoutError)):
        st.error("Cannot reach Ollama. Make sure it is running locally, then retry.")
    else:
        st.error(str(error))


def render_sources(sources):
    if sources:
        with st.expander(f"View sources ({len(sources)})"):
            for source in sources:
                page = f" / Page {source.page}" if source.page else ""
                st.text(f"[{source.label}] {source.filename}{page}")
                st.caption(source.excerpt)



with st.sidebar:
    st.markdown('<div class="brand"><div class="brand-icon">CD</div><div><div class="brand-title">ClarityDesk AI</div><div class="brand-sub">DOCUMENTS. MEETINGS. RESEARCH.</div></div></div>', unsafe_allow_html=True)
    st.divider()
    workspace = st.radio("Workspace", ["Document Assistant", "Meeting Assistant", "Research Assistant"], key="workspace")
    st.divider()
    st.markdown("**Document library**")
    with st.form("upload"):
        files = st.file_uploader("Add PDFs", type="pdf", accept_multiple_files=True,
                                 help="Use text-based PDFs. Scanned pages need OCR.")
        process = st.form_submit_button("Analyze documents", type="primary", use_container_width=True)
    st.caption("Direct answers with page citations. Supporting passages stay under Sources.")
    cloud_consent = True
    if FREE_CLOUD:
        st.info("AI answers send excerpts, meeting transcript text, and research context to Google. Free-tier content may be used to improve Google's products. Avoid confidential information.")
        cloud_consent = st.checkbox("I agree to send analysis text to Google", value=False)
        if not CLOUD_READY:
            st.warning("Cloud AI needs setup: the owner must add GEMINI_API_KEY in Streamlit Secrets.")
        st.caption("Free limits: 5 PDFs / 100 pages, 10 MB per file, 5-minute recordings, research depth 2. Availability depends on Google's quota.")
    with st.expander("Advanced settings"):
        chat_model = st.text_input("Answer model", value=default_chat_model(), disabled=PUBLIC_MODE)
        embedding_model = st.text_input("Embedding model", value=os.getenv("EMBEDDING_MODEL", "nomic-embed-text"), disabled=PUBLIC_MODE or FREE_CLOUD)
        retrieval_count = st.slider("Source passages", 2, 8, 3)
        retrieval_method = st.selectbox("Document search", (["Fast keyword search"] if FREE_CLOUD else ["Fast keyword search", "Semantic search"]), help="Both generate answers. Keyword search avoids embedding time; semantic search can find related wording.")
        st.caption("Answers stream from Gemini within free-tier quotas." if FREE_CLOUD else "Answers stream as they are generated. First use may take longer while Ollama loads a model.")
    if process:
        if not files:
            st.warning("Add a PDF to continue.")
        else:
            if FREE_CLOUD and (len(files) > 5 or any(file.size > 10 * 1024 * 1024 for file in files)):
                st.warning("Free cloud mode allows up to 5 PDFs, at most 10 MB each.")
                st.stop()
            pdfs = [(file.name, file.getvalue()) for file in files]
            digest = hashlib.sha256()
            for name, data in pdfs:
                digest.update(hashlib.sha256(name.encode()).digest())
                digest.update(hashlib.sha256(data).digest())
            search_mode = "keyword" if retrieval_method == "Fast keyword search" else "semantic"
            index_key = (digest.hexdigest(), embedding_model.strip(), search_mode, ai_provider())
            started = time.perf_counter()
            progress = st.progress(0, text="Preparing documents...")
            try:
                cache = st.session_state.index_cache
                reused = index_key in cache
                if reused:
                    store = cache[index_key]
                    cache.move_to_end(index_key)
                else:
                    def update(value, message):
                        progress.progress(value, text=f"{message} ({time.perf_counter() - started:.0f}s)")
                    store = build_vector_store(pdfs, embedding_model, progress_callback=update, search_mode=search_mode)
                    cache[index_key] = store
                    while len(cache) > 2:
                        cache.popitem(last=False)
                if st.session_state.active_key != index_key:
                    st.session_state.history = []
                st.session_state.vector_store = store
                st.session_state.active_key = index_key
                st.session_state.document_names = [name for name, _ in pdfs]
                st.session_state.index_elapsed = time.perf_counter() - started
                progress.empty()
                st.success("Ready - reused your prepared documents." if reused else f"Ready in {st.session_state.index_elapsed:.1f}s")
            except (ResponseError, ValueError, PdfReadError, ConnectionError, TimeoutError) as error:
                progress.empty()
                show_error(error)
    if st.session_state.document_names:
        st.divider()
        for name in st.session_state.document_names:
            st.text(name)
        st.caption(f"{st.session_state.vector_store.index.ntotal} searchable passages")
    st.divider()
    st.caption("Uploads are processed on the server; analysis text goes to Google. Web research queries go to search providers." if FREE_CLOUD else "Uploads are processed on the hosting server for this session. Research queries go to search providers." if PUBLIC_MODE else "Files stay local. Research web queries are sent to search providers when enabled.")

if workspace == "Meeting Assistant":
    st.markdown('<div class="hero"><div class="hero-label">CLARITYDESK AI / MEETING ASSISTANT</div><h1>From conversation to clear next steps.</h1><p>Transcribe a recording, review the text, and create a meeting brief with decisions and action items.</p></div>', unsafe_allow_html=True)
    st.caption("Transcription runs on the hosting server. CPU processing can take time. Speaker identification is not included." if PUBLIC_MODE else "Local transcription. The first run downloads speech-model files. CPU processing can take time. Speaker identification is not included.")
    audio = st.file_uploader("Upload a meeting recording", type=["mp3", "wav", "m4a", "flac", "ogg", "mp4", "webm"], key="meeting_audio")
    speech_size = st.selectbox("Transcription quality", (["tiny"] if PUBLIC_MODE else ["tiny", "base", "small"]), help="Tiny is fastest. Larger models use more memory and processing time.")
    if audio is None:
        st.info("Add a recording to get started. You can review and edit the transcript before generating notes.")
        st.stop()
    audio_data = audio.getvalue()
    audio_key = hashlib.sha256(audio_data).hexdigest() + speech_size
    if st.button("Transcribe recording", type="primary"):
        status = st.empty()
        try:
            if audio_key not in st.session_state.transcripts:
                st.session_state.transcripts[audio_key] = transcribe_audio(audio_data, speech_size, status.info)
                while len(st.session_state.transcripts) > 3:
                    st.session_state.transcripts.popitem(last=False)
            status.success("Transcript ready. Review names, numbers, and deadlines before generating notes.")
        except Exception as error:
            status.error(f"Transcription could not complete: {error}")
    if audio_key in st.session_state.transcripts:
        transcript = st.text_area("Review and edit transcript", value=st.session_state.transcripts[audio_key], height=300, key="transcript_" + audio_key)
        st.download_button("Download transcript", transcript, "meeting-transcript.txt", "text/plain")
        report_key = hashlib.sha256((transcript + chat_model + ai_provider()).encode()).hexdigest()
        if st.button("Create meeting notes", type="primary", disabled=not transcript.strip() or not cloud_consent or not CLOUD_READY):
            status = st.empty()
            try:
                if report_key not in st.session_state.meeting_reports:
                    st.session_state.meeting_reports[report_key] = meeting_notes(transcript, chat_model, status.info)
                    while len(st.session_state.meeting_reports) > 8:
                        st.session_state.meeting_reports.popitem(last=False)
                status.empty()
            except Exception as error:
                status.error(f"Meeting notes could not complete: {error}")
        if report_key in st.session_state.meeting_reports:
            report = st.session_state.meeting_reports[report_key]
            with st.container(border=True):
                st.markdown(report)
            st.download_button("Download meeting notes", report, "meeting-notes.md", "text/markdown")
    st.stop()

if workspace == "Research Assistant":
    st.markdown('<div class="hero"><div class="hero-label">CLARITYDESK AI / RESEARCH ASSISTANT</div><h1>A question. A plan. A sourced report.</h1><p>Explore your documents and web search results through a bounded research workflow.</p></div>', unsafe_allow_html=True)
    st.caption("The assistant chooses follow-up searches and synthesizes the evidence. Web evidence is search-result snippets, not full-page analysis.")
    objective = st.text_area("What would you like to research?", placeholder="Compare the approaches in my documents with publicly available alternatives...", max_chars=2000)
    use_web = st.checkbox("Include web search", value=True, help="Your topic and generated search queries are sent to external search providers. Avoid confidential information in web-enabled objectives.")
    rounds = st.slider("Research depth", 1, 2 if FREE_CLOUD else 5, 2, help="One initial search round, followed by up to this many total rounds. More rounds require additional model calls.")
    if st.session_state.vector_store is None:
        st.info("No documents are prepared. You can research the web now, or add PDFs in the sidebar to include them.")
    report_key = repr((st.session_state.active_key, objective.strip(), use_web, rounds, chat_model, ai_provider()))
    if st.button("Start research", type="primary", disabled=not objective.strip() or not cloud_consent or not CLOUD_READY or (not use_web and st.session_state.vector_store is None)):
        status = st.empty()
        try:
            if report_key not in st.session_state.research_reports:
                st.session_state.research_reports[report_key] = research_report(objective, st.session_state.vector_store, chat_model, use_web, rounds, status.info)
                while len(st.session_state.research_reports) > 8:
                    st.session_state.research_reports.popitem(last=False)
            status.empty()
        except Exception as error:
            status.error(f"Research could not complete: {error}")
    if report_key in st.session_state.research_reports:
        report, sources, activity = st.session_state.research_reports[report_key]
        with st.expander("Research activity", expanded=True):
            for step in activity:
                st.write(f"**{step['tool'].title()}**: {step['query']}")
                st.caption(step["result"])
        with st.container(border=True):
            st.markdown(report)
        render_sources(sources)
        for source in sources:
            if source.filename.startswith(("https://", "http://")):
                st.link_button(f"[{source.label}] Open web source", source.filename)
        exported = report + "\n\n## Sources\n" + "\n".join(f"[{s.label}] {s.filename}" for s in sources)
        st.download_button("Download research report", exported, "research-report.md", "text/markdown")
    st.stop()

processing_label = "Session workspace" if PUBLIC_MODE else "Local &amp; private"
st.markdown(f'<div class="workspace-top"><span>Workspace / Document analysis</span><span class="private-pill">{processing_label}</span></div>', unsafe_allow_html=True)
st.markdown("""<div class="hero"><div class="hero-label">CLARITYDESK AI / DOCUMENT ASSISTANT</div><h1>Turn documents into understanding.</h1><p>Find clear answers, explore the details, and bring your research together in one workspace.</p></div>""", unsafe_allow_html=True)

ready = st.session_state.vector_store is not None
metrics = st.columns(3)
metrics[0].metric("DOCUMENTS", len(st.session_state.document_names))
metrics[1].metric("INDEXED PASSAGES", st.session_state.vector_store.index.ntotal if ready else 0)
metrics[2].metric("ANSWERS", sum(m["role"] == "assistant" for m in st.session_state.history))
st.write("")
selected_question = None
if st.session_state.history:
    export_col, clear_col, _ = st.columns([1, 1, 2])
    transcript = []
    for message in st.session_state.history:
        transcript.extend([f"## {message['role'].title()}", message['content'], ""])
        for source in message.get("sources", []):
            transcript.extend([f"[{source.label}] {source.filename}, page {source.page}", source.excerpt, ""])
    export_col.download_button("Export chat", "\n".join(transcript), "document-chat.md", "text/markdown", use_container_width=True)
    if clear_col.button("New chat", use_container_width=True):
        st.session_state.history = []
        st.rerun()
    st.markdown('<div class="section-label">Your conversation</div>', unsafe_allow_html=True)
    for message in st.session_state.history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            render_sources(message.get("sources", []))
else:
    title = "What would you like to explore?" if ready else "Your next insight starts here"
    description = "Ask a question or choose an analysis below." if ready else "Add PDFs to your library, then select Analyze documents to begin."
    st.markdown(f'<div class="empty-state"><div class="empty-icon">&#10023;</div><h2>{title}</h2><p>{description}</p></div>', unsafe_allow_html=True)
    prompts = [
        ("Summarize", "Summarize the key points and conclusions in the documents.", "Get the main ideas and conclusions."),
        ("Key details", "What dates, deadlines, and requirements are mentioned?", "Find dates, requirements, and specifics."),
        ("Action items", "What actions or next steps are explicitly recommended in the documents?", "Discover recommended next steps."),
    ]
    for column, (label, prompt, description) in zip(st.columns(3), prompts):
        with column:
            with st.container(border=True):
                if st.button(label, use_container_width=True, disabled=not ready or not cloud_consent or not CLOUD_READY):
                    selected_question = prompt
                st.caption(description)

question = st.chat_input("Ask about your documents..." if ready else "Prepare a document to start chatting", disabled=not ready or not cloud_consent or not CLOUD_READY)
question = selected_question or question
if question:
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        output = st.empty()
        parts = []
        output.caption("Finding relevant passages and starting the model...")
        def display_token(token):
            parts.append(token)
            output.markdown("".join(parts))
        started = time.perf_counter()
        try:
            cache_key = (st.session_state.active_key, question.strip(), chat_model.strip(), retrieval_count, ai_provider())
            cached = st.session_state.answer_cache.get(cache_key)
            if cached is not None:
                answer, sources = cached
            else:
                answer, sources = answer_question(st.session_state.vector_store, question, chat_model,
                                                  num_results=retrieval_count, token_callback=display_token)
                st.session_state.answer_cache[cache_key] = (answer, sources)
                while len(st.session_state.answer_cache) > 32:
                    st.session_state.answer_cache.popitem(last=False)
            output.markdown(answer)
            render_sources(sources)
            st.caption(f"Answered in {time.perf_counter() - started:.1f}s")
            message = {"role": "assistant", "content": answer, "sources": sources}
            st.session_state.history.append(message)
        except (ResponseError, ValueError, ConnectionError, TimeoutError) as error:
            output.empty()
            show_error(error)
