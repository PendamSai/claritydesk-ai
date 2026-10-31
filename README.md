# ClarityDesk AI

**Documents. Meetings. Research.**

A local AI workspace combining document question answering, meeting transcription and notes, and bounded agentic research. Built with Streamlit, LangChain, Ollama, Faster Whisper, DDGS, and optional FAISS retrieval. Document answers include source names and page numbers; web research uses search-result snippets. The default Ollama workflow runs model inference on your computer, while web research needs internet.

The current workflow is a local app with Ollama. Optional cloud configuration is documented separately in [FREE_HOSTING.md](FREE_HOSTING.md); no public website is deployed.

## Requirements

- Python 3.10-3.13 recommended. Python 3.14 currently emits a Pydantic compatibility warning from the LangChain dependencies.
- [Ollama](https://ollama.com/) running locally
- Enough memory to run the selected models

## Setup (Windows PowerShell)

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
ollama pull llama3.2
ollama pull nomic-embed-text
Copy-Item .env.example .env
streamlit run app.py
```

If document processing reports that `nomic-embed-text` is missing, run
`ollama pull nomic-embed-text` in a separate PowerShell window, then retry
**Process documents**. Pull the model configured in the sidebar instead if you
changed the embedding model name.

If Ollama reports that its local `/tokenize` endpoint refused a connection,
the model runner has stopped responding. Restart Ollama from the Windows system
tray and retry **Index documents**. Embeddings are sent in small batches; if
the runner still stops, try fewer PDFs at once, update Ollama, and confirm the
selected model is installed with `ollama list`.

If PowerShell blocks virtual-environment activation, use the environment's executable directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open the local Streamlit URL and use the library panel to upload one or more text-based PDFs. Submit with **Index documents** to begin. Progress shows page extraction, passage creation, and embedding batches, with elapsed time and a final passage count. Text is extracted directly from the uploaded bytes without temporary PDF files. The indexer uses 1,500-character passages with 100-character overlap and sends 8 passages per embedding request to reduce indexing work while preserving page citations. First-time indexing can take longer while Ollama loads the embedding model; processing speed depends on PDF length and local hardware.

Ask questions in the chat or use a suggested prompt. The workspace shows document sizes, the number of indexed passages and questions, and supporting excerpts for answers. Use **Passages per answer** under **Model settings** to adjust retrieval, export a conversation as Markdown, or start a new chat without re-indexing your PDFs. The app does not upload documents to a hosted service; Ollama processes them on the local machine.

Set `CHAT_MODEL` and `EMBEDDING_MODEL` in `.env` to use other models already available in Ollama. For example, pull the selected models with `ollama pull <model-name>`. Chat and embedding models serve different roles; choose an embedding model supported by Ollama's embedding API.

## Notes

- The FAISS index and conversation history live in Streamlit session state and are not saved to disk.
- Scanned/image-only PDFs require OCR before they can be searched.
- Answers are grounded in the retrieved excerpts and include source labels such as `[S1]`; check the cited pages for important decisions.
- Reprocessing a different selection of PDFs replaces the current index and clears the chat.

## Tests

```powershell
python -m unittest discover -s tests
```

## Responsiveness and workspace

The workspace now focuses on uploading documents and chatting, with model settings in the sidebar. Select **Prepare documents** after uploading. The two most recently prepared indexes are reused within your session, keyed by PDF names, content, and embedding model. Switching the upload selection does not remove the working index; it changes only after preparation succeeds. Cached indexes are released when the session ends.

Answers stream as tokens arrive, with a 256-token generation limit and a ten-minute chat-model keep-alive. Initial PDF embedding and the first model load still depend on local hardware and document size. These changes avoid repeated indexing and reduce the wait for visible answer text; they do not guarantee faster first-time embedding.

## Direct answers

Every chat question now generates a direct, cited answer with Ollama; the passage-only response mode has been removed from the UI. Supporting passages appear in the collapsed Sources panel. Under Advanced settings, fast keyword retrieval avoids embedding time; semantic retrieval supports related wording at the cost of model-based indexing. Both feed the same answer generator. Existing prepared indexes continue to work. Repeated questions with the same document index, model, and retrieval count reuse up to 32 cached answers within the session. CPU generation can still be slow on new questions.

## AI Document Assistant (RAG Chatbot) workspace

The interface uses a light document library, lavender workspace header, analysis shortcuts, and source-backed chat cards. Select **Analyze documents** after uploading PDFs. Summarize, Key details, and Action items send focused questions through the existing answer generator. Source excerpts remain available beneath each answer.

Verification controls, badges, and automatic verification calls have been removed from the app. Chat no longer waits for a second model review. The standalone verification utility remains available in the backend.

## AI Meeting Assistant

Choose Meeting Assistant in the workspace selector. Upload MP3, WAV, M4A, FLAC, OGG, MP4 or WebM audio (up to 100 MB), select a transcription model, and click Transcribe recording. Faster Whisper runs locally on CPU with int8 inference; first use downloads the chosen speech-model files. Tiny is the fastest default, while base and small need more resources. Transcripts include segment timestamps, not speaker diarization. Review and edit recognition errors before creating meeting notes. Reports contain Summary, Decisions, Action items (owner and due date when stated), and Open questions. Both transcript and report can be downloaded. Long transcripts are reviewed in sections; inputs over 64,000 characters must be split. CPU transcription and multiple model calls can take time.

## Agentic AI Research Assistant

Choose Research Assistant. Enter a topic and enable web search to combine prepared PDFs with public search-result snippets, or disable web search to use PDFs only. The workflow gathers initial evidence, asks the model to select follow-up document/web searches or stop, and writes a cited report. Research depth limits the number of rounds to five. Only document search and web search tools are executable. Activity, unavailable searches, source excerpts and web links are displayed, and reports are downloadable. Web snippets are not full-page analysis, and search providers may fail or rate-limit requests. No API key is required for DDGS search.

Meeting audio and PDFs are processed locally. In web-enabled research, the objective and generated search queries leave the device; avoid confidential objectives and use document-only mode for private topics. The agent is instructed to use public topic terms for web queries, but that instruction is not a privacy guarantee. All reports, transcripts and caches live in the current session; do not rely on them as permanent storage.

## Gemini free cloud mode

All three assistants now support Gemini generation with keyword document retrieval and explicit data-sharing consent. No paid resources are provisioned. Follow [FREE_HOSTING.md](FREE_HOSTING.md) for owner setup, limitations, and secret configuration. Existing Ollama behavior remains available with `AI_PROVIDER=ollama`.
