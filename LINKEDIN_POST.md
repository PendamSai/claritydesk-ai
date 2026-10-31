Introducing ClarityDesk AI ? Documents. Meetings. Research.

Useful information often sits in three places: lengthy PDFs, meeting recordings, and scattered research sources. I wanted a single workspace that helps turn that information into something actionable.

I?ve been building ClarityDesk AI, a local app with three assistants:

?? Document Assistant
Ask questions about uploaded PDFs and get answers with source passages and page citations.

??? Meeting Assistant
Transcribe recordings, review and edit the transcript, and generate summaries, decisions, action items, and open questions.

?? Agentic Research Assistant
Combine uploaded documents with optional web search. A bounded workflow chooses follow-up searches, records its activity, and produces a cited report.

The stack: Python, Streamlit, LangChain, Ollama, Faster Whisper, DDGS, and optional FAISS retrieval.

The default workflow runs AI inference locally through Ollama, without a paid AI API. Web research requires internet and sends search queries to external providers.

Building this has reinforced a practical lesson: generating an answer is only part of the work. Retrieval quality, transcription accuracy, useful citations, and response time matter just as much.

The app is still evolving. Accuracy hasn?t been independently benchmarked, and web research currently uses search-result snippets, so important outputs should be checked against the original evidence.

Which would help you most: document Q&A, meeting notes, or research assistance?

#GenerativeAI #RAG #AgenticAI #Python #Ollama
