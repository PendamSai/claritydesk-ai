# Free public setup: Streamlit Community Cloud + Gemini

The app is prepared for this route but has not been published. No paid hosting resources were created. Live Gemini responses have not been tested because no API key has been configured.

## Owner setup

1. Create or use a GitHub repository containing this app. Keep `.env`, `.streamlit/secrets.toml`, private PDFs and meeting recordings out of Git. Choose an open-source license only if you intend to license your code; none has been added automatically.
2. Create a Gemini API key in Google AI Studio on a project using the unpaid/free tier. Do not enable paid billing or attach this app to a paid project if your budget is zero. The app cannot determine your project's billing tier. Never paste your real key into chat or commit it to Git.
3. In Streamlit Community Cloud, create an app from that GitHub repository with entry point `app.py`. Choose Python 3.12 in Advanced settings. Root `requirements.txt` supplies dependencies.
4. Copy `.streamlit/secrets.toml.example` into the app's Secrets settings and replace only the placeholder key. These server-side secrets configure Gemini, the model and the per-process daily request allowance. Local development can instead use an ignored `.env` file or `.streamlit/secrets.toml`.
5. Deploy. Streamlit supplies a public `*.streamlit.app` URL. Test a small PDF, a short recording and a research question before sharing broadly.

Official instructions:
- https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy
- https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management
- https://ai.google.dev/gemini-api/docs/pricing
- https://ai.google.dev/gemini-api/terms

## Free-mode behavior

Set `AI_PROVIDER=gemini`. All generated answers, meeting notes, research planning and report synthesis use the configured Gemini model. The default is `gemini-2.5-flash-lite`, currently listed with a free tier; availability and quotas can change. Keyword retrieval needs no embedding server. Semantic/Ollama retrieval is unavailable in free cloud mode. Local Ollama remains the default when `AI_PROVIDER` is not changed.

Gemini calls use its native REST API without an added SDK. Streaming is retained for document answers. No automatic retries, alternative-model fallback, paid grounding tools, billing setup or payment actions are performed. An HTTP 429 displays a quota message and stops. Two simultaneous Gemini calls per server process are allowed; extra requests receive a busy message. The default app allowance is 100 attempted AI calls per UTC day per server process; failures count, and process restarts reset this counter. This is not a durable per-user limit, a provider quota prediction, or a billing cap. A research or meeting operation can use multiple calls. Keep billing disabled at Google to avoid paid usage.

## Limits and privacy

- At most five PDFs of 10 MB each and 100 total pages per library in free mode.
- Audio is limited to 10 MB with a known duration of at most five minutes. Faster Whisper tiny runs on the hosting CPU; model files download on first use. Transcription can be slow or exceed free hosting resources. Unknown-duration files must be re-encoded before upload.
- Meeting notes accept at most 14,000 transcript characters. Research depth is limited to two rounds.
- Google receives the selected document excerpts, meeting transcript/notes and research context for generation. The app requires explicit consent before AI requests. Free-tier content may be used to improve Google's products; avoid confidential or sensitive material.
- Meeting audio is decoded on the hosting server rather than sent to Gemini. Web research sends queries to external search providers and uses snippets, not full-page verification.
- Hosting and API free tiers have quotas, availability and resource limits. This setup does not promise unlimited visitors, permanent storage or uninterrupted service. Session indexes and reports are not durable user accounts.

## Local use

Leave `AI_PROVIDER=ollama` to keep the original local workflow. Do not commit a real `GEMINI_API_KEY`. To test cloud mode locally, set the Gemini variables in the ignored `.env` file and restart Streamlit.
