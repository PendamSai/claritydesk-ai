import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
from langchain_core.documents import Document
from assistants import meeting_notes, research_report, transcribe_audio

class MeetingTests(unittest.TestCase):
    def test_transcription_preserves_timestamps(self):
        speech = Mock()
        speech.transcribe.return_value = (iter([SimpleNamespace(start=0.0,end=4.5,text=" Alice will send the report Friday. ")]), SimpleNamespace())
        with patch("assistants.speech_model", return_value=speech):
            text = transcribe_audio(b"recording")
        self.assertEqual(text, "[0.0s - 4.5s] Alice will send the report Friday.")
        self.assertTrue(speech.transcribe.call_args.kwargs["vad_filter"])

    def test_empty_audio_and_no_speech_are_clear_errors(self):
        with self.assertRaises(ValueError):
            transcribe_audio(b"")
        with patch("assistants.speech_model") as model:
            model.return_value.transcribe.return_value = (iter([]), SimpleNamespace())
            with self.assertRaisesRegex(ValueError, "No speech"):
                transcribe_audio(b"silence")

    def test_notes_review_all_sections(self):
        with patch("assistants.ChatOllama") as llm:
            llm.return_value.invoke.return_value = SimpleNamespace(content="Summary and action items")
            text = "[0s] Alice will send the report Friday.\n" * 400
            result = meeting_notes(text, "llama3.2")
            prompts = [call.args[0] for call in llm.return_value.invoke.call_args_list]
            self.assertGreater(len(prompts), 2)
            self.assertIn("Never invent", prompts[0])
            self.assertIn("Evidence timestamp", prompts[-1])
            self.assertEqual(result, "Summary and action items")

class ResearchTests(unittest.TestCase):
    def test_agent_chooses_followup_and_cites_deduplicated_evidence(self):
        store = Mock()
        doc = Document(page_content="Local evidence", metadata={"source":"local.pdf","page":0})
        store.similarity_search.return_value = [doc]
        search = Mock(return_value=[Document(page_content="Web evidence",metadata={"source":"https://example.org"})])
        with patch("assistants.ChatOllama") as llm:
            llm.return_value.invoke.side_effect = [SimpleNamespace(content='{ "tool": "web", "query": "follow up" }'), SimpleNamespace(content="Findings [S1] [S2]")]
            report, sources, activity = research_report("Topic",store,"model",True,2,search_fn=search)
        self.assertEqual(search.call_count, 2)
        self.assertEqual(len(sources), 2)
        self.assertEqual(len(activity), 3)
        self.assertIn("[S1]", report)
        self.assertEqual(sources[0].page, 1)

    def test_document_only_mode_never_calls_web(self):
        store = Mock()
        store.similarity_search.return_value = [Document(page_content="Evidence")]
        search = Mock()
        with patch("assistants.ChatOllama") as llm:
            llm.return_value.invoke.return_value = SimpleNamespace(content="Report [S1]")
            research_report("Topic",store,"model",False,1,search_fn=search)
        search.assert_not_called()

    def test_failed_web_does_not_fabricate_report(self):
        search = Mock(side_effect=RuntimeError("offline"))
        report, sources, activity = research_report("Topic", None, "model", True, 1, search_fn=search)
        self.assertEqual(sources, [])
        self.assertIn("No usable evidence", report)
        self.assertIn("unavailable", activity[0]["result"])

    def test_unknown_tool_cannot_be_executed(self):
        search = Mock(return_value=[Document(page_content="Evidence")])
        with patch("assistants.ChatOllama") as llm:
            llm.return_value.invoke.side_effect = [SimpleNamespace(content='{ "tool": "shell", "query": "anything" }'), SimpleNamespace(content="Report")]
            _, _, activity = research_report("Topic",None,"model",True,2,search_fn=search)
        self.assertEqual(search.call_count, 1)
        self.assertIn("Invalid next action", activity[-1]["result"])
