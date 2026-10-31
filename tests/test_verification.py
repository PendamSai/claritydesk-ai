import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from rag import SourceReference, verify_answer

class VerificationTests(unittest.TestCase):
    def review(self, claims=None, answer="Payment is due in 30 days. [S1]"):
        if claims is None:
            claims = [{"id":"C1", "verdict":"supported", "source_ids":["S1"]}]
        with patch("rag.ChatOllama") as model:
            model.return_value.invoke.return_value = SimpleNamespace(content=json.dumps({"claims": claims}))
            return verify_answer(answer, [SourceReference("S1", "terms.pdf", 1, "Payment is due in 30 days.")], "llama3.2")

    def test_supported_uses_original_source_text(self):
        report = self.review()
        self.assertEqual(report["status"], "supported")
        self.assertIn("Payment is due in 30 days.", report["claims"][0]["evidence"])

    def test_paraphrase_needs_no_exact_claim_or_quote_copy(self):
        report = self.review(answer="You must pay within thirty days. [S1]")
        self.assertEqual(report["status"], "supported")
        self.assertEqual(report["claims"][0]["claim"], "You must pay within thirty days.")

    def test_unknown_sources_and_empty_evidence_are_uncertain(self):
        for labels in [["S99"], [], [None], "S1"]:
            self.assertEqual(self.review([{"id":"C1", "verdict":"supported", "source_ids":labels}])["status"], "uncertain")

    def test_contradictions_remain_unsupported(self):
        self.assertEqual(self.review([{"id":"C1", "verdict":"unsupported", "source_ids":["S1"]}])["status"], "unsupported")

    def test_missing_claim_is_not_silently_dropped(self):
        report = self.review(answer="Payment is due in 30 days. Delivery is Friday. [S1]")
        self.assertEqual(report["status"], "uncertain")
        self.assertEqual(len(report["claims"]), 2)
        self.assertIn("omitted", report["claims"][1]["reason"])

    def test_duplicate_and_unknown_claim_ids_are_uncertain(self):
        raw = {"id":"C1", "verdict":"supported", "source_ids":["S1"]}
        self.assertEqual(self.review([raw, raw])["status"], "uncertain")
        self.assertEqual(self.review([raw, {"id":"C99"}])["status"], "uncertain")

    def test_unknown_answer_citation_is_uncertain(self):
        self.assertEqual(self.review(answer="Payment is due in 30 days. [S99]")["status"], "uncertain")

    def test_invalid_verdict_and_empty_review_are_uncertain(self):
        self.assertEqual(self.review([{"id":"C1", "verdict":"maybe"}])["status"], "uncertain")
        self.assertEqual(self.review([])["status"], "uncertain")

    def test_no_sources_skips_model(self):
        with patch("rag.ChatOllama") as model:
            self.assertEqual(verify_answer("Claim", [], "llama3.2")["status"], "uncertain")
            model.assert_not_called()

    def test_malformed_output_is_uncertain(self):
        with patch("rag.ChatOllama") as model:
            model.return_value.invoke.return_value = SimpleNamespace(content="not JSON")
            self.assertEqual(verify_answer("Claim", [SourceReference("S1","a.pdf",1,"Text")], "llama3.2")["status"], "uncertain")
