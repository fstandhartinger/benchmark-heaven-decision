import unittest
from unittest.mock import patch

from bh_classifier.policy import should_escalate, temper
from bh_classifier.service import classify, systemone


class PolicyTests(unittest.TestCase):
    def test_non_object_payload_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "object"):
            classify([])

    def test_systemone_bounds_questions(self):
        questions = {str(i): {"type": "noul", "instructions": "x", "criteria": {}} for i in range(9)}
        with self.assertRaisesRegex(ValueError, "1..8"):
            systemone({"state": "x", "questions": questions})

    def test_temperature_preserves_winner_and_normalizes(self):
        out = temper({"a": .8, "b": .2})
        self.assertEqual(max(out, key=out.get), "a")
        self.assertAlmostEqual(sum(out.values()), 1)
        self.assertLess(out["a"], .8)

    def test_non_finite_score_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            temper({"a": float("nan"), "b": 1})

    def test_gate_is_strict(self):
        self.assertFalse(should_escalate({"a": .85, "b": .15}))
        self.assertTrue(should_escalate({"a": .849, "b": .151}))

    @patch("bh_classifier.service.classifier_fast")
    def test_confident_answer_is_not_escalated(self, fast):
        fast.return_value = ([{"scores": {"a": .9, "b": .1}, "ms": 2}], .002)
        out = classify({"inputs": ["x"], "labels": ["a", "b"]})
        self.assertEqual(out["usage"]["escalated"], 0)
        self.assertAlmostEqual(sum(out["results"][0]["scores"].values()), 1)

    @patch("bh_classifier.service.gemini_decision")
    @patch("bh_classifier.service.classifier_fast")
    def test_uncertain_answer_gets_scored_escalation(self, fast, gemini):
        fast.return_value = ([{"scores": {"a": .6, "b": .4}, "ms": 2}], .002)
        gemini.return_value = ({"label": "b", "scores": {"a": .1, "b": .9},
                                "usage": {"input_tokens": 10, "output_tokens": 2, "thinking_tokens": 3}}, .5)
        out = classify({"inputs": ["x"], "labels": ["a", "b"]})
        self.assertEqual(out["results"][0]["label"], "b")
        self.assertTrue(out["results"][0]["escalated"])
        self.assertIsNotNone(out["results"][0]["confidence"])

    @patch("bh_classifier.service.classify")
    def test_systemone_choice_shape(self, classify_mock):
        classify_mock.return_value = {"results": [{"label": "red", "scores": {"red": .8, "blue": .2}}],
                                      "usage": {"estimated_cost_usd": .001}}
        out = systemone({"state": "x", "questions": {"color": {"type": "choice", "instructions": "color?",
                                                                    "criteria": {"red": "warm", "blue": "cool"}}}})
        self.assertEqual(out["answers"]["color"]["choice"], "red")
        sent = classify_mock.call_args.args[0]["instructions"]
        self.assertIn("red: warm", sent)


if __name__ == "__main__":
    unittest.main()
