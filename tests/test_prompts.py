import unittest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.prompts.loader import get_prompt, get_prompt_metadata, list_prompts

class TestGetPrompt(unittest.TestCase):

    def test_rag_answer_fills_correctly(self):
        result = get_prompt("rag_answer", context="Some context.", question="What is this?")
        self.assertIn("Some context.", result)
        self.assertIn("What is this?", result)

    def test_query_rewrite_fills_correctly(self):
        result = get_prompt("query_rewrite", message="I can't log in!!!")
        self.assertIn("I can't log in!!!", result)

    def test_fallback_fills_correctly(self):
        result = get_prompt("fallback", question="Do you accept crypto?")
        self.assertIn("Do you accept crypto?", result)

    def test_classifier_zero_shot(self):
        result = get_prompt("classifier_label", examples="", ticket_text="App is broken.")
        self.assertIn("App is broken.", result)
        self.assertNotIn("Examples:", result)

    def test_classifier_few_shot(self):
        examples = "Examples:\nTicket: App crashes.\nCategory: technical\n\n"
        result = get_prompt("classifier_label", examples=examples, ticket_text="Login broken.")
        self.assertIn("Examples:", result)
        self.assertIn("Login broken.", result)

    def test_missing_variable_raises_error(self):
        with self.assertRaises(ValueError):
            get_prompt("rag_answer", context="only context")

    def test_missing_template_raises_error(self):
        with self.assertRaises(FileNotFoundError):
            get_prompt("nonexistent_template", foo="bar")

    def test_version_match_passes(self):
        result = get_prompt("rag_answer", version="v1.0", context="ctx", question="q")
        self.assertIsInstance(result, str)

    def test_version_mismatch_raises_error(self):
        with self.assertRaises(ValueError):
            get_prompt("rag_answer", version="v9.9", context="ctx", question="q")

    def test_header_stripped_from_output(self):
        result = get_prompt("rag_answer", context="ctx", question="q")
        self.assertNotIn("# v1.0", result)

    def test_result_is_string(self):
        result = get_prompt("fallback", question="test")
        self.assertIsInstance(result, str)

    def test_result_not_empty(self):
        result = get_prompt("fallback", question="test")
        self.assertTrue(len(result) > 0)


class TestGetPromptMetadata(unittest.TestCase):

    def test_returns_dict(self):
        meta = get_prompt_metadata("rag_answer")
        self.assertIsInstance(meta, dict)

    def test_has_required_keys(self):
        meta = get_prompt_metadata("rag_answer")
        for key in ["name", "version", "purpose", "path"]:
            self.assertIn(key, meta)

    def test_version_format(self):
        meta = get_prompt_metadata("rag_answer")
        self.assertRegex(meta["version"], r"v\d+\.\d+")

    def test_purpose_not_empty(self):
        meta = get_prompt_metadata("rag_answer")
        self.assertTrue(len(meta["purpose"]) > 0)

    def test_missing_template_raises_error(self):
        with self.assertRaises(FileNotFoundError):
            get_prompt_metadata("nonexistent")

    def test_all_templates_have_metadata(self):
        for name in ["rag_answer", "query_rewrite", "classifier_label", "fallback", "system_base"]:
            meta = get_prompt_metadata(name)
            self.assertIsNotNone(meta["version"])


class TestListPrompts(unittest.TestCase):

    def test_returns_list(self):
        result = list_prompts()
        self.assertIsInstance(result, list)

    def test_has_five_templates(self):
        result = list_prompts()
        self.assertEqual(len(result), 5)

    def test_each_item_is_dict(self):
        for item in list_prompts():
            self.assertIsInstance(item, dict)

    def test_names_are_correct(self):
        names = {p["name"] for p in list_prompts()}
        expected = {"rag_answer", "query_rewrite", "classifier_label", "fallback", "system_base"}
        self.assertEqual(names, expected)

if __name__ == "__main__":
    unittest.main()
