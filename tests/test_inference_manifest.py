import json
import tempfile
import unittest
from pathlib import Path

from mGPT.utils.inference_manifest import load_inference_manifest


class InferenceManifestTest(unittest.TestCase):
    def write_manifest(self, directory, records):
        path = Path(directory) / "manifest.json"
        path.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
        return path

    def test_loads_hyphenated_ids_and_unicode_text(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_manifest(directory, [
                {"id": "csl-1", "text": "你好", "src": "csl"},
                {"id": "-youtube", "text": "first", "src": "how2sign"},
                {"id": "--youtube", "text": "second", "src": "how2sign"},
            ])
            texts, names = load_inference_manifest(path, "how2sign")
        self.assertEqual(texts, ["first", "second"])
        self.assertEqual(names, ["-youtube", "--youtube"])

    def test_rejects_missing_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_manifest(directory, [{"id": "clip", "src": "csl"}])
            with self.assertRaisesRegex(ValueError, "missing fields"):
                load_inference_manifest(path, "csl")

    def test_rejects_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_manifest(directory, [
                {"id": "same", "text": "one", "src": "csl"},
                {"id": "same", "text": "two", "src": "how2sign"},
            ])
            with self.assertRaisesRegex(ValueError, "duplicate ids"):
                load_inference_manifest(path, "csl")

    def test_rejects_absent_source(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_manifest(directory, [
                {"id": "clip", "text": "text", "src": "csl"},
            ])
            with self.assertRaisesRegex(ValueError, "no records for source 'how2sign'"):
                load_inference_manifest(path, "how2sign")


if __name__ == "__main__":
    unittest.main()
