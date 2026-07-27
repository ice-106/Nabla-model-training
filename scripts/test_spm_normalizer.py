"""Regression tests for mBART SentencePiece source normalization.

Run from the repository root:

    python -m unittest scripts.test_spm_normalizer -v

The defaults point at the model and Thai corpus used by SOKE. They can be
overridden with SOKE_MBART_MODEL_PATH and SOKE_THAI_DATA_PATH.
"""

import gzip
import os
import pickle
import unittest
from pathlib import Path
from types import SimpleNamespace

from transformers import MBartTokenizer

from mGPT.archs.mgpt_mbart import Mbart_Based_MLM


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = Path(
    os.environ.get(
        "SOKE_MBART_MODEL_PATH",
        REPO_ROOT / "deps" / "mbart-h2s-csl-phoenix",
    )
)
THAI_DATA_PATH = Path(
    os.environ.get(
        "SOKE_THAI_DATA_PATH",
        REPO_ROOT.parent / "data" / "Thai_Hand4WholePP",
    )
)


class SentencePieceNormalizerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = MBartTokenizer.from_pretrained(
            MODEL_PATH,
            legacy=True,
        )
        # Exercise the production helper without loading the 400M-parameter LM.
        cls.normalizer = SimpleNamespace(tokenizer=cls.tokenizer)

    @classmethod
    def normalize(cls, texts):
        return Mbart_Based_MLM._normalize_source_texts(cls.normalizer, texts)

    @classmethod
    def pieces(cls, text):
        normalized = cls.normalize([text])[0]
        return cls.tokenizer.tokenize(normalized)

    def assert_has_no_unk(self, pieces):
        token_ids = self.tokenizer.convert_tokens_to_ids(pieces)
        self.assertNotIn(self.tokenizer.unk_token_id, token_ids)

    def test_exact_thai_sentence(self):
        sentence = "ฉันกำลังไปโรงเรียน"
        normalized = self.normalize([sentence])[0]
        pieces = self.tokenizer.tokenize(normalized)

        self.assertEqual(normalized, "▁ฉันกําลังไปโรงเรียน")
        self.assertEqual(pieces, ["▁ฉัน", "กําลัง", "ไป", "โรงเรียน"])
        self.assert_has_no_unk(pieces)
        print(f"\nRepresentative: {sentence} -> {pieces}")

    def test_mixed_english_thai_prompt(self):
        prompt = (
            "Please create a motion for the caption: "
            "ฉันกำลังไปโรงเรียน"
        )
        pieces = self.pieces(prompt)

        self.assert_has_no_unk(pieces)
        thai_start = pieces.index("▁ฉัน")
        self.assertEqual(
            pieces[thai_start : thai_start + 4],
            ["▁ฉัน", "กําลัง", "ไป", "โรงเรียน"],
        )

    def test_real_thai_corpus(self):
        annotations = []
        for split in ("train", "val", "test"):
            annotation_path = THAI_DATA_PATH / f"val_vid.{split}"
            with gzip.open(annotation_path, "rb") as annotation_file:
                annotations.extend(pickle.load(annotation_file))

        sentences = [annotation["text"] for annotation in annotations]
        normalized_sentences = self.normalize(sentences)
        tokenized_sentences = [
            self.tokenizer.tokenize(sentence)
            for sentence in normalized_sentences
        ]
        all_pieces = {
            piece
            for sentence_pieces in tokenized_sentences
            for piece in sentence_pieces
        }
        unk_count = sum(
            self.tokenizer.unk_token_id
            in self.tokenizer.convert_tokens_to_ids(sentence_pieces)
            for sentence_pieces in tokenized_sentences
        )

        self.assertEqual(len(sentences), 149)
        self.assertEqual(unk_count, 0)
        self.assertEqual(len(all_pieces), 33)
        print(
            "\nCorpus gate: "
            f"sentences={len(sentences)}, unk_sentences={unk_count}, "
            f"distinct_pieces={len(all_pieces)}"
        )
        print(f"Real SentencePiece tokens ({len(all_pieces)}):")
        print(sorted(all_pieces))


if __name__ == "__main__":
    unittest.main(verbosity=2)
