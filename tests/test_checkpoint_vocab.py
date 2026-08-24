import unittest

import torch

from mGPT.utils.load_checkpoint import _validate_checkpoint_vocab


VOCAB_KEYS = (
    "lm.language_model.main_lm.final_logits_bias",
    "lm.language_model.main_lm.model.shared.weight",
    "lm.language_model.main_lm.model.encoder.embed_tokens.weight",
    "lm.language_model.main_lm.model.decoder.embed_tokens.weight",
    "lm.language_model.main_lm.lm_head.weight",
)


def vocab_state(rows):
    state = {}
    for key in VOCAB_KEYS:
        shape = (1, rows) if key.endswith("final_logits_bias") else (rows, 4)
        state[key] = torch.empty(shape)
    return state


class DummyModel:
    def __init__(self, rows):
        self.rows = rows

    def state_dict(self):
        return vocab_state(self.rows)


class CheckpointVocabularyTest(unittest.TestCase):
    def test_accepts_matching_checkpoint_vocabulary(self):
        _validate_checkpoint_vocab(DummyModel(19504), vocab_state(19504))
        _validate_checkpoint_vocab(DummyModel(19505), vocab_state(19505))

    def test_rejects_crossed_checkpoint_variant(self):
        with self.assertRaises(RuntimeError) as error:
            _validate_checkpoint_vocab(DummyModel(19505), vocab_state(19504))
        message = str(error.exception)
        self.assertIn("Checkpoint vocab size(s): [19504]", message)
        self.assertIn("model vocab size(s): [19505]", message)
        self.assertIn("lm.mbart_h2s_csl_phoenix_thai", message)


if __name__ == "__main__":
    unittest.main()
