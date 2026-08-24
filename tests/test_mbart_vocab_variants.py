import pickle
import unittest

from transformers import MBartTokenizer

from mGPT.archs.mgpt_mbart import get_new_lang_tokens


class MbartVocabularyVariantTest(unittest.TestCase):
    def test_historical_vocabulary_sizes(self):
        model_path = "deps/mbart-h2s-csl-phoenix"
        with open(f"{model_path}/map_ids.pkl", "rb") as file:
            base_mapping = pickle.load(file)

        motion = [f"<motion_id_{index}>" for index in range(99)]
        hand = [f"<hand_id_{index}>" for index in range(195)]
        right_hand = [f"<rhand_id_{index}>" for index in range(195)]

        for include_thai, expected in ((False, 19504), (True, 19505)):
            with self.subTest(include_thai_source_token=include_thai):
                tokenizer = MBartTokenizer.from_pretrained(model_path, legacy=True)
                language = get_new_lang_tokens("mbart_multi", include_thai)
                tokenizer.add_tokens(language, special_tokens=True)
                tokenizer.add_tokens(motion + hand + right_hand)

                mapping = dict(base_mapping)
                next_index = len(mapping)
                for token in [*language, *motion, *hand, *right_hand]:
                    mapping[tokenizer.convert_tokens_to_ids(token)] = next_index
                    next_index += 1

                self.assertEqual(len(mapping), expected)
                self.assertEqual(next_index, expected)


if __name__ == "__main__":
    unittest.main()
