# Rose 27 Jul 2026
"""Regression test for the Thai whole-word token fix.

Runs the real Mbart_Based_MLM init, then asserts the text channel is actually alive.
No training required. Run from the SOKE/ directory:

    python scripts/verify_thai_tokens.py
"""
import gzip, json, pickle, sys, torch
import torch.nn.functional as F

sys.path.insert(0, '.')
from mGPT.archs.mgpt_mbart import Mbart_Based_MLM

MODEL_PATH = './deps/mbart-h2s-csl-phoenix'
DICT_PATH = 'scripts/thai_word2en.json'
THAI_ROOT = '../data/Thai_Hand4WholePP'
FAILED = []


def check(label, ok, detail=''):
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}{(' -- ' + detail) if detail else ''}")
    if not ok:
        FAILED.append(label)


def encode(model, texts):
    """text -> the exact encoder input ids the model trains on."""
    enc = model.tokenizer(texts, padding='longest', truncation=True,
                          max_length=model.max_length, add_special_tokens=True,
                          return_tensors='pt')
    ids = enc.input_ids.clone()
    model.map_ids(ids, direction='token_to_emb')
    return ids


BUILD = dict(model_path=MODEL_PATH, model_type='mbart_multi', motion_codebook_size=512,
             hand_codebook_size=192, rhand_codebook_size=192, num_heads=3)


def main():
    print('=== building baseline (Thai tokens OFF) ===')
    base = Mbart_Based_MLM(**BUILD)
    base_len = len(base.tok_id_to_emb_id)
    base_emb = base.language_model.main_lm.get_input_embeddings().weight.data.clone()
    print(f'  len_token = {base_len}')

    print('\n=== building fixed model (Thai tokens ON, dict-init) ===')
    model = Mbart_Based_MLM(**BUILD, thai_word_tokens_path=DICT_PATH, thai_word_init='dict')
    n_new = len(model.thai_word_str)
    new_len = len(model.tok_id_to_emb_id)
    print(f'  len_token = {new_len}  (+{new_len - base_len})')

    print('\n=== 0. dict file sanity ===')
    spec = json.load(open(DICT_PATH, encoding='utf-8'))
    entries = {k: v for k, v in spec.get('words', spec).items() if not k.startswith('_')}
    check('every entry has a non-empty string donor',
          all(isinstance(v, str) and v.strip() for v in entries.values()))
    check('no duplicate donors hiding a copy-paste slip',
          len(set(entries.values())) == len(entries),
          f'{len(entries) - len(set(entries.values()))} repeated')
    check('no Thai key contains whitespace', not any(k != k.strip() or ' ' in k for k in entries))
    check('loader saw every entry', len(model.thai_word2en) == len(entries),
          f'{len(model.thai_word2en)} loaded / {len(entries)} in file')
    check('_meta not mistaken for a word', '_meta' not in model.thai_word2en)

    print('\n=== 1. vocab / embedding table ===')
    check('all Thai words got an embedding id', new_len == base_len + n_new,
          f'{new_len - base_len} added for {n_new} words')
    emb = model.language_model.main_lm.get_input_embeddings().weight.data
    check('embedding matrix grew to match', emb.shape[0] == new_len, str(tuple(emb.shape)))
    # Only the first len(map_ids.pkl) rows come from the checkpoint. Rows above that are
    # created by resize_token_embeddings() and are randomly drawn per instantiation, so
    # comparing them across two builds is meaningless.
    n_pretrained = len(pickle.load(open(f'{MODEL_PATH}/map_ids.pkl', 'rb')))
    check('pretrained rows untouched',
          torch.equal(emb[:n_pretrained], base_emb[:n_pretrained]), f'first {n_pretrained} rows')

    rows = [model.tok_id_to_emb_id[model.tokenizer.convert_tokens_to_ids(w)]
            for w in model.thai_word_str]
    check('every Thai row is a fresh row', all(r >= base_len for r in rows))
    unk_row = model.tok_id_to_emb_id[model.tokenizer.convert_tokens_to_ids('<unk>')]
    check('no Thai row equals the <unk> row',
          not any(torch.allclose(emb[r], emb[unk_row]) for r in rows))
    check('all Thai rows distinct', len({tuple(emb[r].tolist()) for r in rows}) == len(rows))
    n_ratio = emb[rows].norm(dim=1).mean() / emb[:base_len].norm(dim=1).mean()
    check('Thai row norms match pretrained scale', 0.5 < n_ratio < 2.0, f'ratio {n_ratio:.3f}')

    print('\n=== 2. output vocab masking ===')
    masked = [bool(model.language_model.mask_body[r] == float('-inf')) for r in rows]
    check('Thai tokens masked out of the body head', all(masked),
          f'{sum(masked)}/{len(rows)}')

    print('\n=== 3. text channel liveness (the metric that mattered) ===')
    for split in ['train', 'val', 'test']:
        data = pickle.load(gzip.open(f'{THAI_ROOT}/val_vid.{split}', 'rb'))
        texts = [r['text'] for r in data]
        uniq = len(set(texts))
        ids = encode(model, texts)
        distinct = len({tuple(r.tolist()) for r in ids})
        n_unk = int((ids == unk_row).sum())
        check(f'{split}: distinct encoder inputs == unique texts',
              distinct == uniq, f'{distinct}/{uniq} distinct, {n_unk} <unk> cells')

    print('\n=== 4. -100 loss mask survives map_ids ===')
    probe = torch.tensor([[-100, model.tokenizer.convert_tokens_to_ids('</s>'), -100]])
    model.map_ids(probe, direction='token_to_emb')
    check('-100 passes through untouched', int(probe[0][0]) == -100 and int(probe[0][2]) == -100,
          str(probe.tolist()))

    print('\n=== 5. dict-init semantics ===')
    def vec(w):
        return emb[model.tok_id_to_emb_id[model.tokenizer.convert_tokens_to_ids(w)]]
    def cos(a, b):
        return F.cosine_similarity(vec(a), vec(b), dim=0).item()
    for a, b, c, d in [('เช้า', 'เย็น', 'เช้า', 'ปากกา'),
                       ('พ่อ', 'แม่', 'พ่อ', 'โรงเรียน'),
                       ('เดิน', 'วิ่ง', 'เดิน', 'ทีวี')]:
        check(f'cos({a},{b}) > cos({c},{d})', cos(a, b) > cos(c, d),
              f'{cos(a, b):+.3f} vs {cos(c, d):+.3f}')

    print('\n=== 6. init modes are genuinely different ===')
    def thai_rows(m):
        e = m.language_model.main_lm.get_input_embeddings().weight.data
        return e[[m.tok_id_to_emb_id[m.tokenizer.convert_tokens_to_ids(w)]
                  for w in m.thai_word_str]].clone()

    shuf = Mbart_Based_MLM(**BUILD, thai_word_tokens_path=DICT_PATH,
                           thai_word_init='shuffled', thai_word_init_seed=1)
    shuf2 = Mbart_Based_MLM(**BUILD, thai_word_tokens_path=DICT_PATH,
                            thai_word_init='shuffled', thai_word_init_seed=2)
    non = Mbart_Based_MLM(**BUILD, thai_word_tokens_path=DICT_PATH, thai_word_init='none')
    R = {'dict': thai_rows(model), 'shuffled@1': thai_rows(shuf),
         'shuffled@2': thai_rows(shuf2), 'none': thai_rows(non)}

    check('dict != shuffled', not torch.equal(R['dict'], R['shuffled@1']))
    check('shuffled != none', not torch.equal(R['shuffled@1'], R['none']))
    check('dict != none', not torch.equal(R['dict'], R['none']))
    check('different seeds give different derangements',
          not torch.equal(R['shuffled@1'], R['shuffled@2']))
    same = Mbart_Based_MLM(**BUILD, thai_word_tokens_path=DICT_PATH,
                           thai_word_init='shuffled', thai_word_init_seed=1)
    check('same seed is reproducible', torch.equal(R['shuffled@1'], thai_rows(same)))
    # Same 31 donor vectors in both arms, just attached to different Thai words: the multiset
    # of rows must therefore be identical. This is what makes the comparison controlled.
    check('shuffled reuses exactly the dict donor vectors, only re-paired',
          torch.allclose(R['dict'].sum(1).sort().values,
                         R['shuffled@1'].sum(1).sort().values, atol=1e-5))
    p = shuf.thai_donor_pairing()
    check('derangement leaves ZERO correct pairings',
          sum(p[w] == shuf.thai_word2en[w] for w in p) == 0)
    check("'none' rows really are all the same vector (the degenerate default)",
          bool((torch.cdist(R['none'], R['none']).max() < 1e-2).item()),
          f"max pairwise dist {torch.cdist(R['none'], R['none']).max():.5f}")
    try:
        Mbart_Based_MLM(**BUILD, thai_word_tokens_path=DICT_PATH, thai_word_init='typo')
        check('an unknown init mode is rejected', False, 'no error raised')
    except AssertionError:
        check('an unknown init mode is rejected', True)

    print('\n=== 7. other languages unaffected ===')
    for lbl, t in [('de', 'und nun die wettervorhersage für morgen'),
                   ('en', 'the weather tomorrow'), ('zh', '我今天很高兴')]:
        ids = model.tokenizer(t, add_special_tokens=False).input_ids
        b_ids = base.tokenizer(t, add_special_tokens=False).input_ids
        check(f'{lbl}: tokenization identical to baseline', ids == b_ids)

    print('\n' + '=' * 60)
    if FAILED:
        print(f'FAILED {len(FAILED)}:')
        for f in FAILED:
            print(f'   - {f}')
        sys.exit(1)
    print('ALL CHECKS PASSED')


if __name__ == '__main__':
    main()
