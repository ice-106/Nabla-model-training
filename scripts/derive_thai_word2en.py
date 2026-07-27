# Rose 27 Jul 2026
"""Regenerate the evidence behind scripts/thai_word2en.json.

The dict itself stores only `thai -> english`, because anything derivable should be
derived, not stored where it can rot. This script reproduces the reasoning on demand:

    python scripts/derive_thai_word2en.py                    # evidence for the current dict
    python scripts/derive_thai_word2en.py --corpus _ENMT     # what the ENMT corpus would suggest

Association score for (thai word w, english token t):

    P(t appears | sentence contains w)  -  P(t appears)

High score = t is unusually frequent exactly where w is, i.e. a likely translation.
A near-zero score does NOT mean the gloss is wrong: function words like 'ที่' appear
in almost every sentence, so no statistic can isolate them. Those were chosen by hand.
"""
import argparse, collections, gzip, json, pickle, re, sys

DATA = '../data'


def load_pairs(corpus):
    th = pickle.load(gzip.open(f'{DATA}/Thai_Hand4WholePP/val_vid.train', 'rb'))
    en = {r['name']: r['text']
          for r in pickle.load(gzip.open(f'{DATA}/Thai_Hand4WholePP{corpus}/val_vid.train', 'rb'))}
    return th, en


def segment(text, lexicon):
    """Greedy longest-match, the same rule HF's added-token trie uses."""
    out, i = [], 0
    while i < len(text):
        for w in lexicon:
            if text.startswith(w, i):
                out.append(w)
                i += len(w)
                break
        else:
            i += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dict', default='scripts/thai_word2en.json')
    ap.add_argument('--corpus', default='_EN', help='_EN or _ENMT')
    ap.add_argument('--top', type=int, default=4)
    args = ap.parse_args()

    spec = json.load(open(args.dict, encoding='utf-8'))
    meta = spec.get('_meta', {})
    words = {k: v for k, v in spec.get('words', spec).items() if not k.startswith('_')}
    lexicon = sorted(words, key=len, reverse=True)

    th_data, en_map = load_pairs(args.corpus)
    pairs = [(set(segment(r['text'], lexicon)), re.findall(r'[a-z]+', en_map[r['name']].lower()))
             for r in th_data if r['name'] in en_map]
    n = len(pairs)
    if not n:
        sys.exit(f'no aligned pairs for corpus {args.corpus}')

    doc_freq = collections.Counter()
    for _, en in pairs:
        doc_freq.update(set(en))

    print(f'dict    : {args.dict}  ({len(words)} words)')
    print(f'corpus  : Thai_Hand4WholePP{args.corpus}  ({n} aligned sentence pairs)')
    if meta.get('source_corpus'):
        print(f'declared: {meta["source_corpus"]}'
              + ('   <-- MISMATCH with --corpus' if args.corpus.strip('_') not in meta['source_corpus'] else ''))
    print()
    print(f"{'thai':<11}{'gloss':<11}{'n':>4}  top candidates by association")
    print('-' * 76)

    collisions = collections.defaultdict(list)
    for th in sorted(words, key=lambda w: -sum(1 for s, _ in pairs if w in s)):
        sub = [en for s, en in pairs if th in s]
        if not sub:
            print(f'{th:<11}{words[th]:<11}{0:>4}  (word never appears in a sentence)')
            continue
        c = collections.Counter()
        for en in sub:
            c.update(set(en))
        scored = sorted(((cnt / len(sub) - doc_freq[t] / n, t) for t, cnt in c.items()), reverse=True)
        top = ' '.join(f'{t}({s:+.2f})' for s, t in scored[:args.top] if s > 0.05)
        collisions[tuple(t for _, t in scored[:3])].append(th)
        print(f'{th:<11}{words[th]:<11}{len(sub):>4}  {top or "(no strong signal - function word)"}')

    print()
    print('words whose evidence is IDENTICAL (always co-occur, so statistics cannot separate')
    print('them - these glosses were assigned by hand and deserve review first):')
    for group in collisions.values():
        if len(group) > 1:
            print('   ' + ' + '.join(group) + '  ->  ' + ', '.join(f'{w}={words[w]}' for w in group))


if __name__ == '__main__':
    main()
