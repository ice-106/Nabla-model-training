#!/usr/bin/env python
"""
Assemble the .npz + atlas files written by 03_dtw_extract.py into one self-contained
HTML viewer (no network, no libraries -- plotly is not installed in 01_SOKE/envs).

    cd 01_SOKE/SOKE
    ../envs/bin/python scripts/03_dtw_build_viewer.py \
        --data-dir dtw_viewer_data --out dtw_viewer.html
"""
import argparse
import base64
import json
from pathlib import Path

import numpy as np

PARTS = ('body', 'lhand', 'rhand')
MODES = ('jpe', 'pa')


def b64(raw):
    return base64.b64encode(raw).decode('ascii')


def quantise(C):
    """(N,M) float32 metres -> uint16 base64 + the range needed to invert it.

    Raw float JSON is ~10x larger and the viewer only ever renders these as colours
    or prints them to 0.1 mm, so 16 bits over the observed range is lossless enough.
    """
    lo, hi = float(C.min()), float(C.max())
    span = hi - lo if hi > lo else 1.0
    q = np.clip(np.round((C - lo) / span * 65535.0), 0, 65535).astype('<u2')
    return {'lo': lo, 'hi': hi, 'q': b64(q.tobytes())}


def path_cells(D, path):
    """D values along the path -- the full accumulated matrix is never needed."""
    return [float(D[int(i), int(j)]) for i, j in zip(path[0], path[1])]


def build_sample(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    key = str(d['key'])
    stem = key.split('/')[-1]
    s = {
        'key': key,
        'text': str(d['text']),
        'src': str(d['src']),
        'run': str(d['run']) if 'run' in d else '-',
        'recorded_mode': str(d['recorded_mode']),
        'N': int(d['N']),
        'M': int(d['M']),
        'parts': {},
    }
    for part in PARTS:
        entry = {}
        if f'recorded_{part}' in d:
            entry['recorded'] = float(d[f'recorded_{part}'])
            entry['match'] = bool(d[f'match_{part}'])
        for mode in MODES:
            ck = f'C_{part}_{mode}'
            if ck not in d:
                continue
            C = d[ck]
            path = d[f'path_{part}_{mode}']
            entry[mode] = {
                'score': float(d[f'score_{part}_{mode}']),
                'C': quantise(C),
                'path': [path[0].astype(int).tolist(), path[1].astype(int).tolist()],
                'Dpath': path_cells(d[f'D_{part}_{mode}'], path),
            }
        s['parts'][part] = entry

    for track in ('ref', 'gen'):
        meta = d[f'atlas_{track}'] if f'atlas_{track}' in d else None
        # Name the atlas off the npz filename, not the sample key: with --run-tag the
        # files are <tag>__<key>_<track>.jpg, and deriving from the key would miss them.
        jpg = npz_path.parent / f'{npz_path.stem}_{track}.jpg'
        if meta is None or not jpg.exists():
            # Say so. Silently dropping these yields a viewer with no thumbnails at
            # all and no indication why.
            print(f'  WARNING: {npz_path.name}: no {track} atlas '
                  f'({"not in npz" if meta is None else f"missing {jpg.name}"}) '
                  f'-- thumbnails will be absent for this sample')
            continue
        n, cols, rows, tile = (int(x) for x in meta)
        s[track] = {'n': n, 'cols': cols, 'rows': rows, 'tile': tile,
                    'img': b64(jpg.read_bytes())}
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--template', default=None,
                    help='defaults to 03_dtw_viewer_template.html beside this script')
    a = ap.parse_args()

    data_dir = Path(a.data_dir)
    files = sorted(data_dir.glob('*.npz'))
    if not files:
        raise SystemExit(f'no .npz in {data_dir}')

    samples = [build_sample(f) for f in files]
    # group by run, then dataset, worst-first inside each, so the samples most worth
    # looking at sit near the top of the picker
    samples.sort(key=lambda s: (s['run'], s['src'],
                                -s['parts']['body'].get('recorded',
                                                        s['parts']['body']['jpe']['score'])))
    for s in samples:
        print(f"  {s['run']:8s} {s['src']:9s} {s['key']:45s} N={s['N']:4d} M={s['M']:4d} "
              f"body_jpe={s['parts']['body']['jpe']['score']:.4f}")

    tpl = Path(a.template) if a.template else \
        Path(__file__).with_name('03_dtw_viewer_template.html')
    html = tpl.read_text().replace(
        '/*__DATA__*/null', json.dumps(samples, ensure_ascii=False, separators=(',', ':')))
    out = Path(a.out)
    out.write_text(html, encoding='utf-8')
    mb = out.stat().st_size / 1e6
    print(f'\nwrote      : {out}  ({mb:.1f} MB, {len(samples)} samples)')
    if mb > 14:
        print('warning    : approaching the 16 MB artifact cap -- '
              're-extract with --tile 96 or --jpeg-quality 65')


if __name__ == '__main__':
    main()
