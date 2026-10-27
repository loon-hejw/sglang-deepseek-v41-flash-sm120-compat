#!/usr/bin/env python3
"""Check complete migration accounting, source anchors and original patch hashes."""
import argparse,hashlib,json,re
from pathlib import Path
root=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);args=p.parse_args()
audit=json.loads((root/'compat/migration.json').read_text())
lock=json.loads((root/'stack.lock.json').read_text())
old=root/'compat/upstream-0731'
names=re.findall(r'git apply --verbose /tmp/patches/(\S+)',(old/'Dockerfile').read_text())
assert len(names)==33 and [e['patch'] for e in audit['entries']]==names
ids={p['id'] for p in lock['patches']}
for row in audit['entries']:
    assert row['status'] in {'ported','already_equivalent','not_applicable'}
    assert row['reason'] and row['evidence']
    assert hashlib.sha256((old/'patches'/row['patch']).read_bytes()).hexdigest()==row['original_sha256']
    assert set(row['ported_patch_ids'])<=ids
    assert bool(row['ported_patch_ids'])==(row['status']=='ported')
    for proof in row['evidence']:
        f=args.source/proof['path'];assert proof['anchor'] in f.read_text(),proof
        assert hashlib.sha256(f.read_bytes()).hexdigest()==proof['file_sha256'],proof
for status,n in audit['status_counts'].items():assert n==sum(e['status']==status for e in audit['entries'])
print(json.dumps({'audit_entries':len(names),'status_counts':audit['status_counts']},ensure_ascii=False))
