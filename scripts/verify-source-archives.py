#!/usr/bin/env python3
"""Verify canonical Git trees via official GitHub archives when Git transport is unavailable.

Submodule gitlinks are verified in the parent tree; their worktrees are not
materialized by this source-only check. Image builds still use prepare-sources.py
and its recursive submodule checkout.
"""
from __future__ import annotations
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def git(path, *args, input=None):
    return subprocess.check_output(['git', '-C', str(path), *args], input=input).decode().strip()


def download(url, target):
    for attempt in range(3):
        try:
            req=urllib.request.Request(url,headers={'User-Agent':'dsv41-compat-source-verifier'})
            with urllib.request.urlopen(req,timeout=60) as response, target.open('wb') as out:
                shutil.copyfileobj(response,out)
            return
        except Exception:
            if attempt==2: raise
            time.sleep(attempt+1)


def fetch_source(name, source, dest, cache):
    repository=source['repository'].removeprefix('https://github.com/').removesuffix('.git')
    if repository==source['repository'] or '..' in repository or repository.count('/')!=1:
        raise ValueError('Only explicitly pinned official GitHub sources are supported')
    archive=cache/f'{name}-{source["commit"]}.tar.gz'
    if not archive.exists(): download(f'https://codeload.github.com/{repository}/tar.gz/{source["commit"]}',archive)
    path=dest/name
    if path.exists(): raise FileExistsError(f'{path}: choose an empty destination')
    with tempfile.TemporaryDirectory(dir=dest,prefix='extract-') as tmp:
        with tarfile.open(archive,'r:gz') as tar:
            # Python 3.12 data filter rejects traversal, unsafe links and devices.
            tar.extractall(tmp,filter='data')
        roots=list(Path(tmp).iterdir())
        if len(roots)!=1 or not roots[0].is_dir(): raise ValueError('Unexpected archive layout')
        shutil.move(str(roots[0]),path)
    git(path,'init','-q');git(path,'add','-f','.')
    submodules=[]
    if (path/'.gitmodules').exists():
        tree_file=cache/f'{name}-{source["commit"]}-tree.json'
        if not tree_file.exists():download(f'https://api.github.com/repos/{repository}/git/trees/{source["commit"]}?recursive=1',tree_file)
        response=json.loads(tree_file.read_text())
        if response.get('truncated'):raise ValueError('Truncated Git tree cannot establish provenance')
        submodules=[x for x in response['tree'] if x['mode']=='160000']
        for sub in submodules:
            git(path,'update-index','--add','--cacheinfo',f'160000,{sub["sha"]},{sub["path"]}')
    actual=git(path,'write-tree')
    if actual!=source['tree']:raise ValueError(f'{name}: archive tree {actual} != locked {source["tree"]}')
    return path,submodules,hashlib.sha256(archive.read_bytes()).hexdigest()


def prepare(dest, names, baseline=False):
    if sys.version_info < (3, 12):
        raise SystemExit('Python 3.12+ is required; set COMPAT_PYTHON to your CPU test environment.')
    lock=json.loads((ROOT/'stack.lock.json').read_text())
    dest.mkdir(parents=True,exist_ok=True)
    cache=Path(os.environ.get('COMPAT_ARCHIVE_CACHE',ROOT/'.cache/archives'));cache.mkdir(parents=True,exist_ok=True)
    receipts={}
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(4,len(names))) as executor:
        futures={name:executor.submit(fetch_source,name,lock['sources'][name],dest,cache) for name in names}
        for name,future in futures.items():
            path,submodules,archive_sha=future.result()
            source=lock['sources'][name]
            entries=lock['patches'] if name=='sglang' else lock.get(f'{name}_patches',[])
            if baseline and name=='sglang':entries=entries[:32]
            active=[p for p in entries if 'hold' not in p]
            for patch in active:
                diff=ROOT/patch['path']
                if hashlib.sha256(diff.read_bytes()).hexdigest()!=patch['sha256']:raise ValueError(f'Patch hash mismatch: {diff.name}')
                git(path,'apply','--index',str(diff))
            effective=git(path,'write-tree')
            expected='fe51b5fbe9394b3f1c22fd1154e0284cb455973b' if baseline and name=='sglang' else source.get('effective_tree',source['tree'])
            if effective!=expected:raise ValueError(f'{name}: effective tree {effective} != {expected}')
            receipts[name]={'repository':source['repository'],'commit':source['commit'],'tree':source['tree'],'effective_tree':effective,'patch_count':len(active),'archive_sha256':archive_sha,'submodule_gitlinks':[{k:s[k] for k in ['path','sha']} for s in submodules],'submodule_worktrees_materialized':False}
            print(f'{name}: verified source and {len(active)} patches, tree {effective}',flush=True)
    (dest/'archive-source-receipt.json').write_text(json.dumps(receipts,indent=2)+'\n')
    return receipts


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dest',type=Path,required=True)
    parser.add_argument('--sources',nargs='+',choices=['sglang','flashinfer','deepgemm','triton'],default=['sglang','flashinfer','deepgemm','triton'])
    parser.add_argument('--baseline',action='store_true',help='Prepare the unmodified V4.1 32-patch SGLang baseline')
    args=parser.parse_args();prepare(args.dest,args.sources,args.baseline)
