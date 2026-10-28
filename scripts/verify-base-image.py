#!/usr/bin/env python3
"""Verify a mirrored or imported base against the manifest pinned by the lock."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import urllib.error
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
ACCEPT = ','.join(['application/vnd.oci.image.index.v1+json', 'application/vnd.docker.distribution.manifest.list.v2+json', 'application/vnd.oci.image.manifest.v1+json', 'application/vnd.docker.distribution.manifest.v2+json'])


def digest(data):
    return 'sha256:' + hashlib.sha256(data).hexdigest()


def validate(receipt, locked):
    assert receipt['locked_base'] == locked, 'Receipt is for another base'
    manifest_raw = receipt['manifest_raw'].encode()
    assert digest(manifest_raw) == locked.rsplit('@', 1)[1], 'Manifest digest differs from lock'
    manifest = json.loads(manifest_raw)
    if 'manifests' in manifest:
        selected = next(m for m in manifest['manifests'] if m.get('platform', {}).get('os') == 'linux' and m.get('platform', {}).get('architecture') == 'amd64')
        selected_raw = receipt['platform_manifest_raw'].encode()
        assert digest(selected_raw) == selected['digest'], 'Platform manifest mismatch'
        manifest = json.loads(selected_raw)
    config_raw = receipt['config_raw'].encode()
    assert digest(config_raw) == manifest['config']['digest'], 'Config digest mismatch'
    config = json.loads(config_raw)
    assert (config['os'], config['architecture']) == ('linux', 'amd64'), 'Wrong platform'
    return manifest, config


def capture(source, locked):
    reference, pin = source.rsplit('@', 1)
    assert pin == locked.rsplit('@', 1)[1], 'Mirror must use the locked digest'
    registry, repository = reference.split('/', 1)
    registry = 'registry-1.docker.io' if registry in ('docker.io', 'index.docker.io') else registry
    base = 'https://' + registry
    headers = {'Accept': ACCEPT, 'User-Agent': 'dsv41-compat-base-verifier'}

    def fetch(path):
        try:
            with urllib.request.urlopen(urllib.request.Request(base + path, headers=headers), timeout=30) as response:
                return response.read()
        except urllib.error.HTTPError as error:
            if error.code != 401:
                raise
            challenge = dict(re.findall(r'(\w+)="([^"]+)"', error.headers['WWW-Authenticate']))
            assert challenge['realm'].startswith('https://'), 'Refuse insecure token endpoint'
            url = challenge['realm'] + '?' + urllib.parse.urlencode({k: challenge[k] for k in ('service', 'scope') if k in challenge})
            with urllib.request.urlopen(url, timeout=30) as response:
                auth = json.load(response)
            headers['Authorization'] = 'Bearer ' + (auth.get('token') or auth['access_token'])
            with urllib.request.urlopen(urllib.request.Request(base + path, headers=headers), timeout=30) as response:
                return response.read()

    raw = fetch('/v2/' + repository + '/manifests/' + pin)
    assert digest(raw) == pin, 'Registry returned different content'
    receipt = {'schema': 1, 'locked_base': locked, 'download_source': source, 'manifest_raw': raw.decode()}
    manifest = json.loads(raw)
    if 'manifests' in manifest:
        selected = next(m for m in manifest['manifests'] if m.get('platform', {}).get('os') == 'linux' and m.get('platform', {}).get('architecture') == 'amd64')
        raw = fetch('/v2/' + repository + '/manifests/' + selected['digest'])
        assert digest(raw) == selected['digest']
        receipt['platform_manifest_raw'] = raw.decode()
        manifest = json.loads(raw)
    receipt['config_raw'] = fetch('/v2/' + repository + '/blobs/' + manifest['config']['digest']).decode()
    validate(receipt, locked)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', help='Registry/repository@locked-digest; captures a receipt')
    parser.add_argument('--receipt', type=Path, default=ROOT / 'artifacts/base-receipt.json')
    parser.add_argument('--image', help='Inspect this already pulled or imported local Docker image')
    args = parser.parse_args()
    locked = json.loads((ROOT / 'stack.lock.json').read_text())['base']['image']
    if args.source:
        receipt = capture(args.source, locked)
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps(receipt, indent=2) + '\n')
    else:
        receipt = json.loads(args.receipt.read_text())
    manifest, config = validate(receipt, locked)
    if args.image:
        installed = json.loads(subprocess.check_output(['docker', 'image', 'inspect', args.image]))[0]
        assert installed['Id'] == manifest['config']['digest'], 'Imported image ID mismatch'
        assert installed['Os'] == 'linux' and installed['Architecture'] == 'amd64'
        assert installed['RootFS']['Layers'] == config['rootfs']['diff_ids'], 'RootFS layers mismatch'
    print(json.dumps({'verified': True, 'locked_base': locked, 'config_digest': manifest['config']['digest'], 'platform': 'linux/amd64', 'local_image_verified': bool(args.image), 'compressed_bytes': sum(x['size'] for x in manifest['layers'])}))


if __name__ == '__main__':
    main()
