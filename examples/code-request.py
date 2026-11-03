#!/usr/bin/env python3
"""Apply an explicit coding profile without changing server sampling defaults."""
import argparse
import json
import os
from pathlib import Path
import sys
import urllib.request


def apply_profile(request, profile):
    """Explicit caller fields win, including zero-valued sampling settings."""
    return {**profile, **request}


def code_from_response(response):
    choices = response.get('choices') or []
    if len(choices) != 1:
        raise ValueError('Expected one completion; no code was accepted.')
    choice = choices[0]
    if choice.get('finish_reason') == 'length':
        raise ValueError('The token budget was exhausted; no partial code was accepted. Review the task or explicitly change the request budget.')
    if choice.get('finish_reason') != 'stop':
        raise ValueError('The completion did not finish normally; no code was accepted.')
    content = (choice.get('message') or {}).get('content')
    if not isinstance(content, str) or not content.strip():
        raise ValueError('The response has no final code. Reasoning is not used as a substitute, and the request is not retried automatically.')
    return content


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True, help='JSON request with model and messages; explicit sampling fields override the profile')
    parser.add_argument('--profile', type=Path, default=Path(__file__).with_name('code-request-profile.json'))
    parser.add_argument('--base-url', default=os.environ.get('INFERENCE_BASE_URL', 'http://127.0.0.1:8000/v1'))
    parser.add_argument('--timeout', type=float, default=300)
    args = parser.parse_args()
    body = apply_profile(json.loads(args.request.read_text()), json.loads(args.profile.read_text()))
    if body.get('stream'):
        parser.error('This example reads nonstream completions; use a streaming client for stream=true.')
    if body.get('n', 1) != 1:
        parser.error('This example accepts exactly one completion (n=1).')
    if not body.get('messages') or not body.get('model'):
        parser.error('The request JSON must provide model and messages.')
    headers = {'Content-Type': 'application/json'}
    key = os.environ.get('INFERENCE_API_KEY')
    if key:
        headers['Authorization'] = 'Bearer '+key
    req = urllib.request.Request(args.base_url.rstrip('/')+'/chat/completions',
                                 data=json.dumps(body).encode(), headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=args.timeout) as result:
            response = json.load(result)
        code = code_from_response(response)
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 2
    # The caller decides whether to inspect, save or execute this untrusted code.
    print(code)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
