#!/usr/bin/env python3
"""Apply an explicitly selected task policy to a nonstream chat request.

Explicit request fields win. Tool calls are returned as JSON and never executed.
Experimental policies require --experimental and are not production defaults.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import urllib.request


def apply_profile(request, profile):
    body={**profile,**request}
    kwargs={**(profile.get('chat_template_kwargs') or {}),**(request.get('chat_template_kwargs') or {})}
    if body.get('reasoning_effort') is not None:
        kwargs.setdefault('thinking',body['reasoning_effort']!='none')
    if kwargs:body['chat_template_kwargs']=kwargs
    # The modern field takes precedence at the API. Avoid sending two budgets.
    if request.get('max_completion_tokens') is not None:body.pop('max_tokens',None)
    return body


def select_profile(bundle, category, experimental=False):
    if not bundle.get('qualified') and not experimental:
        raise ValueError('These experimental policies did not qualify for recommendation. Use the existing configuration; --experimental is for deliberate evaluation only.')
    return bundle['profiles'][category]


def validate_response(response):
    choices=response.get('choices') or []
    if len(choices)!=1:raise ValueError('Expected exactly one completion')
    choice=choices[0];finish=choice.get('finish_reason');message=choice.get('message') or {}
    if finish not in ['stop','tool_calls']:raise ValueError('Incomplete response; no partial answer accepted')
    if not ((message.get('content') or '').strip() or message.get('tool_calls')):
        raise ValueError('No final answer or tool call; reasoning is not a substitute')
    for call in message.get('tool_calls') or []:
        fn=call.get('function') or {}
        if not fn.get('name') or not isinstance(json.loads(fn.get('arguments','')),dict):
            raise ValueError('Invalid tool call')
    return response


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--request',type=Path,required=True)
    p.add_argument('--category',required=True,choices=['reasoning','coding','instructions','structured','tools','long_context','penalties'])
    p.add_argument('--profiles',type=Path,default=Path(__file__).with_name('quality-policy-profiles.json'))
    p.add_argument('--experimental',action='store_true')
    p.add_argument('--dry-run',action='store_true',help='Print the effective request without sending it')
    p.add_argument('--base-url',default=os.environ.get('INFERENCE_BASE_URL','http://127.0.0.1:8000/v1'))
    p.add_argument('--timeout',type=float,default=900);args=p.parse_args()
    try:
        request=json.loads(args.request.read_text());bundle=json.loads(args.profiles.read_text())
        body=apply_profile(request,select_profile(bundle,args.category,args.experimental))
        if not body.get('model') or not body.get('messages'):raise ValueError('Provide model and messages in the request')
        if body.get('stream') or body.get('n',1)!=1:raise ValueError('This example supports one nonstream completion')
        if args.dry_run:print(json.dumps(body,ensure_ascii=False,indent=2));return 0
        headers={'Content-Type':'application/json'}
        if os.environ.get('INFERENCE_API_KEY'):headers['Authorization']='Bearer '+os.environ['INFERENCE_API_KEY']
        req=urllib.request.Request(args.base_url.rstrip('/')+'/chat/completions',data=json.dumps(body,ensure_ascii=False).encode(),headers=headers)
        with urllib.request.urlopen(req,timeout=args.timeout) as response:result=validate_response(json.load(response))
        print(json.dumps(result,ensure_ascii=False));return 0
    except (ValueError,KeyError) as e:
        print(str(e),file=sys.stderr);return 2


if __name__=='__main__':raise SystemExit(main())
