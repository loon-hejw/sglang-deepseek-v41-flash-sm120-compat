#!/usr/bin/env python3
"""Synthetic A/B API probe for separately started GPU test servers; never starts/stops services.

Use the same model weights, GPU allocation and launch settings for both URLs.
For a DSPARK-off control, run again against a separately started control server.
This checks protocol and reports timing; it does not establish general model quality.
"""
from __future__ import annotations
import argparse,concurrent.futures,hashlib,json,os,statistics,time,urllib.request
from pathlib import Path


def cases():
    schema={'type':'object','properties':{'value':{'type':'integer'}},'required':['value'],'additionalProperties':False}
    return [
      ('markdown',{'messages':[{'role':'user','content':'Return a Markdown heading named Result and a two-column table with two rows.'}]}),
      ('json_schema',{'messages':[{'role':'user','content':'Return value 7.'}],'response_format':{'type':'json_schema','json_schema':{'name':'result','strict':True,'schema':schema}}}),
      ('tool',{'messages':[{'role':'user','content':'Call record with value 7.'}],'tools':[{'type':'function','function':{'name':'record','parameters':schema,'strict':True}}],'tool_choice':'required'}),
      ('penalties',{'messages':[{'role':'user','content':'Write a short story about a clock.'}],'frequency_penalty':.3,'presence_penalty':.2,'repetition_penalty':1.2}),
      ('min_tokens_logprobs',{'messages':[{'role':'user','content':'Say done.'}],'min_tokens':12,'logprobs':True,'top_logprobs':2}),
      ('thinking',{'messages':[{'role':'user','content':'What is 17 times 23? Check your calculation.'}],'reasoning_effort':'high'}),
    ]


def probe(url,model,key,case,timeout,max_tokens=2048):
    name,extra=case
    body={'model':model,'stream':True,'stream_options':{'include_usage':True},'temperature':0,'top_p':1,'max_tokens':max_tokens,'reasoning_effort':'none',**extra}
    headers={'Content-Type':'application/json'}
    if key:headers['Authorization']='Bearer '+key
    request=urllib.request.Request(url.rstrip('/')+'/chat/completions',data=json.dumps(body).encode(),headers=headers)
    start=time.perf_counter();ttfb=None;done=False;content=[];reasoning=[];calls={};usage={};finishes=[];has_logprobs=False
    with urllib.request.urlopen(request,timeout=timeout) as response:
        for line in response:
            if not line.startswith(b'data:'):continue
            if ttfb is None:ttfb=time.perf_counter()-start
            value=line[5:].strip()
            if value==b'[DONE]':done=True;break
            obj=json.loads(value)
            if 'error' in obj:raise RuntimeError('API error frame')
            usage=obj.get('usage') or usage
            for choice in obj.get('choices',[]):
                delta=choice.get('delta') or {};content.append(delta.get('content') or '');reasoning.append(delta.get('reasoning_content') or '')
                has_logprobs|=bool((choice.get('logprobs') or {}).get('content'))
                if choice.get('finish_reason'):finishes.append(choice['finish_reason'])
                for call in delta.get('tool_calls') or []:
                    target=calls.setdefault(call.get('index',0),{'name':'','arguments':''});fn=call.get('function') or {}
                    if fn.get('name'):target['name']=fn['name']
                    target['arguments']+=fn.get('arguments') or ''
    elapsed=time.perf_counter()-start;text=''.join(content);thought=''.join(reasoning)
    checks={'terminal':done and bool(finishes),'not_truncated':'length' not in finishes,'no_raw_dsml':'｜DSML｜' not in text,'no_raw_think':'<think>' not in text and '</think>' not in text}
    if name=='json_schema':
        value=json.loads(text);checks['schema']=isinstance(value,dict) and set(value)=={'value'} and type(value['value']) is int and value['value']==7
    if name=='tool':
        arguments=[json.loads(c['arguments']) for c in calls.values()]
        checks['tool']=bool(calls) and all(c['name']=='record' for c in calls.values()) and all(isinstance(v,dict) and set(v)=={'value'} and type(v['value']) is int and v['value']==7 for v in arguments)
    if name=='markdown':checks['markdown_skeleton']=text.lstrip().startswith('#') and text.count('|')>=6
    if name=='thinking':checks['reasoning_present']=bool(thought)
    if name=='min_tokens_logprobs':checks.update(min_tokens=usage.get('completion_tokens',0)>=12,logprobs=has_logprobs)
    return {'case':name,'passed':all(checks.values()),'checks':checks,'ttfb_s':ttfb,'total_s':elapsed,'completion_tokens':usage.get('completion_tokens'),'reasoning_chars':len(thought),'content_sha256':hashlib.sha256(text.encode()).hexdigest()}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline-url',required=True,help='Explicit test OpenAI base URL, including /v1')
    p.add_argument('--candidate-url',required=True)
    p.add_argument('--model',default='deepseek-v4.1-flash')
    p.add_argument('--baseline-key-env',default='V41_BASELINE_API_KEY');p.add_argument('--candidate-key-env',default='V41_CANDIDATE_API_KEY')
    p.add_argument('--concurrency',type=int,nargs='+',default=[1,4,16]);p.add_argument('--repetitions',type=int,default=3);p.add_argument('--timeout',type=int,default=180)
    p.add_argument('--max-tokens',type=int,default=2048)
    p.add_argument('--output',type=Path,default=Path('artifacts/gpu-api-comparison.json'));p.add_argument('--dry-run',action='store_true');a=p.parse_args()
    if min(a.concurrency)<1 or a.repetitions<1 or a.max_tokens<1:p.error('Concurrency, repetitions and max-tokens must be positive')
    if a.dry_run:
        print(json.dumps({'cases':[n for n,_ in cases()],'concurrency':a.concurrency,'repetitions':a.repetitions,'requests_per_endpoint':len(cases())*a.repetitions*sum(a.concurrency),'sends_requests':False},indent=2));return
    rows=[]
    for concurrency in a.concurrency:
        # Alternate endpoint order across cells to reduce a fixed warm-cache bias.
        endpoints=[('baseline',a.baseline_url,a.baseline_key_env),('candidate',a.candidate_url,a.candidate_key_env)]
        for rep in range(a.repetitions):
            for name,url,keyenv in endpoints[::1 if rep%2==0 else -1]:
                for case in cases():
                    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
                        futures=[pool.submit(probe,url,a.model,os.environ.get(keyenv),case,a.timeout,a.max_tokens) for _ in range(concurrency)]
                        for future in futures:
                            try:result=future.result()
                            except Exception as error:result={'case':case[0],'passed':False,'error_type':type(error).__name__}
                            rows.append(dict(endpoint=name,concurrency=concurrency,repetition=rep,**result))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    summary=[]
    for endpoint in ['baseline','candidate']:
        for concurrency in a.concurrency:
            for case,_ in cases():
                group=[r for r in rows if r['endpoint']==endpoint and r['concurrency']==concurrency and r['case']==case]
                measured=[r for r in group if 'total_s' in r]
                summary.append({'endpoint':endpoint,'concurrency':concurrency,'case':case,'requests':len(group),'passed':sum(r['passed'] for r in group),'median_total_s':statistics.median(r['total_s'] for r in measured) if measured else None,'median_ttfb_s':statistics.median(r['ttfb_s'] for r in measured if r['ttfb_s'] is not None) if any(r['ttfb_s'] is not None for r in measured) else None})
    a.output.write_text(json.dumps({'model':a.model,'max_tokens':a.max_tokens,'summary':summary,'results':rows,'note':'Controlled synthetic comparison; includes reasoning tokens. TTFB is the first SSE frame, not time to first generated token. No general quality or production throughput claim.'},indent=2)+'\n')
    print(json.dumps({'requests':len(rows),'passed':sum(r['passed'] for r in rows),'report':str(a.output)}))
    if not all(r['passed'] for r in rows):raise SystemExit(1)


if __name__=='__main__':main()
