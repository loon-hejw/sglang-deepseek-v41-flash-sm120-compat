"""OpenAI-compatible requests; raw model text is written only to private output."""
import json
import os
import time
import urllib.error
import urllib.request
from .cases import schema_for
from .policy import digest


class Client:
    def __init__(self, base_url, model):
        self.base=base_url.rstrip('/');self.model=model

    def headers(self):
        h={'Content-Type':'application/json'}
        if os.environ.get('INFERENCE_API_KEY'):h['Authorization']='Bearer '+os.environ['INFERENCE_API_KEY']
        return h

    def json_request(self, path, body):
        req=urllib.request.Request(self.base+path,data=json.dumps(body,ensure_ascii=False).encode(),headers=self.headers())
        with urllib.request.urlopen(req,timeout=120) as response:return json.load(response)

    def token_count(self, prompt):
        return self.json_request('/tokenize',{'model':self.model,'messages':[{'role':'user','content':prompt}],
            'reasoning_effort':'none','chat_template_kwargs':{'thinking':False}})['count']

    def body(self, case, profile, seed, stream=True):
        p=dict(profile)
        body={'model':self.model,'messages':[{'role':'user','content':case['prompt']}],**p,
              'chat_template_kwargs':{'thinking':p['reasoning_effort']!='none'},'seed':seed,'stream':stream}
        if stream:body['stream_options']={'include_usage':True}
        if case.get('strict_schema'):
            body['response_format']={'type':'json_schema','json_schema':{'name':'quality_result','strict':True,
                'schema':case.get('schema') or schema_for(case['expected'])}}
        if case.get('tools'):body.update(tools=case['tools'],tool_choice='required',parallel_tool_calls=True)
        body.update(case.get('request_extra',{}))
        return body

    def generate(self, body):
        req=urllib.request.Request(self.base+'/chat/completions',data=json.dumps(body,ensure_ascii=False).encode(),headers=self.headers())
        start=time.monotonic();first=None;first_answer=None;content=[];reasoning=[];calls={};usage={};finish=[];done=False;error={}
        try:
            with urllib.request.urlopen(req,timeout=900) as response:
                if not body['stream']:
                    obj=json.load(response);usage=obj.get('usage') or {}
                    for ch in obj.get('choices',[]):
                        m=ch.get('message') or {};content.append(m.get('content') or '');reasoning.append(m.get('reasoning_content') or '')
                        if ch.get('finish_reason'):finish.append(ch['finish_reason'])
                        for i,c in enumerate(m.get('tool_calls') or []):calls[i]=c.get('function') or {}
                    done=True
                else:
                    for line in response:
                        if time.monotonic()-start>900:raise TimeoutError('Request exceeded 900-second safety deadline')
                        if not line.startswith(b'data:'):continue
                        data=line[5:].strip()
                        if data==b'[DONE]':done=True;break
                        obj=json.loads(data)
                        if 'error' in obj:raise RuntimeError('API error frame')
                        usage=obj.get('usage') or usage
                        for ch in obj.get('choices',[]):
                            d=ch.get('delta') or {}
                            if first is None and any(d.get(k) for k in ['content','reasoning_content','tool_calls']):first=time.monotonic()-start
                            if first_answer is None and any(d.get(k) for k in ['content','tool_calls']):first_answer=time.monotonic()-start
                            content.append(d.get('content') or '');reasoning.append(d.get('reasoning_content') or '')
                            if ch.get('finish_reason'):finish.append(ch['finish_reason'])
                            for c in d.get('tool_calls') or []:
                                acc=calls.setdefault(c.get('index',0),{'name':'','arguments':''});fn=c.get('function') or {}
                                acc['name']+=fn.get('name') or '';acc['arguments']+=fn.get('arguments') or ''
        except urllib.error.HTTPError as e:error={'error_type':'HTTPError','http_status':e.code,'error_message':e.read(1000).decode(errors='replace')}
        except Exception as e:error={'error_type':type(e).__name__,'error_message':str(e)[:200]}
        return {'content':''.join(content),'tools':[calls[k] for k in sorted(calls)],'usage':usage,
                'reasoning_chars':sum(len(x) for x in reasoning),'reasoning_sha256':digest(''.join(reasoning)),
                'finish':finish,'done':done,'first_token_s':first,'first_answer_s':first_answer,
                'total_s':time.monotonic()-start,'request_sha256':digest(body),**error}
