"""Run the predeclared quality-policy experiment against one dedicated server.

This runner never stops or starts services. Use an external resource supervisor.
Raw output is private and must not be committed; publish only export_summary().
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import time

from . import grading
from .cases import VERSION,CATEGORIES,build_suite,legacy_suite,long_prompt
from .client import Client
from .policy import baseline,digest,initial_profiles,profile_id,rank_profiles,quality_gate,smoke_valid


def save(path, data):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)


class Experiment:
    def __init__(self, args):
        self.args=args;self.root=Path(args.output);self.root.mkdir(parents=True,exist_ok=True,mode=0o700)
        self.client=Client(args.base_url,args.model);grading.SANDBOX_IMAGE=args.sandbox_image
        self.started=time.time();self.rows=[];self.done=set()
        sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}
        self.method={'version':VERSION,'model':args.model,'source_sha256':sources,'new_cases':210,'selection_cases':70,
            'holdout_cases':140,'families':21,'selection_repetitions':3,'holdout_repetitions':5,'legacy_cases':51,
            'primary_baseline':'T1, top_p1; coding low, reasoning high, other none; budgets 12288/4096/2048',
            'quality_concurrency':'Up to four mixed-profile requests; long-context requests singly. Both arms share one server.',
            'request_deadline_s':900,'first_token':'First nonempty content/reasoning/tool delta, not first SSE frame',
            'first_answer':'First nonempty final-content/tool delta; separate from reasoning',
            'speed':'Per-category fixed selection prompt, 1024 output-token cap; 4 warmups per arm; C1/C4, five alternating paired batches per workload. Completion tokens include reasoning; divided by batch wall time including TTFT. Warm-prefix tests; no cache flush.',
            'seed_note':'Supplied seeds are paired; backend deterministic inference is not enabled.',
            'selection_ties':'Higher pass rate, fewer truncations, baseline first, then lower budget/effort/temperature closest to one',
            'quality_gate':{'macro_delta_min':.03,'family_bootstrap_ci95_lower_gt':0,'max_category_decline':.02},
            'speed_gate':{'minimum_ratio_per_category_and_concurrency':.8},
            'no_automatic_retry':True,'no_holdout_retuning':True,
            'smoke_gate':'API completion, nonzero usage and thinking-mode effectiveness; wrong answers are retained, not startup failures',
            'limitations':['Synthetic tasks, not an official benchmark or real production traffic',
                'Only 21 problem families: seven selection and fourteen holdout; instances within a family are correlated',
                'No full 1M input validation','Format compliance is not semantic correctness',
                'More thinking tokens can change token/s without improving useful final output; latency and first-answer time reported separately']}
        mf=self.root/'methodology.json'
        if mf.exists():assert json.loads(mf.read_text())==self.method,'Experiment methodology/source changed; use a new output directory'
        else:save(mf,self.method)
        for line in (self.root/'results.jsonl').read_text().splitlines() if (self.root/'results.jsonl').exists() else []:
            r=json.loads(line);assert r['task_id'] not in self.done;self.rows.append(r);self.done.add(r['task_id'])

    def status(self, phase, **extra):
        data={'phase':phase,'completed_requests':len(self.rows),'at':time.time(),**extra}
        save(self.root/'progress.json',data);print(json.dumps(data,ensure_ascii=False),flush=True)

    def prepare(self):
        f=self.root/'cases.json'
        if f.exists():self.cases=json.loads(f.read_text())
        else:
            self.cases=build_suite()+legacy_suite()
            for i,c in enumerate(self.cases):
                if 'long_spec' not in c:continue
                spec=c['long_spec'];n=max(10,spec['target_tokens']//26)
                for _ in range(5):
                    p=long_prompt(spec,n);count=self.client.token_count(p)
                    if abs(count/spec['target_tokens']-1)<.005:break
                    n=max(10,round(n*spec['target_tokens']/count))
                assert abs(count/spec['target_tokens']-1)<.01, 'Context sizing failed'
                c.update(prompt=p,prepared_prompt_tokens=count)
                self.status('prepare',case=c['id'],prompt_tokens=count)
            save(f,self.cases)
        self.by_id={c['id']:c for c in self.cases}
        manifest={'suite_sha256':digest(self.cases),'cases':[{k:c[k] for k in ['id','category','family','split','prepared_prompt_tokens'] if k in c} for c in self.cases]}
        mf=self.root/'cases-manifest.json'
        if mf.exists():assert json.loads(mf.read_text())==manifest
        else:save(mf,manifest)

    def task(self, phase, case, p, repeat, arm='screen', stream=True):
        task={'phase':phase,'case':case['id'],'category':case['category'],'family':case['family'],
              'profile':p,'profile_id':profile_id(p),'repeat':repeat,'arm':arm,'stream':stream,
              'seed':int(digest([case['id'],repeat])[:7],16)}
        task['task_id']=digest(task)
        return task

    def one(self, task):
        c=self.by_id[task['case']];body=self.client.body(c,task['profile'],task['seed'],task['stream'])
        r=self.client.generate(body)
        # Retain effective settings without copying prompts or secrets into summaries.
        return {**task,'effective_parameters':{k:body[k] for k in ['temperature','top_p','max_tokens','reasoning_effort','chat_template_kwargs']},
                **r,**grading.grade(c,r)}

    def run_tasks(self, phase, tasks):
        todo=[t for t in tasks if t['task_id'] not in self.done];random.Random(20261105).shuffle(todo)
        total=len(tasks);errors=0
        while todo:
            batch=[todo.pop(0)]
            if batch[0]['category']!='long_context':
                while todo and len(batch)<4 and todo[0]['category']!='long_context':batch.append(todo.pop(0))
            self.status(phase,phase_done=total-len(todo)-len(batch),phase_total=total,active_cases=[x['case'] for x in batch])
            with ThreadPoolExecutor(max_workers=len(batch)) as pool:
                for f in as_completed([pool.submit(self.one,t) for t in batch]):
                    r=f.result();r['test_concurrency']=len(batch)
                    with (self.root/'results.jsonl').open('a') as out:out.write(json.dumps(r,ensure_ascii=False)+'\n');out.flush()
                    self.rows.append(r);self.done.add(r['task_id']);errors+=int('error_type' in r)
            if errors>=3:raise RuntimeError('Three API failures in a phase; stop and inspect before continuing')
        self.status(phase,phase_done=total,phase_total=total)

    def screening(self):
        cases=[c for c in self.cases if c['split']=='selection'];profiles={cat:initial_profiles(cat) for cat in CATEGORIES}
        def tasks_for(ps):
            return [self.task('selection',c,p,r) for c in cases for p in ps[c['category']] for r in range(3)]
        self.run_tasks('screen_effort',tasks_for(profiles))
        rows=lambda:[r for r in self.rows if r['phase']=='selection']
        extra={c:[] for c in CATEGORIES}
        for cat in ['reasoning','coding']:
            extra[cat]=[{**p,'max_tokens':24576} for p in rank_profiles(cat,profiles[cat],rows())[:2]]
        self.run_tasks('screen_budget',tasks_for(extra))
        for cat in CATEGORIES:profiles[cat]+=extra[cat]
        extra={c:[] for c in CATEGORIES}
        for cat in CATEGORIES:
            top=rank_profiles(cat,profiles[cat],rows())[0]
            extra[cat]=[{**top,'temperature':t} for t in ([.6] if cat in ['reasoning','coding'] else [.3,.6])]
        self.run_tasks('screen_temperature',tasks_for(extra))
        for cat in CATEGORIES:profiles[cat]+=extra[cat]
        winners={cat:rank_profiles(cat,profiles[cat],rows())[0] for cat in CATEGORIES}
        evidence={cat:[{'parameters':p,'profile_id':profile_id(p),
                       'requests':sum(r['category']==cat and r['profile_id']==profile_id(p) for r in rows()),
                       'passed':sum(r['category']==cat and r['profile_id']==profile_id(p) and r['passed'] for r in rows())}
                      for p in rank_profiles(cat,profiles[cat],rows())] for cat in CATEGORIES}
        selection={'selected':winners,'screening':evidence,'locked_before_holdout':True,'selection_sha256':digest(winners)}
        file=self.root/'selected.json'
        if file.exists():assert json.loads(file.read_text())==selection,'Frozen selection changed'
        else:save(file,selection)
        return winners

    def confirmation(self, winners):
        cases=[c for c in self.cases if c['split']=='holdout']
        self.run_tasks('holdout',[self.task('holdout',c,p,r,arm) for c in cases for r in range(5)
            for arm,p in [('baseline',baseline(c['category'])),('candidate',winners[c['category']])]])
        old=[c for c in self.cases if c['split']=='regression']
        tasks=[]
        for c in old:
            bp={**baseline(c['category']),'max_tokens':c['max_tokens']}
            if c['category']=='coding':bp['reasoning_effort']='high'
            for temp in [0,1]:
                for arm,p in [('baseline',bp),('candidate',winners[c['category']])]:
                    tasks.append(self.task('regression',c,{**p,'temperature':temp},temp,arm))
        self.run_tasks('regression',tasks)
        self.run_tasks('code_low_control',[self.task('code_low_control',c,p,r,arm) for c in old if c['category']=='coding'
            for r in range(5) for arm,p in [('baseline',baseline('coding')),('candidate',winners['coding'])]])
        # Nonstream companion to stream-mode holdout; no real tool execution.
        tools=[c for c in cases if c['category']=='tools']
        self.run_tasks('tools_nonstream',[self.task('tools_nonstream',c,p,r,arm,False) for c in tools for r in range(3)
            for arm,p in [('baseline',baseline('tools')),('candidate',winners['tools'])]])

    def speed(self, winners):
        path=self.root/'speed.jsonl'
        records=[json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []
        done={(x['category'],x['concurrency'],x['repeat'],x['arm']) for x in records}
        for cat in CATEGORIES:
            # Use one fixed screening case per category, not a held-out answer.
            c=next(c for c in self.cases if c['split']=='selection' and c['category']==cat)
            for concurrency in [1,4]:
                ps={'baseline':{**baseline(cat),'max_tokens':1024},'candidate':{**winners[cat],'max_tokens':1024}}
                for arm,p in ps.items():
                    if all((cat,concurrency,r,arm) in done for r in range(5)):continue
                    with ThreadPoolExecutor(max_workers=concurrency) as pool:
                        warm=list(pool.map(lambda i:self.client.generate(self.client.body(c,p,9000+i)),range(4)))
                    assert all(x['done'] and not x.get('error_type') for x in warm),'Speed warmup failed'
                for repeat in range(5):
                    for arm in (['baseline','candidate'] if repeat%2==0 else ['candidate','baseline']):
                        if (cat,concurrency,repeat,arm) in done:continue
                        self.status('speed',category=cat,concurrency=concurrency,repeat=repeat,arm=arm)
                        start=time.monotonic()
                        with ThreadPoolExecutor(max_workers=concurrency) as pool:
                            result=list(pool.map(lambda i:self.client.generate(self.client.body(c,ps[arm],9100+repeat*4+i)),range(concurrency)))
                        elapsed=time.monotonic()-start
                        assert all(x['done'] and not x.get('error_type') and x['usage'].get('completion_tokens') for x in result),'Speed request invalid'
                        row={'category':cat,'case':c['id'],'concurrency':concurrency,'repeat':repeat,'arm':arm,'profile':ps[arm],
                             'elapsed_s':elapsed,'completion_tokens':sum(x['usage']['completion_tokens'] for x in result),
                             'request_total_s':[x['total_s'] for x in result],'first_token_s':[x['first_token_s'] for x in result],
                             'first_answer_s':[x['first_answer_s'] for x in result],
                             'prompt_tokens':[x['usage'].get('prompt_tokens') for x in result],
                             'reasoning_chars':[x['reasoning_chars'] for x in result]}
                        row['tokens_per_second']=row['completion_tokens']/elapsed
                        with path.open('a') as f:f.write(json.dumps(row)+'\n');f.flush()
                        records.append(row);done.add((cat,concurrency,repeat,arm))
        result=[]
        for cat in CATEGORIES:
            for concurrency in [1,4]:
                values={arm:statistics.median(r['tokens_per_second'] for r in records if r['category']==cat and r['concurrency']==concurrency and r['arm']==arm) for arm in ['baseline','candidate']}
                result.append({'category':cat,'concurrency':concurrency,**values,'ratio':values['candidate']/values['baseline']})
        return {'batches':len(records),'comparisons':result,'passed':all(x['ratio']>=.8 for x in result)}

    def export_summary(self, winners, speed):
        hold=[r for r in self.rows if r['phase']=='holdout'];gate=quality_gate(hold)
        summaries={}
        for phase in sorted({r['phase'] for r in self.rows}):
            summaries[phase]={}
            for arm in sorted({r['arm'] for r in self.rows if r['phase']==phase}):
                group=[r for r in self.rows if r['phase']==phase and r['arm']==arm]
                summaries[phase][arm]={cat:{'requests':len(rs),'passed':sum(r['passed'] for r in rs),
                     'truncated':sum('length' in r['finish'] for r in rs),'api_errors':sum('error_type' in r for r in rs),
                     'median_total_s':statistics.median(r['total_s'] for r in rs),
                     'median_completion_tokens':statistics.median(r['usage'].get('completion_tokens',0) for r in rs),
                     'median_reasoning_tokens':statistics.median(v) if (v:=[r['usage'].get('reasoning_tokens',r['usage'].get('completion_tokens_details',{}).get('reasoning_tokens')) for r in rs if r['usage'].get('reasoning_tokens',r['usage'].get('completion_tokens_details',{}).get('reasoning_tokens')) is not None]) else None,
                     'median_first_answer_s':statistics.median(v) if (v:=[r['first_answer_s'] for r in rs if r['first_answer_s'] is not None]) else None}
                    for cat in CATEGORIES if (rs:=[r for r in group if r['category']==cat])}
        legacy=summaries['regression'];code=summaries['code_low_control'];tool=summaries['tools_nonstream']
        regressions_passed=all(legacy['candidate'][cat]['passed']>=legacy['baseline'][cat]['passed'] for cat in CATEGORIES)
        controls_passed=code['candidate']['coding']['passed']>=code['baseline']['coding']['passed'] and tool['candidate']['tools']['passed']>=tool['baseline']['tools']['passed']
        result={'methodology':self.method,'cases_manifest':json.loads((self.root/'cases-manifest.json').read_text()),
                'selection':json.loads((self.root/'selected.json').read_text()),'phase_summaries':summaries,'quality_gate':gate,
                'speed_gate':speed,'legacy_nonregression_passed':regressions_passed,'additional_controls_passed':controls_passed,
                'inference_qualified':gate['passed'] and speed['passed'] and regressions_passed and controls_passed,
                'parser_cpu_regression':'Run separately; source test evidence required before recommendation',
                'production_parameters_changed':False,'completed_epoch':time.time(),
                'raw_results_sha256':hashlib.sha256((self.root/'results.jsonl').read_bytes()).hexdigest()}
        # Allowlist only: no endpoint, raw answer, tool argument, prompt or credential.
        keys=['task_id','phase','case','category','family','profile','profile_id','repeat','arm','stream','seed','passed','checks','code_tests',
              'usage','reasoning_chars','reasoning_sha256','finish','done','first_token_s','first_answer_s','total_s','request_sha256',
              'test_concurrency','error_type','http_status']
        save(self.root/'public-summary.json',result)
        save(self.root/'public-results.json',[{k:r[k] for k in keys if k in r} for r in self.rows])
        save(self.root/'public-speed.json',[{k:v for k,v in r.items()} for r in [json.loads(x) for x in (self.root/'speed.jsonl').read_text().splitlines()]])
        self.status('finished',inference_qualified=result['inference_qualified'],quality_gate=gate['passed'],speed_gate=speed['passed'])

    def main(self):
        self.prepare()
        # API smoke uses existing small tasks and exercises each effective effort.
        c=next(c for c in self.cases if c['id']=='workers')
        self.run_tasks('smoke',[self.task('smoke',c,{**baseline('reasoning'),'reasoning_effort':e},i) for i,e in enumerate(['none','low','medium','high','max'])])
        sm=[r for r in self.rows if r['phase']=='smoke']
        assert smoke_valid(sm),'API completion or thinking mode smoke failed; answer accuracy is scored separately'
        winners=self.screening();self.confirmation(winners);speed=self.speed(winners);self.export_summary(winners,speed)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-url',required=True);p.add_argument('--model',required=True)
    p.add_argument('--sandbox-image',required=True,help='Pinned local image with Python; executed with runc, no network, no GPUs and a read-only filesystem')
    p.add_argument('--output',required=True);args=p.parse_args();os.umask(0o077)
    experiment=Experiment(args)
    try:experiment.main()
    except BaseException as e:
        save(Path(args.output)/'error.json',{'type':type(e).__name__,'error':str(e),'at':time.time()});raise


if __name__=='__main__':main()
