"""Objective graders; generated code runs only in an isolated CPU container."""
import json
import os
import re
import subprocess
import uuid

SANDBOX_IMAGE = None

def equal(a,b):
    if isinstance(b,bool) or b is None:return type(a) is type(b) and a==b
    if isinstance(b,(int,float)):
        return isinstance(a,(int,float)) and not isinstance(a,bool) and abs(a-b)<1e-9
    if isinstance(b,dict):return isinstance(a,dict) and a.keys()==b.keys() and all(equal(a[k],b[k]) for k in b)
    if isinstance(b,list):return isinstance(a,list) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return type(a) is type(b) and a==b


SANDBOX = r'''
import contextlib,copy,io,json,resource,sys
resource.setrlimit(resource.RLIMIT_CPU,(3,3))
resource.setrlimit(resource.RLIMIT_AS,(384*1024*1024,384*1024*1024))
resource.setrlimit(resource.RLIMIT_FSIZE,(1024*1024,1024*1024))
class Limited(io.StringIO):
    def write(self,s):
        if self.tell()+len(s)>65536:raise RuntimeError('excess output')
        return super().write(s)
d=json.load(sys.stdin); results=[]; ns={}
def same(a,b):
    if isinstance(b,bool) or b is None:return type(a) is type(b) and a==b
    if isinstance(b,(int,float)):return isinstance(a,(int,float)) and not isinstance(a,bool) and abs(a-b)<1e-9
    if isinstance(b,list):return isinstance(a,list) and len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    if isinstance(b,dict):return isinstance(a,dict) and a.keys()==b.keys() and all(same(a[k],b[k]) for k in b)
    return type(a) is type(b) and a==b
try:
    with contextlib.redirect_stdout(Limited()),contextlib.redirect_stderr(Limited()):
        exec(compile(d['code'],'<model-code>','exec'),ns)
        fn=ns[d['function']]
        for args,want in d['tests']:
            before=copy.deepcopy(args)
            try:
                got=fn(*args)
                passed=same(got,want)
                if d.get('ordered_keys') and isinstance(want,dict):passed=passed and list(got)==list(want)
                if d.get('immutable'):passed=passed and args==before
                results.append({'passed':bool(passed)})
            except Exception as e:
                results.append({'passed':isinstance(want,dict) and want=={'raises':type(e).__name__}})
    print(json.dumps({'tests':len(results),'passed':sum(x['passed'] for x in results)}))
except BaseException as e:print(json.dumps({'tests':len(d['tests']),'passed':0,'error_type':type(e).__name__}))
'''


def grade_code(case,text):
    if not SANDBOX_IMAGE:
        raise RuntimeError('Explicit CPU sandbox image required for code evaluation')
    code=text.strip()
    m=re.fullmatch(r'```(?:python|py)?\s*\n(.*?)\n```',code,re.S)
    if m:code=m.group(1)
    # Syntax validation does not execute the generated program.
    try:compile(code,'<model-code>','exec')
    except SyntaxError:return {'tests':len(case['tests']),'passed':0,'error_type':'SyntaxError'}
    payload={'code':code,'function':case['function'],'tests':case['tests'],'ordered_keys':case.get('ordered_keys',False),
             'immutable':case.get('immutable',case['id'] in ['code_merge_intervals','code_last_occurrence'])}
    # Identical completions can be graded concurrently for paired repetitions.
    # Each invocation owns its container, including its unconditional cleanup.
    name='quality-code-'+str(os.getpid())+'-'+uuid.uuid4().hex
    cmd=['docker','run','--rm','--name',name,'--runtime','runc','--network','none',
         '--read-only','--cap-drop','ALL','--security-opt','no-new-privileges',
         '--memory','512m','--memory-swap','512m','--cpus','1','--pids-limit','32',
         '--user','65534:65534','--tmpfs','/tmp:rw,noexec,nosuid,size=32m',
         '--env','NVIDIA_VISIBLE_DEVICES=void','--entrypoint','python3','-i',SANDBOX_IMAGE,'-I','-c',SANDBOX]
    try:
        r=subprocess.run(cmd,input=json.dumps(payload),text=True,capture_output=True,timeout=15)
        if r.returncode:return {'tests':len(case['tests']),'passed':0,'error_type':'SandboxExit','returncode':r.returncode}
        return json.loads(r.stdout)
    except subprocess.TimeoutExpired:return {'tests':len(case['tests']),'passed':0,'error_type':'SandboxTimeout'}
    except Exception as e:return {'tests':len(case['tests']),'passed':0,'error_type':type(e).__name__}
    finally:
        subprocess.run(['docker','rm','-f',name],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=10)


def grade(case,r):
    if 'error_type' in r:return {'passed':False,'checks':{'api_success':False}}
    text=r['content']
    checks={'complete':r['done'] and bool(r['finish']) and all(x in ['stop','tool_calls'] for x in r['finish']),
            'no_protocol_leak':not any(s in text for s in ['<think>','</think>','｜DSML｜']) or case.get('literal_content',False)}
    detail={}
    try:
        if case['grader']=='exact':checks['answer']=text.strip()==case['expected']
        elif case['grader']=='json':
            obj=json.loads(text.strip());checks['valid_json']=True;checks['answer']=equal(obj,case['expected'])
        elif case['grader']=='tools':
            actual=[{'name':x['name'],'arguments':json.loads(x['arguments'])} for x in r['tools']]
            key=lambda x:json.dumps(x,ensure_ascii=False,sort_keys=True)
            checks['tool_arguments']=equal(sorted(actual,key=key),sorted(case['expected'],key=key))
            checks['no_extra_prose']=not text.strip()
        elif case['grader']=='code':
            detail=grade_code(case,text);checks['unit_tests']=detail['passed']==detail['tests']
    except (json.JSONDecodeError,TypeError,ValueError,KeyError):checks['parseable_answer']=False
    return {'passed':all(checks.values()),'checks':checks,**({'code_tests':detail} if detail else {})}

