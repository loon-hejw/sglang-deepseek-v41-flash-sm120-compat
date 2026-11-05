"""Guard against optimistic scoring, answer leakage and holdout selection bugs."""
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parent))
from quality_eval import cases,grading,policy
from quality_eval.client import Client
from quality_eval.run import Experiment,save


class EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.cases=cases.build_suite()

    def test_disjoint_families_and_counts(self):
        self.assertEqual(len(self.cases),210)
        self.assertEqual(len(cases.legacy_suite()),51)
        a={c['family'] for c in self.cases if c['split']=='selection'}
        b={c['family'] for c in self.cases if c['split']=='holdout'}
        self.assertEqual((len(a),len(b)),(7,14));self.assertFalse(a&b)

    def test_schema_does_not_force_missing_answers(self):
        c=next(c for c in self.cases if c['family']=='structured/h_versioned_join')
        self.assertEqual(c['schema']['properties']['orders']['items']['properties']['name']['type'],['string','null'])
        for c in self.cases:
            if c.get('strict_schema'):
                raw=json.dumps(c['schema']);self.assertNotIn('"const"',raw);self.assertNotIn('"enum"',raw)

    def test_reference_solvers(self):
        self.assertEqual(cases.discounted_path(3,[[0,1,9],[1,2,3],[0,2,30]],0,2,2),5)
        self.assertIsNone(cases.discounted_path(2,[],0,1,2))
        self.assertEqual(cases.discounted_path(1,[],0,0,2),0)
        self.assertTrue(cases.glob_reference('a\nb','a?b'))
        self.assertTrue(cases.glob_reference('*?',r'\*\?'))
        with self.assertRaises(ValueError):cases.glob_reference('cannot match','x\\')

    def test_types_truncation_and_unknown_finish_fail(self):
        c={'grader':'json','expected':{'n':0}}
        good={'content':'{"n":0}','tools':[],'done':True,'finish':['stop']}
        self.assertTrue(grading.grade(c,good)['passed'])
        for r in [dict(good,content='{"n":false}'),dict(good,finish=['length']),dict(good,finish=['error']),dict(good,done=False)]:
            self.assertFalse(grading.grade(c,r)['passed'])

    def test_tool_semantics_and_literal_preservation(self):
        want={'name':'record','arguments':{'note':'<think>literal</think>','count':0}}
        c={'grader':'tools','expected':[want]}
        r={'done':True,'finish':['tool_calls'],'content':'','tools':[{'name':'record','arguments':json.dumps(want['arguments'])}]}
        self.assertTrue(grading.grade(c,r)['passed'])
        r['tools'][0]['arguments']=json.dumps({'note':'</think>literal</think>','count':0})
        self.assertFalse(grading.grade(c,r)['passed'])

    def test_request_controls_override_case_defaults(self):
        cl=Client('http://localhost/v1','model')
        c=next(c for c in self.cases if c['category']=='structured')
        b=cl.body(c,dict(policy.baseline('structured'),reasoning_effort='medium',temperature=.6),3)
        self.assertEqual(b['reasoning_effort'],'medium');self.assertTrue(b['chat_template_kwargs']['thinking'])
        self.assertEqual(b['temperature'],.6);self.assertTrue(b['response_format']['json_schema']['strict'])

    def test_sse_fragments_and_first_answer(self):
        frames=[{'choices':[{'delta':{'reasoning_content':'check'}}]},
                {'choices':[{'delta':{'tool_calls':[{'index':0,'function':{'name':'rec','arguments':'{"note":"<th'}}]}}]},
                {'choices':[{'delta':{'tool_calls':[{'index':0,'function':{'name':'ord','arguments':'ink>"}'}}]},'finish_reason':'tool_calls'}]},
                {'choices':[],'usage':{'completion_tokens':20}}]
        payload=b''.join(b'data: '+json.dumps(f).encode()+b'\n\n' for f in frames)+b'data: [DONE]\n\n'
        with patch('urllib.request.urlopen',return_value=io.BytesIO(payload)):
            r=Client('http://localhost/v1','model').generate({'stream':True})
        self.assertTrue(r['done']);self.assertEqual(r['tools'],[{'name':'record','arguments':'{"note":"<think>"}'}])
        self.assertGreaterEqual(r['first_answer_s'],r['first_token_s'])

    def test_tied_screen_keeps_baseline(self):
        ps=policy.initial_profiles('coding')
        rows=[{'category':'coding','profile_id':policy.profile_id(p),'passed':True,'finish':['stop']} for p in ps]
        self.assertEqual(policy.rank_profiles('coding',ps,rows)[0],policy.baseline('coding'))

    def test_smoke_keeps_wrong_answer_but_rejects_broken_api(self):
        rows=[{'passed':False,'done':True,'checks':{'complete':True},'usage':{'completion_tokens':8},
               'reasoning_chars':0 if e=='none' else 30,'profile':{'reasoning_effort':e}} for e in ['none','low','medium','high','max']]
        self.assertTrue(policy.smoke_valid(rows))
        self.assertFalse(policy.smoke_valid([{**rows[0],'done':False},*rows[1:]]))
        self.assertFalse(policy.smoke_valid([{**rows[0],'reasoning_chars':20},*rows[1:]]))

    def test_gate_requires_pairs_and_families(self):
        rows=[]
        for cat in cases.CATEGORIES:
            for family in ['a','b']:
                for i in range(10):
                    for arm in ['baseline','candidate']:
                        rows.append({'category':cat,'family':cat+'/'+family,'case':family+str(i),'repeat':0,
                                     'arm':arm,'passed':arm=='candidate' or i<8})
        g=policy.quality_gate(rows,1000)
        self.assertTrue(g['passed']);self.assertAlmostEqual(g['macro_delta'],.2)
        with self.assertRaises(AssertionError):policy.quality_gate(rows[:-1],100)
        with self.assertRaises(AssertionError):policy.quality_gate(rows+rows[:1],100)

    def test_confirmation_is_complete_and_never_retunes(self):
        e=Experiment.__new__(Experiment);e.cases=self.cases+cases.legacy_suite()
        captured={};e.run_tasks=lambda phase,tasks:captured.update({phase:tasks})
        winners={c:policy.baseline(c) for c in cases.CATEGORIES}
        before=json.dumps(winners,sort_keys=True);e.confirmation(winners)
        self.assertEqual({k:len(v) for k,v in captured.items()},
                         {'holdout':1400,'regression':204,'code_low_control':80,'tools_nonstream':120})
        self.assertEqual(json.dumps(winners,sort_keys=True),before)
        ids=[t['task_id'] for tasks in captured.values() for t in tasks]
        self.assertEqual(len(ids),len(set(ids)))

    def test_export_excludes_raw_answers_and_tool_arguments(self):
        with tempfile.TemporaryDirectory() as directory:
            e=Experiment.__new__(Experiment);e.root=Path(directory);e.method={};e.status=lambda *a,**kw:None
            e.rows=[]
            for phase in ['holdout','regression','code_low_control','tools_nonstream']:
                for cat in cases.CATEGORIES:
                    for arm in ['baseline','candidate']:
                        e.rows.append({'phase':phase,'category':cat,'family':cat+'/f','case':cat,'repeat':0,'arm':arm,
                            'passed':True,'finish':['stop'],'total_s':1,'first_answer_s':.5,
                            'usage':{'completion_tokens':12,'reasoning_tokens':4},
                            'content':'PRIVATE_ANSWER_SENTINEL','tools':[{'arguments':'PRIVATE_ARGUMENT_SENTINEL'}]})
            save(e.root/'cases-manifest.json',{});save(e.root/'selected.json',{})
            (e.root/'results.jsonl').write_text('\n'.join(json.dumps(r) for r in e.rows)+'\n')
            (e.root/'speed.jsonl').write_text('{}\n')
            e.export_summary({}, {'passed':True})
            public=(e.root/'public-results.json').read_text()
            self.assertNotIn('PRIVATE_ANSWER_SENTINEL',public);self.assertNotIn('PRIVATE_ARGUMENT_SENTINEL',public)
            summary=json.loads((e.root/'public-summary.json').read_text())
            self.assertFalse(summary['inference_qualified'])


if __name__=='__main__':unittest.main()
