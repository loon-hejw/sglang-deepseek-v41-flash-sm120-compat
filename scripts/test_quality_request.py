"""Request precedence, qualification and partial-answer behavior."""
import importlib.util
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location('quality_request',Path(__file__).resolve().parents[1]/'examples/quality-request.py')
client=importlib.util.module_from_spec(spec);spec.loader.exec_module(client)


class QualityRequestTests(unittest.TestCase):
    def test_explicit_zero_none_and_modern_budget_win(self):
        p={'temperature':1,'reasoning_effort':'high','max_tokens':24576,'top_p':1}
        request={'temperature':0,'reasoning_effort':'none','max_completion_tokens':1000}
        b=client.apply_profile(request,p)
        self.assertEqual(b['temperature'],0);self.assertFalse(b['chat_template_kwargs']['thinking'])
        self.assertNotIn('max_tokens',b);self.assertEqual(b['max_completion_tokens'],1000)
        self.assertEqual(p['max_tokens'],24576)

    def test_explicit_thinking_false_preserved(self):
        b=client.apply_profile({'chat_template_kwargs':{'thinking':False,'custom':'value'}},{'reasoning_effort':'max'})
        self.assertEqual(b['chat_template_kwargs'],{'thinking':False,'custom':'value'})

    def test_null_optional_fields_keep_api_default_semantics(self):
        b=client.apply_profile({'reasoning_effort':None,'chat_template_kwargs':None,'max_completion_tokens':None},
                               {'reasoning_effort':'low','max_tokens':12288})
        self.assertIsNone(b['reasoning_effort']);self.assertIsNone(b['chat_template_kwargs'])
        self.assertEqual(b['max_tokens'],12288)

    def test_unqualified_policy_is_opt_in(self):
        b={'qualified':False,'profiles':{'coding':{'reasoning_effort':'low'}}}
        with self.assertRaisesRegex(ValueError,'did not qualify'):client.select_profile(b,'coding')
        self.assertEqual(client.select_profile(b,'coding',True),b['profiles']['coding'])

    def test_incomplete_and_reasoning_only_fail(self):
        for finish,msg in [('length',{'content':'partial'}),('stop',{'reasoning_content':'thoughts'})]:
            with self.assertRaises(ValueError):client.validate_response({'choices':[{'finish_reason':finish,'message':msg}]})

    def test_tool_arguments_retained_without_execution(self):
        r={'choices':[{'finish_reason':'tool_calls','message':{'tool_calls':[{'function':{'name':'record','arguments':'{"note":"<think>literal</think>"}'}}]}}]}
        self.assertIs(client.validate_response(r),r)
        r['choices'][0]['message']['tool_calls'][0]['function']['arguments']='[]'
        with self.assertRaises(ValueError):client.validate_response(r)


if __name__=='__main__':unittest.main()
