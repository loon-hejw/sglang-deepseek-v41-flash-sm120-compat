"""Request overrides and completion handling for the opt-in coding client."""
import importlib.util
from pathlib import Path
import unittest

path = Path(__file__).resolve().parents[1]/'examples/code-request.py'
spec = importlib.util.spec_from_file_location('code_request', path)
client = importlib.util.module_from_spec(spec)
spec.loader.exec_module(client)


class CodeRequestTests(unittest.TestCase):
    def test_explicit_fields_win(self):
        request = {'temperature': 0, 'max_tokens': 4096, 'reasoning_effort': 'none', 'messages': []}
        profile = {'temperature': 1, 'max_tokens': 12288, 'reasoning_effort': 'low', 'top_p': 1}
        actual = client.apply_profile(request, profile)
        self.assertEqual(actual, {**request, 'top_p': 1})
        self.assertEqual(profile['temperature'], 1)

    def test_normal_final_content(self):
        response = {'choices': [{'finish_reason': 'stop', 'message': {'content': 'def f(): return 1'}}]}
        self.assertEqual(client.code_from_response(response), 'def f(): return 1')

    def test_reject_truncated_even_with_content(self):
        response = {'choices': [{'finish_reason': 'length', 'message': {'content': 'def f(): return 1'}}]}
        with self.assertRaisesRegex(ValueError, 'budget'):
            client.code_from_response(response)

    def test_never_promote_reasoning(self):
        response = {'choices': [{'finish_reason': 'stop', 'message': {'content': '', 'reasoning_content': 'def f(): return 1'}}]}
        with self.assertRaisesRegex(ValueError, 'no final code'):
            client.code_from_response(response)

    def test_reject_abnormal_finish(self):
        for reason in [None, 'tool_calls', 'content_filter']:
            with self.subTest(reason=reason), self.assertRaises(ValueError):
                client.code_from_response({'choices': [{'finish_reason': reason, 'message': {'content': 'partial'}}]})


if __name__ == '__main__':
    unittest.main()
