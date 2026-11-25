import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from minisweagent.agents.default import DefaultAgent
from minisweagent.exceptions import FormatError, Submitted
from worker_reviewer_bench.backends.mini_protocol import ProtocolFeedbackModel, PROTOCOL_APPENDIX
from worker_reviewer_bench.backends.mini import _make_agent
from worker_reviewer_bench.cli import parser


def failure(content='Partial derivation', finish_reason='length'):
    return FormatError({'role': 'user', 'content': 'Missing command', 'extra': {
        'cost': .1, 'response': {'choices': [{'finish_reason': finish_reason,
            'message': {'content': content, 'tool_calls': [{'id': 'invalid', 'function': {'name': 'bash', 'arguments': '{}'}}]}}],
            'usage': {'completion_tokens': 1400}}}})


class ProtocolTests(unittest.TestCase):
    def test_valid_native_response_is_unchanged(self):
        original = {'role': 'assistant', 'extra': {'actions': [{'command': 'printf hello', 'tool_call_id': 'one'}]}}
        model = Mock(); model.query.return_value = original
        wrapper = ProtocolFeedbackModel(model)
        self.assertIs(wrapper.query([]), original)
        self.assertEqual(wrapper.format_recoveries, [])

    def test_truncated_text_kept_without_invalid_tool_calls(self):
        model = Mock(); model.config.model_kwargs = {'max_tokens': 1400}
        error = failure(); model.query.side_effect = error
        with self.assertRaises(FormatError): ProtocolFeedbackModel(model).query([])
        prior, feedback = error.messages
        self.assertEqual(prior['content'], 'Partial derivation')
        self.assertNotIn('tool_calls', prior)
        self.assertNotIn('actions', prior.get('extra', {}))
        self.assertEqual(prior['extra']['cost'], .1)
        self.assertIn('incomplete call did not run', feedback['content'])

    def test_empty_arguments_never_become_commands(self):
        model = Mock(); model.config.model_kwargs = {'max_tokens': 1400}
        error = failure(None, 'tool_calls'); model.query.side_effect = error
        wrapper = ProtocolFeedbackModel(model)
        with self.assertRaises(FormatError): wrapper.query([])
        self.assertEqual(len(error.messages), 1)
        self.assertTrue(wrapper.format_recoveries[0]['truncated'])

    def test_real_agent_recovers_and_submits_native_action(self):
        class Model:
            config = SimpleNamespace(model_kwargs={'max_tokens': 1400})
            def __init__(self): self.calls = 0
            def query(self, messages):
                self.calls += 1
                if self.calls == 1: raise failure()
                assert any(m['role']=='assistant' and m['content']=='Partial derivation' for m in messages)
                return {'role':'assistant','content':None,'extra':{'actions':[{'command':'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT','tool_call_id':'valid'}]}}
            def format_message(self, **kwargs): return kwargs
            def get_template_vars(self): return {}
            def serialize(self): return {}
        class Environment:
            def __init__(self): self.commands = []
            def get_template_vars(self): return {}
            def serialize(self): return {}
            def execute(self, action):
                self.commands.append(action['command'])
                raise Submitted({'role':'exit','content':'done','extra':{'exit_status':'Submitted','submission':''}})
        model, env = Model(), Environment()
        agent=DefaultAgent(ProtocolFeedbackModel(model),env,system_template='system',instance_template='{{task}}',step_limit=160,cost_limit=0)
        self.assertEqual(agent.run('synthetic')['exit_status'],'Submitted')
        self.assertEqual(env.commands,['COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'])
        self.assertEqual(agent.n_calls,2)
        self.assertEqual(agent.config.max_consecutive_format_errors,3)

    def test_opt_in_preserves_old_worker_and_reviewer_prompts(self):
        args=parser().parse_args(['collect','--benchmark','bcb','--backend','mini-swe','--tasks','unused','--output','unused','--workdir','unused'])
        factory,model=Mock(),Mock()
        with tempfile.TemporaryDirectory() as tmp, patch('worker_reviewer_bench.backends.mini._mini_imports',return_value=(factory,Mock(return_value=model))), patch('worker_reviewer_bench.backends.mini.make_environment'):
            for enabled,role,suffix in [(False,'worker',''),(True,'worker',PROTOCOL_APPENDIX),(True,'reviewer','')]:
                args.mini_protocol_compat=enabled
                _make_agent(args,Path(tmp),'original','instance',Path(tmp)/'traj.json',role)
                self.assertEqual(factory.call_args.args[2]['system_template'],'original'+suffix)

    def test_larger_budget_and_timeout_reach_native_model(self):
        args=parser().parse_args(['collect','--benchmark','bcb','--backend','mini-swe','--tasks','unused','--output','unused','--workdir','unused'])
        args.mini_protocol_compat=True
        args.max_tokens=32768
        args.mini_request_timeout=3600
        model_factory=Mock(return_value=Mock())
        with tempfile.TemporaryDirectory() as tmp, patch('worker_reviewer_bench.backends.mini._mini_imports',return_value=(Mock(),model_factory)), patch('worker_reviewer_bench.backends.mini.make_environment'):
            _make_agent(args,Path(tmp),'system','instance',Path(tmp)/'traj.json')
        kwargs=model_factory.call_args.kwargs['config']['model_kwargs']
        self.assertEqual(kwargs['max_tokens'],32768)
        self.assertEqual(kwargs['timeout'],3600)


if __name__=='__main__': unittest.main()
