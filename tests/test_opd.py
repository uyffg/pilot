import argparse
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from worker_reviewer_bench.opd.data import encode_pair, teacher_messages, template_messages
from worker_reviewer_bench.opd.collect import RecordingModel, Recorder, collect_one
from worker_reviewer_bench.opd.reviewer import StateReviewer
from worker_reviewer_bench.benchmarks import adapt
from worker_reviewer_bench.execution import shell
from helpers import CharacterTokenizer


class OPDTests(unittest.TestCase):
    def test_views_align_only_student_action_not_review_or_history(self):
        history = [{'role':'system','content':'WORKER'}, {'role':'user','content':'TASK'},
                   {'role':'assistant','content':'OLD ACTION'}, {'role':'tool','content':'FAILED TEST'}]
        original = deepcopy(history)
        row = {'id':'one', 'student_messages':history, 'review':'PRIVATE REVIEW',
               'action':{'role':'assistant','content':'RUN CHECK'}}
        pair = encode_pair(row, CharacterTokenizer(), 10000)
        self.assertEqual(history, original)
        self.assertNotIn('PRIVATE REVIEW', json.dumps(history))
        target = ''.join(map(chr, pair['target_ids']))
        self.assertIn('RUN CHECK', target)
        self.assertNotIn('PRIVATE REVIEW', target)
        for view in ['student','teacher']:
            start = pair[view]['start']
            self.assertEqual(pair[view]['input_ids'][start+1:], pair['target_ids'])
        with self.assertRaisesRegex(ValueError, 'no truncation'):
            encode_pair(row, CharacterTokenizer(), 8)

    def test_native_tool_arguments_are_normalized_without_mutating_api_messages(self):
        action = {'role':'assistant','content':'','tool_calls':[{'type':'function',
            'function':{'name':'bash','arguments':'{"command":"printf hello"}'}}]}
        rendered = template_messages([action])
        self.assertEqual(rendered[0]['tool_calls'][0]['function']['arguments'], {'command':'printf hello'})
        self.assertIsInstance(action['tool_calls'][0]['function']['arguments'], str)

    def test_full_vocab_kl_value_gradient_and_teacher_stopgrad(self):
        import torch
        from worker_reviewer_bench.opd.loss import full_vocab_kl
        torch.manual_seed(7)
        student = torch.randn(7, 4, requires_grad=True)
        teacher = torch.randn(7, 4, requires_grad=True)
        weight = torch.randn(13, 4)
        saved_shapes = []
        def pack(tensor):
            saved_shapes.append(tuple(tensor.shape))
            return tensor
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
            loss = full_vocab_kl(student, teacher, weight, chunk_size=2)
        self.assertNotIn(tuple(weight.shape), saved_shapes)
        logp = (student @ weight.T).log_softmax(-1)
        logq = (teacher.detach() @ weight.T).log_softmax(-1)
        reference = (logq.exp() * (logq - logp)).sum(-1).mean()
        self.assertTrue(torch.allclose(loss, reference, atol=1e-6))
        actual_grad, = torch.autograd.grad(loss, student, retain_graph=True)
        expected_grad, = torch.autograd.grad(reference, student)
        self.assertTrue(torch.allclose(actual_grad, expected_grad, atol=1e-6))
        loss.backward()
        self.assertIsNone(teacher.grad)
        self.assertIsNotNone(student.grad)
        # Accelerate may promote model outputs to fp32 while the frozen head
        # remains bf16. Projection must accept that mixed-precision boundary.
        mixed_student = student.detach().clone().requires_grad_(True)
        mixed = full_vocab_kl(mixed_student, teacher.detach(), weight.bfloat16(), 2)
        mixed.backward()
        self.assertTrue(torch.isfinite(mixed))
        self.assertTrue(torch.isfinite(mixed_student.grad).all())

    def test_native_qwen_hidden_projection_and_kl_backward(self):
        import torch
        import types
        from transformers import Qwen3_5MoeConfig, Qwen3_5MoeForConditionalGeneration
        from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig, Qwen3_5MoeVisionConfig
        from worker_reviewer_bench.opd.loss import full_vocab_kl
        text = Qwen3_5MoeTextConfig(vocab_size=32, hidden_size=32, num_hidden_layers=1,
            num_attention_heads=4, num_key_value_heads=2, head_dim=8, layer_types=['full_attention'],
            num_experts=4, num_experts_per_tok=2, moe_intermediate_size=16, shared_expert_intermediate_size=16)
        vision = Qwen3_5MoeVisionConfig(depth=1, hidden_size=32, intermediate_size=32,
            num_heads=4, out_hidden_size=32, num_position_embeddings=16)
        model = Qwen3_5MoeForConditionalGeneration(Qwen3_5MoeConfig(text_config=text, vision_config=vision)).eval()
        ids = torch.tensor([[1,2,3,4]])
        with torch.no_grad():
            expected = model(input_ids=ids).logits
        head = model.get_output_embeddings()
        head.requires_grad_(False)
        head.forward = types.MethodType(lambda self, hidden_states: hidden_states, head)
        hidden = model(input_ids=ids).logits
        self.assertTrue(torch.allclose(hidden @ head.weight.T, expected, atol=1e-6))
        with torch.no_grad():
            teacher = model(input_ids=torch.tensor([[1,7,3,4]])).logits[0,1:3].clone()
        full_vocab_kl(hidden[0,1:3], teacher, head.weight, 2).backward()
        self.assertTrue(any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
                            for p in model.parameters() if p.requires_grad))



    def test_reviewer_commands_cannot_mutate_student_workspace(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            workspace = root / 'student'
            workspace.mkdir()
            (workspace/'solution.py').write_text('original')
            (root/'reviews').mkdir()
            args = argparse.Namespace(reviewer_model=None, worker_model='worker', base_url='http://unused',
                api_key='dummy', reviewer_max_tokens=100, reviewer_turns=2, review_root=str(root/'reviews'),
                sandbox='local', agent_env=None, command_timeout=3)
            reviewer = StateReviewer(args)
            reviewer.client = Mock()
            reviewer.client.chat.side_effect = [json.dumps({'action':'bash', 'command':
                "printf overwritten > solution.py; printf private > reviewer_test.txt"}),
                json.dumps({'action':'review','decision':'check','reason':'untested boundary','check':'boundary test','hint':'test a boundary'})]
            review, audit = reviewer([{'role':'system','content':'worker'}, {'role':'user','content':'task'}], workspace)
            self.assertEqual((workspace/'solution.py').read_text(), 'original')
            self.assertFalse((workspace/'reviewer_test.txt').exists())
            self.assertEqual(list((root/'reviews').iterdir()), [])
            self.assertIn('check', review)

    def test_reviewer_final_call_is_constrained_and_failed_review_is_audited(self):
        from worker_reviewer_bench.opd.reviewer import response_format, REVIEW_SCHEMA
        self.assertEqual(response_format(True)['json_schema']['schema'], REVIEW_SCHEMA)
        self.assertIn('oneOf', response_format(False)['json_schema']['schema'])
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            workspace = root/'student'
            workspace.mkdir()
            args = argparse.Namespace(reviewer_model=None, worker_model='worker', base_url='http://unused',
                api_key='local', reviewer_max_tokens=100, reviewer_turns=1, review_root=str(root),
                sandbox='local', agent_env=None, command_timeout=3)
            reviewer = StateReviewer(args)
            reviewer.client = Mock()
            reviewer.client.chat.return_value = '{"action":"bash","command":"echo should_not_execute"}'
            with self.assertRaisesRegex(RuntimeError, 'audit:'):
                reviewer([{'role':'system','content':'worker'}, {'role':'user','content':'task'}], workspace)
            self.assertEqual(reviewer.client.extra_body['response_format'], response_format(True))
            failures = list((root/'failures').glob('*.json'))
            self.assertEqual(len(failures), 1)
            self.assertIn('should_not_execute', failures[0].read_text())

    def test_native_null_content_preserves_tool_calls(self):
        from minisweagent.models.utils.actions_toolcall import BASH_TOOL
        task = adapt({'task_id':'null', 'entry_point':'f', 'instruct_prompt':'implement f', 'code_prompt':''}, 'bcb')
        with tempfile.TemporaryDirectory() as td:
            args = argparse.Namespace(review_every=4, round=0, student_revision='base', backend='mini-swe')
            reviewer = Mock(return_value=('private review', []))
            recorder = Recorder(args, task, task.prepare(td), reviewer)
            action = {'role':'assistant','content':None,'tool_calls':[{'id':'one','type':'function',
                'function':{'name':'bash','arguments':'{"command":"echo hello"}'}}]}
            recorder.observe([{'role':'system','content':'worker'}], action, [BASH_TOOL])
            self.assertEqual(len(recorder.records), 1)
            self.assertEqual(recorder.records[0]['action']['tool_calls'], action['tool_calls'])
            self.assertNotIn('content', recorder.records[0]['action'])

    def test_dependency_overlay_is_mounted_read_only_and_missing_overlay_fails(self):
        import os
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            workspace = root/'workspace'
            workspace.mkdir()
            environment = root/'environment'
            environment.mkdir()
            overlay = root/'overlay'
            overlay.mkdir()
            process = Mock(pid=424242, returncode=0)
            process.communicate.return_value = ('ok', '')
            with patch.dict(os.environ, {'OPD_DEPENDENCY_OVERLAY':str(overlay)}), \
                 patch('worker_reviewer_bench.execution.subprocess.Popen', return_value=process) as popen, \
                 patch('worker_reviewer_bench.execution.os.killpg'):
                shell('python -c "import packaging"', workspace, 5, sandbox='bwrap', agent_env=str(environment))
                command = popen.call_args.args[0]
                self.assertIn('/opd-dependencies', command)
                index = command.index(str(overlay))
                self.assertEqual(command[index-1], '--ro-bind')
                self.assertIn('/opd-dependencies', popen.call_args.kwargs['env']['PYTHONPATH'])
            with patch.dict(os.environ, {'OPD_DEPENDENCY_OVERLAY':str(root/'missing')}):
                with self.assertRaisesRegex(ValueError, 'overlay is missing'):
                    shell('true', workspace, 5, sandbox='bwrap', agent_env=str(environment))

    def test_native_mini_wrapper_preserves_tool_calls_and_observations(self):
        from minisweagent.agents.default import DefaultAgent
        from minisweagent.models.utils.actions_toolcall import BASH_TOOL
        from worker_reviewer_bench.backends.mini_environment import make_environment
        with tempfile.TemporaryDirectory() as td:
            args = argparse.Namespace(shell=shell, command_timeout=3)
            env = make_environment(args, Path(td))
            model = Mock()
            model.get_template_vars.return_value = {}
            model.format_message.side_effect = lambda **kwargs: kwargs
            model.query.return_value = {'role':'assistant', 'content':None,
                'tool_calls':[{'id':'call_1','type':'function','function':{'name':'bash','arguments':'{"command":"printf hello"}'}}],
                'extra':{'actions':[{'command':'printf hello','tool_call_id':'call_1'}]}}
            model.format_observation_messages.side_effect = lambda msg, outputs, vars: [
                {'role':'tool', 'tool_call_id':'call_1', 'content':outputs[0]['output']}]
            recorder = Mock()
            agent = DefaultAgent(RecordingModel(model, recorder), env,
                system_template='worker', instance_template='{{task}}', cost_limit=0)
            agent.messages = [{'role':'system','content':'worker'}, {'role':'user','content':'task'}]
            agent.step()
            self.assertEqual(agent.messages[-1]['role'], 'tool')
            self.assertIn('hello', agent.messages[-1]['content'])
            self.assertEqual(agent.messages[2]['tool_calls'][0]['id'], 'call_1')
            self.assertEqual(recorder.observe.call_args.args[2], [BASH_TOOL])


if __name__ == '__main__':
    unittest.main()
