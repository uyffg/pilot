import argparse
from functools import partial
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from worker_reviewer_bench.backends.mini import _make_agent, run
from worker_reviewer_bench.benchmarks import adapt
from worker_reviewer_bench.cli import parser
from worker_reviewer_bench.execution import shell


class MiniTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec('minisweagent'),'optional mini-swe-agent dependency is absent')
    def test_native_environment_executes_and_recognizes_submission(self):
        from minisweagent.exceptions import Submitted
        from worker_reviewer_bench.backends.mini_environment import make_environment
        with tempfile.TemporaryDirectory() as tmp:
            env=make_environment(argparse.Namespace(shell=shell,command_timeout=5),Path(tmp))
            result=env.execute({'command':'printf hello'})
            self.assertEqual(result['returncode'],0)
            self.assertIn('hello',result['output'])
            with self.assertRaises(Submitted):
                env.execute({'command':"printf 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\\nanswer\\n'"})

    def test_role_configuration_reaches_model_and_agent_factories(self):
        args=parser().parse_args(['collect','--benchmark','bcb','--backend','mini-swe',
            '--tasks','unused','--output','unused','--workdir','unused','--worker-model','worker',
            '--reviewer-model','reviewer','--reviewer-turns','11'])
        agent,model,environment=Mock(),Mock(),Mock()
        with tempfile.TemporaryDirectory() as tmp:
            with patch('worker_reviewer_bench.backends.mini._mini_imports',return_value=(agent,model)), \
                 patch('worker_reviewer_bench.backends.mini.make_environment',return_value=environment):
                for role,expected_model,expected_turns in [('worker','worker',160),('reviewer','reviewer',11)]:
                    _make_agent(args,Path(tmp),'system','instance',Path(tmp)/'trajectory.json',role=role)
                    self.assertEqual(model.call_args.kwargs['config']['model_name'],expected_model)
                    self.assertEqual(agent.call_args.args[2]['step_limit'],expected_turns)

    def test_budget_termination_is_not_a_successful_first_submit(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=parser().parse_args(['collect','--benchmark','lcb','--backend','mini-swe',
                '--tasks','unused','--output','unused','--workdir',tmp,'--no-review'])
            args.reviewer_turns=160
            task=adapt({'question_id':'example','question_content':'task'},'lcb')
            workspace=task.prepare(tmp)
            (workspace/'solution.py').write_text('print(1)')
            agent=Mock()
            agent.messages=[]
            agent.run.return_value={'exit_status':'LimitsExceeded'}
            with patch('worker_reviewer_bench.backends.mini._make_agent',return_value=agent):
                result=run(task,args)
            self.assertFalse(result['submitted'])
            self.assertFalse(result['first_submit_captured'])
            self.assertEqual(result['first_submit_solution'],'')
            self.assertEqual(result['final_solution'],'print(1)')


if __name__=='__main__':
    unittest.main()
