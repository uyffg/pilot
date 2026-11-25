import argparse
import base64
import contextlib
import io
import json
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch
import zlib

from worker_reviewer_bench.cli import main, parser
from worker_reviewer_bench.benchmarks import adapt
from worker_reviewer_bench.evaluation import decode_cases
from worker_reviewer_bench.execution import shell


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def write(self, name, rows):
        path = self.root/name
        path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
        return str(path)

    def invoke(self, argv):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return main(argv)

    def collect_args(self, benchmark, backend):
        row = {'task_id':'BigCodeBench/1','entry_point':'f','instruct_prompt':'Return 1',
               'code_prompt':'def f():', 'question_id':'sample','question_content':'Print 1',
               'test':'SECRET_TESTS', 'canonical_solution':'SECRET_REFERENCE',
               'private_test_cases':'SECRET_PRIVATE'}
        return ['collect','--benchmark',benchmark,'--backend',backend,
            '--tasks',self.write('tasks.jsonl',[row]),'--output',str(self.root/'rows.jsonl'),
            '--workdir',str(self.root/'workspaces'),'--worker-model','worker',
            '--reviewer-model','reviewer','--workers','1','--max-turns','8']




    def test_mini_benchmarks_repair_and_separate_roles(self):
        for benchmark in ['bcb','lcb']:
            with self.subTest(benchmark=benchmark):
                (self.root/'rows.jsonl').unlink(missing_ok=True)
                argv=self.collect_args(benchmark,'mini-swe')
                calls=[]
                def factory(args,root,system,instance,output,role='worker'):
                    calls.append((role,args.worker_model,args.reviewer_model))
                    attempt=int(output.name.split('_')[1].split('.')[0])
                    class FakeAgent:
                        def run(self,prompt):
                            if role=='worker':
                                (root/'solution.py').write_text('first' if attempt==0 else 'fixed')
                            else:
                                (root/'review.json').write_text(json.dumps({'decision':'intervene' if attempt==0 else 'pass'}))
                            return {'exit_status':'Submitted'}
                    return FakeAgent()
                with patch('worker_reviewer_bench.backends.mini._make_agent',side_effect=factory):
                    self.assertEqual(self.invoke(argv),0)
                    self.assertEqual(self.invoke(argv),0)
                row=json.loads((self.root/'rows.jsonl').read_text())
                self.assertEqual(row['first_submit_solution'],'first')
                self.assertEqual(row['final_solution'],'fixed')
                self.assertEqual(len(row['worker_attempts']),2)
                self.assertEqual([c[0] for c in calls],['worker','reviewer','worker','reviewer'])




    def test_workspace_ids_cannot_escape_workdir(self):
        task=adapt({'question_id':'../../escape','question_content':'task'},'lcb')
        workspace=task.prepare(self.root)
        self.assertEqual(workspace.parent,self.root)

    def test_actual_bcb_grading(self):
        tasks=self.write('tests.jsonl',[{'task_id':'t','test':
            'import unittest\nclass Tests(unittest.TestCase):\n def test_f(self): self.assertEqual(f(),1)'}])
        rows=self.write('predictions.jsonl',[{'task_id':'t','first_submit_solution':'def f(): return 0',
            'final_solution':'def f(): return 1'}])
        cache=self.root/'cache.json'
        argv=['grade','--benchmark','bcb','--tasks',tasks,'--rows',rows,
              '--out',str(self.root/'grades.json'),'--cache',str(cache)]
        self.assertEqual(self.invoke(argv),0)
        report=json.loads((self.root/'grades.json').read_text())
        self.assertEqual([r['passed'] for r in report],[0,1])
        with patch('worker_reviewer_bench.evaluation.grade_bcb',side_effect=AssertionError('cache miss')):
            self.assertEqual(self.invoke(argv),0)
        self.assertEqual(len(json.loads(cache.read_text())),2)

    def test_lcb_grading_denominators_and_exclusions(self):
        case=[{'input':'','output':'1','testtype':'stdin'}]
        tasks=self.write('tests.jsonl',[{'question_id':q,'private_test_cases':case} for q in ['a','b','abc337_e']])
        rows=self.write('predictions.jsonl',[{'question_id':'a','final_solution':'print(1)'}])
        self.assertEqual(self.invoke(['grade','--benchmark','lcb','--tasks',tasks,'--rows',rows,
            '--out',str(self.root/'grades.json')]),0)
        first,final=json.loads((self.root/'grades.json').read_text())
        self.assertEqual(first['results']['a']['status'],'missing_snapshot')
        self.assertEqual(final['accuracy'],0.5)
        self.assertEqual(final['evaluated_accuracy'],1.0)
        self.assertEqual(final['missing_tasks'],['b'])
        self.assertEqual(final['excluded_tasks'],['abc337_e'])

    def test_compressed_private_tests_data_only(self):
        cases=[{'input':'','output':'1'}]
        encoded=base64.b64encode(zlib.compress(pickle.dumps(json.dumps(cases)))).decode()
        self.assertEqual(decode_cases(encoded),cases)
        payload=base64.b64encode(zlib.compress(pickle.dumps(Path))).decode()
        with self.assertRaises(pickle.UnpicklingError):decode_cases(payload)



    def test_shell_decodes_invalid_utf8_and_times_out(self):
        self.assertIn('exit_code=0',shell("printf '\\377'",self.root,5))
        self.assertEqual(shell('sleep 3',self.root,0.05),'timeout_after_seconds=0.05')


if __name__ == '__main__':
    unittest.main()
