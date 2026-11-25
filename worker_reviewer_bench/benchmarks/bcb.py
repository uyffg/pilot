from __future__ import annotations
import subprocess, tempfile, re, os, sys
from pathlib import Path

def grade(row, field, timeout):
    code=row.get(field,'') or ''
    if not code: return {'status':'empty','passed':False,'tests_run':0}
    test=row.get('test','')
    with tempfile.TemporaryDirectory(prefix='bcb-grade-') as td:
        root=Path(td); (root/'solution.py').write_text(code,encoding='utf8')
        runner='from solution import *\n'+test+'\nif __name__=="__main__":\n import unittest; unittest.main()\n'
        (root/'runner.py').write_text(runner,encoding='utf8')
        try:
            env = dict(os.environ, MPLBACKEND='Agg', QT_QPA_PLATFORM='offscreen', DISPLAY='', WAYLAND_DISPLAY='', BROWSER='true')
            p=subprocess.run([sys.executable,'runner.py'],cwd=root,capture_output=True,text=True,timeout=timeout,env=env)
            output=p.stdout+'\n'+p.stderr
            count=re.search(r'Ran (\d+) tests?',output)
            n=int(count.group(1)) if count else 0
            passed=p.returncode==0 and n>0 and bool(re.search(r'^OK(?:\s|$)',output,re.M))
            return {'status':'pass' if passed else 'fail','passed':passed,'tests_run':n,'returncode':p.returncode,'tail':output[-2400:]}
        except subprocess.TimeoutExpired: return {'status':'timeout','passed':False,'tests_run':0}
