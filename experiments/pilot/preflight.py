"""Checks of one node before any GPU run. Run it with the training environment's Python.

    source experiments/pilot/env.sh
    python experiments/pilot/preflight.py --gpus 8          # everything
    python experiments/pilot/preflight.py --gpus 8 --skip distributed   # without the GPU trainer test

Each check prints PASS, WARN or FAIL with its evidence. The report is saved as
$PILOT_OUT/preflight/<host>.json, and launch.sh refuses to start on a node whose report is
missing or has a FAIL. Nothing here loads the 35B model on a GPU; smoke.sh does that.

Checks:
  repository   task files and their hashes, the 28-round configurations, the job table
  packages     the modules and symbols the trainer patches or imports, in the versions installed
  gpu          device count and type, a computation on every device, the FLA Triton kernel
  model        pinned revision and provenance file; chat-template alignment of teacher and student
  frontend     Mini-SWE imports and the repository's own unit tests
  sandbox      the recorded Bubblewrap mode still works and isolates (sandbox_probe.py --check)
  storage      locking and atomic directories on PILOT_OUT, free disk, memory, descriptors
  distributed  the real FSDP2 trainer on a tiny Qwen3.5 MoE built from the model's configuration
"""
import argparse
import fcntl
import importlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import time

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
RESULTS = []


def report(name, ok, detail, warn=False):
    level = 'PASS' if ok else ('WARN' if warn else 'FAIL')
    RESULTS.append({'check': name, 'level': level, 'detail': detail})
    print(f'{level:4s} {name}: {detail}', flush=True)
    return ok


def attempt(name, function, warn=False):
    try:
        return report(name, True, function())
    except Exception as error:  # a failed check is a result, not a crash
        return report(name, False, f'{type(error).__name__}: {str(error)[-1500:]}', warn=warn)


def run(command, **kwargs):
    result = subprocess.run(list(map(str, command)), capture_output=True, text=True, **kwargs)
    if result.returncode:
        raise RuntimeError(f'{" ".join(map(str, command[:4]))} ... exited {result.returncode}: '
                           f'{(result.stdout + result.stderr)[-1500:]}')
    return result.stdout + result.stderr


def check_repository():
    def task_files():
        return run([sys.executable, HERE / 'prepare_bcb_full.py', '--check']).strip().splitlines()[-1]

    def configurations():
        for profile in ('General', 'Specific'):
            original = json.loads((REPO / 'experiments' / 'bcb_opd' / f'{profile}.json').read_text())
            ours = json.loads((HERE / 'configs' / f'{profile}_r28.json').read_text())
            changed = {key for key in set(original) | set(ours) if original.get(key) != ours.get(key)}
            if changed != {'rounds'} or ours['rounds'] != 28 or original['rounds'] != 10:
                raise ValueError(f'{profile}: differs from the original in {sorted(changed)}')
        return 'General_r28 and Specific_r28 equal the original configurations except rounds: 10 -> 28'

    def table():
        from experiments.pilot.pilot import Settings, jobs
        listed = jobs(Settings())
        commit = subprocess.run(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
        return f'{len(listed)} jobs; checkout {commit or "unknown"}'

    attempt('repository.task_files', task_files)
    attempt('repository.configurations', configurations)
    attempt('repository.job_table', table)


def check_packages():
    def versions():
        found = {}
        for name in ('torch', 'transformers', 'vllm', 'peft', 'accelerate', 'liger_kernel', 'fla', 'tokenizers',
                     'safetensors', 'huggingface_hub'):
            module = importlib.import_module(name)
            found[name] = getattr(module, '__version__', 'unknown')
        return json.dumps(found)

    def trainer_symbols():
        from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as modeling
        for symbol in ('torch_chunk_gated_delta_rule', 'Qwen3_5MoeRMSNormGated', 'Qwen3_5MoeDecoderLayer'):
            if not hasattr(modeling, symbol):
                raise AttributeError(f'transformers qwen3_5_moe has no {symbol}')
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule  # noqa: F401
        from liger_kernel.transformers.model import qwen3_5_moe as liger_model
        from liger_kernel.transformers.monkey_patch import MODEL_TYPE_TO_APPLY_LIGER_FN
        if not hasattr(liger_model, 'LigerForCausalLMLoss'):
            raise AttributeError('liger qwen3_5_moe has no LigerForCausalLMLoss')
        model_type = json.loads((Path(os.environ['MODEL_PATH']) / 'config.json').read_text())['model_type']
        if model_type not in MODEL_TYPE_TO_APPLY_LIGER_FN:
            raise KeyError(f'Liger does not support model type {model_type}')
        import accelerate.accelerator as accelerator_module
        import accelerate.utils.fsdp_utils as fsdp_utils
        for module in (accelerator_module, fsdp_utils):
            if not hasattr(module, 'fsdp2_apply_ac'):
                raise AttributeError(f'{module.__name__} has no fsdp2_apply_ac')
        from torch.distributed.tensor import DTensor  # noqa: F401
        importlib.import_module('torch.distributed.algorithms._checkpoint.checkpoint_wrapper')
        from transformers import AutoModelForImageTextToText, TrainingArguments  # noqa: F401
        import inspect
        if 'fsdp_config' not in inspect.signature(TrainingArguments.__init__).parameters:
            raise AttributeError('TrainingArguments has no fsdp_config')
        return f'all present; model type {model_type}'

    def dropout_parity():
        return run([sys.executable, '-m', 'worker_reviewer_bench.training.checkpoint_rng_policy'], cwd=REPO).strip()

    def server_options():
        text = ''
        for flag in ('--help=all', '--help'):
            result = subprocess.run([sys.executable, '-m', 'vllm.entrypoints.cli.main', 'serve', flag],
                                    capture_output=True, text=True, timeout=600)
            text = result.stdout + result.stderr
            if 'language-model-only' in text:
                break
        missing = [flag for flag in ('language-model-only', 'enforce-eager', 'enable-prefix-caching',
                                     'tool-call-parser', 'enable-auto-tool-choice') if flag not in text]
        if missing:
            raise RuntimeError(f'not in `vllm serve` help: {missing}')
        return 'the server options of serve.py are known to this vLLM' + ('' if 'qwen3_coder' in text else
                                                                           ' (qwen3_coder not listed in the help text)')

    attempt('packages.versions', versions)
    attempt('packages.trainer_symbols', trainer_symbols)
    attempt('packages.dropout_parity', dropout_parity)
    attempt('packages.server_options', server_options, warn=True)


def check_gpu(expected):
    def cuda_driver():
        """Which libcuda the loader picks, against the kernel module, and whether CUDA starts (no torch)."""
        import ctypes
        kernel = os.environ.get('PILOT_CUDA_KERNEL_DRIVER') or 'unknown'
        dropped = os.environ.get('PILOT_CUDA_DROPPED_LIBRARY_DIRS') or 'none'
        library = ctypes.CDLL('libcuda.so.1')
        with open('/proc/self/maps') as handle:
            loaded = sorted({line.split()[-1] for line in handle if 'libcuda.so' in line})
        status = library.cuInit(0)
        version, count = ctypes.c_int(), ctypes.c_int()
        if status == 0:
            status = library.cuDriverGetVersion(ctypes.byref(version)) or library.cuDeviceGetCount(ctypes.byref(count))
        text = (f'kernel module {kernel}; libcuda loaded from {loaded}; dropped from the library path: {dropped}; '
                f'cuInit/cuDeviceGetCount status {status}, CUDA driver API {version.value}, {count.value} devices')
        if status != 0:
            raise RuntimeError(text + ' (803 means libcuda and the kernel module do not match)')
        return text

    attempt('gpu.cuda_driver', cuda_driver)

    def devices():
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is not available to torch')
        count = torch.cuda.device_count()
        names = sorted({torch.cuda.get_device_name(index) for index in range(count)})
        memory = sorted({round(torch.cuda.get_device_properties(index).total_memory / 2 ** 30) for index in range(count)})
        for index in range(count):
            value = torch.randn(2048, 2048, device=f'cuda:{index}', dtype=torch.bfloat16)
            if not torch.isfinite((value @ value).float().sum()):
                raise RuntimeError(f'non-finite result on cuda:{index}')
        if count < expected:
            raise RuntimeError(f'{count} GPUs visible, {expected} expected')
        return f'{count} x {names} with {memory} GiB; torch {torch.__version__}, CUDA {torch.version.cuda}'

    def kernel():
        import torch
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule
        torch.manual_seed(0)
        batch, length, heads, width = 1, 128, 2, 16
        q, k, v = (torch.randn(batch, length, heads, width, device='cuda') for _ in range(3))
        g = -torch.rand(batch, length, heads, device='cuda')
        beta = torch.rand(batch, length, heads, device='cuda')
        output, _ = chunk_gated_delta_rule(q, k, v, g=g, beta=beta, use_qk_l2norm_in_kernel=True)
        if tuple(output.shape) != (batch, length, heads, width) or not torch.isfinite(output.float()).all():
            raise RuntimeError(f'unexpected kernel output {tuple(output.shape)}')
        return 'FLA chunk_gated_delta_rule compiled and ran on GPU (Triton and a C compiler work)'

    def drivers():
        return run(['nvidia-smi', '--query-gpu=index,name,memory.total,memory.used,driver_version',
                    '--format=csv,noheader']).strip().replace('\n', ' | ')

    attempt('gpu.devices', devices)
    attempt('gpu.fla_kernel', kernel)
    attempt('gpu.nvidia_smi', drivers, warn=True)


def check_model():
    base = Path(os.environ['MODEL_PATH'])

    def provenance():
        expected = {json.loads((REPO / 'experiments' / 'bcb_opd' / f'{name}.json').read_text())['base_revision']
                    for name in ('General', 'Specific')}
        record = json.loads((base / 'bcb_verified.json').read_text())
        if {record['revision']} != expected or record['repo_id'] != 'Qwen/Qwen3.5-35B-A3B':
            raise ValueError(f'{base} records {record["repo_id"]}@{record["revision"]}')
        shards = sorted(base.glob('*.safetensors'))
        size = sum(path.stat().st_size for path in shards)
        if size < 60e9 or 'files' not in record:
            raise ValueError(f'{base}: {len(shards)} weight files, {size / 1e9:.1f} GB, '
                             f'{"no " if "files" not in record else ""}file table; run fetch_model.py')
        if 'files' in record:
            wrong = [name for name, entry in record['files'].items()
                     if not (base / name).is_file() or (base / name).stat().st_size != entry['size']]
            if wrong:
                raise ValueError(f'files missing or of the wrong size: {wrong}')
        return f'{record["repo_id"]}@{record["revision"][:12]}, {len(shards)} weight files, {size / 1e9:.1f} GB'

    def alignment():
        # The original tokenizer preflight: the student and teacher views must tokenize the action identically.
        from transformers import AutoTokenizer
        from worker_reviewer_bench.opd.data import encode_pair
        tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
        history = [{'role': 'system', 'content': 'You are a worker'},
                   {'role': 'user', 'content': 'Implement f in solution.py'}]
        tool = {'type': 'function', 'function': {'name': 'bash', 'description': 'Execute a bash command',
                'parameters': {'type': 'object', 'properties': {'command': {'type': 'string'}}, 'required': ['command']}}}
        action = {'role': 'assistant', 'content': None, 'tool_calls': [{'id': 'one', 'type': 'function',
                  'function': {'name': 'bash', 'arguments': '{"command":"printf hello"}'}}]}
        row = {'id': 'mini-swe', 'student_messages': history, 'review': 'Check unverified requirements',
               'action': action, 'tools': [tool]}
        first = len(encode_pair(row, tokenizer, 131072)['target_ids'])
        row['student_messages'] = history + [action, {'role': 'tool', 'tool_call_id': 'one', 'content': 'hello'}]
        second = len(encode_pair(row, tokenizer, 131072)['target_ids'])
        return f'tool-call action aligns in both views ({first} and {second} target tokens)'

    attempt('model.provenance', provenance)
    attempt('model.chat_template_alignment', alignment)


def check_frontend():
    python = os.environ['OPD_FRONTEND_PYTHON']

    def imports():
        code = ('import importlib.metadata as m, minisweagent, litellm\n'
                'from minisweagent.agents import get_agent\nfrom minisweagent.models import get_model\n'
                'from minisweagent.environments.local import LocalEnvironment\n'
                'from minisweagent.models.utils.actions_toolcall import BASH_TOOL\n'
                'print("mini-swe-agent", m.version("mini-swe-agent"), "litellm", m.version("litellm"), "openai", m.version("openai"))')
        output = run([python, '-c', code]).strip().splitlines()[-1]
        if 'mini-swe-agent 2.4.6' not in output:
            raise RuntimeError(output)
        return output

    def unit_tests():
        # Everything in the Mini-SWE environment; what could not import torch there, in this one.
        def outcomes(interpreter, *arguments):
            output = run([interpreter, HERE / 'run_unit_tests.py', *arguments], cwd=REPO,
                         env=dict(os.environ, PYTHONPATH=str(REPO)))
            return json.loads(output.strip().splitlines()[-1])

        first = outcomes(python)
        need_torch = [name for name, value in first['outcomes'].items()
                      if value == 'error' and "No module named 'torch'" in first['details'].get(name, '')]
        second = outcomes(sys.executable, '--only', *need_torch) if need_torch else {'outcomes': {}, 'details': {}}
        merged = dict(first['outcomes'], **second['outcomes'])
        details = dict(first['details'], **second['details'])
        bad = {name: details.get(name, '')[-400:] for name, value in merged.items() if value in ('fail', 'error')}
        if bad or len(merged) < 40:
            raise RuntimeError(f'{len(merged)} tests, not passing: {json.dumps(bad)}')
        counts = {value: sum(1 for other in merged.values() if other == value) for value in sorted(set(merged.values()))}
        return f'{len(merged)} tests {counts}; {len(need_torch)} of them run in the training environment'

    def agent_roundtrip():
        """The agent side end to end, against a stand-in model server: no GPU, about a minute.

        One worker-only episode (`collect`, as the evaluations run it), one training episode with
        reviewer assessments (`opd-collect`, Specific profile, as the training runs it), both in the
        real sandbox, then the benchmark grading of the first one in the agent environment.
        """
        import socket as sockets
        agent_env = os.environ['PILOT_AGENT_ENV']
        with sockets.socket() as probe:
            probe.bind(('127.0.0.1', 0))
            port = probe.getsockname()[1]
        server = subprocess.Popen([sys.executable, HERE / 'tools' / 'stub_server.py', '--port', str(port)],
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            import urllib.request
            deadline = time.time() + 30
            while True:
                try:
                    urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=2)
                    break
                except OSError:
                    if time.time() > deadline:
                        raise RuntimeError('the stand-in server did not start')
                    time.sleep(0.5)
            with tempfile.TemporaryDirectory(dir=os.environ['PILOT_SCRATCH'], prefix='agent-roundtrip-') as directory:
                work = Path(directory)
                task = (REPO / 'experiments' / 'bcb_opd' / 'public_tasks.jsonl').read_text().splitlines()[0]
                task_id = json.loads(task)['task_id']
                (work / 'task.jsonl').write_text(task + '\n')
                grading = next(line for line in (REPO / 'experiments' / 'bcb_opd' / 'grading_tasks.jsonl').read_text().splitlines()
                               if json.loads(line)['task_id'] == task_id)
                (work / 'grading.jsonl').write_text(grading + '\n')
                env = dict(os.environ, PYTHONPATH=str(REPO))
                common = ['--base-url', f'http://127.0.0.1:{port}/v1', '--api-key', 'local', '--workers', '1',
                          '--max-turns', '5', '--max-tokens', '1400', '--temperature', '0.7',
                          '--sandbox', 'bwrap', '--agent-env', agent_env]
                run([python, '-m', 'worker_reviewer_bench', 'collect', '--benchmark', 'bcb', '--backend', 'mini-swe',
                     '--tasks', work / 'task.jsonl', '--output', work / 'collect.jsonl', '--workdir', work / 'collect',
                     '--no-review', '--disable-thinking', '--review-retries', '0', '--worker-model', 'openai/opd-worker',
                     '--prompt-profile', 'General', '--cost-limit', '0', *common], cwd=REPO, env=env, timeout=300)
                episode = json.loads((work / 'collect.jsonl').read_text().splitlines()[0])
                if episode.get('worker_error') or episode['worker_result'].get('exit_status') != 'Submitted' \
                        or 'def task_func' not in episode.get('final_solution', ''):
                    raise RuntimeError(f"collect episode: exit {episode['worker_result'].get('exit_status')}, "
                                       f"error {episode.get('worker_error')!r}")
                run([python, '-m', 'worker_reviewer_bench', 'opd-collect', '--backend', 'mini-swe',
                     '--tasks', work / 'task.jsonl', '--output', work / 'opd.jsonl', '--pairs', work / 'pairs.jsonl',
                     '--workdir', work / 'opd', '--review-root', work / 'reviews', '--round', '0',
                     '--student-revision', 'preflight', '--worker-model', 'opd-worker', '--prompt-profile', 'Specific',
                     '--review-every', '1', '--reviewer-turns', '1', '--reviewer-max-tokens', '512',
                     '--command-timeout', '60', *common], cwd=REPO, env=env, timeout=300)
                pairs = [json.loads(line) for line in (work / 'pairs.jsonl').read_text().splitlines() if line.strip()]
                if not pairs or not all(pair['action'].get('tool_calls') for pair in pairs):
                    raise RuntimeError(f'opd-collect wrote {len(pairs)} teacher/student pairs, or actions without tool calls')
                grading_env = dict(os.environ, PYTHONNOUSERSITE='1', PYTHONPATH=str(REPO))
                run([Path(agent_env) / 'bin' / 'python', '-m', 'worker_reviewer_bench', 'grade', '--benchmark', 'bcb',
                     '--tasks', work / 'grading.jsonl', '--rows', work / 'collect.jsonl', '--out', work / 'grades.json',
                     '--timeout', '120', '--workers', '1'], cwd=REPO, env=grading_env, timeout=300)
                final = next(entry for entry in json.loads((work / 'grades.json').read_text()) if entry['field'] == 'final_solution')
                result = final['results'][task_id]
                if result.get('status') == 'error' or final['n'] != 1:
                    raise RuntimeError(f'grading: {result}')
                return (f"collect: Submitted; opd-collect: {len(pairs)} pairs with reviews; grade: the stand-in solution "
                        f"ran {result.get('tests_run', 0)} benchmark tests ({result.get('status')}, as expected)")
        finally:
            server.terminate()
            server.wait(timeout=30)

    attempt('frontend.imports', imports)
    attempt('frontend.repository_unit_tests', unit_tests)
    attempt('frontend.agent_roundtrip', agent_roundtrip)


def check_sandbox():
    def probe():
        output = run([sys.executable, HERE / 'sandbox_probe.py', '--check'])
        mode = json.loads((Path(os.environ['PILOT_ROOT']) / 'sandbox_mode.json').read_text())
        return f'{output.strip().splitlines()[-1]}; inside: {mode["checks"].get("identity_and_proc", "")!r}'

    attempt('sandbox.probe', probe)


def check_storage(slots):
    out, root = Path(os.environ['PILOT_OUT']), Path(os.environ['PILOT_ROOT'])

    def locking():
        probe = out / 'preflight' / f'.lock-probe-{socket.gethostname()}-{os.getpid()}'
        probe.parent.mkdir(parents=True, exist_ok=True)
        with probe.open('a') as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(handle, fcntl.LOCK_UN)
        probe.unlink()
        directory = probe.with_suffix('.dir')
        directory.mkdir()
        try:
            directory.mkdir()
            raise RuntimeError('creating an existing directory did not fail')
        except FileExistsError:
            pass
        directory.rmdir()
        return f'flock and atomic mkdir work on {out}'

    def disk():
        free = shutil.disk_usage(root).free / 1e9
        model = 0 if (Path(os.environ['MODEL_PATH']) / 'bcb_verified.json').exists() else 72
        # One merged copy of the model per slot at the worst moment, plus workspaces and logs.
        need = model + 72 * slots + 40
        text = f'{free:.0f} GB free on {root}; {need} GB needed for {slots} slot(s)' + (' and the model' if model else '')
        if free < need:
            raise RuntimeError(text)
        return text

    def resources():
        memory = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / 2 ** 30
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        cores = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count()
        text = f'{memory:.0f} GiB memory, {cores} cores, descriptors {soft}/{hard}'
        # The trainer keeps the model in host memory (CPU offload); a merge loads it once more.
        if memory < 100 * slots or cores < 8 * slots or hard < 4096:
            raise RuntimeError(text + f'; about {100 * slots} GiB and {8 * slots} cores are needed for {slots} slot(s)')
        return text

    attempt('storage.locking', locking)
    attempt('storage.disk', disk)
    attempt('storage.resources', resources, warn=True)


def check_distributed(gpus):
    """The original FSDP preflight: one real distributed OPD update on a tiny native Qwen3.5 MoE."""
    def update():
        import torch
        from transformers import AutoConfig, AutoModelForImageTextToText, AutoTokenizer
        base = Path(os.environ['MODEL_PATH'])
        with tempfile.TemporaryDirectory(dir=os.environ['PILOT_SCRATCH'], prefix='fsdp-preflight-') as directory:
            scratch = Path(directory)
            tiny = scratch / 'tiny_native_model'
            tiny.mkdir()
            config = AutoConfig.from_pretrained(base, local_files_only=True)
            text = config.get_text_config()
            for key, value in dict(hidden_size=32, num_hidden_layers=1, num_attention_heads=4,
                                   num_key_value_heads=2, head_dim=8, num_experts=4, num_experts_per_tok=2,
                                   moe_intermediate_size=16, shared_expert_intermediate_size=16,
                                   layer_types=['full_attention'], max_position_embeddings=4096).items():
                setattr(text, key, value)
            for key, value in dict(depth=1, hidden_size=32, intermediate_size=32, num_heads=4,
                                   out_hidden_size=32, num_position_embeddings=16).items():
                setattr(config.vision_config, key, value)
            model = AutoModelForImageTextToText.from_config(config, dtype=torch.bfloat16, attn_implementation='sdpa')
            model.save_pretrained(tiny)
            del model
            AutoTokenizer.from_pretrained(base, local_files_only=True).save_pretrained(tiny)
            history = [{'role': 'system', 'content': 'You are a coding worker. Return one JSON action.'},
                       {'role': 'user', 'content': 'Implement f(x): return x + 1.'}]
            pairs = scratch / 'tiny_pairs.jsonl'
            pairs.write_text(''.join(json.dumps({
                'id': f'preflight:{index}', 'round': 0, 'student_revision': 'tiny-preflight',
                'student_messages': history,
                'review': 'The code has not been written. Write solution.py before submitting.',
                'action': {'role': 'assistant', 'content': '{"action":"submit"}'}}) + '\n' for index in range(gpus)))
            started = time.time()
            run([sys.executable, '-m', 'torch.distributed.run', '--standalone', f'--nproc_per_node={gpus}',
                 '--module', 'worker_reviewer_bench.opd.train', '--pairs', pairs, '--out', scratch / 'tiny_adapter',
                 '--rank', '2', '--max-len', '4096'], cwd=REPO, env=dict(os.environ, MODEL_PATH=str(tiny)), timeout=3600)
            result = json.loads((scratch / 'tiny_adapter' / 'opd_update.json').read_text())
            if result['optimizer_steps'] != 1 or not result['rank0_mean_kl'] > 0 or result['world_size'] != gpus:
                raise ValueError(f'unexpected update record: {result}')
            return (f'one optimizer step on {gpus} GPUs, mean KL {result["mean_kl"]:.3e}, '
                    f'{time.time() - started:.0f} s')

    attempt('distributed.tiny_fsdp_update', update)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--gpus', type=int, required=True, help='GPUs this node gives to the experiment')
    parser.add_argument('--gpus-per-run', type=int, default=2)
    parser.add_argument('--skip', nargs='*', default=[], help='Check groups to leave out')
    args = parser.parse_args()
    for name in ('PILOT_ROOT', 'PILOT_OUT', 'PILOT_SCRATCH', 'MODEL_PATH', 'OPD_FRONTEND_PYTHON', 'PILOT_AGENT_ENV'):
        if not os.environ.get(name):
            parser.error(f'{name} is not set; source experiments/pilot/env.sh first')
    slots = max(1, args.gpus // args.gpus_per_run)
    groups = {'repository': check_repository, 'packages': check_packages, 'gpu': lambda: check_gpu(args.gpus),
              'model': check_model, 'frontend': check_frontend, 'sandbox': check_sandbox,
              'storage': lambda: check_storage(slots), 'distributed': lambda: check_distributed(args.gpus_per_run)}
    unknown = set(args.skip) - set(groups)
    if unknown:
        parser.error(f'Unknown check group: {sorted(unknown)}')
    for name, function in groups.items():
        if name not in args.skip:
            function()
    failed = [entry['check'] for entry in RESULTS if entry['level'] == 'FAIL']
    out = Path(os.environ['PILOT_OUT']) / 'preflight'
    out.mkdir(parents=True, exist_ok=True)
    record = {'host': socket.gethostname(), 'time': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
              'gpus': args.gpus, 'gpus_per_run': args.gpus_per_run, 'skipped': args.skip,
              'passed': not failed and not args.skip, 'failed': failed, 'results': RESULTS}
    (out / f'{socket.gethostname()}.json').write_text(json.dumps(record, indent=2) + '\n')
    print(f"PILOT_PREFLIGHT_{'FAILED ' + ', '.join(failed) if failed else 'OK'} "
          f"({sum(entry['level'] == 'WARN' for entry in RESULTS)} warning(s){', skipped ' + ' '.join(args.skip) if args.skip else ''})")
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
