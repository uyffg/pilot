"""Dedicated per-round vLLM process; native Mini-SWE tool calls."""
import argparse
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--port', required=True)
    p.add_argument('--gpus', required=True)
    p.add_argument('--seed', default='42')
    args = p.parse_args()
    sys.argv = ['vllm', 'serve', args.model, '--served-model-name', 'opd-worker',
        '--host', '127.0.0.1', '--port', args.port, '--dtype', 'bfloat16',
        '--tensor-parallel-size', args.gpus, '--distributed-executor-backend', 'mp',
        '--max-model-len', '131072', '--max-num-seqs', '8', '--max-num-batched-tokens', '4096',
        '--gpu-memory-utilization', '0.90', '--language-model-only', '--enable-prefix-caching',
        '--enable-auto-tool-choice', '--tool-call-parser', 'qwen3_coder', '--enforce-eager', '--seed', args.seed]
    from vllm.entrypoints.cli.main import main as serve
    serve()


if __name__ == '__main__':
    main()
