"""A stand-in for the model server, for checking the agent side without a GPU.

It answers the OpenAI chat-completions requests that the Mini-SWE worker and the reviewer send:

  * a worker request (it offers tools): first a `bash` tool call that writes `solution.py`, then
    one that submits with `COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`;
  * a reviewer request (it asks for a JSON schema): a valid state assessment.

Standard library only, so it runs with any Python. Used by preflight.py (`frontend.agent_roundtrip`).

    python experiments/pilot/tools/stub_server.py --port 29999
"""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import time

SOLUTION = 'def task_func(*args, **kwargs):\n    return None\n'
WRITE = "cat > solution.py <<'PILOT_EOF'\n" + SOLUTION + 'PILOT_EOF'
SUBMIT = 'echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'
REVIEW = {'action': 'review', 'decision': 'continue', 'reason': 'stub assessment', 'check': '', 'hint': ''}


def reply(request):
    model = request.get('model', 'opd-worker')
    usage = {'prompt_tokens': 10, 'completion_tokens': 10, 'total_tokens': 20}
    if request.get('response_format'):
        message = {'role': 'assistant', 'content': json.dumps(REVIEW)}
        finish = 'stop'
    else:
        earlier = sum(1 for item in request.get('messages', []) if item.get('role') == 'assistant')
        command = WRITE if earlier == 0 else SUBMIT
        message = {'role': 'assistant', 'content': None,
                   'tool_calls': [{'id': f'call_{earlier}', 'type': 'function',
                                   'function': {'name': 'bash', 'arguments': json.dumps({'command': command})}}]}
        finish = 'tool_calls'
    return {'id': f'stub-{time.time_ns()}', 'object': 'chat.completion', 'created': int(time.time()), 'model': model,
            'choices': [{'index': 0, 'message': message, 'finish_reason': finish}], 'usage': usage}


class Handler(BaseHTTPRequestHandler):
    def send(self, value, status=200):
        body = json.dumps(value).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.endswith('/health'):
            self.send({'status': 'ok'})
        elif self.path.endswith('/models'):
            self.send({'object': 'list', 'data': [{'id': 'opd-worker', 'object': 'model'}]})
        else:
            self.send({'error': 'not found'}, 404)

    def do_POST(self):
        length = int(self.headers.get('Content-Length', '0'))
        request = json.loads(self.rfile.read(length) or b'{}')
        if self.path.endswith('/chat/completions'):
            self.send(reply(request))
        else:
            self.send({'error': 'not found'}, 404)

    def log_message(self, *args):
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--port', type=int, required=True)
    args = parser.parse_args()
    ThreadingHTTPServer(('127.0.0.1', args.port), Handler).serve_forever()


if __name__ == '__main__':
    main()
