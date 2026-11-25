"""OpenAI-compatible chat client and JSON response parsing."""
import json
import time
from typing import Any
import requests

class Chat:
    def __init__(self, model, base_url, api_key, temperature, max_tokens=1400):
        self.model, self.base_url, self.api_key = model, base_url.rstrip('/'), api_key
        self.temperature, self.max_tokens = temperature, max_tokens

    def chat(self, messages, max_tokens=None):
        for attempt in range(4):
            try:
                response = requests.post(self.base_url + '/chat/completions',
                    headers={'Authorization': 'Bearer ' + self.api_key},
                    json={'model': self.model, 'messages': messages,
                          'temperature': self.temperature,
                          'max_tokens': max_tokens or self.max_tokens,
                          **getattr(self, 'extra_body', {})}, timeout=180)
                response.raise_for_status()
                return response.json()['choices'][0]['message']['content']
            except requests.RequestException:
                if attempt == 3:
                    raise RuntimeError('Model request failed after 4 attempts') from None
                time.sleep(2 ** attempt)


def parse_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0]
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("model did not return a JSON object")
    value = json.loads(text[start:end + 1])
    if not isinstance(value, dict):
        raise ValueError("model JSON must be an object")
    return value
