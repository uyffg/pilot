class CharacterTokenizer:
    def apply_chat_template(self, messages, add_generation_prompt=False, **kwargs):
        return ''.join(f'<{m["role"]}>{m["content"]}</{m["role"]}>\n' for m in messages) + ('<assistant>' if add_generation_prompt else '')

    def encode(self, text, **kwargs):
        return [ord(c) for c in text]

    def __call__(self, text, **kwargs):
        return {'input_ids': self.encode(text), 'offset_mapping':[(i,i+1) for i in range(len(text))]}
