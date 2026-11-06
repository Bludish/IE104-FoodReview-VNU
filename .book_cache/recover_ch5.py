import json
import os

transcript_path = r'C:\Users\Admin\.gemini\antigravity\brain\3f71b6de-2c0c-456c-9d79-6224f51e8428\.system_generated\logs\transcript_full.jsonl'
if not os.path.exists(transcript_path):
    transcript_path = r'C:\Users\Admin\.gemini\antigravity\brain\3f71b6de-2c0c-456c-9d79-6224f51e8428\.system_generated\logs\transcript.jsonl'

out_path = r'd:\Documents\IE104.R12\distilled_ml_systems\Chapter_05_Model_Development.md'

with open(transcript_path, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        if data.get('step_index') == 13:
            for call in data.get('tool_calls', []):
                if call.get('name') == 'write_file':
                    args = call.get('args', {})
                    content = args.get('Content', '')
                    if isinstance(content, str) and content.startswith('"') and content.endswith('"'):
                        try:
                            content = json.loads(content)
                        except Exception:
                            content = content[1:-1]
                    with open(out_path, 'w', encoding='utf-8') as out_f:
                        out_f.write(content)
                    print(f'Successfully wrote Chapter 5! Size: {len(content)} chars')
                    break

