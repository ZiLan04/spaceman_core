"""Optional OpenAI-compatible tool-call loop; no provider SDK dependency.

Run inside the Genesis Python environment. Keys stay in API_KEY/DEEPSEEK_API_KEY.
This is a minimal adapter, not a deployment of a particular DeepSeek Harness.
"""
import argparse
import json
import os
from pathlib import Path
import urllib.request
import urllib.error
from scene_tool import execute,ROOT


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prompt',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--max-turns',type=int,default=12)
    parser.add_argument('--max-renders',type=int,default=4)
    args=parser.parse_args()
    if not 1<=args.max_turns<=20 or not 1<=args.max_renders<=4:raise ValueError('Limit: 1..20 turns, 1..4 renders')
    key=os.environ.get('API_KEY') or os.environ.get('DEEPSEEK_API_KEY')
    if not key:raise ValueError('Set API_KEY or DEEPSEEK_API_KEY in the execution environment; do not embed it in prompts/files')
    base=os.environ.get('API_BASE_URL','https://api.deepseek.com').rstrip('/')
    model=os.environ.get('API_MODEL','deepseek-chat')
    schema=json.loads(Path(__file__).with_name('scene_tool.schema.json').read_text())
    system=(ROOT/'.agents/skills/spaceman-asset-pair/SKILL.md').read_text(encoding='utf-8')
    system+='\nUse only spaceman_scene for execution. First list assets. Calls are serial. Maximum four renders. After a tool failure stop rendering and report the failure. Unique run IDs required. Explain scope: static inspection only, no grasping/training. Tool output paths are local server files. Do not claim a video was produced unless an artifact exists.'
    messages=[{'role':'system','content':system},{'role':'user','content':args.prompt.read_text(encoding='utf-8-sig')}]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    # Do not overwrite another agent run's transcript.
    with args.output.open('x',encoding='utf-8') as f:f.write('[]')
    trace=[];renders=0;render_failed=False
    for turn in range(args.max_turns):
        payload={'model':model,'messages':messages,'tools':[schema],'tool_choice':'auto','max_tokens':1600}
        req=urllib.request.Request(base+'/chat/completions',data=json.dumps(payload).encode(),
             headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'},method='POST')
        try:
            with urllib.request.urlopen(req,timeout=120) as response:reply=json.load(response)
        except urllib.error.HTTPError as error:raise RuntimeError(f'API HTTP {error.code}; check endpoint/model/credentials') from None
        message=reply['choices'][0]['message'];messages.append(message)
        trace.append({'turn':turn,'assistant':message,'usage':reply.get('usage')})
        calls=message.get('tool_calls') or []
        if not calls:
            args.output.write_text(json.dumps(trace,ensure_ascii=False,indent=2),encoding='utf-8')
            print(message.get('content') or 'Agent finished without text')
            return 0
        for call in calls:
            attempted_render=False
            try:
                if call['function']['name']!='spaceman_scene':raise ValueError('Tool is not allowed')
                request=json.loads(call['function']['arguments'])
                if request.get('action')=='render_scene':
                    attempted_render=True
                    if render_failed or renders>=args.max_renders:raise ValueError('Rendering stopped: failure or run budget exhausted')
                    renders+=1
                result=execute(request)
            except Exception as error:result={'ok':False,'error':type(error).__name__,'message':str(error)}
            if attempted_render and not result.get('ok'):render_failed=True
            messages.append({'role':'tool','tool_call_id':call['id'],'content':json.dumps(result,ensure_ascii=False)})
            trace.append({'tool_call':call,'result':result})
            args.output.write_text(json.dumps(trace,ensure_ascii=False,indent=2),encoding='utf-8')
    raise RuntimeError('Agent turn limit reached; transcript preserved')


if __name__=='__main__':raise SystemExit(main())
