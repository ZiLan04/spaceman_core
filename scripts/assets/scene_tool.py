"""Provider-neutral JSON tool for model-selected satellite/arm inspection."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from asset_pair_api import catalog,make_scene_plan


def execute(request):
    if not isinstance(request,dict):raise ValueError('Request must be a JSON object')
    allowed={'action','carrier','arm','target','seed','run_id'}
    if set(request)-allowed:raise ValueError('Unknown request fields')
    action=request.get('action')
    if action=='list_assets':
        c=catalog()
        return {'ok':True,'satellites':[s for s,v in c['satellites'].items() if v['inspection_eligible']],
                'arms':list(c['arms']),'excluded':['satellite_6'],'purpose':'inspection only'}
    if action not in ['plan_scene','render_scene']:raise ValueError('Unknown action')
    seed=request.get('seed',0)
    if type(seed) is not int:raise ValueError('seed must be an integer')
    plan=make_scene_plan(request.get('carrier'),request.get('arm'),request.get('target'),seed=seed)
    if action=='plan_scene':return {'ok':True,'plan':plan}
    run_id=request.get('run_id')
    if not isinstance(run_id,str) or not re.fullmatch('[A-Za-z0-9_-]{1,64}',run_id):
        raise ValueError('render_scene requires a unique 1..64 character run_id: letters/digits/_/-')
    folder=ROOT/'artifacts/agent_demo'/run_id
    # Refuse clobbering a previous agent run.
    folder.mkdir(parents=True,exist_ok=False)
    (folder/'request.json').write_text(json.dumps(request,indent=2),encoding='utf-8')
    cmd=[sys.executable,str(ROOT/'safe_run.py'),'--seconds','240','--rss-gib','8','--log',str(folder/'render.log'),
         '--',sys.executable,str(ROOT/'src/test/asset_pair_agent_smoke.py'),
         '--carrier',plan['carrier_satellite'],'--arm',plan['arm'],'--target',plan['target_satellite'],
         '--seed',str(seed),'--render','--output',str(folder/'result.json')]
    completed=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True)
    resource=folder/'render.resources.json'
    response={'ok':completed.returncode==0,'returncode':completed.returncode,'request':request,
              'resolved_pair':plan['pair_id'],'resolved_target':plan['target_satellite'],
              'result_path':str(folder/'result.json'),
              'resources':json.loads(resource.read_text()) if resource.exists() else None,
              'scope':'fixed collision-disabled CPU loading/initialization/rendering; no capture or training'}
    if completed.returncode==0:
        result=json.loads((folder/'result.json').read_text());response['renders']=result['renders']
    else:
        response['log_path']=str(folder/'render.log')
        response['runner_message']=(completed.stdout+completed.stderr)[-2000:]
    (folder/'tool_response.json').write_text(json.dumps(response,indent=2),encoding='utf-8')
    return response


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request',type=Path,help='Read JSON file; default reads JSON from stdin')
    args=parser.parse_args()
    try:
        request=json.loads(args.request.read_text(encoding='utf-8-sig') if args.request else sys.stdin.read())
        response=execute(request)
    except Exception as error:
        response={'ok':False,'error':type(error).__name__,'message':str(error)}
    print(json.dumps(response,ensure_ascii=False,indent=2))
    return 0 if response['ok'] else 1


if __name__=='__main__':raise SystemExit(main())
