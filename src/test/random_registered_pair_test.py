"""Standalone reproducible random registered-pair test; no agent needed."""
import argparse,json,random,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from asset_pair_api import catalog

def main():
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=20261105);p.add_argument('--count',type=int,default=4);p.add_argument('--render',action='store_true');p.add_argument('--output',type=Path,default=Path('artifacts/random_20261105'));a=p.parse_args()
 if not 1<=a.count<=8:raise ValueError('count must be 1..8, serial execution only')
 c=catalog();rng=random.Random(a.seed);valid=sorted(k for k,v in c['satellites'].items() if v['inspection_eligible']);pairs=sorted(k for k,v in c['pairs'].items() if v['satellite'] in valid);selected=rng.sample(pairs,a.count);a.output.mkdir(parents=True,exist_ok=True);results=[]
 for i,pair in enumerate(selected):
  row=c['pairs'][pair];target=rng.choice(valid);dest=a.output/f'{i+1:02d}_{pair}.json';log=dest.with_suffix('.log')
  cmd=[sys.executable,str(ROOT/'safe_run.py'),'--seconds','240','--rss-gib','8','--log',str(log),'--',sys.executable,str(ROOT/'src/test/asset_pair_agent_smoke.py'),'--carrier',row['satellite'],'--arm',row['arm'],'--target',target,'--steps','2','--output',str(dest)]
  if a.render:cmd.append('--render')
  print('RUN',i+1,pair,'target='+target,flush=True);proc=subprocess.run(cmd,cwd=ROOT)
  item={'pair':pair,'target':target,'returncode':proc.returncode,'result':str(dest),'log':str(log)};resource=log.with_suffix('.resources.json')
  if resource.exists():item['resources']=json.loads(resource.read_text())
  results.append(item);(a.output/'summary.json').write_text(json.dumps({'seed':a.seed,'count':a.count,'render_requested':a.render,'results':results},indent=2))
  if proc.returncode:print('FAILED; stopped batch, inspect log',log,flush=True);raise SystemExit(1)
 print('RANDOM_TEST_PASSED',a.output/'summary.json',flush=True)
if __name__=='__main__':main()
