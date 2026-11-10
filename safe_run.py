"""Bound one child process tree by RSS, duration and CPU threads (not host-wide)."""
import argparse,os,subprocess,time,json
from pathlib import Path
import psutil
p=argparse.ArgumentParser();p.add_argument('--seconds',type=int,default=240);p.add_argument('--rss-gib',type=float,default=8);p.add_argument('--log',required=True);p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args();cmd=a.command[1:] if a.command[:1]==['--'] else a.command
env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',NUMBA_NUM_THREADS='2',CUDA_VISIBLE_DEVICES='')
path=Path(a.log);path.parent.mkdir(parents=True,exist_ok=True);start=time.monotonic();peak=0;reason=None
with path.open('w') as stream:
 child=subprocess.Popen(cmd,stdout=stream,stderr=subprocess.STDOUT,env=env,start_new_session=True)
 proc=psutil.Process(child.pid)
 while child.poll() is None:
  try:
   group=[proc]+proc.children(recursive=True);rss=sum(x.memory_info().rss for x in group if x.is_running());peak=max(peak,rss)
   if rss>a.rss_gib*1024**3:reason='rss_limit'
   if time.monotonic()-start>a.seconds:reason='time_limit'
   if psutil.virtual_memory().available<16*1024**3:reason='host_available_memory_low'
   if reason:
    for x in reversed(group):
     try:x.kill()
     except psutil.NoSuchProcess:pass
    break
  except psutil.NoSuchProcess:pass
  time.sleep(.5)
 rc=child.wait()
result=dict(returncode=rc,stop_reason=reason,peak_rss_gib=peak/1024**3,elapsed_seconds=time.monotonic()-start);path.with_suffix('.resources.json').write_text(json.dumps(result,indent=2));print(json.dumps(result));raise SystemExit(rc if not reason else 124)
