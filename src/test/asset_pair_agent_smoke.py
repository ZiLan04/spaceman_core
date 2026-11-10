"""Bounded single-pair Genesis inspection; no training or physical capture."""
import argparse,json,sys,os
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from asset_pair_api import make_scene_plan,build_scene_entities,apply_initial_pose

def main():
 p=argparse.ArgumentParser();p.add_argument('--carrier',default='satellite_17');p.add_argument('--arm',default='airbus_vispa');p.add_argument('--target',default='gps');p.add_argument('--seed',type=int,default=0);p.add_argument('--viewer',action='store_true');p.add_argument('--render',action='store_true');p.add_argument('--steps',type=int,default=2);p.add_argument('--output',type=Path,default=Path('artifacts/smoke.json'));a=p.parse_args()
 if not 1<=a.steps<=1000:raise ValueError('steps must be 1..1000')
 plan=make_scene_plan(a.carrier,a.arm,a.target,seed=a.seed)
 if a.render:os.environ.setdefault('PYOPENGL_PLATFORM','egl')
 import genesis as gs
 gs.init(backend=gs.cpu,logging_level='warning')
 scene=gs.Scene(show_viewer=a.viewer,show_FPS=False,sim_options=gs.options.SimOptions(dt=.01,gravity=(0,0,0)),viewer_options=gs.options.ViewerOptions(camera_pos=plan['camera']['position_m'],camera_lookat=(0,0,0),camera_fov=50))
 import numpy as np
 e=build_scene_entities(scene,plan);cameras=[]
 if a.render:
  from asset_pair_api import catalog
  c=catalog();cb=np.array(c['pairs'][plan['pair_id']]['bounds_m'])+plan['carrier_pose']['position_m'];tb=np.array(c['satellites'][a.target]['bounds_m'])+plan['target_pose']['position_m']
  for label,bounds in [('overview',np.array([np.minimum(cb[0],tb[0]),np.maximum(cb[1],tb[1])])),('carrier',cb)]:
   center=bounds.mean(0);span=max(np.linalg.norm(bounds[1]-bounds[0]),1);direction=np.array([.6,-1.,.55]);direction/=np.linalg.norm(direction)
   cameras.append((label,scene.add_camera(res=(960,640),pos=center+direction*span*1.45,lookat=center,fov=40,near=.01,far=500,GUI=False)))
 scene.build();apply_initial_pose(e,plan)
 for _ in range(a.steps):scene.step()
 measured={}
 for n,q in plan['initial_joints'].items():
  ix=e['carrier'].get_joint(n).dof_idx_local;value=float(e['carrier'].get_dofs_position(dofs_idx_local=[ix]).detach().cpu().numpy().reshape(-1)[0]);assert np.isfinite(value) and abs(value-q)<1e-5;measured[n]=value
 result={'status':'passed','plan':plan,'steps':a.steps,'measured_joint_positions':measured,'target_n_links':e['target'].n_links,'carrier_n_links':e['carrier'].n_links,'scope':'single fixed collision-disabled CPU Genesis load, named joint initialization, two steps; not dynamics or capture validation'}
 a.output.parent.mkdir(parents=True,exist_ok=True)
 if cameras:
  from PIL import Image
  result['renders']=[]
  for label,cam in cameras:
   rgb=cam.render(rgb=True)[0];dest=a.output.with_name(a.output.stem+'_'+label+'.png');Image.fromarray(rgb).save(dest);result['renders'].append(str(dest))
 a.output.write_text(json.dumps(result,indent=2),encoding='utf-8');print('ASSET_PAIR_SMOKE_PASSED',a.output,flush=True)
if __name__=='__main__':main()
