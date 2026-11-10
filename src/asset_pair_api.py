"""Agent-neutral asset selection. No Genesis import or simulation at import time.

python src/asset_pair_api.py list
python src/asset_pair_api.py plan --carrier satellite_17 --arm airbus_vispa --target gps
"""
from __future__ import annotations
import argparse,json,math,random
from pathlib import Path
from dataclasses import replace
ROOT=Path(__file__).resolve().parent
CATALOG=ROOT/'assets/registered_library/catalog.json'

def catalog():
    return json.loads(CATALOG.read_text(encoding='utf-8'))

def _vec(value,n):
    if len(value)!=n or not all(math.isfinite(float(x)) for x in value):
        raise ValueError(f'Expected {n} finite values')
    return [float(x) for x in value]

def list_assets():
    c=catalog()
    return {'satellites':c['satellites'],'arms':c['arms'],'pairs':sorted(c['pairs'])}

def make_scene_plan(carrier=None,arm=None,target=None,*,seed=0,allow_partial=False,
                    carrier_position=None,target_position=None,
                    carrier_euler_deg=(0,0,0),target_euler_deg=(0,0,0),
                    initial_joints=None,purpose='inspection'):
    """Resolve an explicit or seeded-random pair, with collision-free overview spacing.

    Rebuild a scene to change topology; this is not live topology hot swapping.
    Calibration readiness is enforced before any task/training use.
    """
    c=catalog();rng=random.Random(seed)
    valid=sorted(k for k,v in c['satellites'].items() if allow_partial or v['inspection_eligible'])
    carrier=carrier or rng.choice(valid);target=target or rng.choice(valid);arm=arm or rng.choice(sorted(c['arms']))
    if carrier not in c['satellites'] or target not in c['satellites'] or arm not in c['arms']:
        raise ValueError('Unknown asset ID; call list_assets()')
    if not allow_partial and any(not c['satellites'][s]['inspection_eligible'] for s in [carrier,target]):
        raise ValueError('Partial source assembly excluded; set allow_partial=True for inspection only')
    if purpose!='inspection':
        raise ValueError('Task/training blocked: grasp sites, TCP, dynamics and task initial poses are not calibrated')
    pair=carrier+'__'+arm;cb=c['pairs'][pair]['bounds_m'];tb=c['satellites'][target]['bounds_m']
    # Bounding spheres remain separated for any orientation, unlike x-only extents.
    radius=lambda b:max(math.sqrt(sum(v*v for v in p)) for p in __import__('itertools').product(*zip(*b)))
    cr,tr=radius(cb),radius(tb);gap=2.;separation=cr+tr+gap
    cp=_vec(carrier_position or [separation/2,0,0],3);tp=_vec(target_position or [-separation/2,0,0],3)
    if math.dist(cp,tp)<cr+tr+.1:raise ValueError('Inspection bounding spheres overlap; increase spawn separation')
    joints=dict(c['arms'][arm]['initial_joint_positions'])
    if initial_joints is not None:
        unknown=set(initial_joints)-set(joints)
        if unknown:raise ValueError(f'Unknown joint names: {sorted(unknown)}')
        if any(float(x)!=0 for x in initial_joints.values()):raise ValueError('Nonzero presets need collision validation; current validated inspection preset is zero')
        joints.update(initial_joints)
    ce=_vec(carrier_euler_deg,3);te=_vec(target_euler_deg,3)
    from scipy.spatial.transform import Rotation
    R=Rotation.from_euler('xyz',te,degrees=True)
    sites=[]
    for site in c['satellites'][target]['grasp_sites']:
        p=R.apply(site['position'])+tp
        w,x,y,z=site['quaternion'];qr=(R*Rotation.from_quat([x,y,z,w])).as_quat()
        sites.append(dict(site,world_position_m=p.tolist(),world_normal=R.apply(site['normal']).tolist(),world_quaternion_wxyz=[float(qr[3]),*map(float,qr[:3])]))
    return {'schema_version':1,'seed':seed,'purpose':purpose,'carrier_satellite':carrier,'arm':arm,'target_satellite':target,'pair_id':pair,
            'carrier_pose':{'position_m':cp,'euler_deg':ce},'target_pose':{'position_m':tp,'euler_deg':te},
            'initial_joints':joints,'gripper_semantics':'1=open, 0=curl; proximal=(1-command), distal=.65*(1-command) radians',
            'target_grasp_sites':sites,'task_ready':False,
            'camera':{'position_m':[0,-separation*1.2,separation*.65],'lookat_m':[0,0,0],'fov_deg':50},
            'inspection':{'fixed':True,'collision':False,'bounding_sphere_separation_m':separation},
            'warnings':['Grasp coordinates are body mount surface candidates, not validated grasp handles.','Initial poses are for separated viewing, not a reachable capture scene.','satellite_6 partial; catch01 has 12 unmeshed faces.']}

def resolve_specs(plan):
    """Return upstream typed specs without starting Genesis."""
    from configs.registry import registry
    if plan.get('purpose')!='inspection' or plan.get('task_ready') is not False:
        raise ValueError('Only inspection plans are supported')
    canonical=make_scene_plan(plan['carrier_satellite'],plan['arm'],plan['target_satellite'],
        allow_partial=True,carrier_position=plan['carrier_pose']['position_m'],target_position=plan['target_pose']['position_m'],
        carrier_euler_deg=plan['carrier_pose']['euler_deg'],target_euler_deg=plan['target_pose']['euler_deg'],initial_joints=plan['initial_joints'])
    if canonical['pair_id']!=plan['pair_id']:raise ValueError('Pair ID mismatch')
    target=registry.get(plan['target_satellite']);carrier=registry.get(plan['pair_id'])
    return target,carrier

def build_scene_entities(scene,plan):
    """Call after gs.init(); returns fixed inspection entities plus apply_initial_pose()."""
    from configs.registry import build_genesis_asset
    t,c=resolve_specs(plan)
    entities={}
    for key,spec,p in [('target',t,plan['target_pose']),('carrier',c,plan['carrier_pose'])]:
        entities[key]=scene.add_entity(**build_genesis_asset(spec,pos=p['position_m'],euler=p['euler_deg'],fixed=True,collision=False))
    return entities

def apply_initial_pose(entities,plan):
    """After scene.build(): resolve joint indices by name, never URDF list order."""
    entity=entities['carrier'];names=list(plan['initial_joints']);indices=[entity.get_joint(n).dof_idx_local for n in names]
    entity.set_dofs_position([plan['initial_joints'][n] for n in names],dofs_idx_local=indices,zero_velocity=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='cmd',required=True);sub.add_parser('list')
    q=sub.add_parser('plan');q.add_argument('--carrier');q.add_argument('--arm');q.add_argument('--target');q.add_argument('--seed',type=int,default=0);q.add_argument('--allow-partial',action='store_true');q.add_argument('--output',type=Path)
    a=p.parse_args()
    result=list_assets() if a.cmd=='list' else make_scene_plan(a.carrier,a.arm,a.target,seed=a.seed,allow_partial=a.allow_partial)
    text=json.dumps(result,ensure_ascii=False,indent=2)
    if getattr(a,'output',None):a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(text,encoding='utf-8')
    print(text)
if __name__=='__main__':main()
