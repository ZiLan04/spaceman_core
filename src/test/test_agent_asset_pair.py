import sys,math,xml.etree.ElementTree as ET
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from asset_pair_api import catalog,make_scene_plan,resolve_specs

def test_all_54_combinations():
 c=catalog();assert len(c['satellites'])==27 and len(c['pairs'])==54
 for pair,row in c['pairs'].items():
  p=make_scene_plan(row['satellite'],row['arm'],'gps',allow_partial=True);t,s=resolve_specs(p);tree=ET.parse(s.urdf)
  joints={e.get('name'):e for e in tree.findall('joint')};links={e.get('name') for e in tree.findall('link')}
  assert len(s.control.initial_dofs)==10 and len(s.arm_joint_names)==6 and len(s.gripper.joint_names)==4
  assert s.tcp.link in links and all(g.link in links for g in s.grasp_sites)
  for name,q in p['initial_joints'].items():
   limit=joints[name].find('limit');assert float(limit.get('lower'))<=q<=float(limit.get('upper'))
  for mesh in tree.findall('.//mesh'):assert (s.urdf.parent/mesh.get('filename')).is_file()

def test_seed_and_coordinates():
 assert make_scene_plan(seed=42)==make_scene_plan(seed=42)
 p=make_scene_plan('satellite_17','airbus_vispa','gps',target_position=[-50,1,2],target_euler_deg=[0,0,90])
 site=p['target_grasp_sites'][0];x,y,z=site['position'];expected=[-50-y,1+x,2+z]
 assert all(abs(a-b)<1e-8 for a,b in zip(site['world_position_m'],expected))

def test_guards():
 for kw in [{'carrier':'bad'},{'carrier':'satellite_6'},{'purpose':'training'},{'carrier_position':[0,0,0],'target_position':[0,0,0]},{'initial_joints':{'bad':0}}]:
  try:make_scene_plan('gps' if 'carrier' not in kw else kw.pop('carrier'),'airbus_vispa','gps',**kw)
  except ValueError:pass
  else:raise AssertionError('Invalid request was accepted')

if __name__=='__main__':
 test_all_54_combinations();test_seed_and_coordinates();test_guards();print('STATIC_CHECKS_PASSED: 54 pairs, seeded selection, transformed coordinates, readiness guards')
