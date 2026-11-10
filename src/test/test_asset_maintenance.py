"""Maintenance must synchronize transforms and invalidate stale evidence."""
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import pytest
import yaml

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/assets'))
from maintain import rebuild


def fixture(root):
    lib=root/'src/assets/registered_library';lib.mkdir(parents=True)
    cfgdir=root/'src/configs/assets';cfgdir.mkdir(parents=True)
    (lib/'sat.urdf').write_text('''<robot name="sat"><link name="body"><visual><geometry><box size="2 2 2"/></geometry></visual></link><link name="mount"/><joint name="fixed" type="fixed"><parent link="body"/><child link="mount"/><origin xyz="0 0 1" rpy="0 0 0"/></joint></robot>''')
    (lib/'arm.urdf').write_text('''<robot name="arm"><link name="arm_root"><visual><geometry><box size="1 1 1"/></geometry></visual></link></robot>''')
    mount={'mount_frame':'mount','mount_xyz_m':[0,0,1],'mount_rpy_rad':[0,0,0],'surface_fit':{'status':'old'}}
    reg={'satellites':[{'satellite_id':'s','satellite_urdf':'sat.urdf','mount_config':'mount.json'}],
         'arms':[{'arm_id':'a','urdf':'arm.urdf','root_link':'arm_root'}],
         'pairs':[{'satellite_id':'s','arm_id':'a','mount_frame':'mount','assembly':'pair.urdf','mount_to_arm_root_xyz_m':[0,0,.5],'mount_to_arm_root_rpy_rad':[0,0,0],'status':'passed','angular_surface_checks':[{'passed':True}]}]}
    cat={'satellites':{'s':{'grasp_sites':[{'position':[9,8,7]}]}},'pairs':{'s__a':{}}}
    for name,data in [('registry.json',reg),('catalog.json',cat),('mount.json',mount)]: (lib/name).write_text(json.dumps(data))
    (lib/'pair.urdf').write_text('<robot name="old"/>')
    for name in ['s','s__a']:(cfgdir/(name+'.yaml')).write_text(yaml.safe_dump({'parameters':{'task_ready':True}}))
    return lib


def test_dry_run_and_apply_sync_and_preserve_independent_grasp(tmp_path):
    lib=fixture(tmp_path);original=(lib/'registry.json').read_bytes()
    result=rebuild(tmp_path,'s',[0,0,3],[0,0,0])
    assert result['dry_run'] and not (tmp_path/'.asset_backups').exists()
    assert (lib/'registry.json').read_bytes()==original
    result=rebuild(tmp_path,'s',[0,0,3],[0,0,0],True)
    assert Path(result['backup']).exists()
    tree=ET.parse(lib/'pair.urdf')
    assert tree.find("joint[@name='fixed']/origin").get('xyz')=='0.0 0.0 3.0'
    cat=json.loads((lib/'catalog.json').read_text())
    assert cat['pairs']['s__a']['bounds_m'][1][2]==4
    assert cat['satellites']['s']['grasp_sites'][0]['position']==[9,8,7]
    pair=json.loads((lib/'registry.json').read_text())['pairs'][0]
    assert pair['status']=='edited_requires_validation' and 'angular_surface_checks' not in pair
    assert yaml.safe_load((tmp_path/'src/configs/assets/s__a.yaml').read_text())['parameters']['task_ready'] is False


def test_failed_rebuild_restores_owned_files(tmp_path):
    lib=fixture(tmp_path)
    (lib/'arm.urdf').write_text('<robot name="broken"><link name="arm_root"><visual><geometry><sphere radius="1"/></geometry></visual></link></robot>')
    owned=['sat.urdf','pair.urdf','mount.json','catalog.json','registry.json']
    before={n:(lib/n).read_bytes() for n in owned}
    with pytest.raises(ValueError,match='primitive unsupported'):
        rebuild(tmp_path,'s',[0,0,6],[0,0,0],True)
    assert before=={n:(lib/n).read_bytes() for n in owned}
