"""Compose a static satellite-arm candidate from registry.json; no simulator started."""
from pathlib import Path
import argparse,copy,json,os,shutil
import xml.etree.ElementTree as ET

def compose_satellite_arm(satellite_id,arm_id,output_path,library_root,portable=True):
    library_root=Path(library_root).resolve();output_path=Path(output_path).resolve();output_path.parent.mkdir(parents=True,exist_ok=True)
    registry=json.loads((library_root/'registry.json').read_text(encoding='utf-8'))
    sat=next(s for s in registry['satellites'] if s['satellite_id']==satellite_id)
    arm=next(a for a in registry['arms'] if a['arm_id']==arm_id)
    pair=next(p for p in registry['pairs'] if p['satellite_id']==satellite_id and p['arm_id']==arm_id)
    satfile=library_root/sat['satellite_urdf'];armfile=library_root/arm['urdf']
    robot=ET.parse(satfile).getroot();robot.set('name',satellite_id+'__'+arm_id)
    parts=[(robot,satfile.parent,'sat')]
    armtree=ET.parse(armfile).getroot();parts.append((armtree,armfile.parent,'arm'))
    copied={}
    for tree,directory,prefix in parts:
        for mesh in tree.findall('.//mesh'):
            source=(directory/mesh.get('filename')).resolve()
            if portable:
                key=(prefix,str(source));target=output_path.parent/'meshes'/(prefix+'_'+source.name)
                if key not in copied:
                    target.parent.mkdir(exist_ok=True);shutil.copyfile(source,target);copied[key]=target
            else:target=source
            mesh.set('filename',Path(os.path.relpath(target,output_path.parent)).as_posix())
    names={e.get('name') for e in robot.findall('link')};assert not names.intersection(e.get('name') for e in armtree.findall('link'))
    joint=ET.SubElement(robot,'joint',name='arm_adapter_fixed',type='fixed');ET.SubElement(joint,'parent',link=pair['mount_frame']);ET.SubElement(joint,'child',link=arm['root_link'])
    ET.SubElement(joint,'origin',xyz=' '.join(map(str,pair['mount_to_arm_root_xyz_m'])),rpy=' '.join(map(str,pair['mount_to_arm_root_rpy_rad'])))
    for child in armtree:robot.append(copy.deepcopy(child))
    ET.indent(robot);ET.ElementTree(robot).write(output_path,encoding='utf-8',xml_declaration=True)
    return output_path

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--satellite',required=True);p.add_argument('--arm',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--library',type=Path,default=Path(__file__).resolve().parent);p.add_argument('--shared-assets',action='store_true');a=p.parse_args()
    print(compose_satellite_arm(a.satellite,a.arm,a.output,a.library,not a.shared_assets))
