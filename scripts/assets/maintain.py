"""Rebuild a registered satellite's derived files after geometry/mount edits.

Dry-run by default. Use a disposable checkout and inspect the diff before --apply.
No simulator, network, training or collision-validation claim is involved.
"""
import argparse
import datetime
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation
import yaml
from compose_satellite_arm import compose_satellite_arm

ROOT = Path(__file__).resolve().parents[2]


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def matrix(origin):
    t = np.eye(4)
    if origin is not None:
        t[:3, :3] = Rotation.from_euler('xyz', np.fromstring(origin.get('rpy', '0 0 0'), sep=' ')).as_matrix()
        t[:3, 3] = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ')
    return t


def bounds(path):
    """Exact zero-joint visual bounds for this library's binary STL/box URDFs."""
    tree = ET.parse(path).getroot()
    joints = list(tree.findall('joint'))
    roots = {x.get('name') for x in tree.findall('link')} - {x.find('child').get('link') for x in joints}
    if len(roots) != 1:
        raise ValueError('URDF must have exactly one root')
    frames = {roots.pop(): np.eye(4)}
    while joints:
        old = len(joints)
        for j in joints[:]:
            parent = j.find('parent').get('link')
            if parent in frames:
                frames[j.find('child').get('link')] = frames[parent] @ matrix(j.find('origin'))
                joints.remove(j)
        if len(joints) == old:
            raise ValueError('Disconnected or cyclic URDF')
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    for link in tree.findall('link'):
        for visual in link.findall('visual'):
            geom = visual.find('geometry'); mesh = geom.find('mesh'); box = geom.find('box')
            if mesh is not None:
                f = (path.parent / mesh.get('filename')).resolve()
                with f.open('rb') as stream:
                    header = stream.read(84)
                count = int.from_bytes(header[80:84], 'little')
                if f.stat().st_size != 84 + count * 50:
                    raise ValueError(f'Expected binary STL, convert new geometry first: {f}')
                data = np.memmap(f, dtype=np.dtype([('n','<f4',(3,)),('v','<f4',(3,3)),('a','<u2')]), mode='r', offset=84, shape=(count,))
                scale = np.fromstring(mesh.get('scale', '1 1 1'), sep=' ')
                t = frames[link.get('name')] @ matrix(visual.find('origin'))
                for start in range(0, count, 100000):
                    points = np.asarray(data['v'][start:start+100000], dtype=float).reshape(-1,3) * scale
                    points = points @ t[:3,:3].T + t[:3,3]
                    lo = np.minimum(lo, points.min(0)); hi = np.maximum(hi, points.max(0))
                del data
            elif box is not None:
                import itertools
                size = np.fromstring(box.get('size'), sep=' ')
                points = np.array(list(itertools.product([-.5,.5], repeat=3))) * size
                t = frames[link.get('name')] @ matrix(visual.find('origin'))
                points = points @ t[:3,:3].T + t[:3,3]
                lo = np.minimum(lo, points.min(0)); hi = np.maximum(hi, points.max(0))
            else:
                raise ValueError('New visual primitive unsupported; add explicit bounds support')
    if not np.isfinite([lo, hi]).all():
        raise ValueError('No finite visual geometry')
    return [lo.tolist(), hi.tolist()]


def rebuild(root, satellite, xyz=None, rpy=None, apply=False):
    root = Path(root).resolve(); lib = root/'src/assets/registered_library'
    registry = load(lib/'registry.json'); catalog = load(lib/'catalog.json')
    sat = next(s for s in registry['satellites'] if s['satellite_id'] == satellite)
    cfg = load(lib/sat['mount_config'])
    for key, value in [('mount_xyz_m',xyz), ('mount_rpy_rad',rpy)]:
        if value is not None:
            if len(value) != 3 or not np.isfinite(value).all():
                raise ValueError('Expected three finite coordinates')
            cfg[key] = list(map(float, value))
    pairs = [p for p in registry['pairs'] if p['satellite_id'] == satellite]
    if not apply:
        return {'dry_run':True,'satellite':satellite,'mount':cfg,'pairs':[p['assembly'] for p in pairs],
                'effect':'rewrite mount joint, regenerate assemblies/bounds/YAML; invalidate prior checks; keep grasp sites independent'}
    files = [lib/'registry.json',lib/'catalog.json',lib/sat['mount_config'],lib/sat['satellite_urdf']]
    files += [lib/p['assembly'] for p in pairs]
    files += [root/'src/configs/assets'/(name+'.yaml') for name in [satellite]+[satellite+'__'+p['arm_id'] for p in pairs]]
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    backup = root/'.asset_backups'/stamp
    for f in files:
        target = backup/f.relative_to(root); target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(f,target)
    try:
        file = lib/sat['satellite_urdf']; tree = ET.parse(file)
        joint = next(j for j in tree.findall('joint') if j.find('child').get('link') == cfg['mount_frame'])
        origin = joint.find('origin')
        if origin is None: origin = ET.SubElement(joint,'origin')
        origin.set('xyz',' '.join(map(str,cfg['mount_xyz_m']))); origin.set('rpy',' '.join(map(str,cfg['mount_rpy_rad'])))
        ET.indent(tree); tree.write(file,encoding='utf-8',xml_declaration=True)
        cfg['status'] = 'edited_needs_surface_support_and_reach_validation'; cfg['dynamics_ready'] = False
        cfg['surface_fit'] = {'status':'invalidated_by_rebuild','previous':cfg.get('surface_fit')}
        write(lib/sat['mount_config'],cfg)
        meta = catalog['satellites'][satellite]
        meta.update(mount_frame=cfg['mount_frame'],mount_xyz_m=cfg['mount_xyz_m'],mount_rpy_rad=cfg['mount_rpy_rad'],bounds_m=bounds(file),task_ready=False)
        for pair in pairs:
            for key in list(pair):
                if 'check' in key or key == 'orientation_selection': pair.pop(key)
            pair['status'] = 'edited_requires_validation'
        write(lib/'registry.json',registry)
        for pair in pairs:
            dest = lib/pair['assembly']
            compose_satellite_arm(satellite,pair['arm_id'],dest,lib,portable=False)
            entry = catalog['pairs'][satellite+'__'+pair['arm_id']]
            entry.update(bounds_m=bounds(dest),status='edited_requires_validation',
                         mount_to_arm_root_xyz_m=pair['mount_to_arm_root_xyz_m'],mount_to_arm_root_rpy_rad=pair['mount_to_arm_root_rpy_rad'])
        for f in files:
            if f.suffix != '.yaml': continue
            conf = yaml.safe_load(f.read_text(encoding='utf-8'))
            conf['mount_site'] = {'link':cfg['mount_frame'],'position':[0,0,0],'euler':[0,0,0]}
            conf.setdefault('parameters',{}).update(task_ready=False,geometry_revision=stamp,
                calibration={'status':'edited_requires_validation','pending':['surface_support','collision','task_reachability','grasp_site','tcp','dynamics']})
            f.write_text(yaml.safe_dump(conf,sort_keys=False),encoding='utf-8')
        catalog.setdefault('revisions',[]).append({'id':stamp,'satellite':satellite,'validation':'invalidated','historical_reports_do_not_apply':True})
        write(lib/'catalog.json',catalog)
    except Exception:
        for f in files: shutil.copy2(backup/f.relative_to(root),f)
        raise
    return {'satellite':satellite,'backup':str(backup),'status':'rebuilt_requires_validation','pairs':len(pairs)}


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--satellite',required=True)
    p.add_argument('--xyz',nargs=3,type=float);p.add_argument('--rpy',nargs=3,type=float);p.add_argument('--apply',action='store_true')
    a=p.parse_args(); print(json.dumps(rebuild(a.root,a.satellite,a.xyz,a.rpy,a.apply),ensure_ascii=False,indent=2))
