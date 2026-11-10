"""Restore exact large STL resources stored as gzip blobs in Git."""
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]
LIB = ROOT/'src/assets/registered_library'


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def main():
    rows=json.loads((LIB/'compressed_meshes.json').read_text())
    for row in rows:
        target=(LIB/row['path']).resolve()
        if not target.is_relative_to(LIB.resolve()): raise ValueError('Invalid mesh path')
        if target.exists():
            if digest(target)!=row['sha256']: raise ValueError(f'Refusing to overwrite modified mesh: {target}')
            continue
        temp=target.with_suffix(target.suffix+'.tmp')
        try:
            with gzip.open(str(target)+'.gz','rb') as src,temp.open('wb') as dst:shutil.copyfileobj(src,dst)
            if temp.stat().st_size!=row['bytes'] or digest(temp)!=row['sha256']:raise ValueError('Mesh integrity failure')
            os.replace(temp,target)
        finally:
            if temp.exists():temp.unlink()
    print(f'Mesh resources ready ({len(rows)} compressed originals verified)')


if __name__=='__main__':main()
