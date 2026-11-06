"""Run genuine upstream train/test entry points with this plugin registered."""
import argparse
import os
from pathlib import Path
import runpy
import sys
p = argparse.ArgumentParser()
p.add_argument('mode', choices=['train','test'])
p.add_argument('--openpcdet', type=Path, required=True)
a,rest = p.parse_known_args()
root = a.openpcdet.resolve()
if not (root/'tools'/f'{a.mode}.py').is_file():
    p.error('Not an OpenPCDet checkout')
sys.path.insert(0,str(root))
sys.path.insert(0,str(root/'tools'))
from spconvtransformer import register
register()
os.chdir(root/'tools')
sys.argv = [a.mode+'.py']+rest
runpy.run_path(str(root/'tools'/f'{a.mode}.py'),run_name='__main__')
