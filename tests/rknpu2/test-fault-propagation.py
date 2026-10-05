#!/usr/bin/env python3
"""Compile the backend with CPU RKNN mocks and inject Release-mode failures."""
from pathlib import Path
import resource
import shlex
import subprocess
import tempfile
import sys

repo=Path(__file__).resolve().parents[2]
build=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else repo/'build'
flags_file=build/'ggml/src/ggml-rknpu2/CMakeFiles/ggml-rknpu2.dir/flags.make'
flags=[]
for prefix in ['CXX_DEFINES = ', 'CXX_INCLUDES = ', 'CXX_FLAGS = ']:
    line=next(x for x in flags_file.read_text().splitlines() if x.startswith(prefix))
    flags+=shlex.split(line[len(prefix):])
cache=(build/'CMakeCache.txt').read_text().splitlines()
compiler=next(x.split('=',1)[1] for x in cache if x.startswith('CMAKE_CXX_COMPILER:'))
with tempfile.TemporaryDirectory(prefix='rknpu2-faults-') as tmp:
    exe=Path(tmp)/'test'
    subprocess.run([compiler,*flags,str(Path(__file__).with_suffix('.cpp')),
                    '-L'+str(build/'bin'),'-Wl,-rpath,'+str(build/'bin'),
                    '-lggml-rknpu2','-lggml-base','-o',str(exe)],check=True)
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    for overlap in [0,1]:
        for fault in range(9):
            subprocess.run([str(exe),str(fault),str(overlap)],check=True,timeout=20)
    fatal=subprocess.run([str(exe),'9','0'],capture_output=True,text=True,timeout=10)
    assert fatal.returncode==-6, fatal.stdout+fatal.stderr
    print('18 graph scenarios passed; void failure terminates with SIGABRT')
