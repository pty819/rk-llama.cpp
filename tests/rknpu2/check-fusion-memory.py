#!/usr/bin/env python3
"""Check static allocations for the 28-layer jina-v5-small Q8_0 fixture."""
import json
import re
import sys
from pathlib import Path
root=Path(sys.argv[1])
values={}
for tag in ('base','qkv','gate','both','nommap','lora-base','lora'):
    p=root/f'{tag}.log'
    if not p.exists(): raise RuntimeError(f"missing configuration: {tag}")
    matches=re.findall(r'RKNPU model buffer size =\s+([0-9.]+) MiB',p.read_text())
    if not matches: raise RuntimeError(f'missing weight allocation: {tag}')
    size=max(map(float,matches))
    if size != 448.0: raise RuntimeError(f'duplicate/static allocation regression: {tag}: {size}')
    memory=json.loads((root/f'{tag}-memory.json').read_text())
    buffers={}
    for text in memory['fdinfo'].values():
        ino=re.search(r'^ino:\s*(\d+)',text,re.M)
        nbytes=re.search(r'^size:\s*(\d+)',text,re.M)
        if ino and nbytes: buffers[ino[1]]=int(nbytes[1])
    values[tag]={'weight_mib':size,'dmabuf_mib':sum(buffers.values())/2**20,'dmabufs':len(buffers),'rss_kib':int(re.search(r'^VmRSS:\s*(\d+)',memory['status'],re.M)[1])}
print(json.dumps(values,indent=2))
(root/'memory-summary.json').write_text(json.dumps(values,indent=2))
