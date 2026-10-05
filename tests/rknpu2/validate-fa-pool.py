#!/usr/bin/env python3
"""Counterbalanced real-model FA pool benchmark on the OPi5 test host."""
from pathlib import Path
import json
import math
import os
import resource
import signal
import statistics
import subprocess
import time
import urllib.request

root = Path(__file__).resolve().parents[2]
out = Path(os.getenv('FA_POOL_DIR', str(root.parent/'fa-pool-evidence-20261005')))
out.mkdir(exist_ok=True)
release = Path('/home/liyifan/services/jina-embed/current/bin').resolve()
resource.setrlimit(resource.RLIMIT_NOFILE, (65536, resource.getrlimit(resource.RLIMIT_NOFILE)[1]))
env = os.environ.copy()
env.update(LD_LIBRARY_PATH=f'{root}/build/bin:{release}', LD_PRELOAD=f'{release}/rknn_poll_deadline.so:{release}/memsync_retry.so', RKNPU_POLL_MAX_MS='30000', RKNPU_SYNC_RETRY_MAX='50', RKNPU_SYNC_RETRY_DELAY_US='2000', OMP_NUM_THREADS='4', RKNPU_HYBRID='W8A8_HADAMARD', RKNPU_FA='1', RKNPU_FA_CPU_FALLBACK='0', RKNPU_FA_MAX_KV='2048', RKNPU_FA_THREADS=os.getenv('FA_POOL_THREADS', '6'), RKNPU_FUSE_QKV='1', RKNPU_FUSE_GATE_UP='1', RKNPU_PROFILE=os.getenv('FA_POOL_PROFILE', '0'))
env.pop('RKNPU_FA_ADAPTIVE', None)
server = None
rows = []

def unit(action, name):
    subprocess.run(['systemctl', '--user', action, name], check=True)

def health(port, proc=None):
    for _ in range(480):
        if proc and proc.poll() is not None: raise RuntimeError(f'server exited {proc.returncode}')
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=2) as response:
                if response.status == 200: return
        except OSError: pass
        time.sleep(.25)
    raise RuntimeError('health timeout')

def post(path, body):
    request = urllib.request.Request(f'http://127.0.0.1:18310{path}', data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=120) as response: return json.load(response)

def workers():
    tasks = Path(f'/proc/{server.pid}/task')
    return sorted(int(p.name) for p in tasks.iterdir() if (p/'comm').read_text().strip() == 'rknpu-fa')

def ticks(ids):
    return sum(sum(map(int, Path(f'/proc/{server.pid}/task/{tid}/stat').read_text().split(') ', 1)[1].split()[11:13])) for tid in ids)

def stop():
    global server
    if server:
        server.terminate()
        try: server.wait(timeout=20)
        except subprocess.TimeoutExpired:
            server.kill(); server.wait(); raise RuntimeError('server shutdown timed out')
        if server.returncode != 0: raise RuntimeError(f'server shutdown status {server.returncode}')
        server = None

def interrupted(signum, frame): raise SystemExit(128 + signum)
for sig in (signal.SIGINT, signal.SIGTERM): signal.signal(sig, interrupted)
active = {name: subprocess.run(['systemctl', '--user', 'is-active', '--quiet', name]).returncode == 0 for name in ('jina-embed', 'jina-gateway')}
try:
    unit('stop', 'jina-gateway'); unit('stop', 'jina-embed')
    modes = os.getenv('FA_POOL_MODES', 'off,on,on,off').split(',')
    lengths = tuple(map(int, os.getenv('FA_POOL_LENGTHS', '53,129,257,547,1023,2048,53').split(',')))
    for iteration, mode in enumerate(modes):
        pooled = mode == 'on'
        runenv = env.copy()
        if mode == 'default': runenv.pop('RKNPU_FA_POOL', None)
        else: runenv['RKNPU_FA_POOL'] = '0' if mode == 'off' else '1'
        logpath = out/f'{iteration}-{mode}.log'
        args = ['taskset', '-c', '4-7', str(root/'build/bin/llama-server'), '-m', '/home/liyifan/models/jina-v5/v5-small-retrieval-Q8_0.gguf', '--embedding', '--pooling', 'last', '-c', '8192', '-b', '4096', '-ub', '4096', '-np', '1', '--kv-unified', '-t', '4', '--host', '127.0.0.1', '--port', '18310']
        with logpath.open('w') as log:
            server = subprocess.Popen(args, env=runenv, stdout=log, stderr=log)
            health(18310, server)
            text = (Path('/home/liyifan/src/rk-llama-jina/bench/texts/long_3000.txt').read_text()+'\n')*3
            tokens = post('/tokenize', {'content': text, 'add_special': True})['tokens']
            expected_ids = None
            for index, n in enumerate(lengths):
                timings, vectors = [], []
                for repeat in range(4):
                    started = time.monotonic()
                    response = post('/v1/embeddings', {'input': tokens[repeat:repeat+n]})
                    timings.append((time.monotonic()-started)*1000)
                    vector = response['data'][0]['embedding']
                    if len(vector)!=1024 or not all(math.isfinite(x) for x in vector): raise RuntimeError('bad embedding')
                    if response['usage']['prompt_tokens']!=n: raise RuntimeError('token count mismatch')
                    vectors.append(vector)
                    ids = workers()
                    if not pooled:
                        if ids: raise RuntimeError('disabled pool has workers')
                    else:
                        if len(ids) != int(runenv['RKNPU_FA_THREADS']) - 1: raise RuntimeError(f'pool size mismatch: {ids}')
                        if expected_ids is None: expected_ids = ids
                        if ids != expected_ids: raise RuntimeError('FA worker threads replaced')
                affinity = {tid: next(line for line in Path(f'/proc/{server.pid}/task/{tid}/status').read_text().splitlines() if line.startswith('Cpus_allowed_list')) for tid in (expected_ids or [])}
                row = {'pooled': pooled, 'worker_affinity': affinity, 'iteration': iteration, 'mode': mode, 'index': index, 'n': n, 'cold_ms': timings[0], 'warm_ms': timings[1:], 'median_ms': statistics.median(timings[1:]), 'worker_ids': expected_ids, 'vectors': vectors}
                rows.append(row)
                (out/'results.json').write_text(json.dumps(rows))
                print(json.dumps({k:v for k,v in row.items() if k!='vectors'}), flush=True)
            if expected_ids:
                before = ticks(expected_ids); time.sleep(.2); delta = ticks(expected_ids) - before
                if delta > 1: raise RuntimeError(f'idle worker CPU use: {delta} ticks')
                print('idle FA worker CPU ticks', delta, flush=True)
            stop()
    minimum = 1.0
    for row in rows:
        reference = next(r for r in rows if r['iteration']==0 and r['index']==row['index'])
        for a,b in zip(row['vectors'],reference['vectors']):
            cosine = sum(x*y for x,y in zip(a,b)) / math.sqrt(sum(x*x for x in a)*sum(y*y for y in b))
            minimum = min(minimum,cosine)
            if cosine < .9999: raise RuntimeError(f"cosine mismatch {row['mode']} {row['n']}: {cosine}")
    print('minimum reference cosine', minimum, flush=True)
    summary = {'embedding_requests': len(rows)*4, 'min_cosine': minimum, 'timings': []}
    for index,n in enumerate(lengths):
        medians = {mode:statistics.median(t for r in rows if r['index']==index and r['pooled']==(mode=='on') for t in r['warm_ms']) for mode in ('off','on')}
        summary['timings'].append({'index':index,'n':n,**medians,'change_pct':100*(medians['on']/medians['off']-1)})
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary), flush=True)
finally:
    try: stop()
    finally:
        if active['jina-embed']: unit('start', 'jina-embed'); health(8310)
        if active['jina-gateway']: unit('start', 'jina-gateway'); health(8311)
        print('Production restored.', flush=True)
