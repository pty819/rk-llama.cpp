#!/usr/bin/env python3
from pathlib import Path
import json
import math
import os
import re
import resource
import signal
import statistics
import subprocess
import time
import urllib.request
import urllib.error

root = Path(__file__).resolve().parents[2]
out = root.parent / 'projection-fusion-evidence-20261005'
out.mkdir(exist_ok=True)
if os.getenv('FUSION_DIR'):
    out = Path(os.environ['FUSION_DIR'])
    out.mkdir(exist_ok=True)
release = Path('/home/liyifan/services/jina-embed/current/bin').resolve()
_, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (65536, hard))
env = os.environ.copy()
env.update(LD_LIBRARY_PATH=f'{root}/build/bin:{release}', LD_PRELOAD=f'{release}/rknn_poll_deadline.so:{release}/memsync_retry.so', RKNPU_POLL_MAX_MS='30000', RKNPU_SYNC_RETRY_MAX='50', RKNPU_SYNC_RETRY_DELAY_US='2000', OMP_NUM_THREADS='4', RKNPU_HYBRID='W8A8_HADAMARD', RKNPU_FA='1', RKNPU_FA_CPU_FALLBACK=os.getenv('FUSION_FA_CPU_FALLBACK', '0'), RKNPU_FA_MAX_KV='2048', RKNPU_PROFILE='1')
env.pop('RKNPU_FA_ADAPTIVE', None)
env['RKNPU_PROFILE'] = os.getenv('FUSION_PROFILE', '1')
server = None
rows = []

def unit(action, name):
    subprocess.run(['systemctl', '--user', action, name], check=True)

def health(port, proc=None):
    for _ in range(480):
        if proc and proc.poll() is not None:
            raise RuntimeError(f'server exited {proc.returncode}')
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=2) as response:
                if response.status == 200:
                    return
        except OSError:
            pass
        time.sleep(.25)
    raise RuntimeError('health timeout')

def post(path, body):
    request = urllib.request.Request(f'http://127.0.0.1:18310{path}', data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        print(error.read().decode(), flush=True)
        raise

def stop():
    global server
    if server:
        server.terminate()
        try:
            server.wait(timeout=20)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()
        server = None

def interrupted(signum, frame):
    raise SystemExit(128 + signum)

for sig in (signal.SIGINT, signal.SIGTERM):
    signal.signal(sig, interrupted)
active = {name: subprocess.run(['systemctl', '--user', 'is-active', '--quiet', name]).returncode == 0 for name in ('jina-embed', 'jina-gateway')}
try:
    unit('stop', 'jina-gateway')
    unit('stop', 'jina-embed')
    for tag in os.getenv('FUSION_MODES', 'base,qkv,gate,both').split(','):
        runenv = env.copy()
        runenv.update(RKNPU_FUSE_QKV='1' if tag in ('qkv', 'both', 'lora', 'nommap') else '0', RKNPU_FUSE_GATE_UP='1' if tag in ('gate', 'both', 'lora', 'nommap') else '0')
        args = ['taskset', '-c', '4-7', str(root/'build/bin/llama-server'), '-m', '/home/liyifan/models/jina-v5/v5-small-retrieval-Q8_0.gguf', '--embedding', '--pooling', 'last', '-c', '8192', '-b', '4096', '-ub', '4096', '-np', '1', '--kv-unified', '-t', '4', '--host', '127.0.0.1', '--port', '18310', '--slots', '-lv', '5']
        if os.getenv('FUSION_TOGGLE', '0') == '1': args += ['--slot-save-path', str(out)]
        if tag == 'nommap': args.append('--no-mmap')
        if tag in ('lora', 'lora-base'): args += ['--lora', os.getenv('FUSION_LORA', str(root.parent/'projection-fusion-evidence-20261005/zero-lora.gguf'))]
        logpath = out/f'{tag}.log'
        with logpath.open('w') as log:
            server = subprocess.Popen(args, env=runenv, stdout=log, stderr=log)
            health(18310, server)
            memory = {'status': Path(f'/proc/{server.pid}/status').read_text(), 'smaps': Path(f'/proc/{server.pid}/smaps_rollup').read_text()}
            memory['fdinfo'] = {p.name: p.read_text() for p in Path(f'/proc/{server.pid}/fdinfo').iterdir() if 'drm-' in p.read_text() or 'exp_name:' in p.read_text()}
            (out/f'{tag}-memory.json').write_text(json.dumps(memory))
            text = (Path('/home/liyifan/src/rk-llama-jina/bench/texts/long_3000.txt').read_text()+'\n')*3
            tokens = post('/tokenize', {'content': text, 'add_special': True})['tokens']
            for n in map(int, os.getenv('FUSION_LENGTHS', '53,129,257,547,1023').split(',')):
                timings, vectors = [], []
                for repeat in range(4):
                    started = time.monotonic()
                    response = post('/v1/embeddings', {'input': tokens[repeat:repeat+n]})
                    timings.append((time.monotonic()-started)*1000)
                    vector = response['data'][0]['embedding']
                    if len(vector)!=1024 or not all(math.isfinite(x) for x in vector): raise RuntimeError('bad embedding')
                    if response['usage']['prompt_tokens']!=n: raise RuntimeError('token count mismatch')
                    vectors.append(vector)
                row = {'tag': tag, 'n': n, 'cold_ms': timings[0], 'warm_ms': timings[1:], 'median_ms': statistics.median(timings[1:]), 'vectors': vectors}
                rows.append(row)
                (out/'results.json').write_text(json.dumps(rows))
                print(json.dumps({k:v for k,v in row.items() if k!='vectors'}), flush=True)
            if tag in ('lora', 'lora-base') and os.getenv('FUSION_LORA') and os.getenv('FUSION_TOGGLE', '0') == '1':
                before = next(r for r in rows if r['tag'] == tag)['vectors'][0]
                n = next(r for r in rows if r['tag'] == tag)['n']
                post('/lora-adapters', [{'id': 0, 'scale': 0.0}])
                post('/slots/0?action=erase', {})
                off = post('/v1/embeddings', {'input': tokens[:n]})['data'][0]['embedding']
                post('/lora-adapters', [{'id': 0, 'scale': 1.0}])
                post('/slots/0?action=erase', {})
                after = post('/v1/embeddings', {'input': tokens[:n]})['data'][0]['embedding']
                error = max(abs(a-b) for a,b in zip(before,after))
                effect = max(abs(a-b) for a,b in zip(before,off))
                reapplied_effect = max(abs(a-b) for a,b in zip(after,off))
                (out/f'{tag}-toggle-vectors.json').write_text(json.dumps({'before':before, 'off':off, 'after':after, 'error':error, 'effect':effect}))
                similarity = sum(a*b for a,b in zip(before,after)) / math.sqrt(sum(a*a for a in before)*sum(b*b for b in after))
                if similarity < .9999 or effect < 1e-5 or reapplied_effect < 1e-5: raise RuntimeError(f'LoRA toggle mismatch: cosine={similarity}, effect={effect}')
                (out/f'{tag}-toggle.json').write_text(json.dumps({'roundtrip_cosine':similarity, 'roundtrip_max_error':error, 'adapter_effect':effect}))
                print('LoRA toggle PASS', error, effect, flush=True)
            stop()
        counts = [int(x) for x in re.findall(r'graph_compute calls=\d+ matmuls=(\d+)', logpath.read_text())]
        print('projection counts', tag, sorted(set(counts)), flush=True)
        if runenv['RKNPU_PROFILE'] == '1':
            expected = {'base': 196, 'qkv': 140, 'gate': 168, 'both': 112, 'lora': 112, 'lora-base': 196, 'nommap': 112}[tag]
            if set(counts) != {expected}: raise RuntimeError(f'unexpected projection counts for {tag}: {counts}')
    def cosine(a, b):
        return sum(x*y for x,y in zip(a,b)) / math.sqrt(sum(x*x for x in a)*sum(x*x for x in b))
    comparison_rows = rows
    if os.getenv('FUSION_REFERENCE_RESULTS'):
        comparison_rows = rows + json.loads(Path(os.environ['FUSION_REFERENCE_RESULTS']).read_text())
    for row in rows:
        if row['tag'] == 'lora-base': continue
        reference_tag = 'lora-base' if row['tag'] == 'lora' else 'base'
        references = [r for r in comparison_rows if r['tag'] == reference_tag and r['n'] == row['n']]
        if references:
            value = min(cosine(a,b) for a,b in zip(row['vectors'], references[0]['vectors']))
            if value < .9999: raise RuntimeError(f"cosine mismatch {row['tag']} {row['n']}: {value}")
            print('cosine', row['tag'], row['n'], value, flush=True)
    adapted = [r for r in comparison_rows if r['tag'] == 'lora-base']
    if adapted and os.getenv('FUSION_LORA'):
        diffs = [max(abs(a-b) for a,b in zip(r['vectors'][0], next(x for x in comparison_rows if x['tag']=='base' and x['n']==r['n'])['vectors'][0])) for r in adapted]
        if max(diffs) < 1e-5: raise RuntimeError('nonzero LoRA did not affect output')
        print('LoRA max differences', diffs, flush=True)
finally:
    stop()
    if active['jina-embed']:
        unit('start', 'jina-embed')
        health(8310)
    if active['jina-gateway']:
        unit('start', 'jina-gateway')
        health(8311)
    print('Production restored.', flush=True)
