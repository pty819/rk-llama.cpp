#!/usr/bin/env python3
"""Serialize NPU comparisons and restore the existing production units."""
from pathlib import Path
import json
import math
import os
import resource
import signal
import statistics
import subprocess
import time
import urllib.error
import urllib.request

root = Path(__file__).resolve().parents[2]
soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (min(65536, hard), hard))
out = root / os.getenv('VALIDATION_DIR', 'validation')
out.mkdir(exist_ok=True)
release = Path('/home/liyifan/services/jina-embed/current/bin').resolve()
server = None

def unit(action, name):
    subprocess.run(['systemctl', '--user', action, name], check=True)

def post(port, path, body):
    request = urllib.request.Request(f'http://127.0.0.1:{port}{path}', data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)

def healthy(port, proc=None):
    until = time.monotonic() + 120
    while time.monotonic() < until:
        if proc and proc.poll() is not None:
            raise RuntimeError(f'test server exited: {proc.returncode}')
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=2) as response:
                if response.status == 200:
                    return
        except (OSError, urllib.error.HTTPError):
            pass
        time.sleep(.25)
    raise RuntimeError(f'health timeout on {port}')

def stop_server():
    global server
    if server:
        server.terminate()
        try:
            server.wait(timeout=15)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()
        server = None

def interrupted(signum, frame):
    raise SystemExit(128 + signum)

for sig in (signal.SIGTERM, signal.SIGINT):
    signal.signal(sig, interrupted)

env = os.environ.copy()
env.update(LD_PRELOAD=f'{release}/rknn_poll_deadline.so:{release}/memsync_retry.so', RKNPU_POLL_MAX_MS='30000', RKNPU_SYNC_RETRY_MAX='50', RKNPU_SYNC_RETRY_DELAY_US='2000', OMP_NUM_THREADS='4', RKNPU_HYBRID='W8A8_HADAMARD', RKNPU_FA='1', RKNPU_FA_MAX_KV='2048', RKNPU_FA_CPU_FALLBACK='0')
running = {name: subprocess.run(['systemctl', '--user', 'is-active', '--quiet', name]).returncode == 0 for name in ('jina-embed.service', 'jina-gateway.service')}
results = []
try:
    unit('stop', 'jina-gateway.service')
    unit('stop', 'jina-embed.service')
    for index, adaptive in enumerate(() if os.getenv('SKIP_MICROBENCH') == '1' else ('0', '1', '1', '0')):
        with (out / f'fa-{index}-{adaptive}.jsonl').open('w') as stdout, (out / f'fa-{index}-{adaptive}.stderr').open('w') as stderr:
            subprocess.run(['taskset', '-c', '4-7', str(root / 'fa-hardware-bench'), adaptive], env=env, stdout=stdout, stderr=stderr, timeout=120, check=True)
        print(f'FA microbench pass adaptive={adaptive} run={index}', flush=True)

    texts = (Path('/home/liyifan/src/rk-llama-jina/bench/texts/long_3000.txt').read_text() + '\n') * 3
    for index, adaptive in enumerate(os.getenv('VALIDATION_MODES', '0,1,0,1').split(',')):
        runenv = env.copy()
        runenv['RKNPU_FA_ADAPTIVE'] = '0' if adaptive == 'cpu' else adaptive
        if adaptive == 'cpu':
            runenv['RKNPU_FA'] = '0'
            runenv['GGML_FA_OPT1'] = '0'
        runenv['LD_LIBRARY_PATH'] = f'{root}/build/bin:{release}'
        with (out / f'server-{index}-{adaptive}.log').open('w') as log:
            server = subprocess.Popen(['taskset', '-c', '4-7', str(root / 'build/bin/llama-server'), '-m', '/home/liyifan/models/jina-v5/v5-small-retrieval-Q8_0.gguf', '--embedding', '--pooling', 'last', '-c', '8192', '-b', '4096', '-ub', '4096', '-np', '1', '--kv-unified', '-t', '4', '--host', '127.0.0.1', '--port', '18310'], env=runenv, stdout=log, stderr=log)
            healthy(18310, server)
            tokens = post(18310, '/tokenize', {'content': texts, 'add_special': True})['tokens']
            if os.getenv('PREWARM_MAX') == '1' and adaptive != 'cpu':
                for repeat in range(2):
                    post(18310, '/v1/embeddings', {'input': tokens[repeat:repeat+2048]})
                print(f'Maximum-shape prewarm pass run={index} adaptive={adaptive}', flush=True)
            seen = {}
            for n in map(int, os.getenv('VALIDATION_LENGTHS', '53,65,127,129,257,547,1023,2048,2049').split(',')):
                if len(tokens) < n + 8:
                    raise RuntimeError('insufficient source tokens')
                timings = []
                vectors = []
                for repeat in range(4):
                    sample = tokens[repeat:repeat+n]
                    started = time.monotonic()
                    response = post(18310, '/v1/embeddings', {'input': sample})
                    elapsed = (time.monotonic() - started) * 1000
                    vector = response['data'][0]['embedding']
                    if not all(math.isfinite(x) for x in vector) or len(vector) != 1024:
                        raise RuntimeError('invalid embedding')
                    if response['usage']['prompt_tokens'] != n:
                        raise RuntimeError(f'unexpected token count: {response["usage"]}')
                    timings.append(elapsed)
                    vectors.append(vector)
                result = {'run': index, 'adaptive': -1 if adaptive == 'cpu' else int(adaptive), 'n': n, 'cold_ms': timings[0], 'warm_ms': timings[1:], 'median_ms': statistics.median(timings[1:]), 'vectors': vectors}
                if n in seen:
                    similarities = [sum(a*b for a,b in zip(u,v)) / math.sqrt(sum(a*a for a in u)*sum(b*b for b in v)) for u,v in zip(seen[n], vectors)]
                    result['shape_switch_min_cos'] = min(similarities)
                    if min(similarities) < .9999:
                        raise RuntimeError(f'shape switching changed previous embedding: n={n}, cos={min(similarities)}')
                seen[n] = vectors
                results.append(result)
                (out / 'embedding-results.json').write_text(json.dumps(results))
                print(json.dumps({k: v for k, v in result.items() if k != 'vectors'}), flush=True)
            stop_server()
finally:
    stop_server()
    if running['jina-embed.service']:
        unit('start', 'jina-embed.service')
        healthy(8310)
    if running['jina-gateway.service']:
        unit('start', 'jina-gateway.service')
        healthy(8311)
    print('Original production services restored.', flush=True)
