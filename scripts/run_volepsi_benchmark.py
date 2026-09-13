"""Run native VOLE-PSI over TLS and compare every returned receiver index."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import platform
import re
import signal
import subprocess
import time

import numpy as np
from run_benchmark import sha,stats,capture

ROOT=Path(__file__).resolve().parents[1]


def encode(ids):
    # The low limb is a bijection of u64 identifiers: unique by construction.
    blocks=np.empty((len(ids),2),dtype='<u8')
    blocks[:,0]=ids*np.uint64(0x9e3779b97f4a7c15)+np.uint64(0xd1b54a32d192ed03)
    blocks[:,1]=ids*np.uint64(0x94d049bb133111eb)+np.uint64(0x8538ec9dc19b337d)
    return blocks


def make_inputs(case,folder,seed):
    folder.mkdir()
    n,m,k=case['sender'],case['receiver'],case['intersection']
    assert 0<=k<=min(n,m)
    receiver=np.random.default_rng(seed).permutation(m).astype(np.uint64)
    expected=np.flatnonzero(receiver<k).astype('<u8')
    encode(receiver).tofile(folder/'receiver.bin')
    sender=np.concatenate((np.arange(k,dtype=np.uint64),np.arange(m,m+n-k,dtype=np.uint64)))
    np.random.default_rng(seed+1).shuffle(sender)
    encode(sender).tofile(folder/'sender.bin')
    return expected


def execute(binary,folder,trial,timeout):
    trial.mkdir()
    cert=ROOT/'vendor/volepsi-tls'
    # Ask the OS for an available loopback port, then start both workers.
    import socket
    with socket.socket() as probe:
        probe.bind(('127.0.0.1',0));port=probe.getsockname()[1]
    commands=[];processes=[];streams=[]
    start=time.perf_counter()
    try:
        for party,role in [(1,'receiver'),(0,'sender')]:
            command=['/usr/bin/time','-f','%e %U %S %M','-o',str(trial/f'resources-P{party}.txt'),
                str(binary),'-r',str(party),'-in',str(folder/f'{role}.bin'),
                '-out',str(trial/'intersection.bin'),'-indexSet','-v','-ssp','40','-useQC',
                '-ip',f'127.0.0.1:{port}','-tls','-CA',str(cert/'ca.crt'),
                '-pk',str(cert/f'p{party}.crt'),'-sk',str(cert/f'p{party}.key')]
            commands.append(command)
            stream=(trial/f'P{party}.log').open('w');streams.append(stream)
            processes.append(subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True))
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(p.wait) for p in processes]
            try:
                for f in futures:f.result(timeout=max(.01,timeout-(time.perf_counter()-start)))
            except BaseException:
                for p in processes:
                    if p.poll() is None:os.killpg(p.pid,signal.SIGKILL)
                raise
    finally:
        for p in processes:
            if p.poll() is None:os.killpg(p.pid,signal.SIGKILL)
            p.wait()
        for stream in streams:stream.close()
        record={'commands':commands,'exit_codes_receiver_sender':[p.returncode for p in processes],
                'wall_seconds':time.perf_counter()-start}
        (trial/'execution.json').write_text(json.dumps(record,indent=2)+'\n')
    if record['exit_codes_receiver_sender']!=[0,0]:raise RuntimeError('Native PSI process failed')
    parties=[]
    for party in (0,1):
        log=(trial/f'P{party}.log').read_text()
        if 'Exception:' in log:raise RuntimeError('Native frontend caught an exception despite exit zero')
        sent=re.search(r'PROFILE_BYTES_SENT=(\d+)',log)
        stage=re.search(r'running PSI\.\.\. (\d+)ms',log)
        if not sent or not stage:raise RuntimeError('Missing success/counter markers')
        elapsed,user,system,rss=map(float,(trial/f'resources-P{party}.txt').read_text().split())
        parties.append({'party':party,'protocol_ms':int(stage[1]),'application_sent_bytes':int(sent[1]),
                        'gnu_elapsed_seconds':elapsed,'cpu_user_seconds':user,'cpu_system_seconds':system,'max_rss_KiB':rss})
    record['parties']=parties
    if not (trial/'intersection.bin').is_file():raise RuntimeError('Missing new intersection output')
    return record


def main():
    config=json.loads((ROOT/'volepsi-benchmark.json').read_text())
    out=ROOT/'results-volepsi';assert (out/'upstream.json').exists()
    binary=ROOT/'vendor/volepsi/out/build/linux/frontend/frontend'
    (out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    (out/'environment.json').write_text(json.dumps({'utc':datetime.now(timezone.utc).isoformat(),
        'platform':platform.platform(),'python':platform.python_version(),'numpy':np.__version__,
        'cpu_count':os.cpu_count(),'run_id':os.environ.get('GITHUB_RUN_ID'),
        'harness_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'image_version':os.environ.get('ImageVersion'),'network':config['network'],
        'binary_sha256':sha(binary)},indent=2)+'\n')
    capture(['lscpu'],ROOT,out/'lscpu.txt')
    capture(['free','-b'],ROOT,out/'memory.txt')
    capture([str(binary),'-u','-list'],ROOT,out/'available-upstream-tests.log')
    # Upstream file mode catches an invalid input exception and may exit zero.
    bad=ROOT/'vendor/invalid-psi.bin';bad.write_bytes(b'not-16-bytes')
    invalid=subprocess.run([str(binary),'-r','1','-in',str(bad)],capture_output=True,text=True,timeout=20)
    (out/'invalid-input.log').write_text(invalid.stdout+invalid.stderr)
    rejected='Exception:' in invalid.stdout and 'Bad file size' in invalid.stdout
    (out/'negative-check.json').write_text(json.dumps({'exit_code':invalid.returncode,'exception_detected':rejected,
        'pass':rejected,'purpose':'harness must reject a native caught error, regardless of exit code'},indent=2)+'\n')
    assert rejected
    all_rows=[];failures=[];case_stats=[]
    try:
        for number,case in enumerate(config['cases']):
            case_out=out/case['name'];case_out.mkdir()
            inputs=ROOT/'vendor'/('psi-inputs-'+case['name'])
            start=time.perf_counter();seed=config['seed']+100*number
            expected=make_inputs(case,inputs,seed)
            expected.tofile(case_out/'expected-indices.bin')
            hashes={p.name:sha(p) for p in inputs.iterdir()}
            (case_out/'dataset.json').write_text(json.dumps({**case,'seed':seed,'generation_seconds':time.perf_counter()-start,
                'file_sha256':hashes,'encoding':'two uint64 little-endian affine limbs; unique low limb',
                'reference':'receiver IDs below intersection size; shuffled row indices',
                'raw_inputs':'regenerate using pinned source/config; original outputs retained'},indent=2)+'\n')
            if max(case['sender'],case['receiver'])<=4096:
                for p in inputs.iterdir():(case_out/p.name).write_bytes(p.read_bytes())
            rows=[]
            for repeat in range(config['repetitions']):
                trial=case_out/f'repeat-{repeat}'
                try:
                    row=execute(binary,inputs,trial,config['timeout_seconds'])
                    raw=(trial/'intersection.bin').read_bytes()
                    assert len(raw)%8==0
                    actual=np.frombuffer(raw,dtype='<u8')
                    exact=np.array_equal(actual,expected)
                    row.update({'case':case['name'],'repetition':repeat,'expected_count':len(expected),
                        'actual_count':len(actual),'exact_indices_match':exact,'pass':exact})
                    (trial/'result.json').write_text(json.dumps(row,indent=2)+'\n')
                    rows.append(row);all_rows.append(row)
                    if not exact:raise RuntimeError('Incorrect intersection indices')
                    print(json.dumps({k:row[k] for k in ['case','repetition','actual_count','pass','wall_seconds']}),flush=True)
                except Exception as error:
                    failure={'case':case['name'],'repetition':repeat,'error':repr(error)}
                    failures.append(failure);(case_out/'failure.json').write_text(json.dumps(failure,indent=2)+'\n')
                    print(json.dumps(failure),flush=True);break
            case_stats.append({**case,'completed_repetitions':len(rows),
                'pass':len(rows)==config['repetitions'] and all(r['pass'] for r in rows),
                'wall_seconds':stats([r['wall_seconds'] for r in rows]) if rows else None,
                'sum_application_sent_bytes':stats([sum(p['application_sent_bytes'] for p in r['parties']) for r in rows]) if rows else None,
                'max_single_party_rss_KiB':max((p['max_rss_KiB'] for r in rows for p in r['parties']),default=None)})
            # These are generated scratch inputs, not raw outputs.
            for p in inputs.iterdir():p.unlink()
            inputs.rmdir()
    finally:
        summary={'status':'PASS' if len(case_stats)==len(config['cases']) and all(c['pass'] for c in case_stats) and not failures else 'INCOMPLETE_OR_FAILED',
            'cases':case_stats,'failures':failures,'formal_acceptance':False,
            'timing':'whole native invocation: file read, TLS, size exchange, real protocol and index output; excludes input generation and build',
            'scope':'native RsPsi using QC, semi-honest ssp40, loopback TLS; receiver-only ordinary PSI output; no production audit'}
        (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        (out/'SHA256SUMS.json').write_text(json.dumps({str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='SHA256SUMS.json'},indent=2)+'\n')
        print(json.dumps(summary,indent=2),flush=True)
    if summary['status']!='PASS':raise SystemExit(1)


if __name__=='__main__':main()
