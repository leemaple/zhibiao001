"""Tie native MPC aggregation to actual gradients of a specified neural network."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import struct
import subprocess
import time

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from run_benchmark import sha,stats,capture,metric

ROOT=Path(__file__).resolve().parents[1]


def execute(dist,program,trial,timeout):
    trial.mkdir();streams=[];processes=[];commands=[]
    import socket
    with socket.socket() as probe:
        probe.bind(('127.0.0.1',0));port=probe.getsockname()[1]
    env=dict(os.environ);env['LD_LIBRARY_PATH']=str(dist)+os.pathsep+env.get('LD_LIBRARY_PATH','')
    start=time.perf_counter()
    try:
        for party in (0,1):
            cmd=['/usr/bin/time','-f','%e %U %S %M','-o',str(trial/f'resources-P{party}.txt'),
                 str(dist/'semi2k-party.x'),str(party),program,'-N','2','-h','localhost','-pn',str(port),'-e']
            commands.append(cmd);stream=(trial/f'P{party}.log').open('w');streams.append(stream)
            processes.append(subprocess.Popen(cmd,cwd=dist,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True))
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(p.wait) for p in processes]
            try:
                for future in futures:future.result(timeout=max(.01,timeout-(time.perf_counter()-start)))
            except BaseException:
                for p in processes:
                    if p.poll() is None:os.killpg(p.pid,signal.SIGKILL)
                raise
    finally:
        for p in processes:
            if p.poll() is None:os.killpg(p.pid,signal.SIGKILL)
            p.wait()
        for stream in streams:stream.close()
        record={'commands':commands,'exit_codes':[p.returncode for p in processes],
                'wall_seconds':time.perf_counter()-start}
        (trial/'execution.json').write_text(json.dumps(record,indent=2)+'\n')
    assert record['exit_codes']==[0,0]
    record['parties']=[]
    for party in (0,1):
        log=(trial/f'P{party}.log').read_text()
        elapsed,user,system,rss=map(float,(trial/f'resources-P{party}.txt').read_text().split())
        record['parties'].append({'party':party,'runtime_seconds':metric(log,r'Time = ([\d.eE+-]+) seconds'),
            'application_sent_decimal_MB':metric(log,r'Data sent = ([\d.eE+-]+) MB'),
            'max_rss_KiB':rss,'user_cpu_seconds':user,'system_cpu_seconds':system})
    return record


def main():
    config=json.loads((ROOT/'gradient-benchmark.json').read_text())
    assert config['affine_layers']==50 and config['parameters']==50914 and config['fraction_bits']==20
    assert config['samples_per_party']==32
    out=ROOT/'results-gradient';out.mkdir(exist_ok=False)
    (out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    install=json.loads((ROOT/'vendor/install.json').read_text());dist=ROOT/'vendor'/install['distribution']
    shutil.copy2(ROOT/'vendor/install.json',out/'upstream-install.json')
    (out/'environment.json').write_text(json.dumps({'utc':datetime.now(timezone.utc).isoformat(),
        'platform':platform.platform(),'python':platform.python_version(),'cpu_count':os.cpu_count(),
        'harness_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'run_id':os.environ.get('GITHUB_RUN_ID'),'image_version':os.environ.get('ImageVersion'),
        'mp_spdz_binary_sha256':sha(dist/'semi2k-party.x'),'network':'same-host two native MPC processes, TLS loopback'},indent=2)+'\n')
    capture(['lscpu'],ROOT,out/'lscpu.txt');capture(['python3','-m','pip','freeze'],ROOT,out/'pip-freeze.txt')
    iris=load_iris();x=iris.data[:100];y=iris.target[:100]
    train,test=train_test_split(np.arange(100),train_size=64,random_state=config['seed'],stratify=y)
    center=np.median(x[train],axis=0);iqr=np.percentile(x[train],75,axis=0)-np.percentile(x[train],25,axis=0)
    scaled=np.clip((x-center)/iqr,-3,3)/3
    np.savez(out/'dataset.npz',x_raw=x,y=y,train_index=train,test_index=test,center=center,iqr=iqr,x_train=scaled[train],y_train=y[train])
    generation_start=time.perf_counter()
    for party in (0,1):
        index=train[party*32:(party+1)*32]
        np.savez(out/f'data-P{party}.npz',x=scaled[index],y=y[index])
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(capture,['python3',str(ROOT/'scripts/model_gradient_worker.py'),str(out/f'data-P{p}.npz'),
            str(out/f'model-P{p}'),'--seed',str(config['seed'])],ROOT,out/f'gradient-worker-P{p}.log') for p in (0,1)]
        worker_seconds=[f.result() for f in futures]
    generation_seconds=time.perf_counter()-generation_start
    gradients=[np.load(out/f'model-P{p}/gradients.npy') for p in (0,1)]
    weights=[np.load(out/f'model-P{p}/weights.npy') for p in (0,1)]
    assert np.array_equal(*weights)
    scale=2**config['fraction_bits'];quantized=[np.rint(g*scale).astype(np.int64) for g in gradients]
    assert all(np.max(np.abs(g*scale))<2**40 for g in gradients)
    expected=quantized[0]+quantized[1];expected.astype('<i8').tofile(out/'expected-sum.bin')
    np.save(out/'expected-float-mean.npy',(gradients[0]+gradients[1])/2)
    for party in (0,1):
        p=out/f'Input-P{party}-0';p.write_text(' '.join(map(str,quantized[party]))+'\n')
        shutil.copy2(p,dist/'Player-Data'/p.name)
    shutil.copy2(ROOT/'programs/gradient_share_sum.mpc',dist/'Programs/Source/gradient_share_sum.mpc')
    n=len(expected);compile_seconds=capture(['python3','compile.py','-R','64','-M','gradient_share_sum',str(n)],dist,out/'compile.log')
    rows=[]
    for repeat in range(config['repetitions']):
        persistence=dist/'Persistence';persistence.mkdir(exist_ok=True)
        for party in (0,1):
            (persistence/f'Transactions-P{party}.data').unlink(missing_ok=True)
            (dist/f'Player-Data/Binary-Output-P{party}-0').unlink(missing_ok=True)
        trial=out/f'repeat-{repeat}'
        row=execute(dist,f'gradient_share_sum-{n}',trial,config['timeout_seconds'])
        shares=[]
        for party in (0,1):
            raw=(persistence/f'Transactions-P{party}.data').read_bytes()
            assert len(raw)==17+8*n
            (trial/f'sum-shares-P{party}.bin').write_bytes(raw)
            shares.append(np.frombuffer(raw[-8*n:],dtype='<u8'))
            diagnostic=dist/f'Player-Data/Binary-Output-P{party}-0'
            assert not diagnostic.exists() or diagnostic.stat().st_size==0
        actual=(shares[0]+shares[1]).view('<i8')
        actual.tofile(trial/'post-run-diagnostic-sum.bin')
        assert np.array_equal(actual,expected)
        error=float(np.max(np.abs(actual/(2*scale)-(gradients[0]+gradients[1])/2)))
        assert error<=.5/scale+1e-15
        row.update({'repetition':repeat,'coordinates':n,'exact_integer_sum':True,
            'max_abs_mean_quantization_error':error,'protocol_plaintext_outputs':0,
            'post_run_public_data_share_reconstruction':True,
            'observed_below_threshold':row['wall_seconds']<=config['observation_threshold_seconds']})
        (trial/'result.json').write_text(json.dumps(row,indent=2)+'\n');rows.append(row)
        print(json.dumps({k:row[k] for k in ['repetition','coordinates','exact_integer_sum','wall_seconds','observed_below_threshold']}),flush=True)
    summary={'status':'PASS_CORRECTNESS','formal_acceptance':False,'model':config['model'],'affine_layers':50,'coordinates':n,
        'gradient_generation_with_imports_seconds':generation_seconds,'gradient_worker_process_seconds':worker_seconds,
        'compile_seconds':compile_seconds,'repetitions':len(rows),'wall_seconds':stats([r['wall_seconds'] for r in rows]),
        'all_observed_below_one_second':all(r['observed_below_threshold'] for r in rows),
        'max_abs_mean_quantization_error':max(r['max_abs_mean_quantization_error'] for r in rows),
        'sum_application_sent_decimal_MB':stats([sum(p['application_sent_decimal_MB'] for p in r['parties']) for r in rows]),
        'scope':config['scope'],'output':'Each party saves additive sum shares; public-data validation reconstructs only after timed protocol exits.',
        'timing':'MPC process launches through exits, including inputs, TLS setup, addition and share persistence; excludes local gradient generation, install, compile and after-run diagnostics.'}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (out/'SHA256SUMS.json').write_text(json.dumps({str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()},indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
