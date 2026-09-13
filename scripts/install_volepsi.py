"""Build fixed upstream native frontend, adding one communication counter only."""
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
out = ROOT/'results-volepsi'
out.mkdir(exist_ok=False)
config = json.loads((ROOT/'volepsi-benchmark.json').read_text())
vendor = ROOT/'vendor/volepsi'
vendor.mkdir(parents=True,exist_ok=False)


def run(args, cwd=vendor):
    subprocess.run(args,cwd=cwd,check=True)


start = time.perf_counter()
run(['git','init'])
run(['git','remote','add','origin',config['repository']])
run(['git','fetch','--depth','1','origin',config['commit']])
run(['git','checkout','--detach','FETCH_HEAD'])
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=vendor,text=True).strip()==config['commit']
source = vendor/'volePSI/fileBased.cpp'
before = source.read_text()
needle = '        }\n        catch (std::exception& e)'
assert before.count(needle)==1
after = before.replace(needle,'            std::cout << "PROFILE_BYTES_SENT=" << chl.bytesSent() << std::endl;\n'+needle)
source.write_text(after)
(out/'instrumentation.diff').write_bytes(subprocess.check_output(['git','diff'],cwd=vendor))
options = ['-DCMAKE_BUILD_TYPE=Release','-DFETCH_AUTO=ON','-DVOLE_PSI_NO_SYSTEM_PATH=ON',
    '-DVOLE_PSI_ENABLE_BOOST=ON','-DVOLE_PSI_ENABLE_OPENSSL=ON',
    '-DVOLE_PSI_ENABLE_BITPOLYMUL=ON','-DPARALLEL_FETCH=2','-DSUDO_FETCH=OFF']
try:
    with (out/'build.log').open('w') as log:
        subprocess.run(['cmake','-S','.','-B','out/build/linux']+options,cwd=vendor,
                       stdout=log,stderr=subprocess.STDOUT,check=True)
        subprocess.run(['cmake','--build','out/build/linux','--parallel','2'],cwd=vendor,
                       stdout=log,stderr=subprocess.STDOUT,check=True)
finally:
    for i,p in enumerate(vendor.rglob('log-*.txt')):
        (out/f'dependency-build-{i}-{p.name}').write_bytes(p.read_bytes())
binary=vendor/'out/build/linux/frontend/frontend'
assert binary.is_file()
commits={}
for p in vendor.rglob('.git'):
    repo=p.parent
    commits[str(repo.relative_to(vendor))]=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
assert config['libote_commit'] in commits.values()
(out/'upstream.json').write_text(json.dumps({'repository':config['repository'],'commit':config['commit'],
    'dependencies':commits,'cmake_options':options,'build_seconds':time.perf_counter()-start,
    'binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),
    'source_instrumentation':'One bytesSent print in fileBased.cpp; no protocol changes'},indent=2)+'\n')
for i,p in enumerate(vendor.rglob('CMakeCache.txt')):
    (out/f'cmake-cache-{i}.txt').write_bytes(p.read_bytes())
cert=ROOT/'vendor/volepsi-tls';cert.mkdir()
with (out/'tls-setup.log').open('w') as log:
    def ssl(args):subprocess.run(['openssl']+args,cwd=cert,stdout=log,stderr=subprocess.STDOUT,check=True)
    ssl(['req','-x509','-newkey','rsa:2048','-nodes','-keyout','ca.key','-out','ca.crt','-days','2','-subj','/CN=public-psi-test-ca'])
    for party in (0,1):
        ssl(['req','-newkey','rsa:2048','-nodes','-keyout',f'p{party}.key','-out',f'p{party}.csr','-subj',f'/CN=public-test-party-{party}'])
        ssl(['x509','-req','-in',f'p{party}.csr','-CA','ca.crt','-CAkey','ca.key','-CAcreateserial','-out',f'p{party}.crt','-days','2'])
print('VOLE-PSI build and certificates ready',flush=True)
