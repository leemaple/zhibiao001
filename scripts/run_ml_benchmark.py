"""Four public-data checks using real MP-SPDZ train/infer executables."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import shutil
import signal
import struct
import subprocess
import time

import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from run_benchmark import capture, metric, sha, stats

ROOT = Path(__file__).resolve().parents[1]


def approx5(z):
    # The public upstream Compiler/ml.py function, evaluated in NumPy float64.
    return np.select([z <= -5, z <= -2.5, z <= 2.5, z <= 5],
                     [0.0001, 0.02776*z + 0.145, 0.17*z + 0.5,
                      0.02776*z + 0.85498], default=0.9999)


def score(name, x, w):
    margin = x @ w[:-1] + w[-1]
    return approx5(margin) if name == "lr" else margin


def labels(name, values):
    return (values > (0.5 if name == "lr" else 0)).astype(int)


def diagnostic_loss(name, x, y, w, regularization):
    z = x @ w[:-1] + w[-1]
    loss = np.logaddexp(0, z) - y*z if name == "lr" else np.maximum(1 - (2*y-1)*z, 0)
    return float(np.mean(loss) + regularization/2 * np.dot(w[:-1], w[:-1]))


def train_reference(name, x, y, config):
    w = np.zeros(x.shape[1] + 1)
    rate = config["learning_rate"]
    history = []
    for epoch in range(config["epochs"]):
        z = x @ w[:-1] + w[-1]
        difference = approx5(z) - y if name == "lr" else -(2*y-1) * ((2*y-1)*z < 1)
        gradient = x.T @ difference / len(x) + config["l2_lambda"] * w[:-1]
        w[:-1] -= rate * gradient
        w[-1] -= rate * np.mean(difference)
        rate *= 1 - 1e-6  # Upstream SGD decay; MPC constants/ops are quantized.
        history.append(diagnostic_loss(name, x, y, w, config["l2_lambda"]))
    return w, history


def execute(dist, program_name, trial, timeout):
    trial.mkdir()
    for party in range(2):
        (dist / f"Player-Data/Binary-Output-P{party}-0").unlink(missing_ok=True)
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = str(dist) + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    port = random.SystemRandom().randrange(16000, 35000)
    commands, processes, streams = [], [], []
    start = time.perf_counter()
    try:
        for party in range(2):
            cmd = ["/usr/bin/time", "-f", "%e %U %S %M", "-o", str(trial/f"resources-P{party}.txt"),
                   str(dist/"semi2k-party.x"), str(party), program_name, "-N", "2", "-h", "localhost", "-pn", str(port), "-e"]
            commands.append(cmd)
            stream = (trial/f"P{party}.log").open("w")
            streams.append(stream)
            processes.append(subprocess.Popen(cmd, cwd=dist, env=env, stdout=stream,
                                              stderr=subprocess.STDOUT, start_new_session=True))
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(p.wait) for p in processes]
            try:
                for future in futures:
                    future.result(timeout=max(0.01, timeout-(time.perf_counter()-start)))
            except BaseException:
                for p in processes:
                    if p.poll() is None:
                        os.killpg(p.pid, signal.SIGKILL)
                raise
    finally:
        for p in processes:
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGKILL)
            p.wait()
        for stream in streams:
            stream.close()
        result = {"commands": commands, "exit_codes": [p.returncode for p in processes],
                  "wall_seconds": time.perf_counter()-start}
        (trial/"execution.json").write_text(json.dumps(result, indent=2)+"\n")
    if result["exit_codes"] != [0, 0]:
        raise RuntimeError(f"MPC execution failed: {trial}")
    shutil.copy2(dist/"Player-Data/Binary-Output-P0-0", trial/"diagnostic-P0.bin")
    result["parties"] = []
    for party in range(2):
        log = (trial/f"P{party}.log").read_text()
        elapsed, user, system, rss = map(float, (trial/f"resources-P{party}.txt").read_text().split())
        result["parties"].append({"party": party,
            "runtime_seconds": metric(log, r"Time = ([\d.eE+-]+) seconds"),
            "application_sent_decimal_MB": metric(log, r"Data sent = ([\d.eE+-]+) MB"),
            "max_rss_KiB": rss, "user_cpu_seconds": user, "system_cpu_seconds": system})
    return result


def write_inputs(dist, target, x, y=None):
    target.mkdir(exist_ok=True)
    for party in range(2):
        values = x[:, party*2:(party+1)*2].flatten()
        if party == 0 and y is not None:
            values = np.concatenate((values, y))
        path = target/f"Input-P{party}-0"
        path.write_text(" ".join(f"{v:.17g}" for v in values)+"\n")
        shutil.copy2(path, dist/"Player-Data"/path.name)


def read_diagnostics(trial, n, training=False):
    data = (trial/"diagnostic-P0.bin").read_bytes()
    count = 2*n + (5 if training else 0)
    assert len(data) == 8*count, (len(data), count)
    values = np.array(struct.unpack(f"<{count}q", data), dtype=np.int64)
    weights = values[:5].astype(float)/65536 if training else None
    start = 5 if training else 0
    return weights, values[start:start+n].astype(float)/65536, values[start+n:]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results-ml")
    args = parser.parse_args()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    config = json.loads((ROOT/"ml-benchmark.json").read_text())
    install = json.loads((ROOT/"vendor/install.json").read_text())
    dist = ROOT/"vendor"/install["distribution"]
    for name in ["ml_common.py", "secure_lr_train.mpc", "secure_lr_infer.mpc", "secure_svm_train.mpc", "secure_svm_infer.mpc"]:
        shutil.copy2(ROOT/"programs"/name, dist/"Programs/Source"/name)
    shutil.copy2(ROOT/"ml-benchmark.json", out/"config.json")
    shutil.copy2(ROOT/"vendor/install.json", out/"upstream-install.json")
    (out/"environment.json").write_text(json.dumps({
        "utc": datetime.now(timezone.utc).isoformat(), "platform": platform.platform(),
        "python": platform.python_version(), "cpu_count": os.cpu_count(),
        "image_version": os.environ.get("ImageVersion"), "github_run_id": os.environ.get("GITHUB_RUN_ID"),
        "harness_commit": subprocess.check_output(["git", "rev-parse", "HEAD"],cwd=ROOT,text=True).strip(),
        "binary_sha256": sha(dist/"semi2k-party.x"),
        "dependencies": {name: importlib.metadata.version(name) for name in ["numpy", "scipy", "scikit-learn"]},
        "source_sha256": {str(p.relative_to(ROOT)): sha(p) for p in [ROOT/"ml-benchmark.json",ROOT/"programs/ml_common.py",Path(__file__).resolve()]},
        "network": "two real processes, one host, TLS loopback; no independent administration or WAN",
    }, indent=2)+"\n")
    capture(["lscpu"],ROOT,out/"lscpu.txt")
    capture(["python3","-m","pip","freeze"],ROOT,out/"pip-freeze.txt")
    dataset = load_iris()
    x, y = dataset.data[:100].astype(float), dataset.target[:100]
    train_index, test_index = train_test_split(np.arange(100),train_size=64,random_state=config["split_seed"],stratify=y)
    center = np.median(x[train_index],axis=0)
    iqr = np.percentile(x[train_index],75,axis=0)-np.percentile(x[train_index],25,axis=0)
    assert np.all(iqr>0)
    scaled = np.clip((x-center)/iqr,-3,3)/3
    xtrain, xtest, ytrain, ytest = scaled[train_index],scaled[test_index],y[train_index],y[test_index]
    np.savez(out/"dataset.npz",x_raw=x,y=y,train_index=train_index,test_index=test_index,
             center=center,iqr=iqr,x_train=xtrain,x_test=xtest,y_train=ytrain,y_test=ytest)
    compiled = {}
    for model in ["lr","svm"]:
        for phase,n in [("train",64),("infer",36)]:
            base=f"secure_{model}_{phase}"
            params=[str(n),"4"] + ([str(config["epochs"])] if phase=="train" else [])
            seconds=capture(["python3","compile.py","-R","64","-M",base]+params,dist,out/f"compile-{model}-{phase}.log")
            compiled[model,phase] = (base+"-"+"-".join(params),seconds)
    records=[]
    for model in ["lr","svm"]:
        mdir=out/model;mdir.mkdir()
        reference,history=train_reference(model,xtrain,ytrain,config)
        baseline=LogisticRegression(C=1/(config["l2_lambda"]*len(xtrain)),solver="lbfgs",max_iter=1000) if model=="lr" else SVC(C=1/(config["l2_lambda"]*len(xtrain)),kernel="linear")
        baseline.fit(xtrain,ytrain)
        (mdir/"reference.json").write_text(json.dumps({"weights":reference.tolist(),"training_diagnostic_loss_history":history,
            "sklearn_test_accuracy":float(baseline.score(xtest,ytest)),
            "sklearn_is_secondary_utility_only":True},indent=2)+"\n")
        for repeat in range(config["repetitions"]):
            trial=mdir/f"repeat-{repeat}";trial.mkdir()
            persistence=dist/"Persistence";persistence.mkdir(exist_ok=True)
            for party in range(2):
                (persistence/f"Transactions-P{party}.data").unlink(missing_ok=True)
            write_inputs(dist,trial/"train-inputs",xtrain,ytrain if model=="lr" else 2*ytrain-1)
            train_result=execute(dist,compiled[model,"train"][0],trial/"train",config["timeout_seconds"])
            weights,train_scores,train_pred=read_diagnostics(trial/"train",64,True)
            saved=trial/"model-shares";saved.mkdir()
            before={}
            for party in range(2):
                path=persistence/f"Transactions-P{party}.data"
                assert path.stat().st_size>0
                before[path.name]=sha(path)
                shutil.copy2(path,saved/path.name)
            weight_error=float(np.max(np.abs(weights-reference)))
            train_error=float(np.max(np.abs(train_scores-score(model,xtrain,reference))))
            train_agreement=float(np.mean(train_pred==labels(model,score(model,xtrain,reference))))
            loss_start=diagnostic_loss(model,xtrain,ytrain,np.zeros(5),config["l2_lambda"])
            loss_end=diagnostic_loss(model,xtrain,ytrain,weights,config["l2_lambda"])
            train_result.update({"model":model,"phase":"train","repetition":repeat,
                "compile_seconds":compiled[model,"train"][1],"max_abs_weight_error":weight_error,
                "max_abs_score_error":train_error,"class_agreement":train_agreement,
                "diagnostic_loss_initial":loss_start,"diagnostic_loss_final":loss_end,
                "weights_diagnostic":weights.tolist(),"accuracy":float(np.mean(train_pred==ytrain)),
                "pass":weight_error<=config["max_abs_weight_error"] and train_error<=config["max_abs_score_error"] and train_agreement>=config["minimum_class_agreement"] and loss_end<loss_start})
            (trial/"train/result.json").write_text(json.dumps(train_result,indent=2)+"\n")
            records.append(train_result)
            print(json.dumps({k:train_result[k] for k in ["model","phase","repetition","pass","wall_seconds","max_abs_weight_error"]}),flush=True)
            write_inputs(dist,trial/"infer-inputs",xtest)
            infer_result=execute(dist,compiled[model,"infer"][0],trial/"infer",config["timeout_seconds"])
            _,test_scores,test_pred=read_diagnostics(trial/"infer",36)
            after={name:sha(persistence/name) for name in before}
            model_unchanged=before==after
            (trial/"model-binding.json").write_text(json.dumps({"before_inference":before,"after_inference":after,"unchanged":model_unchanged},indent=2)+"\n")
            test_error=float(np.max(np.abs(test_scores-score(model,xtest,reference))))
            agreement=float(np.mean(test_pred==labels(model,score(model,xtest,reference))))
            infer_result.update({"model":model,"phase":"infer","repetition":repeat,
                "compile_seconds":compiled[model,"infer"][1],"max_abs_score_error":test_error,
                "max_abs_score_error_from_saved_model":float(np.max(np.abs(test_scores-score(model,xtest,weights)))),
                "class_agreement":agreement,"model_shares_unchanged":model_unchanged,
                "accuracy":float(accuracy_score(ytest,test_pred)),"f1":float(f1_score(ytest,test_pred)),
                "auc":float(roc_auc_score(ytest,test_scores)),
                "pass":test_error<=config["max_abs_score_error"] and agreement>=config["minimum_class_agreement"] and model_unchanged})
            (trial/"infer/result.json").write_text(json.dumps(infer_result,indent=2)+"\n")
            records.append(infer_result)
            print(json.dumps({k:infer_result[k] for k in ["model","phase","repetition","pass","wall_seconds","accuracy","max_abs_score_error"]}),flush=True)
    summary={"status":"PASS_FUNCTIONAL_BASELINE" if all(r["pass"] for r in records) else "FAIL_CHECKS",
        "formal_acceptance":False,"cases":[],"scope":"Public Iris two-class baseline, same-host real two-party MPC; diagnostic model disclosure; not production security or arbitrary dataset utility."}
    for model in ["lr","svm"]:
        for phase in ["train","infer"]:
            rows=[r for r in records if r["model"]==model and r["phase"]==phase]
            summary["cases"].append({"model":model,"phase":phase,"repetitions":len(rows),
                "pass":all(r["pass"] for r in rows),"wall_seconds":stats([r["wall_seconds"] for r in rows]),
                "max_abs_score_error":max(r["max_abs_score_error"] for r in rows),
                "min_class_agreement":min(r["class_agreement"] for r in rows),
                "accuracy":rows[0]["accuracy"],
                "max_abs_weight_error":max(r.get("max_abs_weight_error",0) for r in rows),
                "sum_application_sent_decimal_MB":stats([sum(p["application_sent_decimal_MB"] for p in r["parties"]) for r in rows])})
    (out/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    (out/"SHA256SUMS.json").write_text(json.dumps({str(p.relative_to(out)):sha(p) for p in sorted(out.rglob("*")) if p.is_file()},indent=2)+"\n")
    print(json.dumps(summary,indent=2),flush=True)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"],"a") as f:
            f.write("## Public-data LR/SVM functional baseline\n\n| Model | Phase | Pass | Median wall (s) | Accuracy |\n|---|---|---|---:|---:|\n")
            for row in summary["cases"]:
                f.write(f"| {row['model']} | {row['phase']} | {row['pass']} | {row['wall_seconds']['median']:.4f} | {row['accuracy']:.4f} |\n")
            f.write("\nPublic Iris subset; real MPC on one host. Models persist as shares between independent executions. Diagnostic model output is for testing only.\n")
    if summary["status"]!="PASS_FUNCTIONAL_BASELINE":
        raise SystemExit(1)


if __name__=="__main__":
    main()
