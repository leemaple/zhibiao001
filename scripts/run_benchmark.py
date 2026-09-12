"""Real two-process MPC, complete synthetic-output comparison, immutable run logs."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import re
import shutil
import signal
import statistics
import struct
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def capture(command, cwd, path):
    start = time.perf_counter()
    with path.open("w") as stream:
        subprocess.run(command, cwd=cwd, stdout=stream,
                       stderr=subprocess.STDOUT, check=True)
    return time.perf_counter() - start


def metric(text, pattern):
    found = re.search(pattern, text)
    if not found:
        raise ValueError("Missing runtime metric: " + pattern)
    return float(found.group(1))


def stats(values):
    return {"median": statistics.median(values), "min": min(values), "max": max(values)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results")
    args = parser.parse_args()
    config = json.loads((ROOT / "benchmark.json").read_text())
    install = json.loads((ROOT / "vendor/install.json").read_text())
    dist = ROOT / "vendor" / install["distribution"]
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(ROOT / "benchmark.json", output / "benchmark.json")
    shutil.copy2(ROOT / "vendor/install.json", output / "install.json")
    shutil.copy2(ROOT / "programs/aggregate.mpc", dist / "Programs/Source/aggregate.mpc")
    environment = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(), "architecture": platform.machine(),
        "python": platform.python_version(), "cpu_count": os.cpu_count(),
        "runner_os": os.environ.get("RUNNER_OS"),
        "runner_arch": os.environ.get("RUNNER_ARCH"),
        "runner_image": os.environ.get("ImageOS"),
        "runner_image_version": os.environ.get("ImageVersion"),
        "github_run_id": os.environ.get("GITHUB_RUN_ID"),
        "github_run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
        "harness_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_sha256": {str(p.relative_to(ROOT)): sha(p) for p in
                          [ROOT / "benchmark.json", ROOT / "programs/aggregate.mpc",
                           Path(__file__).resolve(), ROOT / "scripts/install_mpspdz.py"]},
        "mp_spdz_binary_sha256": sha(dist / "semi2k-party.x"),
        "mp_spdz_compiler_sha256": sha(dist / "compile.py"),
        "network": config["network"], "tcp_wire_bytes_measured": False,
        "bandwidth_and_rtt": "Not shaped or independently measured",
    }
    (output / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
    capture(["lscpu"], ROOT, output / "lscpu.txt")
    shutil.copy2("/proc/meminfo", output / "meminfo.txt")
    capture(["ldd", str(dist / "semi2k-party.x")], dist, output / "binary-dependencies.txt")
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = str(dist) + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    records = []
    for n in config["dimensions"]:
        case = output / str(n)
        case.mkdir()
        rng = random.Random(config["seed"] + n)
        bound = config["absolute_input_bound_encoded"]
        left = [rng.randint(-bound, bound) for _ in range(n)]
        right = [rng.randint(-bound, bound) for _ in range(n)]
        # Exercise signed values, cancellation, zeros and the declared bounds.
        left[:8] = [-bound, bound, 0, 1, -1, bound, -bound, 0]
        right[:8] = [-bound, bound, 0, -1, 1, -bound, bound, 1]
        expected = [a + b for a, b in zip(left, right)]
        assert max(map(abs, expected)) < 2 ** 62
        generated = time.perf_counter()
        for party, values in enumerate((left, right)):
            data = case / f"Input-P{party}-0"
            data.write_text(" ".join(map(str, values)) + "\n")
            shutil.copy2(data, dist / "Player-Data" / data.name)
        (case / "expected.bin").write_bytes(struct.pack(f"<{n}q", *expected))
        input_materialization_seconds = time.perf_counter() - generated
        compile_seconds = capture(["python3", "compile.py", "-R", "64", "aggregate", str(n)],
                                  dist, case / "compile.log")
        for repetition in range(config["warmups"] + config["repetitions"]):
            trial = case / f"trial-{repetition}"
            trial.mkdir()
            for party in range(2):
                (dist / f"Player-Data/Binary-Output-P{party}-0").unlink(missing_ok=True)
            port = 15000 + random.SystemRandom().randrange(20000)
            commands, processes, logs = [], [], []
            start = time.perf_counter()
            try:
                for party in range(2):
                    command = ["/usr/bin/time", "-f", "%e %U %S %M", "-o",
                               str(trial / f"resources-P{party}.txt"),
                               str(dist / "semi2k-party.x"), str(party),
                               f"aggregate-{n}", "-N", "2", "-h", "localhost", "-pn", str(port), "-e"]
                    commands.append(command)
                    stream = (trial / f"P{party}.log").open("w")
                    logs.append(stream)
                    processes.append(subprocess.Popen(command, cwd=dist, env=env, stdout=stream,
                                                      stderr=subprocess.STDOUT, start_new_session=True))
                for process in processes:
                    remaining = config["timeout_seconds"] - (time.perf_counter() - start)
                    process.wait(timeout=max(0.001, remaining))
            finally:
                for process in processes:
                    if process.poll() is None:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                for stream in logs:
                    stream.close()
                wall = time.perf_counter() - start
                (trial / "execution.json").write_text(json.dumps({"commands": commands,
                    "exit_codes": [p.returncode for p in processes], "wall_seconds": wall}, indent=2) + "\n")
            if any(p.returncode != 0 for p in processes):
                raise RuntimeError(f"MPC party failed in {trial}")
            actual_path = dist / "Player-Data/Binary-Output-P0-0"
            shutil.copy2(actual_path, trial / "actual-P0.bin")
            actual_bytes = actual_path.read_bytes()
            if len(actual_bytes) != 8 * n:
                raise ValueError(f"Wrong output byte length: {len(actual_bytes)} != {8*n}")
            actual = struct.unpack(f"<{n}q", actual_bytes)
            differences = [abs(a - b) for a, b in zip(actual, expected)]
            party_stats = []
            for party in range(2):
                log = (trial / f"P{party}.log").read_text()
                elapsed, user, system, rss = map(float, (trial / f"resources-P{party}.txt").read_text().split())
                party_stats.append({
                    "party": party, "runtime_seconds": metric(log, r"Time = ([\d.eE+-]+) seconds"),
                    "stage_seconds": {str(i): metric(log, rf"Time{i} = ([\d.eE+-]+) seconds") for i in range(1,5)},
                    "application_sent_decimal_MB": metric(log, r"Data sent = ([\d.eE+-]+) MB"),
                    "approx_rounds": metric(log, r"Data sent = [\d.eE+-]+ MB in ~([\d.eE+-]+) rounds"),
                    "process_seconds_gnu_time": elapsed, "user_cpu_seconds": user,
                    "system_cpu_seconds": system, "max_rss_KiB": rss,
                })
            record = {"dimensions": n, "repetition": repetition, "warmup": repetition < config["warmups"],
                      "checked_elements": n, "mismatched_elements": sum(d != 0 for d in differences),
                      "max_abs_error_encoded": max(differences), "max_abs_error_real": max(differences) / 2**config["fraction_bits"],
                      "actual_sha256": sha(trial / "actual-P0.bin"), "expected_sha256": sha(case / "expected.bin"),
                      "wall_seconds": wall, "compile_seconds": compile_seconds,
                      "input_materialization_seconds": input_materialization_seconds, "parties": party_stats}
            (trial / "result.json").write_text(json.dumps(record, indent=2) + "\n")
            records.append(record)
            print(json.dumps({k: record[k] for k in ["dimensions", "repetition", "warmup", "mismatched_elements", "wall_seconds"]}), flush=True)
            if record["mismatched_elements"]:
                raise ValueError("Full-vector plaintext reference mismatch")
    summary = {"status": "PASS_CORRECTNESS_ONLY", "verified_acceptance": False,
               "stages": {"1": "private inputs", "2": "share addition", "3": "diagnostic reveal to P0", "4": "binary output"},
               "cases": [], "limitations": [
                   "Synthetic integer-encoded vectors, not gradients from a trained model.",
                   "Two processes on one shared host; not two independently administered machines or WAN.",
                   "The diagnostic sum plus P0's own input reveals P1's contribution to P0; no protection against this output leakage is claimed.",
                   "No LR/SVM training or inference, PSI, dropout, malicious-security or production-safety claim.",
                   "Runtime wall excludes dependency setup, TLS certificate generation, input generation and compilation; these are reported separately where measured.",
                   "Communication is MP-SPDZ application counters, not TLS/TCP wire bytes; 3 repetitions do not establish a worst-case bound.",
                   "Addition uses no multiplication triples or fixed-point truncation; it does not validate those backend paths."]}
    for n in config["dimensions"]:
        samples = [r for r in records if r["dimensions"] == n and not r["warmup"]]
        summary["cases"].append({"dimensions": n, "repetitions": len(samples),
            "all_elements_match": all(r["mismatched_elements"] == 0 for r in samples),
            "max_abs_error_encoded": max(r["max_abs_error_encoded"] for r in samples),
            "process_wall_seconds": stats([r["wall_seconds"] for r in samples]),
            "max_party_runtime_seconds": stats([max(p["runtime_seconds"] for p in r["parties"]) for r in samples]),
            "max_party_stage_seconds": {str(i): stats([max(p["stage_seconds"][str(i)] for p in r["parties"]) for r in samples]) for i in range(1,5)},
            "sum_party_application_sent_decimal_MB": stats([sum(p["application_sent_decimal_MB"] for p in r["parties"]) for r in samples]),
            "max_single_party_rss_KiB": max(p["max_rss_KiB"] for r in samples for p in r["parties"])})
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    manifest = {str(p.relative_to(output)): sha(p) for p in sorted(output.rglob("*")) if p.is_file()}
    (output / "SHA256SUMS.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    job_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if job_summary:
        with open(job_summary, "a") as stream:
            stream.write("## Synthetic two-party aggregation: correctness PASS\n\n")
            stream.write("| Dimensions | Repeats | Matches | Wall median (s) |\n|---:|---:|---|---:|\n")
            for case in summary["cases"]:
                stream.write(f"| {case['dimensions']} | {case['repetitions']} | {case['all_elements_match']} | {case['process_wall_seconds']['median']:.6f} |\n")
            stream.write("\nSingle-host TLS loopback; synthetic diagnostic output only. Not an acceptance result.\n")


if __name__ == "__main__":
    main()
