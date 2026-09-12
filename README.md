# Synthetic two-party aggregation baseline

This repository runs the existing [MP-SPDZ](https://github.com/data61/MP-SPDZ) `semi2k` backend on GitHub Actions. It is an independent, generic benchmark using generated data only. It contains no task documents or application dataset.

The pinned v0.4.3 Linux distribution is verified against its release archive SHA-256. Two real MPC processes use private inputs, arithmetic secret sharing in the 64-bit ring, and encrypted loopback channels. The computation sums two signed Q20 integer-encoded vectors. Sizes are 1,000 and 100,000 coordinates; one warmup and three measured repetitions per size. All output coordinates are compared with a Python integer reference, including negative values, cancellation, zero and declared bound cases. Q20 is a benchmark encoding choice, not a full numerical configuration for machine learning.

Run the **Synthetic two-party aggregation** workflow manually in Actions, or push a benchmark change. A GitHub-hosted Ubuntu 24.04 runner is sufficient; no self-hosted runner is required. Download the run artifact for `summary.json`, complete party logs, binary outputs, synthetic inputs, versions, resource records and a SHA-256 manifest.

On Linux x86-64 with the required runtime libraries, Python 3.12+ and GNU `time`:

```bash
python3 scripts/install_mpspdz.py
python3 scripts/run_benchmark.py --output results
```

The output directory must be new for each run. Certificates and private runtime state are never included in the result artifact. Dependencies remain under the ignored `vendor/` directory; the upstream distribution carries its own licenses.

Four MPC timers cover private input sharing, local share addition, diagnostic reconstruction to party 0, and binary output. External wall time covers both process launches through both exits, including connection setup and that diagnostic. Dependency installation, certificate setup, compilation and input materialization are outside this wall-time measurement. MP-SPDZ communication counters are application bytes, not TLS/TCP wire traffic.

**Interpretation:** A success establishes only synthetic aggregation correctness for the recorded configuration. The vectors are gradient-shaped, not gradients from a trained model. The sum is revealed to party 0 strictly for synthetic testing: with two parties, the sum and party 0's own input expose party 1's input to party 0. A production private-gradient pipeline must retain shares or define a different output policy. This benchmark does not establish LR/SVM training or inference, PSI, WAN performance, a model-size performance bound, or production security. Share addition requires neither multiplication triples nor fixed-point truncation, so those backend paths remain untested.
