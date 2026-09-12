# Public two-party MPC baselines

Public-data checks built on [MP-SPDZ v0.4.3](https://github.com/data61/MP-SPDZ/tree/26a605368e40fed3a7e9cee78c9a3f4390b85eb5). The official Linux archive is pinned by SHA-256. A GitHub-hosted Ubuntu 24.04 runner executes two real `semi2k` processes with private inputs, arithmetic shares in the 64-bit ring and encrypted loopback channels.

## Logistic regression and linear SVM

Run **Public LR and linear SVM train and infer** from Actions, or use:

```bash
gh workflow run secure-ml.yml --repo leemaple/zhibiao001 --ref codex/first-aggregate-baseline
```

Four separate programs cover LR training, LR inference, linear SVM training and linear SVM inference. Training saves model shares; newly started inference processes reload those shares. Inference receives only test features and does not instantiate SGD, reset or update model parameters. The runner verifies that share files remain unchanged.

Most computation reuses upstream `ml.Dense`, `ml.Output(approx=5)`, `ml.SGD`, and `sfix.write_to_file/read_from_file`. Small adapters add L2 weight regularization and the linear soft-margin hinge gradient. This is approximate-sigmoid LR and an adapted linear SVM; the latter is not an upstream ready-made trainer.

The fixed first profile uses Iris classes 0/1: 64 training samples, 36 held-out samples, four features split equally between parties, labels at P0. Median/IQR scaling is fitted on training rows only. Both models run 20 full-batch updates with zero initialization, learning rate 0.5, no momentum, L2 coefficient 0.01 and 16 fractional bits. Update order, batch size, regularization and sigmoid approximation match a NumPy float64 reference. Predeclared checks require maximum weight/score error at most 0.001, complete class agreement and a lower final diagnostic loss. sklearn is a secondary utility comparison with a different optimizer.

First successful run: [34699943398](https://github.com/leemaple/zhibiao001/actions/runs/34699943398), code [9955d23](https://github.com/leemaple/zhibiao001/tree/9955d2327065db62f452771cdb9427789ac87f8d).

| Model | Phase | Repetitions | Median process wall | Max score error vs reference |
|---|---|---:|---:|---:|
| Approximate LR | Train, 20 epochs | 3 | 8.4709 s | 0.00001220 |
| Approximate LR | Infer, batch of 36 | 3 | 0.3629 s | 0.00001083 |
| Linear SVM | Train, 20 epochs | 3 | 4.1562 s | 0.00004586 |
| Linear SVM | Infer, batch of 36 | 3 | 0.1447 s | 0.00003318 |

All four checks passed on every repetition. Both models classified 36/36 held-out samples correctly; F1 and AUC were 1.0 on this small, easy public split. Repetitions use the same data. These values establish a functional baseline, not a generalization guarantee or formal acceptance result.

Download `public-secure-ml-<run>-<attempt>` for public inputs, binary diagnostics, model shares, references, versions, party logs, resources and a SHA-256 manifest. Actions retains artifacts for 30 days. Plaintext model diagnostics and both shares are intentionally public for this public-data test; this does not demonstrate production model confidentiality.

On Linux x86-64 with Python 3.12, GNU `time`, and runtime libraries installed by the workflow:

```bash
python3 -m pip install numpy==1.26.4 scipy==1.13.1 scikit-learn==1.6.1
python3 scripts/install_mpspdz.py
python3 scripts/run_ml_benchmark.py --output results-ml
```

Use a new output directory for each run. A profile guard rejects JSON values that disagree with fixed MPC constants; edit the adapter and guard together when changing them. Inference checks scores against both the training reference and the saved diagnostic model; its unmeasured weight-error summary is `null`. Dependencies remain under ignored `vendor/`; upstream retains its licenses. Compilation uses `-R 64 -M` to preserve memory instruction order.

Timing starts before both process launches and ends after both exits and log closure. It includes connections, live cryptographic preprocessing, computation, share persistence during training and diagnostic output. Installation, compilation and input-file preparation are outside this time. MP-SPDZ reports unused preprocessing material at the default batch size; these are untuned measurements including startup. Application communication totals are approximately 1655.949 MB / 48.612 MB for LR train/infer and 690.482 MB / 12.014 MB for SVM train/infer. These are not TLS/TCP wire bytes. Same-host loopback is not WAN or independent-administrator deployment evidence.

## Synthetic aggregation

Run **Synthetic two-party aggregation**, or:

```bash
python3 scripts/install_mpspdz.py
python3 scripts/run_benchmark.py --output results
```

This separate check sums two synthetic signed Q20 integer vectors of 1,000 and 100,000 coordinates, with one warmup and three measurements each. Every output is compared with an integer reference, including negatives, cancellation, zero and bound cases. [First run 34698372827](https://github.com/leemaple/zhibiao001/actions/runs/34698372827) passed; wall medians were 0.064629 s and 0.114797 s.

The aggregate is disclosed to P0 for diagnostics; together with P0's own input it reveals P1's contribution. Production gradient handling requires a separately defined output policy. Aggregation alone does not exercise multiplication, fixed-point truncation, actual model gradients or model training. Internal timers identify code blocks; buffered work can cross their boundaries, so they are not independent network-phase measurements.
