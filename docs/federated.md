# MTOP federated network

Train and validate MTOP across stroke centres without moving patient data. Each hospital runs a
small **site node** next to its registry export; a **coordinator** sends tasks, the nodes answer
with **masked sums** over their own patients, and the coordinator adds them up. Only network-wide
totals ever become readable, and the result is the same model that pooling everyone's data would
give.

The *Multicentre* tab of the app (*Learn together*) shows a working network of four virtual
centres. This document is for the people who decide whether a centre joins: clinicians, registry
and IT staff, and the data protection officer.

## What the demo shows

No other hospital is involved yet: the four centres are FNO's own patients, split by treatment
years into four virtual centres (A 2005–2014, B 2015–2017, C 2018–2020, D 2021–2025, 316–378
patients each). Each centre's data were exported in the MTOP
template and read by its own node. Inside every centre, 5-fold cross-validation kept a fifth of the
patients aside, and each centre scored its own held-out patients with three options: a model
trained on its own patients only (*alone*), the shared model trained by the network (*network*),
and the shared model plus 10 boosting steps on the centre's own patients, done locally
(*adapted*). Model: boosted trees at admission (100 trees, depth 3, learning rate 0.05, at least
25 patients per leaf, monotone constraints).

| Admission model, mean AUROC over the 4 centres | Alone | Network | Adapted |
|---|---|---|---|
| Death within 90 days | 0.742 | **0.780** | 0.778 |
| Functional independence (mRS 0–2) | 0.733 | 0.747 | **0.755** |

- **Death**: every centre gained (+0.02 to +0.07); significant at two centres. Death is the outcome
  with the fewest events per centre, which is where a single centre's data run out first.
- **Small centres**: a centre with only 50 patients reached AUROC 0.59 (mRS 0–2) or 0.57 (death)
  alone, and 0.74 or 0.77 in the network.
- **All settings**: across five stage/outcome settings and four centres, the network (or its local
  adaptation) did better than the centre alone in 19 of 20 comparisons. The exception is
  functional independence at admission for centre D, FNO's most recent era, whose outcomes changed
  after 2020; local adaptation narrows that gap.
- **Validation without moving data**: a network of three trained the model and the fourth tested it
  at home (AUROC 0.69–0.77 for mRS 0–2, 0.75–0.78 for death). Calibration differs between centres,
  which each centre can measure and correct locally.
- **Exactness**: the federated model's 100 trees are identical to pooled training, split for split;
  the spline logistic regression matches pooled fitting to 6 × 10⁻⁸. Over real HTTP connections the
  training took 301 rounds and 5 s, with about 2 MB sent per centre.

Real partner centres differ more in case mix and practice than FNO's eras do. Measuring and
handling that is what the network is for.

Reproduce: `python -m mtop fed-demo` (about 15 minutes; `--quick` for a short run). Results go to
`artifacts/public/federated.json` (aggregates only) and `reports/federated.md`.

## How it works

**Roles.** The *coordinator* (one server, for example at FNO or a national computing centre)
holds the shared model and sends tasks. A *site node* (a Python process inside each hospital)
reads the local registry export, answers tasks, and keeps an audit log. Nodes connect outward to
the coordinator (HTTPS long polling); hospitals open no inbound port.

**A job.** The coordinator offers a job (for example "train the admission model for mRS 0–2,
at most 301 rounds") to the participating centres. A person at each centre approves it in the
node's local console. Each node then makes a fresh key pair for the job, signs it with its
long-term identity key, and the nodes agree pairwise secrets. Then the rounds start.

**A round.** For boosted trees, the coordinator asks for the nodes it is about to split. Every
centre returns, per predictor and per value band, the sums of gradients, Hessians and patient
counts over its own patients in those nodes. The coordinator adds the masked answers, reads the
network totals, picks the best split exactly as LightGBM or XGBoost would, and moves on. Leaf values
follow from the same totals. Logistic regression works the same way with Newton steps
(gradient and Hessian sums), and validation with counts of events per narrow risk band.

**Secure aggregation** (Bonawitz et al., 2017). Every pair of centres derives a shared secret
(X25519 key agreement, HKDF-SHA256). In each round centre *i* adds a ChaCha20-generated mask for
every partner *j* (+ if *i* < *j*, − otherwise) to its vector, working with integers modulo 2⁶⁴.
Each masked vector alone is uniformly random; in the sum, every mask appears once with each sign
and cancels. The coordinator therefore learns the network total and nothing about any single
centre. Job keys are signed with each centre's Ed25519 identity, and centres can pin each other's
fingerprints, so the coordinator cannot slip in keys of its own.

**Why the result is exact.** Each centre rounds its patients' gradients to 2⁻²⁴ before adding
them up, so all sums are whole numbers and the order of addition cannot matter. The coordinator
then sees exactly the totals pooled training would see. The test suite checks this bit for bit.

**Fixed, public value bands.** Splits can only fall on value bands agreed in advance
(`config/federated.yaml`, at most 30 per predictor), never on quantiles of anyone's data.

## Keeping the shared model up to date

Outcomes drift: in FNO's registry the odds of independence at 90 days fell over the years more
than the case mix explains, which is why MTOP is recalibrated to 2021–2025. The update strategies
compared on the *Adaptive* tab run in the network as ordinary jobs restricted to a range of
treatment years. A centre adds the optional `year` column (year of treatment) to its export; it
never leaves the hospital, and a job only selects patients by it
(`"subset": {"kind": "years", "from": 2021, "to": 2025}`). A centre without the column declines
such jobs.

| Strategy | In the network |
|---|---|
| Drift monitor | A validation job per year: network-wide AUROC and calibration from masked counts |
| Intercept recalibration | A few Newton steps on one number, on the most recent years |
| Logistic recalibration | The same with intercept and slope |
| Refit | A new training job on all years |
| Refit + recalibration (MTOP's current policy) | Training on all years, then recalibration on the latest years |
| Sliding window | Training only on the last few years |
| Centre-specific calibration | Each centre shifts the shared model to its own patients locally; nothing is shared |

How often to update, which strategy to use and which alarms trigger an update are agreed by the
network in advance.

## What leaves a hospital, and what does not

| | Leaves the hospital? |
|---|---|
| Patient rows, identifiers, dates | Never |
| The centre's own sums | Only masked; unreadable on their own |
| Network totals (all centres added) | Readable by the coordinator, used to choose splits, then discarded |
| The finished model | Shared with all centres; every leaf summarises at least 25 patients |
| The centre's own validation results | Only if the centre chooses to share them |

**Safeguards enforced by the node** (`src/mtop/fed/site.py`): the protocol version must match;
every job lists at least 3 centres; the centre has at least 50 eligible patients; a person
approves each job; each round must be an operation that job type allows, in increasing order and
within the round budget declared at approval; a job's participants cannot change after approval;
other centres' keys must be validly signed. Every message sent is written to an append-only
audit log with its time, round, size and SHA-256.

**Limits, stated plainly.**
- The coordinator sees network-wide totals inside each tree node, and a narrow band can describe
  few patients. The coordinator should therefore be a trusted institution under a data-processing
  agreement that keeps only the finished model. It cannot tell which centre a number came from.
- Secure aggregation assumes the coordinator does not collude with all other centres against one.
- If a centre drops out in the middle of a round, that round's masks do not cancel and the job
  stops; restarting needs fresh approval everywhere (which also blocks "ask twice with different
  groups" attacks). Recovery from dropouts (secret-shared seeds, as in Bonawitz et al.) is not
  implemented.
- Formal differential privacy could be layered on top (it is not implemented), but at this
  network size it would cost substantial accuracy.
- The nodes and coordinator are research software. Run them over TLS, on managed machines, and
  review them with your IT security team.

## Joining

1. **Agree.** Join the protocol and the shared data dictionary (the MTOP upload template). Patient
   rows never leave the hospital. The coordinator's role is set out in a processing agreement;
   your ethics committee and data protection officer confirm local participation and which
   agreements your institution needs.
2. **Map your registry.** Export one row per patient with the template's columns
   (`python -m mtop.fed template mtop_template.csv` writes the header; the `year` column is
   optional and enables the update jobs above). The *Multicentre* tab (*Validate on your data*)
   checks the mapping in the browser without uploading anything, or run
   `python -m mtop.fed check registry.csv`. English and Czech spellings are recognised; anything
   unrecognised becomes missing and is counted.
3. **Install the node** on a computer inside the hospital that can make outgoing HTTPS connections:

   ```bash
   pip install "mtop[federated] @ git+https://github.com/viktorgajdosik/mt"
   python -m mtop.fed site --coordinator https://COORDINATOR --site-id YOUR-CENTRE \
       --token-file centre.token --data registry.csv          # console on http://127.0.0.1:8751
   ```

   The node keeps its identity key and audit log in `.mtop-site/`. To pin the other centres'
   identities, pass `--pins pins.yaml` (centre id → fingerprint, exchanged out of band).
4. **Approve and learn.** Approve jobs in the console. You receive the shared model, your own
   validation, and the option to adapt the model to your patients locally.

**Running a coordinator.**

```bash
python -m mtop.fed enrol OSTRAVA BRNO PRAHA        # one token per centre; network.yaml stores hashes only
python -m mtop.fed coordinator --network network.yaml --host 0.0.0.0 --port 8750 \
    --cert cert.pem --key key.pem --wait 3 --out network_model.json
```

The dashboard at the coordinator's address shows connected centres, traffic and the training
curve. The trained model is written in the same format the web app uses.

## Code map

| File | Role |
|---|---|
| `config/federated.yaml` | Shared settings: minimum sizes, value bands, model recipes, knots |
| `src/mtop/fed/schema.py` | Reads a registry export in the MTOP template (optional `year` column) |
| `src/mtop/fed/bins.py` | Fixed value bands |
| `src/mtop/fed/secagg.py` | Identities, key agreement, masks, aggregation |
| `src/mtop/fed/gbdt.py` | Boosted trees from network histograms (monotone constraints, missing values) |
| `src/mtop/fed/logreg.py` | Penalised (spline) logistic regression by Newton steps |
| `src/mtop/fed/site.py` | The site node: rules, computations, audit log |
| `src/mtop/fed/coordinator.py` | Jobs, rounds, aggregation; pooled reference for testing |
| `src/mtop/fed/server.py`, `client.py` | HTTP coordinator with dashboard; HTTP node with local console |
| `src/mtop/fed/demo.py` | The four-centre demonstration |
| `tests/test_federated.py` | Masks cancel, keys are checked, exactness, monotonicity, node rules, HTTP |

Requirements: Python 3.10+, numpy, pandas, PyYAML and `cryptography` (for the masks).
