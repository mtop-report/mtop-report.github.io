"""MTOP federated network: train and validate shared models across stroke centres without moving data.

Roles
  coordinator   one server (e.g. at FNO or an HPC centre). Holds the shared model, sends tasks,
                adds up what the sites return. Never receives patient rows.
  site node     one small process inside each hospital. Reads the local registry export
                (MTOP upload template, CSV), answers tasks with sums over its own patients and
                keeps an audit log of everything it sends. Connects outward only.

What travels
  - gradient/hessian histograms on fixed, public bins (gradient-boosted trees),
  - gradient and Hessian sums (penalised logistic regression),
  - counts per score bin (pooled AUROC, calibration),
  all protected by secure aggregation: each site adds random masks agreed pairwise with the other
  sites (X25519 key exchange, ChaCha20 stream), so the coordinator only ever sees the network total.

Guarantees in this implementation
  - training is lossless: the federated model equals the model trained on the pooled data;
  - every leaf of a shared tree summarises at least `min_child` (20) patients of the network;
  - a site refuses to join with fewer than 50 patients and asks for approval of every job;
  - secure aggregation needs at least three sites.

Modules: schema (template parsing), bins, secagg, gbdt, logreg, site, coordinator, transport,
server (HTTP coordinator + dashboard), client (HTTP site node + local console), demo.
"""
PROTOCOL = "mtop-fed/1"
