# ProofID-Research

**Status:** Pre-Hackathon Research Mode

**Purpose:** Feasibility experiments for ProofID identity verification components.

⚠️ **IMPORTANT:** This repository contains **disposable research code only**. It is NOT the final ProofID hackathon submission.

The actual hackathon application will be created fresh in a separate repository (`ProofID/`) starting September 26, 2026.

---

## Research Timeline

- **Milestone 1:** Speaker Verification Feasibility ← **CURRENT**
- **Milestone 2:** Face Verification Feasibility
- **Milestone 3:** Liveness Feasibility
- **Milestone 4:** Fresh Challenge-Response Design
- **Milestone 5:** Unknown Number Context
- **Milestone 6:** Android Local Inference
- **Milestone 7:** Decision Engine Design
- **Milestone 8:** Secure Local Storage
- **Milestone 9:** Performance Benchmarking
- **Milestone 10:** NPU Feasibility (Qualcomm QNN)
- **Milestone 11:** Attack Testing (Replay, Photo, Video)

These milestones are research checkpoints, not a requirement to implement every component. Each milestone must be validated independently, and the project may stop, pivot, or deprioritize a component when experiments show it is unreliable or impractical.

---

## Directory Structure

```
ProofID-Research/
├── README.md                    (this file)
├── decisions.md                 (technical decision log)
├── benchmark-results.md         (measured results across all milestones)
│
├── speaker/                     (Milestone 1: Speaker Verification)
│   ├── README.md
│   ├── data/                    (local-only raw biometric test data - never committed)
│   ├── models/                  (model configs and metadata)
│   ├── scripts/
│   │   ├── 01_model_candidates.py       (evaluate speaker models)
│   │   ├── 02_genuine_vs_impostor.py    (benchmark genuine vs impostor)
│   │   ├── 03_replay_testing.py         (replay attack resistance)
│   │   └── analyze_results.py           (results analysis)
│   └── results/                 (experimental output, graphs, CSV - committed)
│
├── face/                        (Milestone 2: Face Verification)
│   └── (structure similar to speaker)
│
├── liveness/                    (Milestone 3: Liveness)
│   └── (structure similar to speaker)
│
└── challenge/                   (Milestone 4: Challenge-Response)
    └── (research on phrase generation, validation)
```

---

## Constraints

This research WILL NOT:
- ❌ Build the final ProofID Android application
- ❌ Build the final enrollment flow
- ❌ Build the final verification UI
- ❌ Integrate complete end-to-end product
- ❌ Use paid/cloud APIs
- ❌ Require API keys
- ❌ Pre-build hackathon submission code
- ❌ Commit raw biometric data (audio, images, video, embeddings)
- ❌ Commit model checkpoints or caches
- ❌ Commit API keys or secrets

This research WILL:
- ✅ Benchmark open-source speaker/face models
- ✅ Measure actual genuine vs. impostor separation (feasibility dataset only)
- ✅ Test replay attack behavior
- ✅ Assess Android feasibility
- ✅ Document limitations honestly
- ✅ Commit aggregate research results and measurements
- ✅ Record all decisions with evidence

---

## Key Principle

**Reliability > Features**

A well-built simple system beats a broken complex system.

---

## Next Steps

1. Identify suitable lightweight speaker embedding models
2. Set up benchmarking harness
3. Collect test audio samples (local only, not committed)
4. Measure genuine similarity
5. Measure impostor similarity
6. Test replay behavior
7. Report results and recommendations

See `speaker/README.md` for Milestone 1 details.
