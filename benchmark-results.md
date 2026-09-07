# ProofID-Research: Benchmark Results

**Last Updated:** 2026-09-07

---

## Milestone 1: Speaker Verification Feasibility

**Status:** IN PROGRESS

**Research Question:**
"Can a lightweight speaker embedding model reliably distinguish the enrolled speaker from an impostor under controlled conditions?"

**Important Note:**
This benchmark uses a small personal feasibility dataset. Results are NOT statistically robust, NOT generalizable, and intended only to validate the technical pipeline.

---

## Test Matrix (Planned)

| Scenario | Description | Expected Samples |
|----------|-------------|------------------|
| Genuine (Same Phrase) | Enrolled speaker, same phrase, multiple recordings | ≥5 |
| Genuine (Different Phrase) | Enrolled speaker, different phrase | ≥2 |
| Impostor (Different Speaker) | Different person, same phrase | ≥3 per impostor |
| Impostor (Replay Attack) | Recording of enrolled speaker replayed via speaker | ≥2 |
| Environmental (Noise) | Genuine speaker with background noise | ≥1 |

---

## Models Under Test

### 1. ECAPA-TDNN (SpeechBrain spkrec-ecapa-voxceleb)

**Verification Status:** Source verified, checkpoint TBD

| Property | Status | Value |
|----------|--------|-------|
| Model Size (on disk) | PENDING | TBD |
| Inference Latency (CPU) | PENDING | TBD |
| Memory (RAM during inference) | PENDING | TBD |
| Embedding Dimension | VERIFIED | 192 |
| Input Format | VERIFIED | 16 kHz mono waveform |
| Framework | VERIFIED | PyTorch |
| License | VERIFIED | Apache 2.0 |
| ONNX Conversion | PENDING | Unknown |
| TFLite Conversion | PENDING | Unknown |
| EER on Feasibility Dataset | PENDING | TBD |
| Android Feasibility | PENDING | Unknown |

### 2. Resemblyzer (CorentinJ/Real-Time-Voice-Cloning)

| Property | Status | Value |
|----------|--------|-------|
| Model Size (on disk) | PENDING | TBD |
| Inference Latency (CPU) | PENDING | TBD |
| Memory (RAM during inference) | PENDING | TBD |
| Embedding Dimension | UNVERIFIED | TBD |
| Input Format | UNVERIFIED | 16 kHz mono waveform |
| Framework | VERIFIED | PyTorch |
| License | VERIFIED | MIT |
| ONNX Conversion | PENDING | Unknown |
| TFLite Conversion | PENDING | Unknown |
| EER on Feasibility Dataset | PENDING | TBD |
| Android Feasibility | PENDING | Unknown |

---

## Results Summary

(To be populated after Milestone 1 experiments complete)

---

## Milestone 2: Face Verification Feasibility

**Status:** NOT STARTED

---

## Milestone 3: Liveness Feasibility

**Status:** NOT STARTED

---
