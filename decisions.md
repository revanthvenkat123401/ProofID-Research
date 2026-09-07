# ProofID-Research: Technical Decision Log

**Last Updated:** 2026-09-07

---

## Decision 1: Speaker Embedding Model Candidate Selection

**Date:** 2026-09-07

**Status:** Candidate evaluation pending

**Decision:**
Research lightweight, open-source speaker embedding models suitable for Android on-device inference.

**Why:**
Voice verification is the highest-risk component of ProofID. A poor model selection will cascade through all downstream work. We must measure actual performance before committing to any model.

**Candidates Under Evaluation:**

### Candidate 1: ECAPA-TDNN (SpeechBrain)

**Source:**
- Repository: `speechbrain/speechbrain`
- GitHub: https://github.com/speechbrain/speechbrain
- License: Apache 2.0
- Model Hub: HuggingFace `speechbrain/spkrec-ecapa-voxceleb`

**Model Metadata (Pre-verification):**
- Framework: PyTorch
- Embedding Dimension: 192 (from hparams: `emb_dim: 192`, `lin_neurons: 192`)
- Input Format: 16 kHz mono audio waveform
- Feature Pipeline: Fbank (80 mel bins)
- Architecture: Emphasized Channel Attention, Propagation, and Aggregation in Time Delay Neural Network
- Checkpoint Format: PyTorch `.ckpt` files

**Known Properties:**
- Trained on VoxCeleb dataset for speaker verification
- Used in multiple SpeechBrain recipes for speaker identification and verification
- ONNX/TFLite conversion: Status unknown (to be tested)
- Inference requirements: Unknown (to be measured)

**Evidence:**
- Source code available in SpeechBrain repository
- Pre-trained checkpoint available through HuggingFace integration
- Used in published research on VoxCeleb benchmarks

**To Verify:**
- Actual model size on disk
- CPU inference latency on target hardware
- RAM consumption during inference
- TFLite conversion feasibility
- ONNX conversion feasibility
- Actual embedding dimension validation
- Performance on genuine vs. impostor samples

**Measured Result:**
(Pending: Milestone 1 experiments)

**Known Limitation:**
Speaker embedding models are vulnerable to voice cloning with real-time AI systems. This is NOT a deepfake detector; it is identity verification at the enrollment point. A fresh challenge-response is important for reducing the risk of simple replay attacks. It does not guarantee protection against real-time voice cloning or all replay scenarios.

**Next Action:**
Implement benchmarking harness. Download model checkpoint. Measure properties.

---

### Candidate 2: Resemblyzer (d-vector style)

**Source:**
- Repository: `CorentinJ/Real-Time-Voice-Cloning`
- GitHub: https://github.com/CorentinJ/Real-Time-Voice-Cloning
- License: MIT
- Model Format: PyTorch

**Model Metadata (Pre-verification):**
- Framework: PyTorch
- Embedding Dimension: UNVERIFIED - to be measured
- Input Format: 16 kHz mono audio waveform
- Architecture: Speaker embedding using d-vector approach (temporal average of speaker-dependent hidden activations)

**Known Properties:**
- Designed for real-time voice cloning scenarios
- Relatively lightweight compared to large transformer-based models
- ONNX/TFLite conversion: Status unknown (to be tested)

**Evidence:**
- Open-source repository with documented usage
- Used in real-time voice cloning research

**To Verify:**
- Actual model size on disk
- CPU inference latency
- RAM consumption
- TFLite conversion feasibility
- Genuine vs. impostor performance
- Actual embedding dimension on inference output

**Measured Result:**
(Pending: Milestone 1 experiments)

**Known Limitation:**
- Lightweight may trade off accuracy for speed
- Unknown generalization to unseen speakers

**Next Action:**
Download and benchmark alongside ECAPA-TDNN.

---

### Candidate 3: X-Vector (SpeechBrain/PyAnnote variants)

**Source:**
- Repository: `speechbrain/speechbrain`
- License: Apache 2.0
- Model Format: PyTorch

**Model Metadata (Pre-verification):**
- Embedding Dimension: Varies (512 in some configurations)
- Input Format: 16 kHz audio
- Architecture: Time-delay neural network (TDNN) variant

**Status:** Secondary candidate (lower priority than ECAPA-TDNN)

**To Verify:**
- Pre-trained checkpoint availability
- Actual properties
- Comparison to ECAPA-TDNN

---

## Decision 2: Benchmark Dataset Scope

**Date:** 2026-09-07

**Decision:**
Collect small, controlled personal audio dataset for feasibility/sanity-check testing only.

**Why:**
We need to quickly validate that:
1. Models load and run on CPU
2. Embeddings are extractable
3. Genuine and impostor samples produce measurably different similarity scores
4. Replay attacks behave predictably

This is NOT a production benchmark or evaluation on a large corpus.

**Scope:**
- 1-2 enrolled speakers
- 3-4 impostor speakers
- ~5-10 utterances per speaker (varies by speaker)
- Small dataset used for feasibility only

**Important Limitations:**
- Results from a personal dataset are NOT statistically robust
- EER (Equal Error Rate) estimates will not generalize
- No claim of production-grade accuracy
- No claim of general population performance
- Purpose: Verify the technical pipeline works, not to measure security

**Storage:**
- Do NOT commit audio files to GitHub
- Do NOT commit biometric embeddings to GitHub
- Commit only anonymized aggregate measurements and analysis

**Next Action:**
Create `.gitignore` rules for sensitive data.

---

## Decision 3: Similarity Metric

**Date:** 2026-09-07

**Decision:**
Use cosine similarity for all embedding comparisons, as implemented in SpeechBrain's `SpeakerRecognition.verify_batch()`.

**Why:**
- Cosine similarity is standard for speaker embeddings
- Computationally lightweight
- SpeechBrain's implementation uses this by default

**Measured Result:**
(Pending: Baseline from ECAPA-TDNN implementation)

**Next Action:**
Verify cosine similarity produces expected score distributions on feasibility dataset.

---
