# Milestone 1: Speaker Verification Feasibility Research

**Objective:**
Determine if a lightweight speaker embedding model can reliably distinguish an enrolled speaker from impostors in controlled conditions.

**Research Question:**
"Can a lightweight speaker embedding model reliably distinguish the enrolled speaker from an impostor under controlled conditions?"

**Important Scope:**
This research uses a small personal audio dataset for technical feasibility/sanity-check only. Results are NOT production-grade, NOT statistically robust, and NOT generalizable to a population. The purpose is to validate that:
1. The model loads and runs on CPU
2. Embeddings are extractable and comparable
3. Genuine and impostor samples produce measurably different similarity scores
4. Replay attack behavior is observable and documented

**Deliverables:**
1. Model download and checkpoint verification
2. Measured properties (model size, latency, memory)
3. Genuine vs. impostor similarity scores (feasibility dataset only)
4. Replay attack behavior observation
5. Android conversion feasibility assessment
6. Known limitations and caveats
7. Recommendation: proceed to next milestone or pivot

---

## Experiment Design

### Phase 1: Model Candidate Acquisition and Verification

**Objective:** Download candidate models, verify checkpoint integrity, record exact specifications.

**For each candidate model:**

1. Identify exact source and checkpoint
2. Download model checkpoint
3. Verify license compliance
4. Record exact model properties:
   - File size (on disk)
   - Framework and format
   - Embedding dimension (validate against spec)
   - Input requirements (sample rate, format)
5. Test basic inference (forward pass) on CPU
6. Record actual inference time
7. Document findings in `results/01_model_verification.md`

**Output:** `results/01_model_verification.md`

---

### Phase 2: Audio Dataset Collection (Feasibility Scope)

**Objective:** Collect small personal audio dataset for technical validation only.

**Dataset Specifications:**

- **Speakers:** 2 enrolled speakers (A, B) + 2-3 impostor speakers (C, D, E)
- **Enrollment:** 5-10 utterances per enrolled speaker
- **Verification:** 2-3 new utterances per enrolled speaker
- **Impostor:** 3-5 utterances per impostor speaker
- **Replay:** 2 replay-attack samples per enrolled speaker
- **Format:** 16-bit PCM, 16 kHz sample rate, mono
- **Storage:** Do NOT commit to GitHub

**Recording Protocol:**

```
data/
├── speaker_a/
│   ├── enrollment_01.wav
│   ├── enrollment_02.wav
│   ├── verification_01.wav
│   ├── replay_attack.wav
│   └── README.txt (metadata: date, environment, notes)
├── speaker_b/
│   └── (similar structure)
├── impostor_c/
│   ├── impostor_01.wav
│   └── ...
└── README.md (dataset summary)
```

**Important:**
- Do NOT store face images or biometric video
- Do NOT retain audio files longer than experiment duration
- Do NOT commit audio to repository
- Record only in .gitignore

---

### Phase 3: Similarity Score Collection

**Objective:** Extract embeddings and compute similarity scores on feasibility dataset.

**Important Note:** This is a small feasibility experiment for technical validation. Results are NOT statistically robust and will NOT generalize to a population.

**Enrollment Representation (Multi-Utterance Aggregation):**

Instead of using a single enrollment sample, build a robust enrolled speaker representation from multiple utterances:

1. Load all enrollment utterances for Speaker A (e.g., enrollment_01.wav through enrollment_10.wav)
2. Extract embedding for each enrollment utterance using the candidate model
3. Aggregate embeddings using **mean embedding + L2 normalization**:
   ```
   enrolled_embedding = mean([emb_01, emb_02, ..., emb_10])
   enrolled_embedding = enrolled_embedding / ||enrolled_embedding||_2
   ```
4. Document the exact aggregation method and number of enrollment samples used

**Verification Against Enrolled Representation:**

For each candidate model:

1. Use the aggregated enrolled_embedding (from above)
2. Extract embeddings from all verification samples of speaker A
   - Compute cosine similarity scores (genuine samples)
3. Extract embeddings from all impostor samples (C, D, E)
   - Compute cosine similarity scores (impostor samples)
4. Extract embedding from replay-attack sample
   - Compute cosine similarity score
5. Collect all scores

**Output:**
- `results/02_similarity_scores_ecapa.csv` (columns: model, sample_type, speaker, score)
- `results/02_similarity_distribution.png` (histogram plot of genuine vs. impostor scores)
- `results/02_aggregation_metadata.json` (enrollment method, utterance count, normalization approach)

**Output Format Example:**

```csv
model,sample_type,speaker,utterance_count,score
ecapa_tdnn,genuine,a,10,0.92
ecapa_tdnn,genuine,a,10,0.89
ecapa_tdnn,impostor,c,10,0.15
ecapa_tdnn,impostor,c,10,0.22
ecapa_tdnn,replay,a,10,0.35
```

**Metadata Example (02_aggregation_metadata.json):**

```json
{
  "enrollment_representation": {
    "model": "ecapa_tdnn",
    "aggregation_method": "mean_embedding_with_l2_norm",
    "enrollment_utterances": 10,
    "utterance_list": [
      "speaker_a/enrollment_01.wav",
      "speaker_a/enrollment_02.wav",
      "...",
      "speaker_a/enrollment_10.wav"
    ],
    "embedded_dimension": 192,
    "normalization": "L2 (Euclidean) normalization after mean aggregation"
  },
  "note": "This is a small feasibility experiment. Enrollment with 10 utterances is reasonable for this sanity-check phase but NOT sufficient for production. Results will NOT generalize to population."
}
```

---

### Phase 4: Replay Attack Observation

**Objective:** Document replay attack behavior.

**Procedure:**

1. For enrolled speaker A:
   - Record original audio file (enrollment_01.wav)
   - Play back recording through speaker at normal volume
   - Re-record with microphone from different distance/angle
   - Extract embedding from replay-attack recording
   - Compute similarity to enrolled speaker embedding

2. Compare:
   - Original genuine similarity distribution (Phase 3)
   - Replay-attack similarity score
   - Document observations

**Output:** `results/03_replay_observations.md`

**Example Observations:**
- "Replay recording produced similarity score of 0.45, within impostor range"
- "Replay contained noticeable audio artifacts"
- "Replay behaved more like impostor than genuine sample"

---

### Phase 5: Results Analysis

**Objective:** Aggregate findings and generate recommendation.

**Important Disclaimer:** This feasibility dataset is small and personal. Results are NOT statistically robust, NOT production-grade, and will NOT generalize to a broader population. The purpose is to validate that the technical pipeline works, not to measure security or accuracy.

**Analysis Steps:**

1. Calculate statistics from feasibility dataset:
   - Mean genuine similarity: X ± σ
   - Mean impostor similarity: Y ± σ
   - Score separation: X - Y
   - Observed overlap (if any)
   - **Explicitly note:** "These statistics are from a feasibility dataset (N=~5-10 genuine, N=~3-5 impostor per speaker, 2-3 speakers). They do NOT estimate population-level performance."

2. Assess model properties:
   - Inference latency acceptable? (target: <500ms, engineering goal only)
   - Model size manageable? (target: <50MB, engineering goal only)
   - Memory consumption reasonable? (target: <100MB RAM, engineering goal only)

3. Record Android conversion status:
   - ONNX conversion: Possible / Difficult / Not attempted
   - TFLite conversion: Possible / Difficult / Not attempted
   - Expected mobile latency: TBD after conversion

4. Document limitations explicitly:
   - Dataset size and composition (personal, small)
   - No population-level generalization (tiny N)
   - Replay attack only tested on personal audio (limited scope)
   - No cross-speaker robustness testing
   - No claim of security or accuracy
   - Aggregation method affects results (mean + L2 norm may not be optimal, but reasonable for feasibility check)

**Output:** `results/MILESTONE_1_REPORT.md`

**Report Structure:**

```markdown
# Milestone 1 Report: Speaker Verification Feasibility

## Disclaimer
This report documents a small feasibility/sanity-check experiment. Results are NOT statistically robust, NOT production-grade, and do NOT estimate population-level performance or security guarantees.

## Models Tested
- ECAPA-TDNN
- Resemblyzer
- (others as applicable)

## Methodology
- Enrollment: Multi-utterance aggregation (mean embedding + L2 norm)
- Verification: Cosine similarity against enrolled representation
- Dataset: Personal, small (2 enrolled speakers, 3-4 impostors, ~5-10 utterances per speaker)

## Results (Feasibility Dataset Only)
[Similarity score distributions, separation metrics, latency measurements]

## Known Limitations
[List all limitations]

## Recommendation
- Proceed to Milestone 2: Face Verification
- OR: Pivot to alternative speaker verification approach
- OR: Proceed with face + liveness pipeline (voice verification TBD)
```

---

## Critical Constraints

⚠️ **DO NOT:**
- Commit audio files to GitHub
- Commit biometric embeddings to GitHub
- Commit face images to GitHub
- Use paid APIs or cloud services
- Claim production readiness
- Claim general population accuracy
- Ignore inconvenient results

✅ **DO:**
- Measure actual numbers from experiments
- Document all limitations
- Record decision rationale with evidence
- Preserve feasibility dataset locally (not in GitHub)
- Focus on technical validation, not security claims

---

## Timeline

- Phase 1 (Model Acquisition): 30 mins - 1 hour
- Phase 2 (Data Collection): 1-2 hours
- Phase 3 (Similarity Scoring): 30 mins - 1 hour
- Phase 4 (Replay Testing): 30 mins - 1 hour
- Phase 5 (Analysis & Report): 1-2 hours

**Total:** ~5-7 hours (depends on data collection and iteration)

---

## Success Criteria (Feasibility Check)

At completion, Milestone 1 is successful if:

1. ✅ At least one model loads and runs on CPU
2. ✅ Embeddings are extractable and valid
3. ✅ Genuine and impostor samples show measurable score separation
4. ✅ Replay attack behavior is observed and documented
5. ✅ Android conversion status is assessed
6. ✅ Exact model specifications are recorded
7. ✅ All limitations are clearly documented
8. ✅ Decision: Proceed to Milestone 2 or pivot to alternative approach

**Failure Mode:** If no model produces clear separation, recommend alternative architectures or proceed with backup pipeline (face + liveness only, without voice).

---
