#!/usr/bin/env python3
"""
Phase 1: Model Candidate Download and Verification

Objective:
Download and verify speaker embedding model candidates.
Measure actual properties: model size, embedding dimension, inference latency, memory.

Candidates:
1. ECAPA-TDNN (SpeechBrain spkrec-ecapa-voxceleb)
2. Resemblyzer (CorentinJ/Real-Time-Voice-Cloning)

Do NOT claim security, accuracy, or generalization.
This is a feasibility/sanity-check phase only.

Synthetic test audio is created in /tmp, not stored in repo.
"""

import os
import sys
import json
import time
import warnings
import traceback
import tempfile
from pathlib import Path
import numpy as np

# Optional psutil import for memory measurement
try:
    import psutil
except ImportError:
    psutil = None

# Suppress non-critical warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# ============================================================================
# CONFIGURATION
# ============================================================================

RESEARCH_ROOT = Path(__file__).parent.parent
RESULTS_DIR = RESEARCH_ROOT / 'results'
MODELS_DIR = RESEARCH_ROOT / 'models'

# Ensure directories exist
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

VERIFICATION_REPORT = RESULTS_DIR / '01_model_verification.md'
MEASUREMENTS_JSON = RESULTS_DIR / '01_model_measurements.json'

# Synthetic test audio created in system temp, NOT in repo
TEST_AUDIO_TEMP_DIR = tempfile.gettempdir()

print("="*80)
print("PHASE 1: MODEL CANDIDATE DOWNLOAD AND VERIFICATION")
print("="*80)
print(f"\nResearch Root: {RESEARCH_ROOT}")
print(f"Results Dir: {RESULTS_DIR}")
print(f"Models Dir: {MODELS_DIR}")
print(f"Temp Audio Dir (not in repo): {TEST_AUDIO_TEMP_DIR}")

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_process_memory_mb():
    """
    Get current process memory in MB.
    Uses psutil if available, otherwise returns None (TBD).
    """
    if psutil is None:
        return None
    try:
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        return mem_info.rss / (1024 * 1024)  # Convert to MB
    except:
        return None

def create_synthetic_test_audio(duration_seconds=1.0, sample_rate=16000):
    """
    Create a synthetic test audio signal for pipeline validation.
    Stored in system temp directory, NOT in repo.
    
    WARNING: This is NOT real speech.
    It is used only to validate that the model pipeline works.
    Do NOT use this for any speaker verification claims.
    
    Returns: (audio_path, success, message)
    """
    try:
        import scipy.io.wavfile as wavfile
        
        # Generate a simple sine wave at 440 Hz (A4 note)
        t = np.linspace(0, duration_seconds, int(sample_rate * duration_seconds), False)
        frequency = 440  # Hz
        signal = 0.3 * np.sin(2 * np.pi * frequency * t)
        
        # Convert to 16-bit PCM
        signal_int16 = np.int16(signal * 32767)
        
        # Write to temp directory (NOT repo)
        temp_audio_path = Path(TEST_AUDIO_TEMP_DIR) / 'proofid_phase1_test_audio_TEMP.wav'
        wavfile.write(temp_audio_path, sample_rate, signal_int16)
        file_size_kb = temp_audio_path.stat().st_size / 1024
        
        return temp_audio_path, True, f"Created synthetic test signal in temp ({file_size_kb:.1f} KB)"
    except Exception as e:
        return None, False, f"Failed to create synthetic test audio: {e}"

def load_synthetic_audio(audio_path, sample_rate=16000):
    """
    Load synthetic audio file.
    """
    try:
        import scipy.io.wavfile as wavfile
        loaded_sr, audio_data = wavfile.read(audio_path)
        
        if loaded_sr != sample_rate:
            return None, f"Sample rate mismatch: expected {sample_rate}, got {loaded_sr}"
        
        if audio_data.dtype != np.int16:
            audio_data = np.int16(audio_data)
        
        # Convert to float [-1, 1]
        audio_float = audio_data.astype(np.float32) / 32768.0
        
        return audio_float, None
    except Exception as e:
        return None, f"Failed to load audio: {e}"

def get_file_size_mb(file_path):
    """Get file size in MB."""
    try:
        return Path(file_path).stat().st_size / (1024 * 1024)
    except:
        return None

def cleanup_temp_audio(audio_path):
    """
    Delete temporary audio file.
    """
    try:
        if audio_path and Path(audio_path).exists():
            Path(audio_path).unlink()
            return True, f"Cleaned up temp audio: {audio_path}"
    except Exception as e:
        return False, f"Failed to clean up temp audio: {e}"
    return False, "No temp audio to clean up"

# ============================================================================
# CANDIDATE 1: ECAPA-TDNN (SpeechBrain)
# ============================================================================

def verify_ecapa_tdnn():
    """
    Verify ECAPA-TDNN model from SpeechBrain.
    
    Source: speechbrain/speechbrain
    Model Hub: HuggingFace speechbrain/spkrec-ecapa-voxceleb
    """
    print("\n" + "="*80)
    print("CANDIDATE 1: ECAPA-TDNN (SpeechBrain)")
    print("="*80)
    
    results = {
        "model_name": "ECAPA-TDNN",
        "source": "SpeechBrain",
        "source_url": "https://github.com/speechbrain/speechbrain",
        "model_hub": "HuggingFace",
        "model_hub_id": "speechbrain/spkrec-ecapa-voxceleb",
        "model_hub_url": "https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb",
        "code_license": "Apache 2.0",
        "checkpoint_license_status": "UNVERIFIED - requires manual check of HuggingFace model card",
        "loading_status": "PENDING",
        "inference_status": "PENDING",
        "errors": [],
        "measurements": {}
    }
    
    temp_audio_path = None
    
    try:
        print("\n[1] Attempting to import SpeechBrain...")
        mem_before = get_process_memory_mb()
        print(f"    Memory before import: {mem_before:.1f} MB" if mem_before else "    Memory: TBD")
        
        from speechbrain.inference.speaker import SpeakerRecognition
        print("    ✓ SpeechBrain imported successfully")
        
        print("\n[2] Attempting to load model checkpoint...")
        print("    Model ID: speechbrain/spkrec-ecapa-voxceleb")
        
        # Create a temporary directory for model download
        model_cache_dir = MODELS_DIR / 'ecapa_tdnn_cache'
        model_cache_dir.mkdir(parents=True, exist_ok=True)
        
        mem_before_load = get_process_memory_mb()
        start_time = time.time()
        
        # Load the model
        classifier = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=str(model_cache_dir)
        )
        
        load_time = time.time() - start_time
        mem_after_load = get_process_memory_mb()
        
        print(f"    ✓ Model loaded in {load_time:.2f}s")
        results["measurements"]["load_time_seconds"] = load_time
        results["loading_status"] = "LOADED / PIPELINE VERIFIED"
        
        if mem_before_load and mem_after_load:
            mem_delta = mem_after_load - mem_before_load
            print(f"    Memory before: {mem_before_load:.1f} MB, after: {mem_after_load:.1f} MB (delta: {mem_delta:+.1f} MB)")
            results["measurements"]["memory_before_load_mb"] = mem_before_load
            results["measurements"]["memory_after_load_mb"] = mem_after_load
            results["measurements"]["memory_delta_mb"] = mem_delta
        else:
            print(f"    Memory: TBD")
            results["measurements"]["memory_mb"] = "TBD"
        
        print("\n[3] Testing inference on synthetic audio...")
        
        # Create synthetic test audio in temp
        audio_path, success, msg = create_synthetic_test_audio()
        if not success:
            print(f"    Warning: {msg}")
            results["errors"].append(msg)
        else:
            temp_audio_path = audio_path
            print(f"    ✓ {msg}")
        
        # Load and test on synthetic audio
        if temp_audio_path and Path(temp_audio_path).exists():
            audio, err = load_synthetic_audio(temp_audio_path)
            if audio is not None:
                print(f"    Audio loaded: shape {audio.shape}, dtype {audio.dtype}")
                
                # Convert to torch tensor
                import torch
                audio_tensor = torch.from_numpy(audio).float()
                
                print(f"    Running inference...")
                mem_before_infer = get_process_memory_mb()
                start_time = time.time()
                
                with torch.no_grad():
                    embedding = classifier.encode_batch(audio_tensor.unsqueeze(0))
                
                inference_time = time.time() - start_time
                mem_after_infer = get_process_memory_mb()
                embedding_np = embedding.cpu().numpy()
                
                print(f"    ✓ Inference completed in {inference_time:.4f}s")
                print(f"    Embedding shape: {embedding_np.shape}")
                print(f"    Embedding dtype: {embedding_np.dtype}")
                
                results["measurements"]["inference_time_seconds"] = inference_time
                results["measurements"]["embedding_shape"] = list(embedding_np.shape)
                results["measurements"]["embedding_dtype"] = str(embedding_np.dtype)
                results["measurements"]["embedding_dimension"] = int(embedding_np.shape[-1])
                results["measurements"]["embedding_min"] = float(embedding_np.min())
                results["measurements"]["embedding_max"] = float(embedding_np.max())
                results["inference_status"] = "INFERENCE COMPLETED / PIPELINE VERIFIED"
                
                if mem_before_infer and mem_after_infer:
                    mem_delta = mem_after_infer - mem_before_infer
                    print(f"    Memory before inference: {mem_before_infer:.1f} MB, after: {mem_after_infer:.1f} MB (delta: {mem_delta:+.1f} MB)")
                    results["measurements"]["memory_before_inference_mb"] = mem_before_infer
                    results["measurements"]["memory_after_inference_mb"] = mem_after_infer
                    results["measurements"]["memory_inference_delta_mb"] = mem_delta
            else:
                error_msg = f"Failed to load audio: {err}"
                results["errors"].append(error_msg)
                print(f"    Error: {error_msg}")
        
        print("\n[4] Checking checkpoint size...")
        checkpoint_files = list(model_cache_dir.glob('**/*.ckpt'))
        if checkpoint_files:
            total_size_mb = sum(get_file_size_mb(f) for f in checkpoint_files)
            print(f"    Found {len(checkpoint_files)} checkpoint file(s)")
            print(f"    Total size: {total_size_mb:.2f} MB")
            results["measurements"]["checkpoint_count"] = len(checkpoint_files)
            results["measurements"]["checkpoint_total_size_mb"] = total_size_mb
            
            for ckpt_file in checkpoint_files:
                size_mb = get_file_size_mb(ckpt_file)
                print(f"      - {ckpt_file.name}: {size_mb:.2f} MB")
        
        print("\n✓ ECAPA-TDNN verification COMPLETE")
        
    except ImportError as e:
        error_msg = f"SpeechBrain not installed: {e}"
        print(f"\n✗ Error: {error_msg}")
        results["errors"].append(error_msg)
        results["loading_status"] = "FAILED - IMPORT ERROR"
        
    except Exception as e:
        error_msg = f"Verification failed: {e}"
        print(f"\n✗ Error: {error_msg}")
        print(traceback.format_exc())
        results["errors"].append(error_msg)
        results["loading_status"] = "FAILED"
    
    finally:
        # Clean up temp audio
        if temp_audio_path:
            success, msg = cleanup_temp_audio(temp_audio_path)
            print(f"\n[Cleanup] {msg}")
    
    return results

# ============================================================================
# CANDIDATE 2: RESEMBLYZER
# ============================================================================

def verify_resemblyzer():
    """
    Verify Resemblyzer model.
    
    Source: CorentinJ/Real-Time-Voice-Cloning
    Model Hub: GitHub (not on standard hub)
    """
    print("\n" + "="*80)
    print("CANDIDATE 2: RESEMBLYZER (d-vector)")
    print("="*80)
    
    results = {
        "model_name": "Resemblyzer",
        "source": "CorentinJ/Real-Time-Voice-Cloning",
        "source_url": "https://github.com/CorentinJ/Real-Time-Voice-Cloning",
        "model_hub": "GitHub (local repo)",
        "model_hub_id": "N/A - requires local installation",
        "code_license": "MIT",
        "checkpoint_license_status": "UNVERIFIED - requires manual check of repository",
        "loading_status": "PENDING",
        "inference_status": "PENDING",
        "errors": [],
        "measurements": {}
    }
    
    temp_audio_path = None
    
    try:
        print("\n[1] Attempting to import Resemblyzer...")
        mem_before = get_process_memory_mb()
        print(f"    Memory before import: {mem_before:.1f} MB" if mem_before else "    Memory: TBD")
        
        # Try standard import
        try:
            from resemblyzer import VoiceEncoder, preprocess_wav
            print("    ✓ Resemblyzer imported successfully")
        except ImportError:
            # Try alternative import
            try:
                from encoder import VoiceEncoder
                from utils import preprocess_wav
                print("    ✓ Resemblyzer modules imported (alternative path)")
            except ImportError as e:
                raise ImportError(f"Resemblyzer not installed and not found in sys.path: {e}")
        
        print("\n[2] Attempting to load voice encoder...")
        
        mem_before_load = get_process_memory_mb()
        start_time = time.time()
        encoder = VoiceEncoder()
        load_time = time.time() - start_time
        mem_after_load = get_process_memory_mb()
        
        print(f"    ✓ VoiceEncoder loaded in {load_time:.2f}s")
        results["measurements"]["load_time_seconds"] = load_time
        results["loading_status"] = "LOADED / PIPELINE VERIFIED"
        
        if mem_before_load and mem_after_load:
            mem_delta = mem_after_load - mem_before_load
            print(f"    Memory before: {mem_before_load:.1f} MB, after: {mem_after_load:.1f} MB (delta: {mem_delta:+.1f} MB)")
            results["measurements"]["memory_before_load_mb"] = mem_before_load
            results["measurements"]["memory_after_load_mb"] = mem_after_load
            results["measurements"]["memory_delta_mb"] = mem_delta
        else:
            print(f"    Memory: TBD")
            results["measurements"]["memory_mb"] = "TBD"
        
        print("\n[3] Checking encoder properties...")
        
        # Get encoder information
        if hasattr(encoder, 'embed_size'):
            embedding_dim = encoder.embed_size
            print(f"    Embedding dimension: {embedding_dim}")
            results["measurements"]["embedding_dimension"] = embedding_dim
        
        print("\n[4] Testing inference on synthetic audio...")
        
        # Create synthetic test audio in temp
        audio_path, success, msg = create_synthetic_test_audio()
        if not success:
            print(f"    Warning: {msg}")
            results["errors"].append(msg)
        else:
            temp_audio_path = audio_path
            print(f"    ✓ {msg}")
        
        # Load and test on synthetic audio
        if temp_audio_path and Path(temp_audio_path).exists():
            audio, err = load_synthetic_audio(temp_audio_path)
            if audio is not None:
                print(f"    Audio loaded: shape {audio.shape}, dtype {audio.dtype}")
                
                print(f"    Running inference...")
                mem_before_infer = get_process_memory_mb()
                start_time = time.time()
                
                try:
                    # Resemblyzer expects audio in a specific format
                    embedding = encoder.embed_utterance(audio)
                    inference_time = time.time() - start_time
                    mem_after_infer = get_process_memory_mb()
                    
                    print(f"    ✓ Inference completed in {inference_time:.4f}s")
                    print(f"    Embedding shape: {embedding.shape}")
                    print(f"    Embedding dtype: {embedding.dtype}")
                    
                    results["measurements"]["inference_time_seconds"] = inference_time
                    results["measurements"]["embedding_shape"] = list(embedding.shape)
                    results["measurements"]["embedding_dtype"] = str(embedding.dtype)
                    results["measurements"]["embedding_dimension"] = int(embedding.shape[-1])
                    results["measurements"]["embedding_min"] = float(embedding.min())
                    results["measurements"]["embedding_max"] = float(embedding.max())
                    results["inference_status"] = "INFERENCE COMPLETED / PIPELINE VERIFIED"
                    
                    if mem_before_infer and mem_after_infer:
                        mem_delta = mem_after_infer - mem_before_infer
                        print(f"    Memory before inference: {mem_before_infer:.1f} MB, after: {mem_after_infer:.1f} MB (delta: {mem_delta:+.1f} MB)")
                        results["measurements"]["memory_before_inference_mb"] = mem_before_infer
                        results["measurements"]["memory_after_inference_mb"] = mem_after_infer
                        results["measurements"]["memory_inference_delta_mb"] = mem_delta
                except Exception as e:
                    error_msg = f"Inference failed: {e}"
                    print(f"    Error during inference: {error_msg}")
                    results["errors"].append(error_msg)
                    results["inference_status"] = "INFERENCE FAILED"
            else:
                error_msg = f"Failed to load audio: {err}"
                print(f"    Error: {error_msg}")
                results["errors"].append(error_msg)
        
        print("\n✓ RESEMBLYZER verification COMPLETE")
        
    except ImportError as e:
        error_msg = f"Resemblyzer not installed: {e}"
        print(f"\n✗ Error: {error_msg}")
        results["errors"].append(error_msg)
        results["loading_status"] = "FAILED - IMPORT ERROR"
        
    except Exception as e:
        error_msg = f"Verification failed: {e}"
        print(f"\n✗ Error: {error_msg}")
        print(traceback.format_exc())
        results["errors"].append(error_msg)
        results["loading_status"] = "FAILED"
    
    finally:
        # Clean up temp audio
        if temp_audio_path:
            success, msg = cleanup_temp_audio(temp_audio_path)
            print(f"\n[Cleanup] {msg}")
    
    return results

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("\n" + "#"*80)
    print("# PHASE 1: MODEL CANDIDATE VERIFICATION")
    print("# This is a feasibility/sanity-check experiment only.")
    print("# Do NOT claim security, accuracy, or generalization from these results.")
    print("#"*80)
    
    all_results = {}
    
    # Verify ECAPA-TDNN
    print("\n[PHASE 1.1] Verifying ECAPA-TDNN...")
    ecapa_results = verify_ecapa_tdnn()
    all_results["ecapa_tdnn"] = ecapa_results
    
    # Verify Resemblyzer
    print("\n[PHASE 1.2] Verifying Resemblyzer...")
    resemblyzer_results = verify_resemblyzer()
    all_results["resemblyzer"] = resemblyzer_results
    
    # Save results
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    # Save as JSON for machine readability
    print(f"\nSaving measurements to: {MEASUREMENTS_JSON}")
    with open(MEASUREMENTS_JSON, 'w') as f:
        json.dump(all_results, f, indent=2)
    print("✓ JSON saved")
    
    # Generate markdown report
    print(f"\nGenerating verification report: {VERIFICATION_REPORT}")
    
    report_content = generate_verification_report(all_results)
    with open(VERIFICATION_REPORT, 'w') as f:
        f.write(report_content)
    print("✓ Markdown report saved")
    
    # Print summary
    print("\n" + "="*80)
    print("PHASE 1 SUMMARY")
    print("="*80)
    
    for model_name, model_results in all_results.items():
        loading_status = model_results["loading_status"]
        inference_status = model_results["inference_status"]
        print(f"\n{model_name.upper()}")
        print(f"  Loading: {loading_status}")
        print(f"  Inference: {inference_status}")
        if model_results["errors"]:
            print(f"  Errors:")
            for err in model_results["errors"]:
                print(f"    - {err}")
        else:
            print(f"  Measurements:")
            for key, value in model_results["measurements"].items():
                print(f"    - {key}: {value}")
    
    print("\n" + "="*80)
    print("✓ PHASE 1 COMPLETE")
    print("="*80)

def generate_verification_report(all_results):
    """
    Generate a markdown verification report.
    """
    report = """# Model Candidate Verification Report (Phase 1)

**Date:** 2026-09-07  
**Status:** Feasibility/Sanity-Check Phase  
**Disclaimer:** This is a small feasibility experiment. Results do NOT estimate population performance or provide security guarantees.

---

## Overview

Phase 1 verifies that documented speaker embedding model candidates can be:
1. Downloaded and loaded locally
2. Executed on CPU with synthetic test audio
3. Produce embeddings with measurable properties

**Test Audio:** Synthetic sine wave (440 Hz, 1 second, 16 kHz, 16-bit PCM) created in system temp directory (NOT stored in repo)  
**Purpose:** Pipeline validation only - NOT for speaker verification performance testing

---

"""
    
    for model_name, model_results in all_results.items():
        report += f"\n## {model_results['model_name']}\n\n"
        report += f"**Source:** {model_results['source']}  \n"
        report += f"**Repository:** {model_results['source_url']}  \n"
        report += f"**Model Hub:** {model_results['model_hub']}  \n\n"
        
        report += f"### Status\n\n"
        report += f"- **Loading Status:** {model_results['loading_status']}\n"
        report += f"- **Inference Status:** {model_results['inference_status']}\n\n"
        
        report += f"### License\n\n"
        report += f"- **Code License:** {model_results['code_license']}\n"
        report += f"- **Checkpoint License:** {model_results['checkpoint_license_status']}\n\n"
        
        if model_results["errors"]:
            report += f"### Errors\n\n"
            for err in model_results["errors"]:
                report += f"- {err}\n"
            report += "\n"
        
        if model_results["measurements"]:
            report += f"### Measured Properties\n\n"
            report += "| Property | Value |\n"
            report += "|----------|-------|\n"
            for key, value in model_results["measurements"].items():
                report += f"| {key} | {value} |\n"
            report += "\n"
    
    report += """---

## Important Notes

⚠️ **This is a feasibility experiment ONLY**

- Test audio is SYNTHETIC (not real speech)
- Results do NOT estimate speaker verification performance
- Results do NOT claim security or accuracy
- Embedding similarity has NOT been tested
- Genuine vs. impostor separation has NOT been measured
- This validates only that the models can load and produce embeddings
- Temporary test audio files were created in system temp and deleted after testing (NOT stored in repo)

---

## Next Steps

1. Collect real audio data for genuine vs. impostor testing (Phase 2-3)
2. Measure similarity distributions (Phase 3)
3. Test replay attack behavior (Phase 4)
4. Analyze results and make recommendations (Phase 5)

"""
    
    return report

if __name__ == "__main__":
    main()
