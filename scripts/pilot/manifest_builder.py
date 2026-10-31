#!/usr/bin/env python3
"""
Manifest Builder for H2 Spatial Tax Pilot (S1_prereg_pilot.md §4, §12)

Creates manifest.json with:
- Model revision hash
- Tokenizer files (list)
- SHA-256 of safetensors file
- Environment specification
- External timestamp DOI (added post-freeze per §12 step 5)
"""

import hashlib
import json
import platform
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List


MODEL_REVISION = "2e1fd397ee46e1388853d2af2c993145b0f1098a"
MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
MODEL_SNAPSHOT_DIR = Path("models/hf_cache/hub/models--Qwen--Qwen2.5-Coder-1.5B-Instruct/snapshots") / MODEL_REVISION

# Tokenizer files per §4
TOKENIZER_FILES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "config.json",
    "generation_config.json",
]

# Safetensors file
SAFETENSORS_FILE = "model.safetensors"


def compute_sha256(filepath: Path) -> str:
    """Compute SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def get_tokenizer_file_info(snapshot_dir: Path) -> List[Dict[str, Any]]:
    """Get info for each tokenizer file."""
    files_info = []
    for fname in TOKENIZER_FILES:
        fpath = snapshot_dir / fname
        if not fpath.exists():
            raise FileNotFoundError(f"Required tokenizer file missing: {fpath}")
        stat = fpath.stat()
        files_info.append({
            "filename": fname,
            "size_bytes": stat.st_size,
            "sha256": compute_sha256(fpath),
        })
    return files_info


def get_safetensors_sha256(snapshot_dir: Path) -> str:
    """Get SHA-256 of model.safetensors."""
    fpath = snapshot_dir / SAFETENSORS_FILE
    if not fpath.exists():
        raise FileNotFoundError(f"Model weights file missing: {fpath}")
    return compute_sha256(fpath)


def get_environment_info() -> Dict[str, Any]:
    """Collect environment information per §4."""
    return {
        "hardware": "CPU only. Intel Core Ultra 7 265K (20 cores; pinned to 16 threads via torch.set_num_threads(16)). No discrete GPU.",
        "operating_system": platform.platform(),
        "python_version": sys.version.split()[0],
        "pytorch_version": "2.15.0+cpu",
        "transformers_version": "5.21.1",
        "tokenizers_version": "0.23.3",
        "safetensors_version": "0.8.0",
        "numpy_version": "2.5.4",
        "precision": "bfloat16 (native model weight precision)",
        "model_storage": "Hugging Face release stored and loaded natively as bfloat16 (dtype=torch.bfloat16)",
        "virtual_environment": ".venv in project folder; requirements.txt pins versions",
    }


def get_pilot_parameters() -> Dict[str, Any]:
    """Get fixed pilot parameters from §3."""
    return {
        "depths": [1, 3, 5],
        "tasks_per_depth": 30,
        "total_test_tasks": 90,
        "conditions": ["A", "B", "C"],
        "total_generations": 270,
        "demonstration_depths": [1, 2, 3],
        "demonstration_seed": 9000,
        "task_seed": 1234,
        "unit_test_inputs": [[], [1, 2, 3], [-1, 5, 0, 2]],
        "statements_per_block": {1: 3, 2: 2, 3: 2, 5: 1},
        "constant_pool": "1-99 unique integers",
        "mutant_catch_rate_threshold": 0.80,
    }


def get_inference_parameters() -> Dict[str, Any]:
    """Get inference parameters from §5."""
    return {
        "model_loading": "Transformers, bfloat16 (dtype=torch.bfloat16), CPU (torch.set_num_threads(16))",
        "prompt_mode": "raw completion (no chat template)",
        "decoding": "greedy (do_sample=False), max_new_tokens=512",
        "seed": 42,
        "determinism_check": "Run 3 demonstration tasks across all 3 conditions (9 prompts) three times; all outputs must be identical",
    }


def build_manifest() -> Dict[str, Any]:
    """Build the complete manifest per §4 and §12."""
    snapshot_dir = MODEL_SNAPSHOT_DIR
    
    manifest = {
        "pilot": "H2 Spatial Tax Pilot",
        "pre_registration_version": "1.0",
        "pre_registration_date": "2026-10-31",
        "status": "Draft",
        "model": {
            "name": MODEL_NAME,
            "revision": MODEL_REVISION,
            "snapshot_directory": str(snapshot_dir),
        },
        "tokenizer_files": get_tokenizer_file_info(snapshot_dir),
        "model_safetensors_sha256": get_safetensors_sha256(snapshot_dir),
        "environment": get_environment_info(),
        "pilot_parameters": get_pilot_parameters(),
        "inference_parameters": get_inference_parameters(),
        "external_timestamp_doi": None,  # Filled in after freeze per §12 step 5
        "created_at": datetime.utcnow().isoformat() + "Z",
        "git_commit": None,  # To be filled at freeze commit
    }
    
    return manifest


def write_manifest(manifest: Dict[str, Any], output_path: Path = Path("manifest.json")):
    """Write manifest to JSON file."""
    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest written to {output_path}")


def verify_manifest(manifest_path: Path = Path("manifest.json")) -> bool:
    """Verify manifest can be read and has required fields."""
    with open(manifest_path) as f:
        data = json.load(f)
    
    required_fields = [
        "pilot", "pre_registration_version", "model", "tokenizer_files",
        "model_safetensors_sha256", "environment", "pilot_parameters",
        "inference_parameters", "external_timestamp_doi", "created_at"
    ]
    
    for field in required_fields:
        if field not in data:
            raise KeyError(f"Manifest verification failed: missing field '{field}'")

    if not data["model_safetensors_sha256"] or len(data["model_safetensors_sha256"]) != 64:
        raise ValueError(f"Manifest verification failed: invalid safetensors SHA-256: {data['model_safetensors_sha256']}")

    if len(data.get("tokenizer_files", [])) < 6:
        raise ValueError(f"Manifest verification failed: expected 6 tokenizer files, found {len(data.get('tokenizer_files', []))}")
    
    print("Manifest verification: PASSED")
    print(f"  Model revision: {data['model']['revision']}")
    print(f"  Safetensors SHA-256: {data['model_safetensors_sha256'][:16]}...")
    print(f"  Tokenizer files: {len(data['tokenizer_files'])}")
    print(f"  Created at: {data['created_at']}")
    return True


def main():
    """Build and write manifest."""
    print("Building manifest per §4...")
    manifest = build_manifest()
    write_manifest(manifest)
    verify_manifest()
    
    # Print key values for verification
    print("\nKey manifest values:")
    print(f"  Model revision: {manifest['model']['revision']}")
    print(f"  Safetensors SHA-256: {manifest['model_safetensors_sha256']}")
    for tf in manifest['tokenizer_files']:
        if tf.get('sha256'):
            print(f"  {tf['filename']}: {tf['sha256'][:16]}... ({tf['size_bytes']} bytes)")


if __name__ == "__main__":
    main()