#!/usr/bin/env python3
"""
Neuromorphic Benchmark Suite
============================
Systematic A/B testing for GRAIL-V paper statistical validation.

Test Matrix (per Grok recommendation):
- 3-4 different characters/subjects
- 4-5 emotional arcs
- 3-5 source images
- 5 seeds per test (for variance measurement)

Goal: Demonstrate 20% efficiency gain with statistical significance.
"""

import os
import json
import time
import requests
from dataclasses import dataclass
from typing import List, Dict, Tuple
from datetime import datetime
from seed_scaffolding import CognitiveSeedScaffolder, CognitiveFunction

COMFYUI_SERVER = "http://192.168.0.133:8188"
OUTPUT_DIR = "/home/scott/grail_paper/benchmark_results"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)


@dataclass
class TestCase:
    """Single test case configuration"""
    name: str
    image_path: str
    stock_prompt: str
    neuro_prompt: str
    subject_type: str  # woman, man, child, etc.
    emotional_arc: str  # realization, defiance, warmth, etc.


# =============================================================================
# TEST CASES
# =============================================================================

TEST_CASES = [
    # --- SOPHIA VICTORIAN (Woman, Portrait) ---
    TestCase(
        name="sophia_realization",
        image_path="/home/scott/sophia_victorian_frame.png",
        stock_prompt="Victorian woman portrait, subtle head movement, slight smile, blinking eyes, warm lighting",
        neuro_prompt="The young woman's eyes brighten with quiet realization, a knowing smile forming as inspiration takes hold, warmth spreading across her expression",
        subject_type="woman",
        emotional_arc="realization"
    ),
    TestCase(
        name="sophia_contemplation",
        image_path="/home/scott/sophia_victorian_frame.png",
        stock_prompt="Victorian woman portrait, looking thoughtful, gentle movements, soft lighting",
        neuro_prompt="Her gaze turns inward with deep contemplation, a subtle shift from curiosity to understanding, quiet wisdom settling in her features",
        subject_type="woman",
        emotional_arc="contemplation"
    ),
    TestCase(
        name="sophia_determination",
        image_path="/home/scott/sophia_victorian_frame.png",
        stock_prompt="Victorian woman portrait, serious expression, focused look, slight movement",
        neuro_prompt="Quiet determination hardens in her eyes, jaw setting with newfound resolve, inner fire building behind composed exterior",
        subject_type="woman",
        emotional_arc="determination"
    ),

    # --- ELYAN LABS (Two characters - test sequential arcs) ---
    TestCase(
        name="elyan_sophia_focus",
        image_path="/home/scott/grail_paper/sophia_elyan_labs.png",
        stock_prompt="Victorian exhibition, woman working on machine, man watching, gaslight flickering",
        neuro_prompt="The young woman works with fierce concentration, confident hands moving with purpose, quiet authority radiating as she masters the brass machinery",
        subject_type="woman_focus",
        emotional_arc="confidence"
    ),
    TestCase(
        name="elyan_claude_focus",
        image_path="/home/scott/grail_paper/sophia_elyan_labs.png",
        stock_prompt="Victorian exhibition, older man gesturing, woman at machine, warm lighting",
        neuro_prompt="The older gentleman's skepticism softens to grudging respect, pride wounded but giving way to reluctant admiration",
        subject_type="man_focus",
        emotional_arc="respect"
    ),

    # --- DEBATE SCENE (Dynamic interaction) ---
    TestCase(
        name="debate_passion",
        image_path="/home/scott/sophia_claude_i2v_debate_preview.png",
        stock_prompt="Two people in conversation, gesturing, fireplace glowing, Victorian study",
        neuro_prompt="Passionate intellectual exchange, conviction burning in their eyes, the electricity of clashing ideas filling the air between them",
        subject_type="interaction",
        emotional_arc="passion"
    ),
    TestCase(
        name="debate_tension",
        image_path="/home/scott/sophia_claude_i2v_debate_preview.png",
        stock_prompt="Two people talking, subtle movements, warm firelight, period room",
        neuro_prompt="Tension crackling between them, unspoken challenge in their gazes, the air thick with intellectual rivalry",
        subject_type="interaction",
        emotional_arc="tension"
    ),
]

# Emotional arc vocabulary for analysis
EMOTIONAL_ARCS = {
    "realization": ["brighten", "dawning", "understanding", "clarity", "inspiration"],
    "contemplation": ["inward", "thoughtful", "wisdom", "quiet", "settling"],
    "determination": ["resolve", "fire", "hardening", "strength", "purpose"],
    "confidence": ["authority", "mastery", "assured", "commanding", "power"],
    "respect": ["softening", "admiration", "reluctant", "acknowledging", "yielding"],
    "passion": ["burning", "conviction", "electricity", "intensity", "fire"],
    "tension": ["crackling", "challenge", "rivalry", "charged", "unspoken"],
}


def build_workflow(prompt: str, negative: str, params: dict, image_path: str, prefix: str) -> dict:
    """Build LTX-2 I2V workflow"""
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "ltx-2-19b-dev-fp8.safetensors"}},
        "2": {"class_type": "LoadImage", "inputs": {"image": os.path.basename(image_path)}},
        "3": {"class_type": "LTXAVTextEncoderLoader", "inputs": {"text_encoder": "gemma_3_12B_it_fp4_mixed.safetensors", "ckpt_name": "ltx-2-19b-dev-fp8.safetensors", "device": "default"}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["3", 0]}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"text": negative, "clip": ["3", 0]}},
        "6": {"class_type": "LTXVConditioning", "inputs": {"positive": ["4", 0], "negative": ["5", 0], "frame_rate": 24.0}},
        "7": {"class_type": "LTXVImgToVideo", "inputs": {"positive": ["6", 0], "negative": ["6", 1], "vae": ["1", 2], "image": ["2", 0], "width": params["width"], "height": params["height"], "length": params["frames"], "batch_size": 1, "strength": params.get("img_strength", 1.0)}},
        "8": {"class_type": "ModelSamplingLTXV", "inputs": {"model": ["1", 0], "max_shift": params["max_shift"], "base_shift": params["base_shift"], "latent": ["7", 2]}},
        "9": {"class_type": "LTXVScheduler", "inputs": {"steps": params["steps"], "max_shift": params["max_shift"], "base_shift": params["base_shift"], "stretch": True, "terminal": params.get("terminal", 0.1), "latent": ["7", 2]}},
        "10": {"class_type": "RandomNoise", "inputs": {"noise_seed": params["seed"]}},
        "11": {"class_type": "CFGGuider", "inputs": {"model": ["8", 0], "positive": ["7", 0], "negative": ["7", 1], "cfg": params["guidance_scale"]}},
        "12": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "13": {"class_type": "LTXVBaseSampler", "inputs": {"model": ["8", 0], "vae": ["1", 2], "width": params["width"], "height": params["height"], "num_frames": params["frames"], "guider": ["11", 0], "sampler": ["12", 0], "sigmas": ["9", 0], "noise": ["10", 0], "optional_cond_images": ["2", 0], "optional_cond_indices": "0", "strength": params.get("denoise_strength", 0.9)}},
        "14": {"class_type": "VAEDecode", "inputs": {"samples": ["13", 0], "vae": ["1", 2]}},
        "15": {"class_type": "SaveAnimatedWEBP", "inputs": {"images": ["14", 0], "filename_prefix": prefix, "fps": 24.0, "lossless": False, "quality": 90, "method": "default"}}
    }


def upload_image(image_path: str) -> bool:
    """Upload image to ComfyUI"""
    try:
        with open(image_path, 'rb') as f:
            files = {'image': (os.path.basename(image_path), f, 'image/png')}
            resp = requests.post(f"{COMFYUI_SERVER}/upload/image", files=files, timeout=30)
            return resp.status_code == 200
    except:
        return False


def queue_prompt(workflow: dict) -> Tuple[str, float]:
    """Queue prompt and return (prompt_id, queue_time)"""
    start = time.time()
    try:
        resp = requests.post(f"{COMFYUI_SERVER}/prompt", json={"prompt": workflow}, timeout=30)
        data = resp.json()
        return data.get('prompt_id', ''), time.time() - start
    except Exception as e:
        return str(e), 0


def run_single_test(test: TestCase, seed: int, test_type: str) -> dict:
    """Run a single STOCK or NEURO test"""

    negative = "worst quality, blurry, distorted, frozen, static, motionless, deformed"

    if test_type == "stock":
        prompt = test.stock_prompt
        params = {
            "guidance_scale": 7.5,
            "steps": 30,
            "width": 512, "height": 320, "frames": 49,  # ~2 seconds
            "max_shift": 2.05,
            "base_shift": 0.95,
            "terminal": 0.1,
            "img_strength": 1.0,
            "denoise_strength": 0.9,
            "seed": seed
        }
        prefix = f"BENCH_{test.name}_STOCK_s{seed}"
    else:
        prompt = test.neuro_prompt
        params = {
            "guidance_scale": 8.0,
            "steps": 24,  # 20% fewer
            "width": 512, "height": 320, "frames": 49,
            "max_shift": 2.10,
            "base_shift": 0.98,
            "terminal": 0.1,
            "img_strength": 1.0,
            "denoise_strength": 0.9,
            "seed": seed
        }
        prefix = f"BENCH_{test.name}_NEURO_s{seed}"

    workflow = build_workflow(prompt, negative, params, test.image_path, prefix)
    prompt_id, queue_time = queue_prompt(workflow)

    return {
        "test_name": test.name,
        "test_type": test_type,
        "seed": seed,
        "steps": params["steps"],
        "prompt_id": prompt_id,
        "queue_time": queue_time,
        "prefix": prefix,
        "subject_type": test.subject_type,
        "emotional_arc": test.emotional_arc
    }


def run_benchmark_suite(num_seeds: int = 3, tests_to_run: List[str] = None):
    """Run the full benchmark suite"""

    print("=" * 70)
    print("NEUROMORPHIC BENCHMARK SUITE")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Seeds per test: {num_seeds}")
    print(f"Test cases: {len(TEST_CASES)}")
    print(f"Total runs: {len(TEST_CASES) * num_seeds * 2} (stock + neuro)")
    print("=" * 70)

    # Generate seeds using cognitive scaffolding
    scaffolder = CognitiveSeedScaffolder(base_seed=42424242)
    scaffold = scaffolder.generate_scaffold(CognitiveFunction.LANGUAGE, num_frames=num_seeds * 10)
    test_seeds = scaffold.frame_seeds[:num_seeds]

    results = []
    images_uploaded = set()

    for i, test in enumerate(TEST_CASES):
        if tests_to_run and test.name not in tests_to_run:
            continue

        print(f"\n[{i+1}/{len(TEST_CASES)}] {test.name}")
        print(f"    Subject: {test.subject_type} | Arc: {test.emotional_arc}")

        # Upload image if needed
        if test.image_path not in images_uploaded:
            print(f"    Uploading: {os.path.basename(test.image_path)}...")
            if upload_image(test.image_path):
                images_uploaded.add(test.image_path)
                print("    Done")
            else:
                print("    FAILED - skipping test")
                continue

        # Run tests for each seed
        for seed_idx, seed in enumerate(test_seeds):
            # Stock test
            print(f"    [{seed_idx+1}/{num_seeds}] STOCK (30 steps, seed={seed})...", end=" ")
            stock_result = run_single_test(test, seed, "stock")
            results.append(stock_result)
            print(f"queued: {stock_result['prompt_id'][:8]}...")

            # Neuro test
            print(f"    [{seed_idx+1}/{num_seeds}] NEURO (24 steps, seed={seed})...", end=" ")
            neuro_result = run_single_test(test, seed, "neuro")
            results.append(neuro_result)
            print(f"queued: {neuro_result['prompt_id'][:8]}...")

    # Save results manifest
    manifest = {
        "timestamp": datetime.now().isoformat(),
        "num_seeds": num_seeds,
        "test_cases": len(TEST_CASES),
        "total_runs": len(results),
        "seeds_used": test_seeds,
        "results": results
    }

    manifest_path = os.path.join(OUTPUT_DIR, f"benchmark_manifest_{int(time.time())}.json")
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print("\n" + "=" * 70)
    print("BENCHMARK QUEUED")
    print("=" * 70)
    print(f"Total tests: {len(results)}")
    print(f"Stock tests: {len([r for r in results if r['test_type'] == 'stock'])}")
    print(f"Neuro tests: {len([r for r in results if r['test_type'] == 'neuro'])}")
    print(f"Manifest: {manifest_path}")
    print("\nStep comparison:")
    print("  STOCK: 30 steps")
    print("  NEURO: 24 steps (20% fewer)")
    print("\nWait for ComfyUI to complete, then run analysis.")

    return manifest


def quick_test(num_tests: int = 2, num_seeds: int = 2):
    """Quick test with subset for validation"""
    print("QUICK TEST MODE")
    print(f"Running {num_tests} tests with {num_seeds} seeds each\n")

    test_names = [t.name for t in TEST_CASES[:num_tests]]
    return run_benchmark_suite(num_seeds=num_seeds, tests_to_run=test_names)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        quick_test(num_tests=2, num_seeds=2)
    elif len(sys.argv) > 1 and sys.argv[1] == "full":
        run_benchmark_suite(num_seeds=5)
    else:
        print("Usage:")
        print("  python neuromorphic_benchmark_suite.py quick  # 2 tests, 2 seeds each")
        print("  python neuromorphic_benchmark_suite.py full   # All tests, 5 seeds each")
        print("\nRunning quick test by default...")
        quick_test(num_tests=2, num_seeds=2)
