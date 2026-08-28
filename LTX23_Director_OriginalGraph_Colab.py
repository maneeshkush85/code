"""LTX23_Director_OriginalGraph_Colab.py

Faithfully re-executes the ORIGINAL ComfyUI graph
`LTX-2.3_Director_2.0-MV-Workflow-30s.json` on a Colab T4 runtime.

Architecture (NON-NEGOTIABLE):
    JSON  ->  pure-Python parser  ->  internal node/link/widgets graph
          ->  ComfyUI node registry  ->  memory-aware executor
          ->  the REAL original ComfyUI node classes  ->  original output.

The LTXDirector node (id 131) is the MASTER timeline controller; its serialized
timeline lives in `widgets_values[6]` (a JSON string) and drives the 5 image
segments + 1 audio segment. We NEVER rebuild that timeline ourselves; we parse
and replay it.

`LTX23_Director_Master_V2.py` is used ONLY as a memory-infrastructure reference
(malloc_trim, page-cache drop, VRAM/RAM thresholds, comfy cleanup helpers).

`LTX23_Director_Master_V3_Production.py`'s 5-segment / linear_blend_overlap /
per-scene latent architecture is FORBIDDEN. V3 is used ONLY for its known-good
HuggingFace download URLs and custom-node repo URLs.

This file is laid out as Google Colab cells delimited by `# ===== CELL N =====`.
CELLS 1-10 (environment, system-info, install/download scaffolding, and the
pure-Python GRAPH PARSER + GRAPH VALIDATOR) run with the Python STANDARD LIBRARY
ONLY, so the `--validate-only` smoke test works with no GPU / CUDA / ComfyUI /
torch present. CELLS 11-19 (the actual GPU execution pipeline) are stubs here
and are implemented in a later feature.

Standalone smoke test (CPU, stdlib only):
    python3 LTX23_Director_OriginalGraph_Colab.py --validate-only
"""

# ===== CELL 1 =====
# CELL 1: Environment + memory protection (GPU-only imports are guarded/lazy)
import os
import sys
import json
import gc
import argparse
import glob
import subprocess
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Memory-protection env vars (lifted from LTX23_Director_Master_V2.py lines 40-42).
# Must be set BEFORE torch/CUDA are ever imported.
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF',
                       'expandable_segments:True,garbage_collection_threshold:0.8')
os.environ.setdefault('TORCH_CUDNN_V8_API_ENABLED', '1')
os.environ.setdefault('MALLOC_TRIM_THRESHOLD_', '65536')

# ctypes is stdlib; the libc calls it powers are only made lazily at runtime.
import ctypes  # noqa: E402

# ---------------------------------------------------------------------------
# Capability flags. GPU-only libraries are imported lazily so that the parser
# and validator below run with stdlib alone (no torch / comfy / psutil / etc).
# ---------------------------------------------------------------------------
HAS_TORCH = False
HAS_PSUTIL = False
HAS_SAFETENSORS = False


def _probe_capabilities() -> None:
    """Detect optional GPU/runtime libraries WITHOUT importing them eagerly.

    Uses importlib.util.find_spec so we never actually import torch here (which
    would be slow and would fail in a CPU-only sandbox). The real imports happen
    lazily inside the functions that need them.
    """
    global HAS_TORCH, HAS_PSUTIL, HAS_SAFETENSORS
    import importlib.util
    HAS_TORCH = importlib.util.find_spec('torch') is not None
    HAS_PSUTIL = importlib.util.find_spec('psutil') is not None
    HAS_SAFETENSORS = importlib.util.find_spec('safetensors') is not None


_probe_capabilities()


def in_colab_env() -> bool:
    """Best-effort detection of a Colab-like environment.

    All install/download cells (3-7) are guarded by this so that importing the
    module (or running --validate-only) never touches the network or the GPU.
    """
    if os.path.isdir('/content'):
        return True
    try:
        import google.colab  # noqa: F401
        return True
    except Exception:
        return False


def run_cmd(cmd: str, silent: bool = True) -> int:
    """Run a shell command (pattern lifted from V2/V3). Returns the exit code."""
    if silent:
        res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return res.returncode
    return subprocess.run(cmd, shell=True).returncode


# ===== CELL 2 =====
# CELL 2: System info (degrades gracefully with no GPU / no psutil)
def print_system_info() -> None:
    """Print Python / GPU / RAM info. Never crashes when torch/psutil absent."""
    print("=" * 64)
    print(" SYSTEM INFO")
    print("=" * 64)
    print("  Python           : %s" % sys.version.split()[0])
    print("  Platform         : %s" % sys.platform)

    if HAS_TORCH:
        try:
            import torch
            if torch.cuda.is_available():
                print("  GPU              : %s" % torch.cuda.get_device_name(0))
                try:
                    free_b, total_b = torch.cuda.mem_get_info()
                    print("  VRAM total       : %.2f GB" % (total_b / (1024 ** 3)))
                    print("  VRAM free        : %.2f GB" % (free_b / (1024 ** 3)))
                except Exception:
                    print("  VRAM             : query unavailable")
            else:
                print("  GPU              : torch present but CUDA not available")
        except Exception as exc:
            print("  GPU              : torch import failed (%s)" % exc)
    else:
        print("  GPU              : not available in this environment (no torch)")

    if HAS_PSUTIL:
        try:
            import psutil
            vm = psutil.virtual_memory()
            print("  RAM total        : %.2f GB" % (vm.total / (1024 ** 3)))
            print("  RAM available    : %.2f GB" % (vm.available / (1024 ** 3)))
        except Exception as exc:
            print("  RAM              : psutil query failed (%s)" % exc)
    else:
        print("  RAM              : not available in this environment (no psutil)")
    print("=" * 64)


# ===== CELL 3 =====
# CELL 3: Install ComfyUI (Colab-guarded)
COMFYUI_REPO = "https://github.com/comfyanonymous/ComfyUI.git"
COMFYUI_DIR = "/content/ComfyUI"


def install_comfyui() -> None:
    """Clone ComfyUI and install its requirements. Colab-only."""
    if not in_colab_env():
        print("[CELL 3] Skipping ComfyUI install (not a Colab environment).")
        return
    if not os.path.isdir(COMFYUI_DIR):
        print("[CELL 3] Cloning ComfyUI ...")
        rc = run_cmd("git clone %s %s" % (COMFYUI_REPO, COMFYUI_DIR))
        if rc != 0:
            raise RuntimeError("Failed to clone ComfyUI from %s" % COMFYUI_REPO)
    else:
        print("[CELL 3] ComfyUI already present at %s" % COMFYUI_DIR)
    req = os.path.join(COMFYUI_DIR, "requirements.txt")
    if os.path.exists(req):
        print("[CELL 3] Installing ComfyUI requirements ...")
        run_cmd("%s -m pip install -r %s" % (sys.executable, req))


# ===== CELL 4 =====
# CELL 4: Install the EXACT custom-node repos + verify required classes exist.
#
# The source workflow JSON does NOT embed commit hashes, so by default we clone
# each repo at its latest HEAD. To make a run fully reproducible, populate
# CUSTOM_NODE_PINS below with a commit SHA per repo key; a non-None pin causes
# CELL 4 to check out that exact commit after cloning. This is intentionally a
# data-driven dict (no dangling TODOs): leave a value None to track latest.
CUSTOM_NODE_REPOS: Dict[str, str] = {
    # key                        : git URL
    "WhatDreamsCost-ComfyUI":      "https://github.com/WhatDreamscost/WhatDreamsCost-ComfyUI",
    "ComfyUI-KJNodes":             "https://github.com/kijai/ComfyUI-KJNodes.git",
    "ComfyUI-GGUF":                "https://github.com/city96/ComfyUI-GGUF.git",
    "ComfyUI-LTXVideo":            "https://github.com/Lightricks/ComfyUI-LTXVideo",
    "ComfyUI-VideoHelperSuite":    "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite",
    "rgthree-comfy":               "https://github.com/rgthree/rgthree-comfy",
}

# Optional per-repo pinned commit SHA. None => track latest HEAD.
CUSTOM_NODE_PINS: Dict[str, Optional[str]] = {
    "WhatDreamsCost-ComfyUI":   None,
    "ComfyUI-KJNodes":          None,
    "ComfyUI-GGUF":             None,
    "ComfyUI-LTXVideo":         None,
    "ComfyUI-VideoHelperSuite": None,
    "rgthree-comfy":            None,
}

# Custom-node classes the original graph relies on. If any is missing after
# install/import, CELL 4 raises the Section-25 hard error (NO silent fallback).
REQUIRED_CUSTOM_NODE_CLASSES: List[str] = [
    "LTXDirector",
    "LTXDirectorGuide",
    "LTXDirectorCropGuides",
    "Power Lora Loader (rgthree)",
    "ModelPreviewOverrideKJ",
    "LTXVLatentUpsampler",
    "LTXVConcatAVLatent",
    "LTXVSeparateAVLatent",
    "LTXVConditioning",
    "LTXVAudioVAEDecode",
]


def install_custom_nodes() -> None:
    """Clone (and optionally pin) each custom-node repo. Colab-only."""
    if not in_colab_env():
        print("[CELL 4] Skipping custom-node install (not a Colab environment).")
        return
    nodes_dir = os.path.join(COMFYUI_DIR, "custom_nodes")
    Path(nodes_dir).mkdir(parents=True, exist_ok=True)
    for key, url in CUSTOM_NODE_REPOS.items():
        dest = os.path.join(nodes_dir, key)
        if not os.path.isdir(dest):
            print("[CELL 4] Cloning %s ..." % key)
            rc = run_cmd("git clone %s %s" % (url, dest))
            if rc != 0:
                raise RuntimeError("Failed to clone custom node repo %s (%s)" % (key, url))
        else:
            print("[CELL 4] %s already present." % key)
        pin = CUSTOM_NODE_PINS.get(key)
        if pin:
            print("[CELL 4] Pinning %s -> %s" % (key, pin))
            rc = run_cmd("cd %s && git fetch --all --quiet && git checkout %s" % (dest, pin))
            if rc != 0:
                raise RuntimeError("Failed to pin %s to commit %s" % (key, pin))
        req = os.path.join(dest, "requirements.txt")
        if os.path.exists(req):
            run_cmd("%s -m pip install -r %s" % (sys.executable, req))


def verify_required_node_classes(node_class_mappings: Dict[str, Any]) -> None:
    """Raise the Section-25 HARD error if any required node class is missing.

    `node_class_mappings` is ComfyUI's NODE_CLASS_MAPPINGS after all custom
    nodes are imported. This is called from CELL 11 (registry build) on real
    Colab; there is deliberately NO silent fallback.
    """
    missing = [c for c in REQUIRED_CUSTOM_NODE_CLASSES if c not in node_class_mappings]
    if missing:
        raise RuntimeError(
            "HARD ERROR (Section 25): required ComfyUI node classes are missing "
            "after installing custom nodes: %s. The original graph cannot be "
            "executed faithfully without them. Check the CUSTOM_NODE_REPOS clones "
            "in %s/custom_nodes and their import errors."
            % (", ".join(missing), COMFYUI_DIR)
        )


# ===== CELL 5 =====
# CELL 5: Download core models (Colab-guarded). URLs reused verbatim from V3.
# Each entry: (url, dest_dir, filename, [optional symlink targets]).
MODEL_DOWNLOADS: List[Dict[str, Any]] = [
    {
        "url": "https://huggingface.co/vantagewithai/LTX-2.3-GGUF/resolve/main/dev/ltx-2-3-22b-dev-Q4_K_M.gguf",
        "dir": COMFYUI_DIR + "/models/unet",
        "name": "ltx-2-3-22b-dev-Q4_K_M.gguf",
        "links": [COMFYUI_DIR + "/models/diffusion_models/ltx-2-3-22b-dev-Q4_K_M.gguf"],
    },
    {
        "url": "https://huggingface.co/Comfy-Org/ltx-2/resolve/main/split_files/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors",
        "dir": COMFYUI_DIR + "/models/text_encoders",
        "name": "gemma_3_12B_it_fp4_mixed.safetensors",
        "links": [COMFYUI_DIR + "/models/clip/gemma_3_12B_it_fp4_mixed.safetensors"],
    },
    {
        "url": "https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/text_encoders/ltx-2.3_text_projection_bf16.safetensors",
        "dir": COMFYUI_DIR + "/models/text_encoders",
        "name": "ltx-2.3_text_projection_bf16.safetensors",
        "links": [COMFYUI_DIR + "/models/clip/ltx-2.3_text_projection_bf16.safetensors"],
    },
    {
        "url": "https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/vae/LTX23_video_vae_bf16.safetensors",
        "dir": COMFYUI_DIR + "/models/vae",
        "name": "LTX23_video_vae_bf16.safetensors",
        "links": [],
    },
    {
        "url": "https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/vae/LTX23_audio_vae_bf16.safetensors",
        "dir": COMFYUI_DIR + "/models/vae",
        "name": "LTX23_audio_vae_bf16.safetensors",
        "links": [],
    },
    {
        "url": "https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/vae/taeltx2_3.safetensors",
        "dir": COMFYUI_DIR + "/models/vae",
        "name": "taeltx2_3.safetensors",
        "links": [COMFYUI_DIR + "/models/vae_approx/taeltx2_3.safetensors"],
    },
    {
        "url": "https://huggingface.co/vidfom/aimusic/resolve/main/ComfyUI/models/latent_upscale_models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
        "dir": COMFYUI_DIR + "/models/latent_upscale_models",
        "name": "ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
        "links": [COMFYUI_DIR + "/models/upscale_models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors"],
    },
]


def download_file(url: str, dest_dir: str, filename: Optional[str] = None) -> Optional[str]:
    """Download via aria2c (pattern lifted from V3). Skips if already present."""
    Path(dest_dir).mkdir(parents=True, exist_ok=True)
    if filename is None:
        filename = url.split('/')[-1].split('?')[0]
    dest = os.path.join(dest_dir, filename)
    if os.path.exists(dest) and os.path.getsize(dest) > 1024:
        print("  [FOUND] %s" % filename)
        return filename
    cmd = ['aria2c', '--console-log-level=error', '-c', '-x', '16',
           '-s', '16', '-k', '1M', '-d', dest_dir, '-o', filename, url]
    print("  downloading %s ..." % filename, end=' ', flush=True)
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    if os.path.exists(dest) and os.path.getsize(dest) > 1024:
        print("done")
        return filename
    raise RuntimeError("Download failed or produced an empty file: %s" % dest)


def link_file_safe(src_path: str, dst_path: str) -> None:
    """Symlink src->dst, falling back to a copy. Only links when src exists."""
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    if os.path.exists(dst_path) or not os.path.exists(src_path):
        return
    try:
        os.symlink(src_path, dst_path)
    except Exception:
        import shutil
        shutil.copyfile(src_path, dst_path)


def download_models() -> None:
    """Download all core models + create the expected symlinks. Colab-only."""
    if not in_colab_env():
        print("[CELL 5] Skipping model download (not a Colab environment).")
        return
    print("[CELL 5] Downloading LTX-2.3 core models ...")
    for entry in MODEL_DOWNLOADS:
        download_file(entry["url"], entry["dir"], entry["name"])
        src = os.path.join(entry["dir"], entry["name"])
        for link in entry.get("links", []):
            link_file_safe(src, link)


# ===== CELL 6 =====
# CELL 6: Download the 4 LoRAs (Colab-guarded). URLs reused verbatim from V3.
LORA_DOWNLOADS: List[Dict[str, str]] = [
    {
        "url": "https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/loras/ltx-2.3-22b-distilled-lora-dynamic_fro09_avg_rank_105_bf16.safetensors",
        "name": "ltx-2.3-22b-distilled-lora-dynamic_fro09_avg_rank_105_bf16.safetensors",
    },
    {
        "url": "https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/loras/LTX-2.3-OmniNFT-RL-Lora_bf16.safetensors",
        "name": "LTX-2.3-OmniNFT-RL-Lora_bf16.safetensors",
    },
    {
        "url": "https://huggingface.co/joyfox/LTX-2.3-Transition-LORA/resolve/main/ltx2.3-transition.safetensors",
        "name": "ltx2.3-transition.safetensors",
    },
    {
        "url": "https://huggingface.co/vidfom/aimusic/resolve/main/ComfyUI/models/loras/LTX2.3-MVCamera-drclips.safetensors",
        "name": "LTX2.3-MVCamera-drclips.safetensors",
    },
]


def download_loras() -> None:
    """Download all 4 LoRAs into models/loras. Colab-only."""
    if not in_colab_env():
        print("[CELL 6] Skipping LoRA download (not a Colab environment).")
        return
    lora_dir = COMFYUI_DIR + "/models/loras"
    print("[CELL 6] Downloading LoRAs ...")
    for entry in LORA_DOWNLOADS:
        download_file(entry["url"], lora_dir, entry["name"])


# ===== CELL 7 =====
# CELL 7: Download audio + verify reference images (Colab-guarded).
AUDIO_URL = "https://huggingface.co/vidfom/aimusic/resolve/main/Late%20night%20trap.mp3"
AUDIO_NAME = "Late night trap.mp3"
INPUT_SUBFOLDER = "whatdreamscost"

# The reference images the timeline segments point at (imageFile fields).
REQUIRED_REFERENCE_IMAGES: List[str] = ["1.png", "2.png", "3.png", "4.png", "5.3.png"]
# If/when direct URLs become known they can be filled in here (name -> url).
REFERENCE_IMAGE_URLS: Dict[str, str] = {}


def download_audio() -> None:
    """Download the background track into input/whatdreamscost/. Colab-only."""
    if not in_colab_env():
        print("[CELL 7] Skipping audio download (not a Colab environment).")
        return
    audio_dir = os.path.join(COMFYUI_DIR, "input", INPUT_SUBFOLDER)
    print("[CELL 7] Downloading audio ...")
    download_file(AUDIO_URL, audio_dir, AUDIO_NAME)


def verify_reference_images(comfyui_dir: str = COMFYUI_DIR) -> None:
    """HARD error (Section 37) if any reference image is missing.

    We NEVER generate placeholder images. A missing image reports its exact
    filename, the directory it is expected in, and a download URL if known.
    """
    input_dir = os.path.join(comfyui_dir, "input", INPUT_SUBFOLDER)
    missing: List[str] = []
    for name in REQUIRED_REFERENCE_IMAGES:
        path = os.path.join(input_dir, name)
        if not (os.path.exists(path) and os.path.getsize(path) > 0):
            missing.append(name)
    if missing:
        lines = ["HARD ERROR (Section 37): required reference image(s) are missing. "
                 "Placeholder images are NOT generated; you must supply them."]
        for name in missing:
            url = REFERENCE_IMAGE_URLS.get(name)
            lines.append("  - %s  (expected in: %s)%s"
                         % (name, input_dir,
                            ("  URL: " + url) if url else "  URL: unknown, supply manually"))
        raise FileNotFoundError("\n".join(lines))
    print("[CELL 7] All %d reference images present in %s"
          % (len(REQUIRED_REFERENCE_IMAGES), input_dir))


# ===== CELL 8 =====
# CELL 8: Load the workflow JSON (stdlib only).
DEFAULT_WORKFLOW_FILENAME = "LTX-2.3_Director_2.0-MV-Workflow-30s.json"


def default_workflow_path() -> str:
    """Resolve the source-of-truth JSON that sits next to this script."""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, DEFAULT_WORKFLOW_FILENAME)


def load_workflow_json(path: Optional[str] = None) -> Dict[str, Any]:
    """Load and sanity-check the ComfyUI workflow JSON. Returns the raw dict."""
    if path is None:
        path = default_workflow_path()
    if not os.path.exists(path):
        raise FileNotFoundError("Workflow JSON not found: %s" % path)
    with open(path, "r", encoding="utf-8") as fh:
        workflow = json.load(fh)
    required_keys = ["nodes", "links", "groups", "config", "extra", "version"]
    missing = [k for k in required_keys if k not in workflow]
    if missing:
        raise ValueError("Workflow JSON missing top-level keys: %s" % ", ".join(missing))
    return workflow


# ===== CELL 9 =====
# CELL 9: Parse the workflow into an internal graph + LTXDirector timeline.
LTXDIRECTOR_NODE_ID = 131
LTXDIRECTOR_TIMELINE_WIDGET_INDEX = 6


def parse_ltxdirector_timeline(node131_widgets: List[Any]) -> Dict[str, Any]:
    """Parse LTXDirector (id 131) timeline. Values are READ, never hardcoded.

    widgets_values[6] is a serialized JSON string holding the full timeline
    (segments / audioSegments / motionSegments + track flags + global prompt).
    Scalar timeline fields are read from the surrounding widget indices.
    """
    raw_timeline = node131_widgets[LTXDIRECTOR_TIMELINE_WIDGET_INDEX]
    timeline_config = json.loads(raw_timeline)

    def _norm_segment(seg: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "id": seg.get("id"),
            "type": seg.get("type"),
            "image_file": seg.get("imageFile"),
            "image_b64_ref": seg.get("imageB64"),
            "prompt": seg.get("prompt", ""),
            "start": seg.get("start"),
            "length": seg.get("length"),
            "is_end_frame": seg.get("isEndFrame", False),
        }

    def _norm_audio(seg: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "id": seg.get("id"),
            "type": seg.get("type"),
            "audio_file": seg.get("audioFile"),
            "file_name": seg.get("fileName"),
            "start": seg.get("start"),
            "length": seg.get("length"),
            "trim_start": seg.get("trimStart"),
            "audio_duration_frames": seg.get("audioDurationFrames"),
        }

    timeline_segments = [_norm_segment(s) for s in timeline_config.get("segments", [])]
    audio_segments = [_norm_audio(a) for a in timeline_config.get("audioSegments", [])]
    motion_segments = list(timeline_config.get("motionSegments", []))

    # Scalar timeline widgets (read from the widget list, NOT hardcoded).
    wv = node131_widgets
    scalars = {
        "index0": wv[0],
        "duration": wv[1],           # 31.5 (seconds)
        "duration2": wv[2],          # 31.5
        "startFrame": wv[3],         # 0
        "endFrame": wv[4],           # 756
        "frames": wv[5],             # 756
        "fps": wv[14],               # 24
    }

    return {
        "raw_serialized": raw_timeline,
        "timeline_config": timeline_config,
        "timeline_segments": timeline_segments,
        "audio_segments": audio_segments,
        "motion_segments": motion_segments,
        "main_track_enabled": timeline_config.get("mainTrackEnabled"),
        "audio_track_enabled": timeline_config.get("audioTrackEnabled"),
        "motion_track_enabled": timeline_config.get("motionTrackEnabled"),
        "override_audio": timeline_config.get("overrideAudio"),
        "inpaint_audio": timeline_config.get("inpaint_audio"),
        "global_prompt": timeline_config.get("global_prompt"),
        "normal_start_frame": timeline_config.get("normalStartFrame"),
        "normal_duration_frames": timeline_config.get("normalDurationFrames"),
        "scalars": scalars,
        # Convenience top-level scalars (read, not hardcoded):
        "duration": scalars["duration"],
        "frames": scalars["frames"],
        "fps": scalars["fps"],
        "start_frame": scalars["startFrame"],
        "end_frame": scalars["endFrame"],
    }


def parse_graph(workflow: Dict[str, Any]) -> Dict[str, Any]:
    """Parse the raw workflow dict into an internal representation.

    Returns a dict with:
        nodes_by_id : {id: {type, inputs, outputs, widgets_values, mode, order}}
        links       : [(link_id, from_node, from_slot, to_node, to_slot, type)]
        groups      : normalized group dicts
        timeline    : parsed LTXDirector timeline (see parse_ltxdirector_timeline)
        counts      : {nodes, links, groups}
        version     : workflow schema version
    """
    nodes_by_id: Dict[int, Dict[str, Any]] = {}
    for node in workflow.get("nodes", []):
        nodes_by_id[node["id"]] = {
            "id": node["id"],
            "type": node.get("type"),
            "inputs": node.get("inputs", []) or [],
            "outputs": node.get("outputs", []) or [],
            "widgets_values": node.get("widgets_values"),
            "mode": node.get("mode", 0),
            "order": node.get("order"),
        }

    # ComfyUI 0.4 link format: [link_id, from_node, from_slot, to_node, to_slot, type]
    links: List[Tuple[Any, ...]] = []
    for link in workflow.get("links", []):
        if isinstance(link, (list, tuple)) and len(link) >= 6:
            links.append((link[0], link[1], link[2], link[3], link[4], link[5]))
        else:
            raise ValueError("Unexpected link format in workflow JSON: %r" % (link,))

    groups: List[Dict[str, Any]] = []
    for g in workflow.get("groups", []):
        groups.append({
            "id": g.get("id"),
            "title": g.get("title"),
            "color": g.get("color"),
            "bounding": g.get("bounding"),
        })

    if LTXDIRECTOR_NODE_ID not in nodes_by_id:
        raise ValueError("LTXDirector node (id %d) not found in graph."
                         % LTXDIRECTOR_NODE_ID)
    director = nodes_by_id[LTXDIRECTOR_NODE_ID]
    if director["type"] != "LTXDirector":
        raise ValueError("Node id %d is %r, expected LTXDirector."
                         % (LTXDIRECTOR_NODE_ID, director["type"]))
    timeline = parse_ltxdirector_timeline(director["widgets_values"])

    return {
        "nodes_by_id": nodes_by_id,
        "links": links,
        "groups": groups,
        "timeline": timeline,
        "version": workflow.get("version"),
        "counts": {
            "nodes": len(nodes_by_id),
            "links": len(links),
            "groups": len(groups),
        },
    }


# ===== CELL 10 =====
# CELL 10: Graph validator + Section-27 report (stdlib only).
#
# EXPECTED_* below is the ground truth derived from the source-of-truth JSON.
# The validator compares the PARSED graph against these and prints a PASS/FAIL
# line per Section-27 category. Any mismatch prints exact expected-vs-found and
# makes validate() return False so the caller can stop before GPU execution.

EXPECTED_COUNTS = {"nodes": 32, "links": 65, "groups": 4}
EXPECTED_VERSION = 0.4

EXPECTED_NODE_TYPES: Dict[int, str] = {
    17: "CFGGuider", 18: "LTXVConcatAVLatent", 20: "KSamplerSelect",
    132: "LTXDirectorGuide", 14: "LTXVLatentUpsampler", 19: "SamplerCustomAdvanced",
    28: "CFGGuider", 133: "LTXDirectorGuide", 128: "ConditioningZeroOut",
    27: "LTXVConditioning", 8: "VAELoader", 36: "VAELoader",
    10: "ModelPreviewOverrideKJ", 34: "LTXVSeparateAVLatent", 22: "LTXVSeparateAVLatent",
    54: "LTXDirectorCropGuides", 6: "VAELoaderKJ", 13: "LatentUpscaleModelLoader",
    55: "LTXDirectorCropGuides", 30: "RandomNoise", 31: "SamplerCustomAdvanced",
    32: "KSamplerSelect", 33: "BasicScheduler", 29: "LTXVConcatAVLatent",
    24: "LTXVAudioVAEDecode", 1: "VAEDecode", 139: "VHS_VideoCombine",
    135: "UnetLoaderGGUF", 138: "Power Lora Loader (rgthree)", 21: "BasicScheduler",
    12: "DualCLIPLoader", 131: "LTXDirector",
}

EXPECTED_LTXDIRECTOR_OUTPUTS = [
    "model", "positive", "video_latent", "audio_latent",
    "guide_data", "motion_guide_data", "frame_rate", "combined_audio",
]

EXPECTED_LORAS = [
    ("ltx-2.3-22b-distilled-lora-dynamic_fro09_avg_rank_105_bf16.safetensors", 0.4),
    ("LTX-2.3-OmniNFT-RL-Lora_bf16.safetensors", 0.6),
    ("ltx2.3-transition.safetensors", 0.7),
    ("LTX2.3-MVCamera-drclips.safetensors", 0.9),
]

EXPECTED_MODELS = {
    "unet_135": "ltx-2-3-22b-dev-Q4_K_M.gguf",
    "dualclip_12": ["gemma_3_12B_it_fp4_mixed.safetensors",
                    "ltx-2.3_text_projection_bf16.safetensors", "ltxv", "default"],
    "vae_8_audio": "LTX23_audio_vae_bf16.safetensors",
    "vae_36_video": "LTX23_video_vae_bf16.safetensors",
    "vaeloaderkj_6": ["taeltx2_3.safetensors", "main_device", "bf16"],
    "upscaler_13": "ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
}

# Timeline ground truth (scalars + segment/audio facts).
EXPECTED_TIMELINE = {
    "frames": 756,
    "fps": 24,
    "duration": 31.5,
    "start_frame": 0,
    "end_frame": 756,
    "num_segments": 5,
    "num_audio_segments": 1,
    "num_motion_segments": 0,
    "segment_images": ["whatdreamscost/1.png", "whatdreamscost/2.png",
                       "whatdreamscost/3.png", "whatdreamscost/4.png",
                       "whatdreamscost/5.3.png"],
    "segment_starts": [0, 226.01059340956584, 387.3291931757404,
                       518.785491487707, 744.2918243643325],
    "segment_lengths": [226.01059340956584, 161.31859976617454, 131.45629831196658,
                        225.5063328766255, 83.22765271847516],
}

EXPECTED_AUDIO = {
    "audio_file": "whatdreamscost/Late night trap.mp3",
    "file_name": "Late night trap.mp3",
    "start": 0,
    "trim_start": 446.9222739141953,
    "audio_duration_frames": 2880,
}

# Stage 1 / Stage 2 sampler graph ground truth.
EXPECTED_SAMPLERS = {20: "euler", 32: "euler"}
EXPECTED_CFG = {17: 1, 28: 1}
EXPECTED_SCHEDULERS = {
    33: ["linear_quadratic", 8, 1],
    21: ["linear_quadratic", 4, 0.42],
}
EXPECTED_GUIDE_STRENGTHS = {132: 1.0, 133: 0.5}  # LTXDirectorGuide widget index 2
EXPECTED_RANDOMNOISE_30 = [0, "fixed"]
EXPECTED_LTXVCONDITIONING_27 = [24]
EXPECTED_MODELPREVIEW_10 = [0, 80, True, 240, 24, ""]

EXPECTED_VHS = {
    "frame_rate": 24,
    "crf": 8,
    "pix_fmt": "yuv420p",
    "format": "video/h264-mp4",
    "filename_prefix": "LTX2.3/Video",
    "save_metadata": False,
    "trim_to_audio": False,
    "pingpong": False,
    "save_output": True,
}

_FLOAT_TOL = 1e-6


def _approx(a: Any, b: Any, tol: float = _FLOAT_TOL) -> bool:
    try:
        return abs(float(a) - float(b)) <= tol
    except (TypeError, ValueError):
        return a == b


class _Report:
    """Accumulates Section-27 report lines and an overall pass/fail flag."""

    def __init__(self) -> None:
        self.all_pass = True
        self._lines: List[str] = []

    def check(self, label: str, ok: bool, detail: str = "") -> None:
        status = "PASS" if ok else "FAIL"
        if not ok:
            self.all_pass = False
        line = "  [%s] %-14s" % (status, label)
        if detail:
            line += " : %s" % detail
        self._lines.append(line)

    def render(self) -> str:
        header = "=" * 64 + "\n GRAPH VALIDATION REPORT (Section 27)\n" + "=" * 64
        footer = "=" * 64 + "\n RESULT: %s\n" % ("ALL PASS" if self.all_pass else "FAILED") + "=" * 64
        return "\n".join([header] + self._lines + [footer])


def _lora_entries(director_widgets: Any) -> List[Dict[str, Any]]:
    """Extract active LoRA dicts from Power Lora Loader (rgthree) widgets_values."""
    entries = []
    if isinstance(director_widgets, list):
        for w in director_widgets:
            if isinstance(w, dict) and "lora" in w and "strength" in w:
                entries.append(w)
    return entries


def build_validator(graph: Dict[str, Any]):
    """Return a validate() closure that checks `graph` and prints Section-27."""

    def validate(verbose: bool = True) -> bool:
        rep = _Report()
        nodes = graph["nodes_by_id"]
        counts = graph["counts"]
        tl = graph["timeline"]

        # --- Nodes ---------------------------------------------------------
        node_ok = counts["nodes"] == EXPECTED_COUNTS["nodes"]
        detail = "" if node_ok else "expected %d, found %d" % (
            EXPECTED_COUNTS["nodes"], counts["nodes"])
        type_mismatches = []
        for nid, ntype in EXPECTED_NODE_TYPES.items():
            found = nodes.get(nid, {}).get("type")
            if found != ntype:
                type_mismatches.append("id %s expected %r found %r" % (nid, ntype, found))
        if type_mismatches:
            node_ok = False
            detail = (detail + "; " if detail else "") + "; ".join(type_mismatches)
        director_ok = LTXDIRECTOR_NODE_ID in nodes
        d_outputs = [o.get("name") for o in nodes.get(LTXDIRECTOR_NODE_ID, {}).get("outputs", [])]
        if d_outputs != EXPECTED_LTXDIRECTOR_OUTPUTS:
            node_ok = False
            detail = (detail + "; " if detail else "") + \
                "LTXDirector outputs expected %r found %r" % (EXPECTED_LTXDIRECTOR_OUTPUTS, d_outputs)
        rep.check("Nodes", node_ok and director_ok, detail)

        # --- Connections ---------------------------------------------------
        conn_ok = counts["links"] == EXPECTED_COUNTS["links"] and \
            counts["groups"] == EXPECTED_COUNTS["groups"] and \
            _approx(graph.get("version"), EXPECTED_VERSION)
        conn_detail = ""
        if counts["links"] != EXPECTED_COUNTS["links"]:
            conn_detail += "links expected %d found %d; " % (EXPECTED_COUNTS["links"], counts["links"])
        if counts["groups"] != EXPECTED_COUNTS["groups"]:
            conn_detail += "groups expected %d found %d; " % (EXPECTED_COUNTS["groups"], counts["groups"])
        if not _approx(graph.get("version"), EXPECTED_VERSION):
            conn_detail += "version expected %s found %s" % (EXPECTED_VERSION, graph.get("version"))
        rep.check("Connections", conn_ok, conn_detail.strip("; "))

        # --- Models --------------------------------------------------------
        model_ok = True
        model_detail = []

        def _wv(nid):
            return nodes.get(nid, {}).get("widgets_values")

        checks = [
            ("unet 135", _wv(135), [EXPECTED_MODELS["unet_135"]]),
            ("dualclip 12", _wv(12), EXPECTED_MODELS["dualclip_12"]),
            ("vae 8 (audio)", _wv(8), [EXPECTED_MODELS["vae_8_audio"]]),
            ("vae 36 (video)", _wv(36), [EXPECTED_MODELS["vae_36_video"]]),
            ("vaeloaderkj 6", _wv(6), EXPECTED_MODELS["vaeloaderkj_6"]),
            ("upscaler 13", _wv(13), [EXPECTED_MODELS["upscaler_13"]]),
        ]
        for name, found, exp in checks:
            if found != exp:
                model_ok = False
                model_detail.append("%s expected %r found %r" % (name, exp, found))
        rep.check("Models", model_ok, "; ".join(model_detail))

        # --- LoRAs ---------------------------------------------------------
        lora_ok = True
        lora_detail = []
        entries = _lora_entries(_wv(138))
        if len(entries) != len(EXPECTED_LORAS):
            lora_ok = False
            lora_detail.append("expected %d loras found %d" % (len(EXPECTED_LORAS), len(entries)))
        else:
            for i, (exp_name, exp_strength) in enumerate(EXPECTED_LORAS):
                e = entries[i]
                if e.get("lora") != exp_name:
                    lora_ok = False
                    lora_detail.append("lora[%d] name expected %r found %r"
                                       % (i + 1, exp_name, e.get("lora")))
                if not _approx(e.get("strength"), exp_strength):
                    lora_ok = False
                    lora_detail.append("lora[%d] strength expected %s found %s"
                                       % (i + 1, exp_strength, e.get("strength")))
                if e.get("on") is not True:
                    lora_ok = False
                    lora_detail.append("lora[%d] not enabled (on=%r)" % (i + 1, e.get("on")))
        rep.check("LoRAs", lora_ok, "; ".join(lora_detail))

        # --- Timeline ------------------------------------------------------
        tl_ok = True
        tl_detail = []
        scalar_checks = [
            ("frames", tl["frames"], EXPECTED_TIMELINE["frames"]),
            ("fps", tl["fps"], EXPECTED_TIMELINE["fps"]),
            ("duration", tl["duration"], EXPECTED_TIMELINE["duration"]),
            ("start_frame", tl["start_frame"], EXPECTED_TIMELINE["start_frame"]),
            ("end_frame", tl["end_frame"], EXPECTED_TIMELINE["end_frame"]),
            ("num_segments", len(tl["timeline_segments"]), EXPECTED_TIMELINE["num_segments"]),
            ("num_audio_segments", len(tl["audio_segments"]), EXPECTED_TIMELINE["num_audio_segments"]),
            ("num_motion_segments", len(tl["motion_segments"]), EXPECTED_TIMELINE["num_motion_segments"]),
        ]
        for name, found, exp in scalar_checks:
            if not _approx(found, exp):
                tl_ok = False
                tl_detail.append("%s expected %s found %s" % (name, exp, found))
        seg_images = [s["image_file"] for s in tl["timeline_segments"]]
        if seg_images != EXPECTED_TIMELINE["segment_images"]:
            tl_ok = False
            tl_detail.append("segment images expected %r found %r"
                             % (EXPECTED_TIMELINE["segment_images"], seg_images))
        for i, exp_start in enumerate(EXPECTED_TIMELINE["segment_starts"]):
            found = tl["timeline_segments"][i]["start"] if i < len(tl["timeline_segments"]) else None
            if not _approx(found, exp_start):
                tl_ok = False
                tl_detail.append("segment[%d] start expected %s found %s" % (i, exp_start, found))
        for i, exp_len in enumerate(EXPECTED_TIMELINE["segment_lengths"]):
            found = tl["timeline_segments"][i]["length"] if i < len(tl["timeline_segments"]) else None
            if not _approx(found, exp_len):
                tl_ok = False
                tl_detail.append("segment[%d] length expected %s found %s" % (i, exp_len, found))
        rep.check("Timeline", tl_ok, "; ".join(tl_detail))

        # --- Audio ---------------------------------------------------------
        audio_ok = True
        audio_detail = []
        if not tl["audio_segments"]:
            audio_ok = False
            audio_detail.append("no audio segment parsed")
        else:
            a = tl["audio_segments"][0]
            if a.get("audio_file") != EXPECTED_AUDIO["audio_file"]:
                audio_ok = False
                audio_detail.append("audio_file expected %r found %r"
                                    % (EXPECTED_AUDIO["audio_file"], a.get("audio_file")))
            if a.get("file_name") != EXPECTED_AUDIO["file_name"]:
                audio_ok = False
                audio_detail.append("file_name expected %r found %r"
                                    % (EXPECTED_AUDIO["file_name"], a.get("file_name")))
            if not _approx(a.get("trim_start"), EXPECTED_AUDIO["trim_start"]):
                audio_ok = False
                audio_detail.append("trim_start expected %s found %s"
                                    % (EXPECTED_AUDIO["trim_start"], a.get("trim_start")))
            if a.get("audio_duration_frames") != EXPECTED_AUDIO["audio_duration_frames"]:
                audio_ok = False
                audio_detail.append("audio_duration_frames expected %s found %s"
                                    % (EXPECTED_AUDIO["audio_duration_frames"],
                                       a.get("audio_duration_frames")))
        rep.check("Audio", audio_ok, "; ".join(audio_detail))

        # --- Stage 1 (base sampling: nodes 33/32/28/133/31/30) -------------
        stage1_ok = True
        stage1_detail = []
        if _wv(33) != EXPECTED_SCHEDULERS[33]:
            stage1_ok = False
            stage1_detail.append("scheduler 33 expected %r found %r"
                                 % (EXPECTED_SCHEDULERS[33], _wv(33)))
        if _wv(32) != [EXPECTED_SAMPLERS[32]]:
            stage1_ok = False
            stage1_detail.append("sampler 32 expected %r found %r"
                                 % ([EXPECTED_SAMPLERS[32]], _wv(32)))
        if _wv(28) != [EXPECTED_CFG[28]]:
            stage1_ok = False
            stage1_detail.append("cfg 28 expected %r found %r" % ([EXPECTED_CFG[28]], _wv(28)))
        g133 = _wv(133)
        if not (isinstance(g133, list) and len(g133) > 2 and _approx(g133[2], EXPECTED_GUIDE_STRENGTHS[133])):
            stage1_ok = False
            stage1_detail.append("guide 133 strength expected %s found %r"
                                 % (EXPECTED_GUIDE_STRENGTHS[133], g133))
        if _wv(30) != EXPECTED_RANDOMNOISE_30:
            stage1_ok = False
            stage1_detail.append("randomnoise 30 expected %r found %r"
                                 % (EXPECTED_RANDOMNOISE_30, _wv(30)))
        rep.check("Stage 1", stage1_ok, "; ".join(stage1_detail))

        # --- Stage 2 (upscale sampling: nodes 21/20/17/132/19/27/10) -------
        stage2_ok = True
        stage2_detail = []
        if _wv(21) != EXPECTED_SCHEDULERS[21]:
            stage2_ok = False
            stage2_detail.append("scheduler 21 expected %r found %r"
                                 % (EXPECTED_SCHEDULERS[21], _wv(21)))
        if _wv(20) != [EXPECTED_SAMPLERS[20]]:
            stage2_ok = False
            stage2_detail.append("sampler 20 expected %r found %r"
                                 % ([EXPECTED_SAMPLERS[20]], _wv(20)))
        if _wv(17) != [EXPECTED_CFG[17]]:
            stage2_ok = False
            stage2_detail.append("cfg 17 expected %r found %r" % ([EXPECTED_CFG[17]], _wv(17)))
        g132 = _wv(132)
        if not (isinstance(g132, list) and len(g132) > 2 and _approx(g132[2], EXPECTED_GUIDE_STRENGTHS[132])):
            stage2_ok = False
            stage2_detail.append("guide 132 strength expected %s found %r"
                                 % (EXPECTED_GUIDE_STRENGTHS[132], g132))
        if _wv(27) != EXPECTED_LTXVCONDITIONING_27:
            stage2_ok = False
            stage2_detail.append("ltxvconditioning 27 expected %r found %r"
                                 % (EXPECTED_LTXVCONDITIONING_27, _wv(27)))
        if _wv(10) != EXPECTED_MODELPREVIEW_10:
            stage2_ok = False
            stage2_detail.append("modelpreview 10 expected %r found %r"
                                 % (EXPECTED_MODELPREVIEW_10, _wv(10)))
        rep.check("Stage 2", stage2_ok, "; ".join(stage2_detail))

        # --- Final Video (VHS_VideoCombine node 139) -----------------------
        final_ok = True
        final_detail = []
        vhs = _wv(139)
        if not isinstance(vhs, dict):
            final_ok = False
            final_detail.append("VHS widgets not a dict: %r" % (vhs,))
        else:
            for key, exp in EXPECTED_VHS.items():
                found = vhs.get(key)
                if found != exp:
                    final_ok = False
                    final_detail.append("%s expected %r found %r" % (key, exp, found))
        rep.check("Final Video", final_ok, "; ".join(final_detail))

        if verbose:
            print(rep.render())
        return rep.all_pass

    return validate


# ===== CELL 11 =====
# CELL 11: Node registry (build ComfyUI NODE_CLASS_MAPPINGS). Implemented in FEAT-002.
def build_node_registry(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError(
        "CELL 11 (node registry) is implemented in FEAT-002. It requires the "
        "ComfyUI runtime + installed custom nodes, which are unavailable in the "
        "validate-only / CPU sandbox path."
    )


# ===== CELL 12 =====
# CELL 12: Memory manager (VRAM/RAM aware cleanup). Implemented in FEAT-002.
def build_memory_manager(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError(
        "CELL 12 (memory manager) is implemented in FEAT-002. It requires torch/"
        "psutil/comfy runtime, which are unavailable in the validate-only path."
    )


# ===== CELL 13 =====
# CELL 13: Load components (models, VAEs, LoRAs, CLIP). Implemented in FEAT-002.
def load_components(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError(
        "CELL 13 (load components) is implemented in FEAT-002. It requires the "
        "downloaded model weights + ComfyUI loaders (GPU only)."
    )


# ===== CELL 14 =====
# CELL 14: Initialize LTXDirector timeline into runtime state. Implemented in FEAT-002.
def init_ltxdirector_timeline(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError(
        "CELL 14 (init LTXDirector timeline runtime) is implemented in FEAT-002. "
        "Note: the pure-Python timeline PARSE lives in CELL 9 and already runs here."
    )


# ===== CELL 15 =====
# CELL 15: Execute the parsed graph via the memory-aware executor. Implemented in FEAT-002.
def execute_graph(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError(
        "CELL 15 (graph executor) is implemented in FEAT-002. It topologically "
        "runs the real ComfyUI node classes (GPU only)."
    )


# ===== CELL 16 =====
# CELL 16: Decode video + audio latents. Implemented in FEAT-002.
def decode_video_audio(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError(
        "CELL 16 (decode video/audio) is implemented in FEAT-002. It requires the "
        "VAE decoders + GPU."
    )


# ===== CELL 17 =====
# CELL 17: VHS_VideoCombine (mux to final MP4). Implemented in FEAT-002.
def vhs_combine(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError(
        "CELL 17 (VHS combine) is implemented in FEAT-002. It requires the "
        "VideoHelperSuite node + ffmpeg (GPU/Colab only)."
    )


# ===== CELL 18 =====
# CELL 18: Validate output MP4. Implemented in FEAT-002.
def validate_output(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError(
        "CELL 18 (validate output) is implemented in FEAT-002. It inspects the "
        "produced MP4, which only exists after a real Colab T4 run."
    )


# ===== CELL 19 =====
# CELL 19: Display / download the final video. Implemented in FEAT-002.
def display_download(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError(
        "CELL 19 (display/download) is implemented in FEAT-002. It uses the Colab "
        "display + files.download APIs and the produced MP4."
    )


# ===== ENTRY POINT =====
def _run_validate_only(workflow_path: Optional[str]) -> int:
    """Load JSON, parse the graph, run the validator. Returns process exit code.

    This path imports NOTHING GPU-related (stdlib only).
    """
    try:
        workflow = load_workflow_json(workflow_path)
        graph = parse_graph(workflow)
    except Exception as exc:
        print("VALIDATION SETUP ERROR: %s" % exc)
        traceback.print_exc()
        return 1

    counts = graph["counts"]
    print("Loaded workflow: %d nodes, %d links, %d groups (version %s)"
          % (counts["nodes"], counts["links"], counts["groups"], graph["version"]))
    validate = build_validator(graph)
    ok = validate(verbose=True)
    return 0 if ok else 1


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Faithfully re-execute the original LTX-2.3 Director ComfyUI graph."
    )
    parser.add_argument(
        "--validate-only", action="store_true",
        help="CPU/stdlib-only smoke test: load the JSON, parse the graph, run the "
             "graph validator, print the Section-27 report, and exit. No GPU/ComfyUI."
    )
    parser.add_argument(
        "--workflow", default=None,
        help="Path to the workflow JSON (defaults to the source-of-truth JSON next "
             "to this script)."
    )
    args = parser.parse_args(argv)

    if args.validate_only:
        return _run_validate_only(args.workflow)

    # Full pipeline requires a real Colab T4 environment (FEAT-002).
    print("This script is designed to run cell-by-cell in Google Colab (T4 GPU).")
    print("For a CPU/stdlib-only smoke test of the graph parser + validator, run:")
    print("    python3 %s --validate-only" % os.path.basename(__file__))
    return 0


if __name__ == "__main__":
    sys.exit(main())
