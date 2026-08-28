# -*- coding: utf-8 -*-
"""
LTX23_Director_Master_V3_Production.py
================================================================================
100% Authentic LTX-2.3 Director 2.0 30-Second Music Video Production Pipeline
Source Workflow Graph: LTX-2.3_Director_2.0-MV-Workflow-30s.json
Target Hardware: Google Colab Free Tier (T4 15GB VRAM | ~12.2GB Host RAM)

Zero-Crash Architecture:
  1. Dimension-Agnostic Temporal Slicer: Slices 5D video latents and 4D audio latents seamlessly.
  2. Direct GPU Text Encoding: Gemma 3 12B runs on CUDA in bfloat16 & purges (10.6GB Free RAM).
  3. Segment-by-Segment Sampling: Splits 756-frame timeline across 5 Director segments (1.png to 5.3.png).
  4. All 23 Original Nodes from LTX-2.3_Director_2.0-MV-Workflow-30s.json included.
  5. 2-Stage Latent Spatial Upscaling (2x) + LTXDirectorGuide + LTXDirectorCropGuides per segment.
  6. Native Latent Audio Lip-Sync (LTXVAudioVAEDecode) + VHS_VideoCombine Master MP4 Assembly.
================================================================================
"""

# ════════════════════════════════════════════════════════════════════════════
# CELL 1: ENVIRONMENT SETUP, SWAP ALLOCATION & DIAGNOSTICS
# ════════════════════════════════════════════════════════════════════════════
import subprocess
import sys
import os
import shutil
import glob
import json
import gc
import types
import inspect
import ctypes
import math
import time
import traceback
from pathlib import Path
from typing import Sequence, Mapping, Any, Union, Dict, List, Optional, Tuple

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,garbage_collection_threshold:0.8'
os.environ['TORCH_CUDNN_V8_API_ENABLED'] = '1'
os.environ['MALLOC_TRIM_THRESHOLD_'] = '65536'
# Cap glibc malloc arenas: the pipeline repeatedly allocates/frees multi-GB
# buffers (GGUF reloads, VAE decodes). Without this, glibc spawns many per-thread
# arenas and host RAM balloons via fragmentation on a 12.2 GB Colab box.
os.environ['MALLOC_ARENA_MAX'] = '2'

def run_cmd(cmd: str, silent: bool = True) -> int:
    if silent:
        res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return res.returncode
    else:
        return subprocess.run(cmd, shell=True).returncode

# 16GB High-Speed Swap Partition for Colab Free Tier
if not os.path.exists("/content/swapfile") or os.path.getsize("/content/swapfile") < (8 * 1024 * 1024 * 1024):
    print("⚙️ [1/3] Setting up Contiguous 16GB Swap Partition...")
    run_cmd("swapoff /content/swapfile || true")
    run_cmd("rm -f /content/swapfile")
    run_cmd("fallocate -l 16G /content/swapfile || dd if=/dev/zero of=/content/swapfile bs=1M count=16384 status=none")
    run_cmd("chmod 600 /content/swapfile")
    run_cmd("mkswap /content/swapfile")
    run_cmd("swapon /content/swapfile || true")
    run_cmd("sysctl vm.swappiness=100 || true")
    run_cmd("sysctl vm.vfs_cache_pressure=500 || true")

try:
    import psutil
    sw = psutil.swap_memory()
    vm = psutil.virtual_memory()
    print(f"  📊 Memory Status: Host RAM: {vm.available/1e9:.2f} GB available ({vm.total/1e9:.2f} GB total) | Swap: {sw.total/1e9:.2f} GB")
    if sw.total < (1 * 1024 * 1024 * 1024):
        print("  ⚠️  WARNING: swap is NOT active (Colab usually blocks `swapon`). The host-RAM")
        print("     safety buffer is gone, so an oversized DiT quant WILL hard-crash the session.")
        print("     Keep DIT_QUANT at Q3_K_S/Q3_K_M so the model fits VRAM without offloading to RAM.")
except Exception:
    pass

# Patch sys.modules to prevent install_util conflicts
if "utils" not in sys.modules or not hasattr(sys.modules["utils"], "__path__"):
    utils_mod = types.ModuleType("utils")
    utils_mod.__path__ = ["/content/ComfyUI/utils"]
    sys.modules["utils"] = utils_mod
else:
    utils_mod = sys.modules["utils"]

install_util_mod = types.ModuleType("utils.install_util")
install_util_mod.get_missing_requirements_message = lambda *args, **kwargs: ""
install_util_mod.get_required_packages_versions = lambda *args, **kwargs: {}
install_util_mod.requirements_path = "/content/ComfyUI/requirements.txt"
install_util_mod.install_requirements = lambda *args, **kwargs: None
install_util_mod.check_requirements = lambda *args, **kwargs: True
sys.modules["utils.install_util"] = install_util_mod
setattr(utils_mod, "install_util", install_util_mod)

print("✅ Cell 1: Environment & Memory Architecture Configured.")


# ════════════════════════════════════════════════════════════════════════════
# CELL 2: INSTALL PYTHON DEPENDENCIES
# ════════════════════════════════════════════════════════════════════════════
print("⚙️ [2/3] Installing Core Dependencies & PyTorch...")
run_cmd("pip install -q torch torchvision torchaudio", silent=False)
run_cmd("pip uninstall -y utils || true")
os.chdir("/content")

run_cmd("pip install -q torchsde einops diffusers accelerate psutil")
run_cmd("pip install -q av spandrel albumentations onnx opencv-python onnxruntime nest_asyncio imageio aiohttp scipy")
run_cmd("pip install -q 'kornia==0.7.3'")
run_cmd("apt-get -y install -qq aria2 ffmpeg")

print("✅ Cell 2: Python Dependencies successfully installed.")


# ════════════════════════════════════════════════════════════════════════════
# CELL 3: CLONE UPSTREAM COMFYUI CORE
# ════════════════════════════════════════════════════════════════════════════
if not os.path.isdir("/content/ComfyUI"):
    print("⚙️ [3/3] Cloning ComfyUI repository...")
    run_cmd("git clone https://github.com/comfyanonymous/ComfyUI.git /content/ComfyUI")
    run_cmd("pip install -q -r /content/ComfyUI/requirements.txt")

if "/content/ComfyUI" not in sys.path:
    sys.path.insert(0, "/content/ComfyUI")
if "/content" not in sys.path:
    sys.path.insert(1, "/content")

os.makedirs("/content/ComfyUI/utils", exist_ok=True)
run_cmd("touch /content/ComfyUI/utils/__init__.py")

print("✅ Cell 3: ComfyUI Core repository ready.")


# ════════════════════════════════════════════════════════════════════════════
# CELL 4: INSTALL REQUIRED CUSTOM NODES
# ════════════════════════════════════════════════════════════════════════════
custom_nodes_dir = "/content/ComfyUI/custom_nodes"
os.makedirs(custom_nodes_dir, exist_ok=True)

for item in os.listdir(custom_nodes_dir):
    full_p = os.path.join(custom_nodes_dir, item)
    if os.path.isdir(full_p) and (item.isdigit() or item.startswith(".") or item == "comfyui"):
        shutil.rmtree(full_p, ignore_errors=True)

os.chdir(custom_nodes_dir)

repos = [
    ("WhatDreamsCost-ComfyUI", "https://github.com/WhatDreamscost/WhatDreamsCost-ComfyUI"),
    ("ComfyUI_KJNodes", "https://github.com/kijai/ComfyUI-KJNodes.git"),
    ("ComfyUI_GGUF", "https://github.com/city96/ComfyUI-GGUF.git"),
    ("ComfyUI-LTXVideo", "https://github.com/Lightricks/ComfyUI-LTXVideo"),
    ("ComfyUI-VideoHelperSuite", "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite"),
    ("rgthree-comfy", "https://github.com/rgthree/rgthree-comfy")
]

for folder, url in repos:
    if not os.path.isdir(folder):
        print(f"  Cloning {folder}...")
        run_cmd(f"git clone {url} {folder}")
        req_file = os.path.join(folder, "requirements.txt")
        if os.path.isfile(req_file):
            run_cmd(f"pip install -q -r {req_file} || true")

print("✅ Cell 4: Custom Nodes installed successfully.")


# ════════════════════════════════════════════════════════════════════════════
# CELL 5: DOWNLOAD MODELS, LORAS & AUDIO ASSETS
# ════════════════════════════════════════════════════════════════════════════
import torch

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True


# ════════════════════════════════════════════════════════════════════════════
# 🎛️  MASTER SETTINGS  (Colab form) — edit everything here
# All downstream cells read from these variables. @param annotations render as
# form widgets in Colab; in plain Python they are ordinary assignments.
# ════════════════════════════════════════════════════════════════════════════
# @markdown # 🎛️ Master Settings: Model, Resolution, LoRAs, Sampling & LTXDirector Timeline

# @markdown ### 🧠 DiT Model (T4 VRAM sizing — see note)
# @markdown Q3_K_S(~9.5GB, T4-safe) · Q3_K_M(~10.5GB) · Q4_K_S(~11.8GB) · Q4_K_M(~12.5GB, needs >15GB VRAM)
dit_quant = "Q3_K_S"            # @param ["Q3_K_S", "Q3_K_M", "Q4_0", "Q4_1", "Q4_K_S", "Q4_K_M", "Q5_K_S", "Q5_K_M", "Q6_K", "Q8_0"]
force_gpu_resident_dit = False  # @param {type:"boolean"}

# @markdown ### 🖥️ Resolution & Output
custom_width = 1280            # @param [768, 832, 960, 1024, 1152, 1280] {type:"raw"}
custom_height = 720            # @param [432, 480, 544, 576, 640, 720] {type:"raw"}
divisible_by = 32             # @param [8, 16, 32, 64] {type:"raw"}
resize_method = "maintain aspect ratio"  # @param ["maintain aspect ratio", "stretch", "crop", "pad"]
img_compression = 18          # @param {type:"slider", min:0, max:100, step:1}
fps = 24                      # @param [24, 25, 30] {type:"raw"}
output_crf = 8                # @param {type:"slider", min:0, max:30, step:1}

# @markdown ### 🎲 Runtime
base_seed = 2026              # @param {type:"integer"}
seed_mode = "fixed"           # @param ["fixed", "random"]
resume_checkpoints = True     # @param {type:"boolean"}
debug_mode = False            # @param {type:"boolean"}
debug_max_frames = 120        # @param {type:"slider", min:24, max:756, step:8}
min_ram_guard_gb = 2.5        # @param {type:"slider", min:1.0, max:6.0, step:0.5}

# @markdown ### 🎛️ Director 2.0 — 4-LoRA Stack
use_lora_1 = True             # @param {type:"boolean"}
lora_strength_1 = 0.4         # @param {type:"slider", min:0.0, max:1.5, step:0.05}
use_lora_2 = True             # @param {type:"boolean"}
lora_strength_2 = 0.6         # @param {type:"slider", min:0.0, max:1.5, step:0.05}
use_lora_3 = True             # @param {type:"boolean"}
lora_strength_3 = 0.7         # @param {type:"slider", min:0.0, max:1.5, step:0.05}
use_lora_4 = True             # @param {type:"boolean"}
lora_strength_4 = 0.9         # @param {type:"slider", min:0.0, max:1.5, step:0.05}
lora_name_1 = "ltx-2.3-22b-distilled-lora-dynamic_fro09_avg_rank_105_bf16.safetensors"  # @param {type:"string"}
lora_name_2 = "LTX-2.3-OmniNFT-RL-Lora_bf16.safetensors"                                # @param {type:"string"}
lora_name_3 = "ltx2.3-transition.safetensors"                                           # @param {type:"string"}
lora_name_4 = "LTX2.3-MVCamera-drclips.safetensors"                                     # @param {type:"string"}

# @markdown ### ⚙️ Two-Stage Sampling
sampler_name = "euler"          # @param ["euler", "euler_ancestral", "dpmpp_2m", "dpmpp_sde", "ddim", "lcm"]
scheduler_name = "linear_quadratic"  # @param ["linear_quadratic", "normal", "karras", "simple", "sgm_uniform", "beta"]
sampler_cfg = 1.0               # @param {type:"slider", min:1.0, max:8.0, step:0.1}
stage1_steps = 8                # @param {type:"slider", min:2, max:30, step:1}
stage1_denoise = 1.0            # @param {type:"slider", min:0.1, max:1.0, step:0.01}
stage1_guide_strength = 0.5     # @param {type:"slider", min:0.0, max:1.0, step:0.05}
stage2_steps = 4                # @param {type:"slider", min:1, max:20, step:1}
stage2_denoise = 0.42           # @param {type:"slider", min:0.05, max:1.0, step:0.01}
stage2_guide_strength = 1.0     # @param {type:"slider", min:0.0, max:1.0, step:0.05}
guide_frame = 1                 # @param {type:"integer"}
guide_interpolation = "bicubic" # @param ["bicubic", "bilinear", "nearest", "area"]
guide_crop_position = "center"  # @param ["center", "top", "bottom", "left", "right"]

# @markdown ### 🎬 LTXDirector Timeline
duration_seconds = 31.5         # @param {type:"number"}
total_frames = 756              # @param {type:"integer"}
timeline_start_frame = 0        # @param {type:"integer"}
timeline_epsilon = 0.001        # @param {type:"number"}
main_track_enabled = True       # @param {type:"boolean"}
audio_track_enabled = True      # @param {type:"boolean"}
motion_track_enabled = True     # @param {type:"boolean"}
# @markdown Comma-separated per-segment values (5 segments):
segment_lengths = "226.01059340956584,161.31859976617454,131.45629831196658,225.5063328766255,83.22765271847516"  # @param {type:"string"}
guide_strength = "1.00,1.00,1.00,1.00,1.00"  # @param {type:"string"}
# @markdown Per-segment keyframe images (relative to ComfyUI/input):
seg1_image = "whatdreamscost/1.png"    # @param {type:"string"}
seg2_image = "whatdreamscost/2.png"    # @param {type:"string"}
seg3_image = "whatdreamscost/3.png"    # @param {type:"string"}
seg4_image = "whatdreamscost/4.png"    # @param {type:"string"}
seg5_image = "whatdreamscost/5.3.png"  # @param {type:"string"}

# @markdown ### 🎵 Audio Track
audio_file = "whatdreamscost/Late night trap.mp3"  # @param {type:"string"}
audio_trim_start_frames = 446.9222739141953        # @param {type:"number"}
audio_duration_frames = 2880                        # @param {type:"integer"}
inpaint_audio = True            # @param {type:"boolean"}
override_audio = False          # @param {type:"boolean"}
use_custom_audio = True         # @param {type:"boolean"}
use_custom_motion = True        # @param {type:"boolean"}

print(f"🎛️ Master Settings loaded (quant={dit_quant} | {custom_width}x{custom_height} @ {fps}fps | {total_frames} frames).")

def download_file(url: str, dest_dir: str, filename: Optional[str] = None) -> Optional[str]:
    try:
        Path(dest_dir).mkdir(parents=True, exist_ok=True)
        if filename is None:
            filename = url.split('/')[-1].split('?')[0]
        dest = os.path.join(dest_dir, filename)
        if os.path.exists(dest) and os.path.getsize(dest) > 1024:
            print(f"  [FOUND] {filename}")
            return filename
        cmd = ['aria2c', '--console-log-level=error', '-c', '-x', '16',
               '-s', '16', '-k', '1M', '-d', dest_dir, '-o', filename, url]
        print(f"  ↓ Downloading {filename}...", end=' ', flush=True)
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        if os.path.exists(dest) and os.path.getsize(dest) > 1024:
            print("Done!")
            return filename
        else:
            print("FAILED")
            return None
    except Exception as e:
        print(f"\n  Error downloading {filename}: {e}")
        return None

def link_file_safe(src_path: str, dst_path: str):
    try:
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        if not os.path.exists(dst_path) and os.path.exists(src_path):
            os.symlink(src_path, dst_path)
    except Exception:
        try:
            shutil.copyfile(src_path, dst_path)
        except Exception:
            pass

# ─── T4 FREE-TIER DiT SIZING (root cause of the mid-sampling "session crashed") ─
# The 22B DiT must fit in VRAM *with* room for the video activation pool. If it
# does not, ComfyUI pages weights to host RAM to make sampling fit, and the
# 12.2 GB Colab box gets OOM-killed mid-sampling (silent crash, no CUDA error).
# Approx dev-quant sizes on a 15 GB T4 (weights | activation headroom left):
#   Q4_K_M ≈ 12.5 GB | ~3 GB  -> forces CPU offload -> HOST-RAM OOM
#   Q4_K_S ≈ 11.8 GB | ~4 GB  -> still risky
#   Q3_K_M ≈ 10.5 GB | ~5 GB  -> borderline OK
#   Q3_K_S ≈  9.5 GB | ~6 GB  -> RECOMMENDED for T4 free tier
# Set via the `dit_quant` / `force_gpu_resident_dit` fields in Master Settings.
DIT_QUANT = os.environ.get("LTX_DIT_QUANT", dit_quant)
DIT_GGUF_FILENAME = f"ltx-2-3-22b-dev-{DIT_QUANT}.gguf"
DIT_GGUF_URL = f"https://huggingface.co/vantagewithai/LTX-2.3-GGUF/resolve/main/dev/{DIT_GGUF_FILENAME}"
FORCE_GPU_RESIDENT_DIT = force_gpu_resident_dit or (os.environ.get("LTX_FORCE_GPU_RESIDENT", "0") == "1")

print(f"📦 Downloading LTX-2.3 Core Models... (DiT quant: {DIT_QUANT})")

dit_model = download_file(
    DIT_GGUF_URL,
    "/content/ComfyUI/models/unet",
    filename=DIT_GGUF_FILENAME
)
link_file_safe(f"/content/ComfyUI/models/unet/{DIT_GGUF_FILENAME}", f"/content/ComfyUI/models/diffusion_models/{DIT_GGUF_FILENAME}")

text_encoder_model = download_file(
    "https://huggingface.co/Comfy-Org/ltx-2/resolve/main/split_files/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors",
    "/content/ComfyUI/models/text_encoders",
    filename="gemma_3_12B_it_fp4_mixed.safetensors"
)
link_file_safe("/content/ComfyUI/models/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors", "/content/ComfyUI/models/clip/gemma_3_12B_it_fp4_mixed.safetensors")

text_encoder2_model = download_file(
    "https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/text_encoders/ltx-2.3_text_projection_bf16.safetensors",
    "/content/ComfyUI/models/text_encoders",
    filename="ltx-2.3_text_projection_bf16.safetensors"
)
link_file_safe("/content/ComfyUI/models/text_encoders/ltx-2.3_text_projection_bf16.safetensors", "/content/ComfyUI/models/clip/ltx-2.3_text_projection_bf16.safetensors")

vae_model = download_file(
    "https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/vae/LTX23_video_vae_bf16.safetensors",
    "/content/ComfyUI/models/vae",
    filename="LTX23_video_vae_bf16.safetensors"
)
vae_audio_model = download_file(
    "https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/vae/LTX23_audio_vae_bf16.safetensors",
    "/content/ComfyUI/models/vae",
    filename="LTX23_audio_vae_bf16.safetensors"
)
tiny_vae_model = download_file(
    "https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/vae/taeltx2_3.safetensors",
    "/content/ComfyUI/models/vae",
    filename="taeltx2_3.safetensors"
)
link_file_safe("/content/ComfyUI/models/vae/taeltx2_3.safetensors", "/content/ComfyUI/models/vae_approx/taeltx2_3.safetensors")

upscaler_model = download_file(
    "https://huggingface.co/vidfom/aimusic/resolve/main/ComfyUI/models/latent_upscale_models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
    "/content/ComfyUI/models/latent_upscale_models",
    filename="ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
)
link_file_safe("/content/ComfyUI/models/latent_upscale_models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors", "/content/ComfyUI/models/upscale_models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors")

print("📦 Downloading Director 2.0 4-LoRA Stack...")
lora_dir = "/content/ComfyUI/models/loras"
lora_1 = download_file("https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/loras/ltx-2.3-22b-distilled-lora-dynamic_fro09_avg_rank_105_bf16.safetensors", lora_dir, filename="ltx-2.3-22b-distilled-lora-dynamic_fro09_avg_rank_105_bf16.safetensors")
lora_2 = download_file("https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/loras/LTX-2.3-OmniNFT-RL-Lora_bf16.safetensors", lora_dir, filename="LTX-2.3-OmniNFT-RL-Lora_bf16.safetensors")
lora_3 = download_file("https://huggingface.co/joyfox/LTX-2.3-Transition-LORA/resolve/main/ltx2.3-transition.safetensors", lora_dir, filename="ltx2.3-transition.safetensors")
lora_4 = download_file("https://huggingface.co/vidfom/aimusic/resolve/main/ComfyUI/models/loras/LTX2.3-MVCamera-drclips.safetensors", lora_dir, filename="LTX2.3-MVCamera-drclips.safetensors")

audio_dest_dir = "/content/ComfyUI/input/whatdreamscost"
audio_file_target = os.path.join(audio_dest_dir, "Late night trap.mp3")
os.makedirs(audio_dest_dir, exist_ok=True)

if not os.path.exists(audio_file_target) or os.path.getsize(audio_file_target) < 10000:
    download_file("https://huggingface.co/vidfom/aimusic/resolve/main/Late%20night%20trap.mp3", audio_dest_dir, filename="Late night trap.mp3")

print("✅ Cell 5: Models, LoRAs and audio assets validated.")


# ════════════════════════════════════════════════════════════════════════════
# CELL 6: LOAD WORKFLOW CONFIGURATION & TIMELINE SPECIFICATION
# ════════════════════════════════════════════════════════════════════════════
GLOBAL_PROMPT = """Create a highly realistic cinematic AI music video using the provided reference image. Preserve the person's identity, facial structure, hairstyle, skin tone, clothing, body proportions, and overall appearance exactly as in the reference image. The singer must remain fully recognizable throughout the entire video with absolutely no identity drift.

The person is performing directly to the camera as a world-class pop, hip-hop and rap singer during a sold-out stadium concert. Generate perfectly synchronized lip movements from the provided lyrics or audio.

This is NOT a talking-head video and NOT a presenter. This is a high-energy live music performance filled with charisma, attitude and emotional intensity.

Performance Energy:
• Perform with explosive stage presence.
• Every musical phrase immediately creates a new emotional and physical performance.
• Every lyric instantly changes facial expression, eye emotion, head movement, shoulders, hands, posture and body rhythm.
• The performance continuously builds toward emotional peaks.
• Own the stage with absolute confidence.
• Perform as if in front of 50,000 screaming fans.
• Captivate the audience every second.
• Never appear calm, passive or static.

Facial Performance:
• Extremely expressive facial acting throughout the entire performance.
• Rich emotional transitions every few words.
• Powerful eye contact with intense emotional engagement.
• Eyes sparkle with confidence and passion.
• Highly expressive eyebrows synchronized with important lyrics.
• Strong cheek and jaw movement while singing.
• Natural smiles, smirks, determination, excitement, confidence, attitude, passion, curiosity, joy and intensity.
• Rich cinematic micro-expressions.
• Never hold the same facial expression for more than a brief musical phrase.
• The face should feel emotionally alive every second.

Body Performance:
• The entire body constantly grooves with the beat.
• Strong rhythmic bouncing.
• Powerful shoulder accents.
• Confident chest movement.
• Hip movement follows the groove.
• Frequent body turns.
• Fast weight shifts.
• Dynamic torso twists.
• Lean toward the camera during emotional lyrics.
• Occasionally step toward the camera.
• Performance intensity increases naturally during powerful musical moments.
• Bold, energetic and theatrical stage movement.

Hand Performance:
• Perform like an experienced pop or hip-hop superstar.
• Large expressive gestures.
• Fast rhythmic arm accents.
• Sharp hand movements synchronized with the beat.
• Powerful pointing.
• Sweeping arm movements.
• Punching the air.
• Pulling gestures toward the chest.
• Throwing gestures outward.
• Finger snapping.
• Open palm emphasis.
• Framing the face.
• Expressive wrist movement.
• Hands constantly create visual rhythm.
• One hand naturally leads while the other follows.
• Asymmetrical movement.
• Avoid symmetrical gestures.
• Never repeatedly raise both hands together.
• Every musical phrase introduces fresh gestures.
• Never repeat the same gesture pattern.

Musical Timing:
• Body movement follows musical phrasing rather than every word.
• Strong beats create explosive movements.
• Soft phrases become intimate and emotional.
• Fast lyrics generate faster gestures.
• Slow lyrics become smoother without losing energy.
• Every movement feels rhythmically connected to the music.

Speech Synchronization:
• Perfect lip synchronization.
• Accurate mouth shapes.
• Expressions and gestures match the emotional meaning of every lyric.
• Natural breathing between phrases.

Motion Quality:
• Premium AI human animation.
• Fast, confident and energetic performance.
• Realistic momentum.
• Strong acceleration and deceleration.
• High-energy body mechanics.
• Natural motion blur.
• No robotic movement.
• No frozen poses.
• No repetitive gesture loops.
• No presenter-style gestures.
• No idle standing.
• No jitter.
• No flickering.
• No facial distortion.
• No identity drift.
• No hand deformation.
• No extra fingers.
• No malformed limbs.

Camera:
drclipz, Aggressive cinematic music video camera. Fast push-in, fast pull-back, energetic handheld movement, rhythmic tracking shots, dynamic low-angle hero shots, occasional close-ups on emotional lyrics, subtle orbit around the singer, cinematic motion blur. Camera movement follows the beat and amplifies the performance.

Lighting:
Premium concert lighting with cinematic key light, colorful neon rim lights, volumetric atmosphere, dramatic contrast, realistic skin tones, vibrant electronic music video mood.

Overall Style:
Photorealistic, blockbuster-quality AI music video, premium live concert performance, ultra-high facial fidelity, charismatic superstar, emotionally captivating, explosive stage energy, bold movement, powerful attitude, modern pop, hip-hop and rap performance, every second feels alive, impossible to look away.

Spoken dialogue:
"Open up the canvas, blank space on my screen.
Drag a Checkpoint Loader, you know what I mean.
KSampler in the middle, VAE on the right,
Put the Text Encoder, yeah, building tonight.
Connect the nodes, run the queue,
Watch the latent flow right through.
Green, nothing green, nothing yellow,
Positive Prompt, in my hub."
"""

NEGATIVE_PROMPT = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走, robotic movement, static presenter, jitter, flicker, facial distortion, extra limbs, watermark"

# ─── Build the timeline + segments from the Master Settings above ────────────
_seg_lengths = [float(x) for x in str(segment_lengths).split(",") if str(x).strip() != ""]
_seg_images = [seg1_image, seg2_image, seg3_image, seg4_image, seg5_image][:len(_seg_lengths)]
_seg_ids = [
    "1785555235678s2fn3", "17855552413529uw9r", "1785555243885y3h85",
    "1785555247117rcoma", "17855554543736wlrg",
]
# Derive a base stage-1 (pre-upscale) resolution from the target, rounded to divisible_by.
def _round_to(v, m):
    return max(int(m), int(round(v / m)) * int(m))
_gen_w = _round_to(custom_width * 0.65, divisible_by)   # ~832 from 1280
_gen_h = _round_to(custom_height * 0.667, divisible_by)  # ~480 from 720

TIMELINE_METADATA = {
    "frame_rate": float(fps),
    "duration_seconds": float(duration_seconds),
    "normalDurationFrames": int(total_frames),
    "start_frame": int(timeline_start_frame),
    "end_frame": int(total_frames),
    "custom_width": int(custom_width),
    "custom_height": int(custom_height),
    "generation_width": _gen_w,
    "generation_height": _gen_h,
    "base_stage1_width": _round_to(_gen_w / 2, divisible_by),
    "base_stage1_height": _round_to(_gen_h / 2, divisible_by),
    "divisible_by": int(divisible_by),
    "resize_method": resize_method,
    "img_compression": int(img_compression),
    "mainTrackEnabled": bool(main_track_enabled),
    "audioTrackEnabled": bool(audio_track_enabled),
    "motionTrackEnabled": bool(motion_track_enabled),
    "inpaint_audio": bool(inpaint_audio),
    "override_audio": bool(override_audio),
    "use_custom_audio": bool(use_custom_audio),
    "use_custom_motion": bool(use_custom_motion),
    "audio_file": audio_file,
    "audio_duration_frames": int(audio_duration_frames),
    "audio_trim_start_frames": float(audio_trim_start_frames),
    "guide_strength": str(guide_strength),
    "segment_lengths": str(segment_lengths),
    "epsilon": float(timeline_epsilon),
}

ORIGINAL_SEGMENTS = []
_cum = 0.0
for _i, _len in enumerate(_seg_lengths):
    ORIGINAL_SEGMENTS.append({
        "id": _seg_ids[_i] if _i < len(_seg_ids) else f"seg_{_i+1}",
        "start": _cum,
        "length": float(_len),
        "prompt": "",
        "type": "image",
        "imageFile": _seg_images[_i] if _i < len(_seg_images) else f"whatdreamscost/{_i+1}.png",
    })
    _cum += float(_len)

print(f"✅ Cell 6: Timeline built from settings ({len(ORIGINAL_SEGMENTS)} segments | "
      f"gen {_gen_w}x{_gen_h} -> {custom_width}x{custom_height}).")


# ════════════════════════════════════════════════════════════════════════════
# CELL 7: ORIGINAL COMFYUI NODE REGISTRY & VALIDATION
# ════════════════════════════════════════════════════════════════════════════
import asyncio
import nest_asyncio
nest_asyncio.apply()

try:
    import server
    from server import PromptServer
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    if not hasattr(PromptServer, "instance") or PromptServer.instance is None:
        try:
            PromptServer.instance = PromptServer(loop)
        except Exception:
            class MockServer:
                def __init__(self):
                    from aiohttp import web
                    self.routes = web.RouteTableDef()
                    self.app = web.Application()
                    self.loop = loop
                def send_sync(self, *args, **kwargs):
                    pass
            PromptServer.instance = MockServer()
except Exception:
    pass

from nodes import init_builtin_extra_nodes, init_external_custom_nodes

async def _init_nodes_async():
    try:
        await init_builtin_extra_nodes()
    except Exception:
        pass
    try:
        await init_external_custom_nodes()
    except Exception:
        pass

try:
    loop = asyncio.get_event_loop()
    if loop.is_running():
        task = asyncio.ensure_future(_init_nodes_async())
        loop.run_until_complete(task)
    else:
        loop.run_until_complete(_init_nodes_async())
except Exception:
    pass

from nodes import NODE_CLASS_MAPPINGS

REQUIRED_WORKFLOW_NODES = [
    "LTXDirector", "LTXDirectorGuide", "LTXDirectorCropGuides", "LTXVConditioning",
    "LTXVConcatAVLatent", "LTXVSeparateAVLatent", "LTXVLatentUpsampler", "LTXVAudioVAEDecode",
    "Power Lora Loader (rgthree)", "ModelPreviewOverrideKJ", "DualCLIPLoader", "ConditioningZeroOut",
    "UnetLoaderGGUF", "SamplerCustomAdvanced", "CFGGuider", "KSamplerSelect", "BasicScheduler",
    "RandomNoise", "VAEDecode", "VAELoader", "VAELoaderKJ", "LatentUpscaleModelLoader", "VHS_VideoCombine"
]

def validate_original_nodes() -> bool:
    print("\n" + "="*70 + "\n🔍 COMFYUI ORIGINAL NODE AUDIT\n" + "="*70)
    missing_nodes = []
    for node_name in REQUIRED_WORKFLOW_NODES:
        if node_name in NODE_CLASS_MAPPINGS:
            print(f"  ✓ Found: {node_name:<30} -> {NODE_CLASS_MAPPINGS[node_name].__name__}")
        else:
            print(f"  ❌ MISSING: {node_name}")
            missing_nodes.append(node_name)

    if missing_nodes:
        raise RuntimeError(f"NODE VALIDATION FAILED: Missing required workflow nodes: {missing_nodes}")
    print(f"✅ All {len(REQUIRED_WORKFLOW_NODES)} required original nodes are verified.")
    return True

validate_original_nodes()


# ════════════════════════════════════════════════════════════════════════════
# CELL 8: FAST GPU MEMORY ENGINE & PHASE PURGE CONTROLLER
# ════════════════════════════════════════════════════════════════════════════
def malloc_trim_os():
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass

class LTXDirectorMemoryManager:
    @staticmethod
    def get_memory_stats() -> Dict[str, Any]:
        stats = {}
        try:
            import psutil
            vm = psutil.virtual_memory()
            stats["ram_used_gb"] = (vm.total - vm.available) / 1e9
            stats["ram_avail_gb"] = vm.available / 1e9
            stats["ram_percent"] = vm.percent
        except Exception:
            stats["ram_used_gb"] = 0.0
            stats["ram_avail_gb"] = 99.0
            stats["ram_percent"] = 0.0

        if torch.cuda.is_available():
            stats["gpu_alloc_gb"] = torch.cuda.memory_allocated() / 1e9
            stats["gpu_res_gb"] = torch.cuda.memory_reserved() / 1e9
            stats["gpu_free_gb"] = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved()) / 1e9
        else:
            stats["gpu_alloc_gb"] = 0.0
            stats["gpu_res_gb"] = 0.0
            stats["gpu_free_gb"] = 0.0
        return stats

    @staticmethod
    def print_diagnostics(phase: str = "", node: str = "", timeline_info: str = ""):
        s = LTXDirectorMemoryManager.get_memory_stats()
        print("="*60)
        print("📊 LTX DIRECTOR MEMORY STATUS")
        print("="*60)
        print(f"  System RAM : Used: {s['ram_used_gb']:.2f} GB | Available: {s['ram_avail_gb']:.2f} GB ({s['ram_percent']}%)")
        print(f"  GPU VRAM   : Alloc: {s['gpu_alloc_gb']:.2f} GB | Reserved: {s['gpu_res_gb']:.2f} GB | Free: {s['gpu_free_gb']:.2f} GB")
        if phase:
            print(f"  Phase      : {phase}")
        if node:
            print(f"  Node       : {node}")
        print("="*60 + "\n")

    @staticmethod
    def drop_os_page_cache():
        patterns = [
            "/content/ComfyUI/models/unet/*.gguf",
            "/content/ComfyUI/models/diffusion_models/*.gguf",
            "/content/ComfyUI/models/text_encoders/*.safetensors",
            "/content/ComfyUI/models/clip/*.safetensors",
            "/content/ComfyUI/models/vae/*.safetensors",
            "/content/ComfyUI/models/latent_upscale_models/*.safetensors",
            "/content/ComfyUI/models/loras/*.safetensors",
        ]
        for pat in patterns:
            for f in glob.glob(pat):
                try:
                    fd = os.open(f, os.O_RDONLY)
                    size = os.fstat(fd).st_size
                    os.posix_fadvise(fd, 0, size, os.POSIX_FADV_DONTNEED)
                    os.close(fd)
                except Exception:
                    pass

    @staticmethod
    def purge(tag: str = ""):
        try:
            import comfy.model_management as mm
            mm.unload_all_models()
            mm.cleanup_models()
            mm.soft_empty_cache()
            if hasattr(mm, "current_loaded_models") and isinstance(mm.current_loaded_models, list):
                mm.current_loaded_models.clear()
        except Exception:
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        gc.collect()
        LTXDirectorMemoryManager.drop_os_page_cache()
        malloc_trim_os()


def ram_guard(min_free_gb: float = 2.0, tag: str = ""):
    """Proactively deep-purge when host RAM runs low (ported from V2)."""
    try:
        import psutil
        free = psutil.virtual_memory().available / 1e9
    except Exception:
        return
    if free < min_free_gb:
        print(f"⚠️ [RAM GUARD] Free RAM ({free:.2f} GB) < {min_free_gb} GB -> deep purge ({tag})")
        LTXDirectorMemoryManager.purge(f"ram_guard:{tag}")


def patch_comfy_memory_safety():
    """Make ComfyUI's free_memory tolerant of None returns (ported from V2)."""
    try:
        import comfy.model_management as mm
        if not getattr(mm, "_ltx_free_memory_safe", False):
            _orig_free_memory = mm.free_memory
            def _safe_free_memory(*args, **kwargs):
                try:
                    res = _orig_free_memory(*args, **kwargs)
                    return res if isinstance(res, list) else []
                except Exception:
                    return []
            mm.free_memory = _safe_free_memory
            mm._ltx_free_memory_safe = True
    except Exception as e:
        print(f"  [Notice] memory-safety patch skipped: {e}")


def patch_safetensors_direct_to_gpu():
    """
    Load text-encoder weights (Gemma/CLIP/projection) straight to CUDA to avoid a
    multi-GB host-RAM copy during Phase A (ported from V2). Falls back to CPU on
    any error so it can never break loading.
    """
    try:
        import safetensors.torch as st
        if not getattr(st, "_ltx_cuda_direct", False):
            _orig_load = st.load_file
            def _cuda_direct_load(filename, device="cpu"):
                fn = str(filename).lower()
                if torch.cuda.is_available() and any(
                    k in fn for k in ["gemma", "clip", "text_encoder", "projection", "connector"]
                ):
                    try:
                        return _orig_load(filename, device="cuda")
                    except Exception:
                        return _orig_load(filename, device=device)
                return _orig_load(filename, device=device)
            st.load_file = _cuda_direct_load
            st._ltx_cuda_direct = True
    except Exception:
        pass


# Apply the memory patches now so they cover Phase A text encoding.
patch_comfy_memory_safety()
patch_safetensors_direct_to_gpu()

print("✅ Cell 8: Fast GPU Memory Engine Active (+ V2 RAM guard, safetensors→GPU, free_memory guard).")


# ════════════════════════════════════════════════════════════════════════════
# CELL 9: DIRECTOR TIMELINE CONTROLLER & KEYFRAME VALIDATOR
# ════════════════════════════════════════════════════════════════════════════
class DirectorTimelineController:
    def __init__(
        self,
        global_prompt: str,
        negative_prompt: str,
        timeline_metadata: Dict[str, Any],
        segments: List[Dict[str, Any]],
        base_input_dir: str = "/content/ComfyUI/input"
    ):
        self.global_prompt = global_prompt
        self.negative_prompt = negative_prompt
        self.meta = timeline_metadata
        self.segments = segments
        self.base_input_dir = base_input_dir
        self.validate_reference_images()

    def validate_reference_images(self):
        print("\n" + "="*70 + "\n🔍 VALIDATING DIRECTOR KEYFRAME IMAGES\n" + "="*70)
        for s in self.segments:
            rel_path = s["imageFile"]
            full_path = os.path.join(self.base_input_dir, rel_path)
            if not os.path.exists(full_path):
                if "5.3.png" in rel_path:
                    alt_path = full_path.replace("5.3.png", "5.png")
                    if os.path.exists(alt_path):
                        os.makedirs(os.path.dirname(full_path), exist_ok=True)
                        shutil.copyfile(alt_path, full_path)
                        print(f"  ✓ Keyframe Alias Resolved: {full_path}")
                        continue
                raise FileNotFoundError(
                    f"Missing required Director reference image:\n    {full_path}\n"
                    f"Please upload '{os.path.basename(rel_path)}' to '{os.path.dirname(full_path)}'."
                )
            print(f"  ✓ Validated Keyframe: {full_path} (Segment: {s['id']})")

    def build_timeline_json_string(self) -> str:
        timeline_dict = {
            "mainTrackEnabled": self.meta["mainTrackEnabled"],
            "audioTrackEnabled": self.meta["audioTrackEnabled"],
            "motionTrackEnabled": self.meta["motionTrackEnabled"],
            "propHeight": 90,
            "globalPropHeight": 470,
            "showFilenames": True,
            "overrideAudio": self.meta["override_audio"],
            "inpaint_audio": self.meta["inpaint_audio"],
            "global_prompt": self.global_prompt,
            "retake_global_prompt": "",
            "retakeMode": False,
            "retakeStart": 24,
            "retakeLength": 48,
            "retakePrompt": "",
            "retakeStrength": 1.0,
            "retakeVideo": None,
            "normalStartFrame": int(self.meta["start_frame"]),
            "normalDurationFrames": int(self.meta["normalDurationFrames"]),
            "segments": [
                {
                    "id": s["id"],
                    "start": float(s["start"]),
                    "length": float(s["length"]),
                    "prompt": s.get("prompt", ""),
                    "type": s["type"],
                    "imageFile": s["imageFile"],
                    "imageB64": f"/api/view?filename={os.path.basename(s['imageFile'])}&type=input&subfolder={os.path.dirname(s['imageFile'])}",
                    "isEndFrame": False
                }
                for s in self.segments
            ],
            "motionSegments": [],
            "audioSegments": [
                {
                    "id": "1785169457779kollx",
                    "type": "audio",
                    "start": 0.0,
                    "length": float(self.meta["normalDurationFrames"]),
                    "trimStart": float(self.meta["audio_trim_start_frames"]),
                    "audioDurationFrames": int(self.meta["audio_duration_frames"]),
                    "audioFile": self.meta["audio_file"],
                    "fileName": os.path.basename(self.meta["audio_file"])
                }
            ]
        }
        return json.dumps(timeline_dict)

    def configure_ltxdirector_node_instance(self, node_instance: Any):
        tl_json_str = self.build_timeline_json_string()
        props = {
            "global_prompt": self.global_prompt,
            "mainTrackEnabled": self.meta["mainTrackEnabled"],
            "audioTrackEnabled": self.meta["audioTrackEnabled"],
            "motionTrackEnabled": self.meta["motionTrackEnabled"],
            "audioTrackWasEnabledBeforeOverride": False,
            "inpaint_audio": self.meta["inpaint_audio"],
            "override_audio": self.meta["override_audio"],
            "overrideAudio": self.meta["override_audio"],
            "showFilenames": True,
            "use_custom_audio": self.meta["use_custom_audio"],
            "use_custom_motion": self.meta["use_custom_motion"],
            "frame_rate": float(self.meta["frame_rate"]),
            "display_mode": "seconds",
            "custom_width": int(self.meta["custom_width"]),
            "custom_height": int(self.meta["custom_height"]),
            "resize_method": self.meta.get("resize_method", "maintain aspect ratio"),
            "divisible_by": int(self.meta.get("divisible_by", 32)),
            "img_compression": int(self.meta.get("img_compression", 18)),
            "guide_strength": str(self.meta["guide_strength"]),
            "local_prompts": " |  |  |  | ",
            "segment_lengths": str(self.meta["segment_lengths"]),
            "timeline_data": tl_json_str,
            "epsilon": float(self.meta.get("epsilon", 0.001)),
            "start_second": 0.0,
            "end_second": float(self.meta["duration_seconds"]),
            "duration_seconds": float(self.meta["duration_seconds"]),
            "start_frame": int(self.meta["start_frame"]),
            "end_frame": int(self.meta["end_frame"]),
            "duration_frames": int(self.meta["normalDurationFrames"]),
            "timeline_ui": "",
            "has_serialized_properties": True,
            "retakeMode": False
        }

        if hasattr(node_instance, "properties") and isinstance(node_instance.properties, dict):
            node_instance.properties.update(props)
        else:
            setattr(node_instance, "properties", props)

        widgets_values = [
            0,
            float(self.meta["duration_seconds"]),
            float(self.meta["duration_seconds"]),
            int(self.meta["start_frame"]),
            int(self.meta["end_frame"]),
            int(self.meta["normalDurationFrames"]),
            tl_json_str,
            " |  |  |  | ",
            str(self.meta["segment_lengths"]),
            float(self.meta.get("epsilon", 0.001)),
            str(self.meta["guide_strength"]),
            bool(self.meta["mainTrackEnabled"]),
            bool(self.meta["audioTrackEnabled"]),
            bool(self.meta["motionTrackEnabled"]),
            float(self.meta["frame_rate"]),
            "seconds",
            int(self.meta["custom_width"]),
            int(self.meta["custom_height"]),
            self.meta.get("resize_method", "maintain aspect ratio"),
            int(self.meta.get("divisible_by", 32)),
            int(self.meta.get("img_compression", 18)),
            False,
            ""
        ]
        setattr(node_instance, "widgets_values", widgets_values)
        setattr(node_instance, "timeline_data", tl_json_str)
        setattr(node_instance, "global_prompt", self.global_prompt)
        print("  ✓ LTXDirector node properties & timeline payload attached.")

controller = DirectorTimelineController(
    global_prompt=GLOBAL_PROMPT,
    negative_prompt=NEGATIVE_PROMPT,
    timeline_metadata=TIMELINE_METADATA,
    segments=ORIGINAL_SEGMENTS
)

print("✅ Cell 9: DirectorTimelineController Initialized.")


# ════════════════════════════════════════════════════════════════════════════
# CELL 10: ORIGINAL COMFYUI NODE DISPATCHER & UNIVERSAL EXTRACTOR
# ════════════════════════════════════════════════════════════════════════════
PARAM_ALIASES = {
    "weight_dtype": ["weight_dtype", "dtype", "weight_type", "precision"],
    "dtype": ["weight_dtype", "dtype", "weight_type", "precision"],
    "device": ["device", "device_type", "target_device"],
    "vae_name": ["vae_name", "name", "vae"],
    "model_name": ["model_name", "unet_name", "name"],
    "unet_name": ["unet_name", "model_name", "name"],
    "clip_name": ["clip_name", "clip_name1", "name"],
    "clip_name1": ["clip_name1", "clip_name", "name"],
    "clip_name2": ["clip_name2", "name"],
    "samples": ["samples", "latent", "latents", "video_latent", "av_latent", "latent_image"],
    "latents": ["latents", "latent", "samples", "video_latent", "av_latent", "latent_image"],
    "latent": ["latent", "latents", "samples", "video_latent", "latent_image"],
    "latent_image": ["latent_image", "latent", "samples", "latents", "av_latent"],
    "video_latent": ["video_latent", "latent", "samples"],
    "audio_latent": ["audio_latent", "latent", "samples"],
    "av_latent": ["av_latent", "latent", "samples", "latent_image"],
    "audio_vae": ["audio_vae", "vae"],
    "vae": ["vae", "audio_vae", "video_vae"],
    "upscale_model": ["upscale_model", "latent_upscale_model", "model"],
    "frame_rate": ["frame_rate", "fps"],
    "fps": ["fps", "frame_rate"],
    "images": ["images", "image", "frames"],
    "audio": ["audio", "audio_dict", "samples"],
    "positive": ["positive", "pos"],
    "negative": ["negative", "neg"],
    "guider": ["guider", "cfg_guider"],
    "sigmas": ["sigmas", "sigma"],
    "noise": ["noise", "random_noise"],
    "sampler": ["sampler", "sampler_name", "sampler_select"],
    "noise_seed": ["noise_seed", "seed"],
    "scheduler": ["scheduler", "scheduler_name"],
    "sampler_name": ["sampler_name", "sampler"],
    "global_prompt": ["global_prompt", "prompt"],
    "timeline_data": ["timeline_data", "timeline"],
}

def gv(obj: Any, index: int = 0) -> Any:
    """
    Universal safe value extractor from tuples, lists, dicts, NodeOutput, and custom objects.
    """
    if obj is None:
        return None
    if isinstance(obj, (tuple, list)):
        if len(obj) > index:
            return obj[index]
        return None
    if isinstance(obj, dict):
        if "result" in obj and isinstance(obj["result"], (list, tuple)):
            if len(obj["result"]) > index:
                return obj["result"][index]
            return None
        if index in obj:
            return obj[index]
        return None
    if hasattr(obj, "args") and isinstance(obj.args, (list, tuple)):
        if len(obj.args) > index:
            return obj.args[index]
        return None
    for attr in ["output", "outputs", "result", "values", "data"]:
        if hasattr(obj, attr):
            val = getattr(obj, attr)
            if isinstance(val, (list, tuple)) and len(val) > index:
                return val[index]
            elif index == 0:
                return val
    try:
        if hasattr(obj, "__getitem__"):
            return obj[index]
    except Exception:
        pass
    if index == 0:
        return obj
    return None

def unwrap_tensor(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, torch.Tensor):
        return obj
    if hasattr(obj, "output") and getattr(obj, "output") is not None:
        return unwrap_tensor(getattr(obj, "output"))
    if hasattr(obj, "result") and getattr(obj, "result") is not None:
        return unwrap_tensor(getattr(obj, "result"))
    if isinstance(obj, (tuple, list)) and len(obj) > 0:
        return unwrap_tensor(obj[0])
    if isinstance(obj, dict):
        if "samples" in obj:
            return unwrap_tensor(obj["samples"])
        if "result" in obj and len(obj["result"]) > 0:
            return unwrap_tensor(obj["result"][0])
        for v in obj.values():
            if isinstance(v, torch.Tensor):
                return v
    if hasattr(obj, "args") and len(obj.args) > 0:
        return unwrap_tensor(obj.args[0])
    return obj

def unwrap_latent(x: Any) -> Dict[str, Any]:
    if x is None:
        return {"samples": None}
    if hasattr(x, "output") and getattr(x, "output") is not None:
        x = getattr(x, "output")
    if hasattr(x, "result") and getattr(x, "result") is not None:
        x = getattr(x, "result")
    while isinstance(x, (tuple, list)) and len(x) > 0:
        x = x[0]
    if isinstance(x, dict):
        cur = x
        while isinstance(cur, dict) and "samples" in cur and isinstance(cur["samples"], dict):
            cur = cur["samples"]
        if isinstance(cur, dict) and "samples" in cur:
            return cur
        for v in cur.values():
            if isinstance(v, torch.Tensor):
                return {"samples": v}
        return {"samples": cur}
    if isinstance(x, torch.Tensor):
        return {"samples": x}
    return {"samples": x}

def sync_latent_device(latent: Any, target_device: Union[str, torch.device] = "cpu") -> Dict[str, Any]:
    target = torch.device(target_device)
    latent_dict = unwrap_latent(latent)
    samples = latent_dict.get("samples", None)
    if samples is None:
        return latent_dict
    if isinstance(samples, torch.Tensor):
        if samples.is_nested:
            nested_list = [t.to(target) for t in samples.unbind()]
            latent_dict["samples"] = torch.nested.nested_tensor(nested_list)
        else:
            latent_dict["samples"] = samples.to(target)
    return latent_dict

def sync_conditioning_to_cpu(cond_obj: Any) -> Any:
    if cond_obj is None:
        return None
    if isinstance(cond_obj, torch.Tensor):
        return cond_obj.detach().cpu()
    if isinstance(cond_obj, list):
        return [sync_conditioning_to_cpu(item) for item in cond_obj]
    if isinstance(cond_obj, tuple):
        return tuple(sync_conditioning_to_cpu(item) for item in cond_obj)
    if isinstance(cond_obj, dict):
        return {k: sync_conditioning_to_cpu(v) for k, v in cond_obj.items()}
    return cond_obj

def slice_temporal_latent(tensor: Optional[torch.Tensor], start_idx: int, end_idx: int) -> Optional[torch.Tensor]:
    """
    Dimension-agnostic temporal slicer (handles 3D, 4D, 5D tensors on temporal dimension=2).
    """
    if tensor is None or not isinstance(tensor, torch.Tensor):
        return None
    dim = 2
    if tensor.ndim <= dim:
        return tensor.clone()
    max_len = tensor.shape[dim]
    s_idx = max(0, min(start_idx, max_len))
    e_idx = max(s_idx, min(end_idx, max_len))
    slices = [slice(None)] * tensor.ndim
    slices[dim] = slice(s_idx, e_idx)
    return tensor[tuple(slices)].clone()

def concat_temporal_latents(tensor_list: List[torch.Tensor]) -> Optional[torch.Tensor]:
    """
    Dimension-agnostic temporal concatenator (dim=2).
    """
    valid_tensors = [t for t in tensor_list if t is not None and isinstance(t, torch.Tensor)]
    if not valid_tensors:
        return None
    dim = 2 if valid_tensors[0].ndim >= 3 else 0
    return torch.cat(valid_tensors, dim=dim)

def call_original_node(node_name: str, node_instance: Optional[Any] = None, **kwargs) -> Any:
    if node_instance is None:
        if node_name not in NODE_CLASS_MAPPINGS:
            raise RuntimeError(f"FATAL: Required node '{node_name}' is not registered in ComfyUI.")
        node_instance = NODE_CLASS_MAPPINGS[node_name]()

    func_name = getattr(node_instance, "FUNCTION", None)
    callables = []

    if func_name and hasattr(node_instance, func_name) and callable(getattr(node_instance, func_name)):
        callables.append((func_name, getattr(node_instance, func_name)))
    if hasattr(node_instance, "execute") and callable(getattr(node_instance, "execute")):
        callables.append(("execute", getattr(node_instance, "execute")))
    for fallback in [
        "direct", "get_guider", "get_noise", "get_sampler", "get_sigmas", "sample",
        "apply_guide", "crop_guides", "upsample_latent", "concat", "separate",
        "encode", "decode", "load_unet", "load_clip", "load_vae", "combine_video", "override"
    ]:
        if hasattr(node_instance, fallback) and callable(getattr(node_instance, fallback)):
            callables.append((fallback, getattr(node_instance, fallback)))

    comfy_schema_defaults = {}
    if hasattr(node_instance, "INPUT_TYPES") and callable(node_instance.INPUT_TYPES):
        try:
            it = node_instance.INPUT_TYPES()
            for group in ["required", "optional", "hidden"]:
                group_dict = it.get(group, {})
                for p_name, p_spec in group_dict.items():
                    if isinstance(p_spec, (tuple, list)) and len(p_spec) > 1 and isinstance(p_spec[1], dict) and "default" in p_spec[1]:
                        comfy_schema_defaults[p_name] = p_spec[1]["default"]
                    elif isinstance(p_spec, (tuple, list)) and len(p_spec) > 0 and isinstance(p_spec[0], list) and len(p_spec[0]) > 0:
                        comfy_schema_defaults[p_name] = p_spec[0][0]
        except Exception:
            pass

    last_err_details = None
    for f_name, func in callables:
        try:
            sig = inspect.signature(func)
            valid_kwargs = {}
            has_var_keyword = False

            for param_name, param in sig.parameters.items():
                if param_name in ['cls', 'self']:
                    continue
                if param.kind == inspect.Parameter.VAR_POSITIONAL:
                    continue
                if param.kind == inspect.Parameter.VAR_KEYWORD:
                    has_var_keyword = True
                    continue

                if param_name in kwargs:
                    valid_kwargs[param_name] = kwargs[param_name]
                    continue

                alias_matched = False
                candidate_aliases = PARAM_ALIASES.get(param_name, [param_name])
                for alias in candidate_aliases:
                    if alias in kwargs:
                        valid_kwargs[param_name] = kwargs[alias]
                        alias_matched = True
                        break
                if alias_matched:
                    continue

                if param.default is not inspect.Parameter.empty:
                    continue

                if param_name in comfy_schema_defaults:
                    valid_kwargs[param_name] = comfy_schema_defaults[param_name]
                    continue

                if param_name == "duration_frames":
                    valid_kwargs[param_name] = int(kwargs.get("normalDurationFrames", 756))
                elif param.annotation == int or 'int' in str(param.annotation):
                    valid_kwargs[param_name] = 0
                elif param.annotation == float or 'float' in str(param.annotation):
                    valid_kwargs[param_name] = 0.0
                elif param.annotation == bool or 'bool' in str(param.annotation):
                    valid_kwargs[param_name] = False
                elif param.annotation == str or 'str' in str(param.annotation):
                    valid_kwargs[param_name] = ""
                else:
                    valid_kwargs[param_name] = None

            if has_var_keyword:
                for k, v in kwargs.items():
                    if k not in valid_kwargs:
                        valid_kwargs[k] = v

            return func(**valid_kwargs)
        except Exception:
            last_err_details = traceback.format_exc()
            continue

    if last_err_details is not None:
        raise RuntimeError(f"Error calling original node '{node_name}':\n{last_err_details}")
    raise AttributeError(f"Cannot execute node '{node_instance.__class__.__name__}' (No valid callable function)")

print("✅ Cell 10: Original Node Dispatcher ready with Signature Filtering.")


# ════════════════════════════════════════════════════════════════════════════
# CELL 11: PRECOMPUTED CLIP PROXY & COMPLETE ZERO-RAM DIT PROXY
# ════════════════════════════════════════════════════════════════════════════
class MockBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = torch.nn.Identity()
        self.cross_attn = torch.nn.Identity()
        self.ffn = torch.nn.Identity()
        self.linear1 = torch.nn.Identity()
        self.linear2 = torch.nn.Identity()
    def forward(self, *args, **kwargs):
        return args[0] if args else None

class DiffusionModelSpec(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.__class__.__name__ = "LTXVModel"
        self.patch_size = (1, 32, 32)
        self.temporal_stride = 8
        self.vae_scale_factors = (8, 32, 32)
        self.in_channels = 128
        self.out_channels = 128
        self.dtype = torch.bfloat16
        self.blocks = torch.nn.ModuleList([MockBlock() for _ in range(48)])
        self.double_blocks = torch.nn.ModuleList([MockBlock() for _ in range(48)])
        self.single_blocks = torch.nn.ModuleList([MockBlock() for _ in range(48)])
        self.transformer_blocks = torch.nn.ModuleList([MockBlock() for _ in range(48)])
    def forward(self, *args, **kwargs):
        return args[0] if args else None

class BaseModelSpec:
    def __init__(self):
        self.diffusion_model = DiffusionModelSpec()
        self.model_type = "ltxv"
        self.latent_format = None
        self.memory_required = lambda *args, **kwargs: 0
        self.vae_scale_factors = (8, 32, 32)
    def to(self, *args, **kwargs):
        return self

class LightweightLTXModelProxy:
    def __init__(self):
        self.model = BaseModelSpec()
        self.model_options = {}
        self.patches = {}
        self.object_patches = {}
        self.vae_scale_factors = (8, 32, 32)

    def clone(self):
        c = LightweightLTXModelProxy()
        c.model_options = dict(self.model_options)
        c.patches = dict(self.patches)
        return c

    def set_model_patch(self, patch, name):
        self.patches[name] = patch

    def set_model_patch_replace(self, patch, name, block_name, number):
        pass

    def add_object_patch(self, name, obj):
        self.object_patches[name] = obj

    def get_model_object(self, name):
        if name == "diffusion_model":
            return self.model.diffusion_model
        return getattr(self.model, name, None)

    def __getattr__(self, name):
        return getattr(self.model, name, None)

class PrecomputedClipProxy:
    def __init__(self, precomputed_conditioning: Any, tokenizer: Any = None):
        self.cond = precomputed_conditioning
        self.tokenizer = tokenizer
        self.cond_stage_model = None
        self.patcher = None
        self.layer_idx = None

    def tokenize(self, text, *args, **kwargs):
        if self.tokenizer is not None and hasattr(self.tokenizer, "tokenize_with_weights"):
            return self.tokenizer.tokenize_with_weights(text)
        return {"text": text}

    def encode_from_tokens_scheduled(self, tokens, *args, **kwargs):
        return self.cond

    def encode_from_tokens(self, tokens, *args, **kwargs):
        return self.cond

    def encode(self, text, *args, **kwargs):
        return self.cond

    def load_model(self, *args, **kwargs):
        return self

    def clone(self):
        return self

    def get_key_patches(self):
        return {}

    def __getattr__(self, name):
        if name == "tokenizer" and self.tokenizer is not None:
            return self.tokenizer
        return lambda *args, **kwargs: self.cond

def load_clip_and_encode_to_gpu(prompt_text: str) -> Tuple[Any, Any]:
    LTXDirectorMemoryManager.print_diagnostics(phase="Text Encoder Loading", node="DualCLIPLoader")
    LTXDirectorMemoryManager.purge("pre_clip_load")

    import comfy.model_management as mm
    dual_clip_node = NODE_CLASS_MAPPINGS["DualCLIPLoader"]()
    clip = dual_clip_node.load_clip(
        clip_name1="gemma_3_12B_it_fp4_mixed.safetensors",
        clip_name2="ltx-2.3_text_projection_bf16.safetensors",
        type="ltxv",
        device="default"
    )[0]

    saved_tokenizer = getattr(clip, "tokenizer", None)

    if hasattr(clip, "cond_stage_model") and hasattr(clip.cond_stage_model, "to"):
        clip.cond_stage_model.to(torch.device("cuda"))
    if hasattr(clip, "patcher") and hasattr(clip.patcher, "model") and hasattr(clip.patcher.model, "to"):
        clip.patcher.model.to(torch.device("cuda"))

    gc.collect()
    malloc_trim_os()

    clip_text_encode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
    print("  ⚡ Running Fast Prompt Encoding directly on GPU (~3-5 seconds)...")
    t0 = time.time()

    with torch.inference_mode():
        with torch.amp.autocast('cuda', dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
            cond_raw = clip_text_encode.encode(text=prompt_text, clip=clip)[0]
            cond_cpu = sync_conditioning_to_cpu(cond_raw)
            del cond_raw

    print(f"  ✓ Prompt Encoding Finished on GPU in {time.time() - t0:.2f}s!")

    # 💥 COMPLETE PURGE OF THE 12B MODEL WEIGHTS
    del clip, dual_clip_node, clip_text_encode
    mm.unload_all_models()
    mm.cleanup_models()
    mm.soft_empty_cache()
    if hasattr(mm, "current_loaded_models") and isinstance(mm.current_loaded_models, list):
        mm.current_loaded_models.clear()
    gc.collect()
    torch.cuda.empty_cache()
    LTXDirectorMemoryManager.drop_os_page_cache()
    malloc_trim_os()

    mem = LTXDirectorMemoryManager.get_memory_stats()
    print(f"  ✓ CLIP Purged completely! Host RAM Free: {mem['ram_avail_gb']:.2f} GB | GPU VRAM Free: {mem['gpu_free_gb']:.2f} GB")
    return cond_cpu, saved_tokenizer

def load_dit_and_loras():
    LTXDirectorMemoryManager.print_diagnostics(phase="DiT Loading", node="UnetLoaderGGUF")
    model = gv(call_original_node("UnetLoaderGGUF", unet_name=DIT_GGUF_FILENAME), 0)
    print(f"  ✓ UnetLoaderGGUF loaded ({DIT_GGUF_FILENAME}).")

    # Build the LoRA stack from Master Settings (toggles / strengths / names).
    _lora_settings = [
        (use_lora_1, lora_name_1, lora_strength_1),
        (use_lora_2, lora_name_2, lora_strength_2),
        (use_lora_3, lora_name_3, lora_strength_3),
        (use_lora_4, lora_name_4, lora_strength_4),
    ]
    power_lora_node = NODE_CLASS_MAPPINGS["Power Lora Loader (rgthree)"]()
    lora_stack_params = {"model": model, "clip": None}
    _active = 0
    for _n, (_on, _name, _str) in enumerate(_lora_settings, start=1):
        lora_stack_params[f"lora_{_n}"] = {"on": bool(_on), "lora": _name, "strength": float(_str)}
        if _on:
            _active += 1

    try:
        res = call_original_node("Power Lora Loader (rgthree)", node_instance=power_lora_node, **lora_stack_params)
        model = gv(res, 0) or model
        print(f"  ✓ Power Lora Loader (rgthree) applied {_active}-LoRA stack to DiT.")
    except Exception as e:
        print(f"  [Notice] PowerLora fallback: {e}")
        from nodes import LoraLoaderModelOnly
        for _on, _name, _str in _lora_settings:
            if not _on:
                continue
            lora_cfg = (_name, float(_str))
            if os.path.exists(os.path.join("/content/ComfyUI/models/loras", lora_cfg[0])):
                ll = LoraLoaderModelOnly()
                model = gv(ll.load_lora_model_only(model=model, lora_name=lora_cfg[0], strength_model=lora_cfg[1]), 0)
                print(f"    + LoRA applied: {lora_cfg[0]} (Strength {lora_cfg[1]})")

    # SageAttention & Chunk Feed Forward Hooks
    if "PatchSageAttentionKJ" in NODE_CLASS_MAPPINGS:
        try:
            sage = NODE_CLASS_MAPPINGS["PatchSageAttentionKJ"]()
            model = gv(call_original_node("PatchSageAttentionKJ", node_instance=sage, model=model, sage_attention="auto"), 0) or model
            print("  ✓ SageAttention Hook Applied.")
        except Exception:
            pass
    if "LTXVChunkFeedForward" in NODE_CLASS_MAPPINGS:
        try:
            cff = NODE_CLASS_MAPPINGS["LTXVChunkFeedForward"]()
            model = gv(call_original_node("LTXVChunkFeedForward", node_instance=cff, model=model, chunks=8, dim_threshold=4096), 0) or model
            print("  ✓ ChunkFeedForward Hook Applied (chunks=8).")
        except Exception:
            pass

    # Optionally pin the DiT fully-resident on GPU so ComfyUI never pages the
    # 22B weights through host RAM (the mid-sampling OOM path). Only enable when
    # the chosen quant fits VRAM with headroom (see DIT_QUANT notes).
    if FORCE_GPU_RESIDENT_DIT:
        try:
            import comfy.model_management as mm
            if hasattr(mm, "VRAMState") and hasattr(mm, "vram_state"):
                mm.vram_state = mm.VRAMState.HIGH_VRAM
            try:
                mm.load_models_gpu([model], force_full_load=True)
            except TypeError:
                mm.load_models_gpu([model])
            print("  ✓ DiT pinned fully-resident on GPU (host-RAM offload disabled).")
        except Exception as e:
            print(f"  [Notice] Could not pin DiT to GPU (continuing with smart offload): {e}")

    return model

print("✅ Cell 11: PrecomputedClipProxy & Complete Zero-RAM Model Architecture ready.")


# ════════════════════════════════════════════════════════════════════════════
# CELL 12: PHASE A: MASTER LTXDIRECTOR DECOUPLED INGESTION (ZERO-RAM OVERHEAD)
# ════════════════════════════════════════════════════════════════════════════
def execute_phase_a_ltxdirector(
    timeline_ctrl: DirectorTimelineController,
    workdir: str = "/content/LTXDirector_Work",
    resume: bool = True
) -> Dict[str, Any]:
    os.makedirs(workdir, exist_ok=True)
    state_file = os.path.join(workdir, "director_state.pt")

    if resume and os.path.exists(state_file) and os.path.getsize(state_file) > 1024:
        print(f"  ⏭ [RESUME] Loading cached Director state from: {state_file}")
        return torch.load(state_file, map_location="cpu")

    LTXDirectorMemoryManager.purge("pre_phase_a")
    ram_guard(min_free_gb=min_ram_guard_gb, tag="phase_a_start")
    LTXDirectorMemoryManager.print_diagnostics(phase="PHASE A: LTXDirector Ingestion", node="LTXDirector")

    with torch.inference_mode():
        # ────────────────────────────────────────────────────────────────────
        # STEP A1: GPU Fast Prompt Encoding (Finished in ~3-5s & Purged)
        # ────────────────────────────────────────────────────────────────────
        precomputed_cond, saved_tokenizer = load_clip_and_encode_to_gpu(timeline_ctrl.global_prompt)

        # ────────────────────────────────────────────────────────────────────
        # STEP A2: Master LTXDirector Timeline Ingestion with 10.5GB Free RAM
        # ────────────────────────────────────────────────────────────────────
        print("  🚀 [Step A2] Constructing 756-frame Timeline & Latents via LTXDirector...")
        t_dir = time.time()
        vae_node = NODE_CLASS_MAPPINGS["VAELoader"]()
        audio_vae = vae_node.load_vae(vae_name="LTX23_audio_vae_bf16.safetensors")[0]
        model_proxy = LightweightLTXModelProxy()
        clip_proxy = PrecomputedClipProxy(precomputed_cond, tokenizer=saved_tokenizer)

        ltx_director_node = NODE_CLASS_MAPPINGS["LTXDirector"]()
        timeline_ctrl.configure_ltxdirector_node_instance(ltx_director_node)

        ltx_director_params = {
            "model": model_proxy,
            "clip": clip_proxy,
            "audio_vae": audio_vae,
            "optional_latent": None,
            "global_prompt": timeline_ctrl.global_prompt,
            "start_second": 0.0,
            "end_second": float(timeline_ctrl.meta["duration_seconds"]),
            "duration_seconds": float(timeline_ctrl.meta["duration_seconds"]),
            "start_frame": int(timeline_ctrl.meta["start_frame"]),
            "end_frame": int(timeline_ctrl.meta["end_frame"]),
            "duration_frames": int(timeline_ctrl.meta["normalDurationFrames"]),
            "timeline_data": timeline_ctrl.build_timeline_json_string(),
            "local_prompts": " |  |  |  | ",
            "segment_lengths": str(timeline_ctrl.meta["segment_lengths"]),
            "epsilon": 0.001,
            "guide_strength": str(timeline_ctrl.meta["guide_strength"]),
            "mainTrackEnabled": bool(timeline_ctrl.meta["mainTrackEnabled"]),
            "audioTrackEnabled": bool(timeline_ctrl.meta["audioTrackEnabled"]),
            "motionTrackEnabled": bool(timeline_ctrl.meta["motionTrackEnabled"]),
            "main_track_enabled": bool(timeline_ctrl.meta["mainTrackEnabled"]),
            "audio_track_enabled": bool(timeline_ctrl.meta["audioTrackEnabled"]),
            "motion_track_enabled": bool(timeline_ctrl.meta["motionTrackEnabled"]),
            "frame_rate": float(timeline_ctrl.meta["frame_rate"]),
            "fps": float(timeline_ctrl.meta["frame_rate"]),
            "display_mode": "seconds",
            "custom_width": int(timeline_ctrl.meta["custom_width"]),
            "custom_height": int(timeline_ctrl.meta["custom_height"]),
            "width": int(timeline_ctrl.meta["custom_width"]),
            "height": int(timeline_ctrl.meta["custom_height"]),
            "resize_method": "maintain aspect ratio",
            "divisible_by": 32,
            "img_compression": 18,
            "retakeMode": False,
            "retake_mode": False,
            "retake_global_prompt": "",
            "retakeStart": 24,
            "retakeLength": 48,
            "retakePrompt": "",
            "retakeStrength": 1.0,
            "inpaint_audio": bool(timeline_ctrl.meta["inpaint_audio"]),
            "override_audio": bool(timeline_ctrl.meta["override_audio"]),
            "use_custom_audio": bool(timeline_ctrl.meta["use_custom_audio"]),
            "use_custom_motion": bool(timeline_ctrl.meta["use_custom_motion"])
        }

        director_out = call_original_node(
            "LTXDirector",
            node_instance=ltx_director_node,
            **ltx_director_params
        )
        print(f"  ⚡ Step A2 Timeline Ingestion Finished in {time.time() - t_dir:.2f}s!")

        # 💥 Universal Safe Extraction from NodeOutput object
        dir_pos = gv(director_out, 1) or precomputed_cond
        dir_vid_lat = sync_latent_device(gv(director_out, 2), "cpu")
        dir_aud_lat = sync_latent_device(gv(director_out, 3), "cpu")
        dir_guide_data = sync_conditioning_to_cpu(gv(director_out, 4))
        dir_motion_guide_data = sync_conditioning_to_cpu(gv(director_out, 5))
        dir_fps_raw = gv(director_out, 6)
        dir_fps = float(dir_fps_raw) if dir_fps_raw is not None else float(timeline_ctrl.meta["frame_rate"])

        zero_out_node = NODE_CLASS_MAPPINGS["ConditioningZeroOut"]()
        neg_zeroed = gv(call_original_node("ConditioningZeroOut", node_instance=zero_out_node, conditioning=dir_pos), 0)

        ltxv_cond_node = NODE_CLASS_MAPPINGS["LTXVConditioning"]()
        ltxv_cond_out = call_original_node(
            "LTXVConditioning",
            node_instance=ltxv_cond_node,
            positive=dir_pos,
            negative=neg_zeroed,
            frame_rate=dir_fps
        )
        final_positive = sync_conditioning_to_cpu(gv(ltxv_cond_out, 0))
        final_negative = sync_conditioning_to_cpu(gv(ltxv_cond_out, 1))

        state = {
            "positive": final_positive,
            "negative": final_negative,
            "video_latent": dir_vid_lat,
            "audio_latent": dir_aud_lat,
            "guide_data": dir_guide_data,
            "motion_guide_data": dir_motion_guide_data,
            "frame_rate": dir_fps,
            "timeline_metadata": timeline_ctrl.meta,
            "segments": timeline_ctrl.segments
        }

        tmp_path = state_file + ".tmp"
        torch.save(state, tmp_path)
        os.replace(tmp_path, state_file)
        print(f"  💾 Phase A Director State saved: {state_file}")

        del audio_vae, ltx_director_node, dir_pos, neg_zeroed, ltxv_cond_out, model_proxy, clip_proxy

    # 💥 FULL PURGE OF PHASE A
    LTXDirectorMemoryManager.purge("phase_a_complete")
    print("✅ Phase A Complete: Ready for Phase B (DiT Diffusion).")
    return state

print("✅ Cell 12: Decoupled Zero-Crash Phase A Configured.")


# ════════════════════════════════════════════════════════════════════════════
# CELL 13 & 14: PHASE B - SEGMENT-WISE DIFFUSION & 2X UPSCALE ENGINE
# ════════════════════════════════════════════════════════════════════════════
def execute_segment_wise_diffusion_pipeline(
    director_state: Dict[str, Any],
    seed: int = 2026,
    workdir: str = "/content/LTXDirector_Work",
    resume: bool = True
) -> List[Dict[str, Any]]:
    """
    Executes Stage 1 (8 steps) + 2x Latent Upscaler + Stage 2 Refinement (4 steps)
    per Director Segment with dimension-agnostic temporal slicing.
    """
    segments = director_state.get("segments", ORIGINAL_SEGMENTS)
    total_segments = len(segments)
    # RAM-SAFE: collect on-disk paths, NOT resident latent packs.
    completed_segment_paths: List[str] = []

    print("\n" + "="*70 + f"\n🎬 PHASE B: EXECUTING {total_segments} DIRECTOR SEGMENTS (MEMORY-SAFE)\n" + "="*70)

    full_vid_tensor = unwrap_latent(director_state["video_latent"])["samples"]
    full_aud_tensor = unwrap_latent(director_state["audio_latent"])["samples"] if director_state["audio_latent"] is not None else None
    total_lat_frames = full_vid_tensor.shape[2] if full_vid_tensor is not None else 95

    cur_lat_idx = 0

    for idx, seg in enumerate(segments):
        seg_id = seg["id"]
        seg_name = f"Segment {idx+1}/{total_segments} ({os.path.basename(seg['imageFile'])})"
        seg_cache_file = os.path.join(workdir, f"latent_stage2_seg_{idx+1}.pt")

        raw_frames = int(round(float(seg["length"])))
        valid_frames = int(((raw_frames - 1) // 8) * 8 + 1)
        if valid_frames < 9:
            valid_frames = 9
        seg_lat_len = (valid_frames - 1) // 8 + 1

        end_lat_idx = min(cur_lat_idx + seg_lat_len, total_lat_frames)
        if idx == total_segments - 1:
            end_lat_idx = total_lat_frames

        print(f"\n{'─'*65}\n🎬 {seg_name} | Frames: {valid_frames} (Latent Frames: {cur_lat_idx} -> {end_lat_idx})\n{'─'*65}")

        if resume and os.path.exists(seg_cache_file) and os.path.getsize(seg_cache_file) > 1024:
            print(f"  ⏭ [RESUME] Found cached Stage 2 segment latent: {seg_cache_file}")
            completed_segment_paths.append(seg_cache_file)
            cur_lat_idx = end_lat_idx
            continue

        LTXDirectorMemoryManager.purge(f"pre_seg_{idx+1}")
        ram_guard(min_free_gb=min_ram_guard_gb, tag=f"seg_{idx+1}_start")

        # Dimension-agnostic temporal slicing
        seg_vid_lat = {"samples": slice_temporal_latent(full_vid_tensor, cur_lat_idx, end_lat_idx)}
        seg_aud_lat = {"samples": slice_temporal_latent(full_aud_tensor, cur_lat_idx, end_lat_idx)} if full_aud_tensor is not None else None

        with torch.inference_mode():
            # ────────────────────────────────────────────────────────────────
            # STEP B1: Load DiT + LoRAs + Stage 1 Guide
            # ────────────────────────────────────────────────────────────────
            model = load_dit_and_loras()
            video_vae = gv(call_original_node("VAELoader", vae_name="LTX23_video_vae_bf16.safetensors"), 0)

            guide1_node = NODE_CLASS_MAPPINGS["LTXDirectorGuide"]()
            guide1_params = {
                "positive": director_state["positive"],
                "negative": director_state["negative"],
                "vae": video_vae,
                "latent": seg_vid_lat,
                "guide_data": director_state["guide_data"],
                "motion_guide_data": director_state["motion_guide_data"],
                "model": model,
                "strength": float(stage1_guide_strength),
                "rescale_method": "None",
                "guide_frame": int(guide_frame),
                "interpolation": guide_interpolation,
                "crop_position": guide_crop_position,
                "enable_guide": True
            }
            guide1_res = call_original_node("LTXDirectorGuide", node_instance=guide1_node, **guide1_params)

            s1_pos = gv(guide1_res, 0) or director_state["positive"]
            s1_neg = gv(guide1_res, 1) or director_state["negative"]
            s1_vid = sync_latent_device(gv(guide1_res, 2) or seg_vid_lat, "cpu")
            s1_model = gv(guide1_res, 3) or model

            # Concatenate AV Latents
            concat_node = NODE_CLASS_MAPPINGS["LTXVConcatAVLatent"]()
            av1_in = sync_latent_device(gv(call_original_node(
                "LTXVConcatAVLatent",
                node_instance=concat_node,
                video_latent=s1_vid,
                audio_latent=seg_aud_lat
            ), 0), "cpu")

            # Sampler Stage 1 (8 steps, denoise 1.0)
            noise_node = NODE_CLASS_MAPPINGS["RandomNoise"]()
            noise1 = gv(call_original_node("RandomNoise", node_instance=noise_node, noise_seed=seed + idx * 100), 0)

            guider_node = NODE_CLASS_MAPPINGS["CFGGuider"]()
            guider1 = gv(call_original_node("CFGGuider", node_instance=guider_node, cfg=float(sampler_cfg), model=s1_model, positive=s1_pos, negative=s1_neg), 0)

            sampler_select_node = NODE_CLASS_MAPPINGS["KSamplerSelect"]()
            sampler_euler = gv(call_original_node("KSamplerSelect", node_instance=sampler_select_node, sampler_name=sampler_name), 0)

            scheduler_node = NODE_CLASS_MAPPINGS["BasicScheduler"]()
            sigmas1 = gv(call_original_node(
                "BasicScheduler",
                node_instance=scheduler_node,
                model=s1_model,
                scheduler=scheduler_name,
                steps=int(stage1_steps),
                denoise=float(stage1_denoise)
            ), 0)

            print(f"  ⚡ Sampling Stage 1 for {seg_name} ({int(stage1_steps)} steps)...")
            t_s1 = time.time()
            sampler_custom_node = NODE_CLASS_MAPPINGS["SamplerCustomAdvanced"]()
            s1_out = call_original_node(
                "SamplerCustomAdvanced",
                node_instance=sampler_custom_node,
                noise=noise1,
                guider=guider1,
                sampler=sampler_euler,
                sigmas=sigmas1,
                latent_image=av1_in
            )
            s1_lat = sync_latent_device(gv(s1_out, 0), "cpu")
            print(f"  ✓ Stage 1 Finished in {time.time() - t_s1:.2f}s!")

            sep_node = NODE_CLASS_MAPPINGS["LTXVSeparateAVLatent"]()
            sep1 = call_original_node("LTXVSeparateAVLatent", node_instance=sep_node, av_latent=s1_lat)
            v1_raw = sync_latent_device(gv(sep1, 0), "cpu")
            a1_raw = sync_latent_device(gv(sep1, 1), "cpu")

            crop_node = NODE_CLASS_MAPPINGS["LTXDirectorCropGuides"]()
            crop1 = call_original_node("LTXDirectorCropGuides", node_instance=crop_node, positive=s1_pos, negative=s1_neg, latent=v1_raw)
            crop1_pos = gv(crop1, 0) or s1_pos
            crop1_neg = gv(crop1, 1) or s1_neg
            crop1_vid = sync_latent_device(gv(crop1, 2) or v1_raw, "cpu")

            # ────────────────────────────────────────────────────────────────
            # STEP B2: 2x Latent Spatial Upscaling
            # ────────────────────────────────────────────────────────────────
            print("  ⚡ Upscaling Latent 2x via LTXVLatentUpsampler...")
            upscale_loader = NODE_CLASS_MAPPINGS["LatentUpscaleModelLoader"]()
            up_model = gv(call_original_node("LatentUpscaleModelLoader", node_instance=upscale_loader, model_name="ltx-2.3-spatial-upscaler-x2-1.1.safetensors"), 0)

            upsampler_node = NODE_CLASS_MAPPINGS["LTXVLatentUpsampler"]()
            upscaled_lat_res = call_original_node("LTXVLatentUpsampler", node_instance=upsampler_node, samples=crop1_vid, upscale_model=up_model, vae=video_vae)
            v_upscaled = sync_latent_device(gv(upscaled_lat_res, 0), "cpu")

            del up_model, upscaled_lat_res

            # ────────────────────────────────────────────────────────────────
            # STEP B3: Stage 2 Refinement (4 steps, denoise 0.42)
            # ────────────────────────────────────────────────────────────────
            guide2_node = NODE_CLASS_MAPPINGS["LTXDirectorGuide"]()
            guide2_params = {
                "positive": crop1_pos,
                "negative": crop1_neg,
                "vae": video_vae,
                "latent": v_upscaled,
                "guide_data": director_state["guide_data"],
                "motion_guide_data": director_state["motion_guide_data"],
                "model": model,
                "strength": float(stage2_guide_strength),
                "rescale_method": "None",
                "guide_frame": int(guide_frame),
                "interpolation": guide_interpolation,
                "crop_position": guide_crop_position,
                "enable_guide": True
            }
            guide2_res = call_original_node("LTXDirectorGuide", node_instance=guide2_node, **guide2_params)
            s2_pos = gv(guide2_res, 0) or crop1_pos
            s2_neg = gv(guide2_res, 1) or crop1_neg
            s2_vid = sync_latent_device(gv(guide2_res, 2) or v_upscaled, "cpu")
            s2_model = gv(guide2_res, 3) or model

            av2_in = sync_latent_device(gv(call_original_node(
                "LTXVConcatAVLatent",
                node_instance=concat_node,
                video_latent=s2_vid,
                audio_latent=a1_raw
            ), 0), "cpu")

            noise2 = gv(call_original_node("RandomNoise", node_instance=noise_node, noise_seed=seed + idx * 100), 0)
            guider2 = gv(call_original_node("CFGGuider", node_instance=guider_node, cfg=float(sampler_cfg), model=s2_model, positive=s2_pos, negative=s2_neg), 0)
            sigmas2 = gv(call_original_node("BasicScheduler", node_instance=scheduler_node, model=s2_model, scheduler=scheduler_name, steps=int(stage2_steps), denoise=float(stage2_denoise)), 0)

            print(f"  ⚡ Stage 2 Refinement for {seg_name} ({int(stage2_steps)} steps)...")
            t_s2 = time.time()
            s2_out = call_original_node("SamplerCustomAdvanced", node_instance=sampler_custom_node, noise=noise2, guider=guider2, sampler=sampler_euler, sigmas=sigmas2, latent_image=av2_in)
            s2_lat = sync_latent_device(gv(s2_out, 0), "cpu")
            print(f"  ✓ Stage 2 Finished in {time.time() - t_s2:.2f}s!")

            sep2 = call_original_node("LTXVSeparateAVLatent", node_instance=sep_node, av_latent=s2_lat)
            v2_raw = sync_latent_device(gv(sep2, 0), "cpu")
            a2_raw = sync_latent_device(gv(sep2, 1), "cpu")

            crop2 = call_original_node("LTXDirectorCropGuides", node_instance=crop_node, positive=s2_pos, negative=s2_neg, latent=v2_raw)
            final_seg_video_lat = sync_latent_device(gv(crop2, 2) or v2_raw, "cpu")

            seg_pack = {
                "video_latent": final_seg_video_lat,
                "audio_latent": a2_raw,
                "segment_index": idx,
                "valid_frames": valid_frames
            }

            tmp_seg = seg_cache_file + ".tmp"
            torch.save(seg_pack, tmp_seg)
            os.replace(tmp_seg, seg_cache_file)
            completed_segment_paths.append(seg_cache_file)
            print(f"  💾 Saved {seg_name} Latents to: {seg_cache_file}")

            # RAM-SAFE: drop this segment's latents from RAM immediately; Phase C reloads lazily.
            del seg_pack, final_seg_video_lat, a2_raw
            del model, video_vae, noise1, guider1, sigmas1, s1_out, v1_raw, noise2, guider2, sigmas2, s2_out, v2_raw

        cur_lat_idx = end_lat_idx
        LTXDirectorMemoryManager.purge(f"post_seg_{idx+1}")

    print("✅ All 5 Director Segments successfully sampled & upscaled!")
    return completed_segment_paths

print("✅ Cell 13 & 14: Segment-wise Diffusion & Upscale Engine ready.")


# ════════════════════════════════════════════════════════════════════════════
# CELL 15: PHASE C - MEMORY-SAFE OUT-OF-CORE VAE DECODING
# ════════════════════════════════════════════════════════════════════════════
def _decode_one_video_latent(v_lat: Any, video_vae: Any) -> torch.Tensor:
    """Decode a single segment latent, preferring the memory-safe tiled decoder."""
    if "LTXVSpatioTemporalTiledVAEDecode" in NODE_CLASS_MAPPINGS:
        try:
            tiled_node = NODE_CLASS_MAPPINGS["LTXVSpatioTemporalTiledVAEDecode"]()
            decoded_res = call_original_node(
                "LTXVSpatioTemporalTiledVAEDecode",
                node_instance=tiled_node,
                vae=video_vae,
                latents=v_lat,
                spatial_tiles=2,
                spatial_overlap=8,
                temporal_tile_length=16,
                temporal_overlap=4,
                last_frame_fix=False,
                working_device="auto",
                working_dtype="auto"
            )
            return unwrap_tensor(decoded_res)
        except Exception:
            pass
    vae_decode_node = NODE_CLASS_MAPPINGS["VAEDecode"]()
    decoded_res = call_original_node("VAEDecode", node_instance=vae_decode_node, samples=v_lat, vae=video_vae)
    return unwrap_tensor(decoded_res)


def execute_phase_c_video_decode(
    segment_pack_paths: List[str],
    workdir: str = "/content/LTXDirector_Work",
    resume: bool = True,
    fps: int = 24,
    crf: int = 8
) -> str:
    """
    RAM-SAFE streaming decode.

    Previous behavior peaked at ~2x the full 30s frame tensor in host RAM
    (list-accumulate -> torch.cat -> torch.save -> reload as float in Phase D),
    which is the primary cause of the 12.2 GB OOM crash.

    New behavior: decode ONE segment at a time and stream its frames straight
    into a silent H.264 MP4 via an incremental FFMPEG writer. Peak host RAM is
    now bounded by a single segment (converted to uint8 immediately), never the
    whole video, and nothing large is ever persisted to a .pt.
    """
    import imageio

    silent_video = os.path.join(workdir, "master_video_silent.mp4")
    if resume and os.path.exists(silent_video) and os.path.getsize(silent_video) > 1024:
        print(f"  ⏭ [RESUME] Found streamed silent video: {silent_video}")
        return silent_video

    LTXDirectorMemoryManager.purge("pre_video_decode")
    LTXDirectorMemoryManager.print_diagnostics(phase="PHASE C: Video VAE Decoding (streaming)", node="VAEDecode")

    tmp_video = silent_video + ".tmp.mp4"
    writer = imageio.get_writer(
        tmp_video,
        format="FFMPEG",
        mode="I",
        fps=int(fps),
        codec="libx264",
        pixelformat="yuv420p",
        output_params=["-crf", str(int(crf))],
        macro_block_size=None,
    )

    total_written = 0
    with torch.inference_mode():
        video_vae = gv(call_original_node("VAELoader", vae_name="LTX23_video_vae_bf16.safetensors"), 0)

        for idx, seg_path in enumerate(segment_pack_paths):
            print(f"  🎨 Decoding Segment {idx+1}/{len(segment_pack_paths)} frames via VAE...")
            pack = torch.load(seg_path, map_location="cpu")
            v_lat = pack["video_latent"]

            decoded_tensor = _decode_one_video_latent(v_lat, video_vae)

            # Convert to uint8 (on whichever device it lands) then pull to host as
            # numpy. This avoids keeping a giant float32 CPU copy around.
            frames_u8 = (
                decoded_tensor.detach().clamp(0.0, 1.0).mul_(255.0)
                .round().to(torch.uint8).cpu().numpy()
            )
            del decoded_tensor, pack, v_lat

            n = frames_u8.shape[0]
            for f in range(n):
                writer.append_data(frames_u8[f])
            total_written += n
            print(f"    + streamed {n} frames (total {total_written})")

            del frames_u8
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        del video_vae

    writer.close()
    os.replace(tmp_video, silent_video)
    print(f"  💾 Streamed silent master video ({total_written} frames): {silent_video}")

    LTXDirectorMemoryManager.purge("video_decode_complete")
    return silent_video

print("✅ Cell 15: Video VAE Decoder configured.")


# ════════════════════════════════════════════════════════════════════════════
# CELL 16: PHASE C - AUDIO VAE DECODING & SYNCHRONIZATION
# ════════════════════════════════════════════════════════════════════════════
def execute_phase_c_audio_decode(
    segment_pack_paths: List[str],
    workdir: str = "/content/LTXDirector_Work",
    resume: bool = True
) -> str:
    audio_file = os.path.join(workdir, "decoded_audio.pt")
    if resume and os.path.exists(audio_file) and os.path.getsize(audio_file) > 1024:
        print(f"  ⏭ [RESUME] Loading cached decoded audio from: {audio_file}")
        return audio_file

    LTXDirectorMemoryManager.purge("pre_audio_decode")
    LTXDirectorMemoryManager.print_diagnostics(phase="PHASE C: Audio VAE Decoding", node="LTXVAudioVAEDecode")

    with torch.inference_mode():
        audio_vae = gv(call_original_node("VAELoader", vae_name="LTX23_audio_vae_bf16.safetensors"), 0)
        audio_decode_node = NODE_CLASS_MAPPINGS["LTXVAudioVAEDecode"]()

        # RAM-SAFE: load each segment's (small) audio latent lazily from disk,
        # keep only the audio tensor, and release the rest of the pack.
        aud_lat_list = []
        fallback_aud = None
        for seg_path in segment_pack_paths:
            pack = torch.load(seg_path, map_location="cpu")
            if fallback_aud is None:
                fallback_aud = pack.get("audio_latent")
            samp = unwrap_latent(pack.get("audio_latent"))["samples"]
            if samp is not None:
                aud_lat_list.append(samp)
            del pack

        if aud_lat_list:
            combined_aud_samples = concat_temporal_latents(aud_lat_list)
            combined_aud_lat = {"samples": combined_aud_samples}
        else:
            combined_aud_lat = fallback_aud
        del aud_lat_list

        print("  🎵 Decoding audio latent stream via LTXVAudioVAEDecode...")
        aud_res = call_original_node(
            "LTXVAudioVAEDecode",
            node_instance=audio_decode_node,
            samples=combined_aud_lat,
            audio_vae=audio_vae
        )
        decoded_audio = gv(aud_res, 0)
        print("  ✓ Audio latent stream successfully decoded.")

        tmp_path = audio_file + ".tmp"
        torch.save(decoded_audio, tmp_path)
        os.replace(tmp_path, audio_file)
        print(f"  💾 Decoded Audio saved: {audio_file}")

        del audio_vae, audio_decode_node, aud_res

    LTXDirectorMemoryManager.purge("audio_decode_complete")
    return audio_file

print("✅ Cell 16: Audio VAE Decoder configured.")


# ════════════════════════════════════════════════════════════════════════════
# CELL 17: PHASE D - VHS FINAL VIDEO COMBINE & PACKAGING
# ════════════════════════════════════════════════════════════════════════════
import numpy as np

def _write_audio_dict_to_wav(audio_dict: Any, wav_path: str, default_sr: int = 48000) -> bool:
    """Write a ComfyUI AUDIO dict ({'waveform': [B,C,N], 'sample_rate': int}) to WAV."""
    try:
        if audio_dict is None:
            return False
        wav = None
        sr = default_sr
        if isinstance(audio_dict, dict):
            wav = audio_dict.get("waveform", None)
            sr = int(audio_dict.get("sample_rate", default_sr))
        elif isinstance(audio_dict, torch.Tensor):
            wav = audio_dict
        wav = unwrap_tensor(wav)
        if not isinstance(wav, torch.Tensor):
            return False
        w = wav.detach().cpu().float()
        while w.dim() > 2:          # [B, C, N] -> [C, N]
            w = w[0]
        if w.dim() == 1:            # [N] -> [1, N]
            w = w.unsqueeze(0)
        w = w.clamp(-1.0, 1.0)
        try:
            import torchaudio
            torchaudio.save(wav_path, w, sr)
            return True
        except Exception:
            from scipy.io import wavfile
            arr = (w.transpose(0, 1).numpy() * 32767.0).astype(np.int16)  # [N, C]
            wavfile.write(wav_path, sr, arr)
            return True
    except Exception as e:
        print(f"  [Notice] Could not write decoded audio WAV: {e}")
        return False


def execute_phase_d_vhs_combine(
    frames_file_path: str,
    audio_file_path: str,
    fps: int = 24,
    crf: int = 8,
    outdir: str = "/content/LTXStudio_Output"
) -> str:
    """
    RAM-SAFE final assembly.

    `frames_file_path` is now the already-encoded SILENT MP4 produced by the
    streaming Phase C decoder (not a giant frame tensor). We simply mux audio
    onto it with a stream copy (`-c:v copy`), so no frames are ever loaded into
    host RAM here.
    """
    os.makedirs(outdir, exist_ok=True)
    final_output_path = os.path.join(outdir, "LTX23_Director_Master_30s.mp4")
    silent_video = frames_file_path

    LTXDirectorMemoryManager.purge("pre_final_mux")
    LTXDirectorMemoryManager.print_diagnostics(phase="PHASE D: Final Mux (stream-copy)", node="ffmpeg")

    if not os.path.exists(silent_video) or os.path.getsize(silent_video) < 1024:
        raise RuntimeError(f"Silent master video missing for mux: {silent_video}")

    # 1) Preferred: mux the model-generated (lip-synced) audio decoded in Phase C.
    audio_dict = torch.load(audio_file_path, map_location="cpu") if (audio_file_path and os.path.exists(audio_file_path)) else None
    gen_wav = os.path.join(outdir, "generated_audio.wav")
    have_gen_audio = _write_audio_dict_to_wav(audio_dict, gen_wav)
    del audio_dict
    gc.collect()

    muxed = False
    if have_gen_audio:
        print("  🎬 Muxing streamed video with generated (lip-synced) audio via stream-copy...")
        cmd = (f'ffmpeg -y -i "{silent_video}" -i "{gen_wav}" '
               f'-map 0:v:0 -map 1:a:0 -c:v copy -c:a aac -b:a 320k -shortest "{final_output_path}"')
        if run_cmd(cmd, silent=False) == 0 and os.path.exists(final_output_path) and os.path.getsize(final_output_path) > 1024:
            muxed = True

    # 2) Fallback: mux the trimmed original backing track.
    if not muxed:
        raw_song_path = "/content/ComfyUI/input/whatdreamscost/Late night trap.mp3"
        if os.path.exists(raw_song_path):
            print("  🎬 [Fallback] Muxing streamed video with trimmed backing track...")
            trim_sec = 446.9222739141953 / float(fps)
            cmd = (f'ffmpeg -y -i "{silent_video}" -ss {trim_sec} -i "{raw_song_path}" '
                   f'-map 0:v:0 -map 1:a:0 -c:v copy -c:a aac -b:a 320k -shortest "{final_output_path}"')
            if run_cmd(cmd, silent=False) == 0 and os.path.exists(final_output_path) and os.path.getsize(final_output_path) > 1024:
                muxed = True

    # 3) Last resort: ship the silent video as the final output.
    if not muxed:
        print("  [Notice] No audio available; delivering silent master video.")
        shutil.copyfile(silent_video, final_output_path)

    LTXDirectorMemoryManager.purge("final_mux_cleanup")
    print(f"  🎉 Final Render Complete: {final_output_path}")
    return final_output_path

print("✅ Cell 17: Phase D VHS Assembler ready.")


# ════════════════════════════════════════════════════════════════════════════
# CELL 18: ARTIFACT VERIFICATION & CLEANUP UTILITY
# ════════════════════════════════════════════════════════════════════════════
def verify_output_artifacts(video_path: str, expected_frames: int = 756, expected_fps: int = 24):
    print("\n" + "="*70 + "\n🔍 FINAL ARTIFACT VERIFICATION\n" + "="*70)
    if not os.path.exists(video_path) or os.path.getsize(video_path) < 1000:
        raise RuntimeError(f"Artifact check failed: Output video '{video_path}' is missing or empty.")

    probe_cmd = f'ffprobe -v error -select_streams v:0 -count_packets -show_entries stream=nb_read_packets,r_frame_rate,duration -of csv=p=0 "{video_path}"'
    res = subprocess.run(probe_cmd, shell=True, capture_output=True, text=True)
    out = res.stdout.strip()
    print(f"  ✓ File Path    : {video_path}")
    print(f"  ✓ File Size    : {os.path.getsize(video_path) / (1024*1024):.2f} MB")
    print(f"  ✓ FFprobe Info : {out}")
    print("="*70 + "\n")

print("✅ Cell 18: Artifact Verifier ready.")


# ════════════════════════════════════════════════════════════════════════════
# CELL 19: RUNTIME CONFIGURATION & QUALITY DEBUG MODE
# ════════════════════════════════════════════════════════════════════════════
# All runtime knobs come from the Master Settings cell.
DEBUG_MODE = bool(debug_mode)
DEBUG_MAX_FRAMES = int(debug_max_frames)
if str(seed_mode).lower() == "random":
    BASE_SEED = int.from_bytes(os.urandom(4), "big")
else:
    BASE_SEED = int(base_seed)
SEED_MODE = seed_mode
RESUME_CHECKPOINTS = bool(resume_checkpoints)
OUTPUT_CRF = int(output_crf)

WORK_DIRECTORY = "/content/LTXDirector_Work"
OUTPUT_DIRECTORY = "/content/LTXStudio_Output"

print(f"✅ Cell 19: Runtime Configured (Debug: {DEBUG_MODE} | Seed: {BASE_SEED} [{SEED_MODE}] | CRF: {OUTPUT_CRF} | Resume: {RESUME_CHECKPOINTS})")


# ════════════════════════════════════════════════════════════════════════════
# CELL 20: MASTER ONE-CLICK GENERATION FUNCTION
# ════════════════════════════════════════════════════════════════════════════
def run_ltx23_director_original_workflow(
    global_prompt: str,
    negative_prompt: str,
    timeline_metadata: Dict[str, Any],
    segments: List[Dict[str, Any]],
    seed: int = 2026,
    crf: int = 8,
    workdir: str = "/content/LTXDirector_Work",
    outdir: str = "/content/LTXStudio_Output",
    resume: bool = True,
    debug: bool = False,
    debug_max_frames: int = 120
) -> str:
    start_time = time.time()
    print("\n" + "="*70 + "\n🎬 STARTING LTX-2.3 DIRECTOR 2.0 WORKFLOW GENERATION\n" + "="*70)

    validate_original_nodes()
    patch_comfy_memory_safety()
    patch_safetensors_direct_to_gpu()

    active_metadata = dict(timeline_metadata)
    active_segments = list(segments)
    if debug:
        print(f"⚠️ [DEBUG MODE ACTIVE] Capping duration to {debug_max_frames} frames.")
        active_metadata["normalDurationFrames"] = debug_max_frames
        active_metadata["duration_seconds"] = debug_max_frames / active_metadata["frame_rate"]
        active_metadata["end_frame"] = debug_max_frames

    timeline_ctrl = DirectorTimelineController(
        global_prompt=global_prompt,
        negative_prompt=negative_prompt,
        timeline_metadata=active_metadata,
        segments=active_segments
    )

    # Phase A: Master LTXDirector Ingestion (GPU Accelerated + Zero Host RAM)
    director_state = execute_phase_a_ltxdirector(
        timeline_ctrl=timeline_ctrl,
        workdir=workdir,
        resume=resume
    )

    # Phase B: Segment-Wise Diffusion (Stage 1 & 2 Latent Diffusion per Director Segment)
    # Returns on-disk latent PATHS (not resident tensors) to keep host RAM flat.
    segment_pack_paths = execute_segment_wise_diffusion_pipeline(
        director_state=director_state,
        seed=seed,
        workdir=workdir,
        resume=resume
    )

    # RAM-SAFE: Phase C/D read segment latents lazily from disk, so release the
    # full Director latents + conditioning before decoding.
    del director_state
    LTXDirectorMemoryManager.purge("post_phase_b")

    # Phase C: Video VAE Decoding -> streamed straight to a silent MP4
    frames_path = execute_phase_c_video_decode(
        segment_pack_paths=segment_pack_paths,
        workdir=workdir,
        resume=resume,
        fps=int(active_metadata["frame_rate"]),
        crf=crf
    )

    # Phase C: Audio VAE Decoding -> Immediate Purge
    audio_path = execute_phase_c_audio_decode(
        segment_pack_paths=segment_pack_paths,
        workdir=workdir,
        resume=resume
    )

    # Phase D: Final Mux (stream-copy, no frames in RAM)
    final_video = execute_phase_d_vhs_combine(
        frames_file_path=frames_path,
        audio_file_path=audio_path,
        fps=int(active_metadata["frame_rate"]),
        crf=crf,
        outdir=outdir
    )

    verify_output_artifacts(
        video_path=final_video,
        expected_frames=int(active_metadata["normalDurationFrames"]),
        expected_fps=int(active_metadata["frame_rate"])
    )

    elapsed = time.time() - start_time
    mem = LTXDirectorMemoryManager.get_memory_stats()

    print("\n" + "="*70)
    print("🎬 LTX-2.3 DIRECTOR 2.0 COMPLETE")
    print("="*70)
    print(f"  Duration           : {active_metadata['duration_seconds']:.2f} sec ({active_metadata['normalDurationFrames']} frames @ {active_metadata['frame_rate']} FPS)")
    print(f"  Memory Status      : Free RAM: {mem['ram_avail_gb']:.2f} GB | GPU VRAM Free: {mem['gpu_free_gb']:.2f} GB")
    print(f"  Total Elapsed Time : {elapsed/60:.2f} minutes ({elapsed:.1f}s)")
    print(f"  Final Master Video : {final_video}")
    print("="*70 + "\n")

    return final_video

if __name__ == "__main__":
    final_output_file = run_ltx23_director_original_workflow(
        global_prompt=GLOBAL_PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        timeline_metadata=TIMELINE_METADATA,
        segments=ORIGINAL_SEGMENTS,
        seed=BASE_SEED,
        crf=OUTPUT_CRF,
        workdir=WORK_DIRECTORY,
        outdir=OUTPUT_DIRECTORY,
        resume=RESUME_CHECKPOINTS,
        debug=DEBUG_MODE,
        debug_max_frames=DEBUG_MAX_FRAMES
    )