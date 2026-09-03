
# -- coding: utf-8 --
"""
LTX23_Director_Master_V6.py  · LTX-2.5 / LTX-5 Integration + All-In-One Workflow
Target: Colab Free Tier (T4 15GB VRAM | ~12.2GB Host RAM)
Feature: 100% VRAM Leak Patched (Nuclear Purge), Restored Colab UI Sliders,
         All-In-One Workflow Integration (9 Generation Modes)
"""

# ════════════════════════════════════════════════════════════════════════════════
# CELL 1: ENVIRONMENT SETUP & MEMORY PROTECTION  [FIXED: utils module shadowing]
# ════════════════════════════════════════════════════════════════════════════════
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

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = ('expandable_segments:True,'
                                         'garbage_collection_threshold:0.8,'
                                         'max_split_size_mb:64')
os.environ['TORCH_CUDNN_V8_API_ENABLED'] = '1'
os.environ['MALLOC_TRIM_THRESHOLD_'] = '65536'

# Fetch Hugging Face Token for gated LTX-2.5 downloads
HF_TOKEN = os.environ.get("HF_TOKEN", "")
try:
    from google.colab import userdata
    HF_TOKEN = userdata.get('HF_TOKEN') or HF_TOKEN
except Exception:
    pass

# ── PATH CONSTANTS ───────────────────────────────────────────────────────────
COMFY_ROOT   = os.environ.get("LTX_COMFY_ROOT", "/content/ComfyUI")
CONTENT_ROOT = os.environ.get("LTX_CONTENT_ROOT", "/content")
MODELS_DIR   = os.path.join(COMFY_ROOT, "models")
INPUT_DIR    = os.path.join(COMFY_ROOT, "input")
WHATDREAMS_INPUT = os.path.join(INPUT_DIR, "whatdreamscost")

# ── LIGHTWEIGHT LOGGING ──────────────────────────────────────────────────────
_LOG_LEVELS = {"DEBUG": 10, "INFO": 20, "WARN": 30, "ERROR": 40}
LOG_LEVEL = os.environ.get("LTX_LOG_LEVEL", "INFO").upper()

def log(msg: str, level: str = "INFO"):
    if _LOG_LEVELS.get(level, 20) >= _LOG_LEVELS.get(LOG_LEVEL, 20):
        prefix = {"DEBUG": "  🔎 ", "INFO": "  ", "WARN": "  ⚠️ ", "ERROR": "  ❌ "}.get(level, "  ")
        print(f"{prefix} {msg}")

def _dbg(msg: str):
    log(msg, "DEBUG")

def run_cmd(cmd: str, silent: bool = True) -> int:
    if silent:
        res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return res.returncode
    return subprocess.run(cmd, shell=True).returncode

try:
    import psutil
    vm = psutil.virtual_memory()
    print(f"  📊 Memory: Host RAM {vm.available/1e9:.2f} GB free / {vm.total/1e9:.2f} GB")
except Exception:
    pass

# ══════════════════════════════════════════════════════════════════════════════
# 🔧 FIX: Do NOT create a fake 'utils' module — it shadows ComfyUI's real
#    utils package (which contains mime_types.py, etc.) and breaks server.py.
#    Instead, just ensure /content/ComfyUI is importable.
# ══════════════════════════════════════════════════════════════════════════════
# Remove any previously-broken fake 'utils' from sys.modules if it exists
# from a prior cell run, so the real package can be found.
if "utils" in sys.modules:
    _existing = sys.modules["utils"]
    # If it's a fake ModuleType without a real __path__ to ComfyUI's utils, remove it
    if isinstance(_existing, types.ModuleType) and not hasattr(_existing, "__file__"):
        _fake_path = getattr(_existing, "path", [])
        if not any("/content/ComfyUI/utils" in str(p) and os.path.isdir(str(p)) for p in (
            _fake_path if isinstance(_fake_path, list) else [_fake_path]
        )):
            # It's likely our old fake module — remove it so real utils can load
            for _k in list(sys.modules.keys()):
                if _k == "utils" or _k.startswith("utils."):
                    del sys.modules[_k]
            print("  🧹 Cleaned stale fake 'utils' module from sys.modules")

print("✅ Cell 1: Environment & Memory Protection configured.")

# ════════════════════════════════════════════════════════════════════════════════
# CELL 2: INSTALL PYTHON DEPENDENCIES
# ════════════════════════════════════════════════════════════════════════════════
print("⚙️ [2/3] Installing Core Dependencies & PyTorch...")
run_cmd("pip install -q torch torchvision torchaudio", silent=False)
run_cmd("pip uninstall -y utils || true")
os.chdir("/content")
run_cmd("pip install -q torchsde einops diffusers accelerate psutil")
run_cmd("pip install -q av spandrel albumentations onnx opencv-python onnxruntime nest_asyncio imageio aiohttp scipy")
run_cmd("pip install -q 'kornia==0.7.3'")
run_cmd("apt-get -y install -qq aria2 ffmpeg")
run_cmd("pip install -q sageattention || true")
print("✅ Cell 2: Dependencies installed.")

# ════════════════════════════════════════════════════════════════════════════════
# CELL 3: CLONE UPSTREAM COMFYUI CORE
# ════════════════════════════════════════════════════════════════════════════════
if not os.path.isdir("/content/ComfyUI"):
    print("⚙️ [3/3] Cloning ComfyUI repository...")
    run_cmd("git clone https://github.com/comfyanonymous/ComfyUI.git /content/ComfyUI")
    run_cmd("pip install -q -r /content/ComfyUI/requirements.txt")

if "/content/ComfyUI" not in sys.path:
    sys.path.insert(0, "/content/ComfyUI")
if "/content" not in sys.path:
    sys.path.insert(1, "/content")

# Ensure the real utils package directory exists and is a proper package
os.makedirs("/content/ComfyUI/utils", exist_ok=True)
_init_path = "/content/ComfyUI/utils/__init__.py"
if not os.path.exists(_init_path):
    run_cmd(f"touch {_init_path}")

# 🔧 FIX: Verify the real utils package is importable BEFORE any node loading
try:
    import utils as _real_utils
    print(f"  ✓ Real 'utils' package loaded from: {getattr(_real_utils, '__file__', 'unknown')}")
except ImportError as e:
    print(f"  ⚠️  Could not import real 'utils' package: {e}")

print("✅ Cell 3: ComfyUI Core ready.")

# ════════════════════════════════════════════════════════════════════════════════
# CELL 4: INSTALL REQUIRED CUSTOM NODES (EXTENDED FOR ALL-IN-ONE)
# ════════════════════════════════════════════════════════════════════════════════
custom_nodes_dir = "/content/ComfyUI/custom_nodes"
os.makedirs(custom_nodes_dir, exist_ok=True)

for item in os.listdir(custom_nodes_dir):
    full_p = os.path.join(custom_nodes_dir, item)
    if os.path.isdir(full_p) and (item.isdigit() or item.startswith(".") or item == "comfyui"):
        shutil.rmtree(full_p, ignore_errors=True)

os.chdir(custom_nodes_dir)
repos = [
    # Original Director nodes
    ("WhatDreamsCost-ComfyUI", "https://github.com/WhatDreamscost/WhatDreamsCost-ComfyUI"),
    ("ComfyUI_KJNodes", "https://github.com/kijai/ComfyUI-KJNodes.git"),
    ("ComfyUI_GGUF", "https://github.com/city96/ComfyUI-GGUF.git"),
    ("ComfyUI-LTXVideo", "https://github.com/Lightricks/ComfyUI-LTXVideo"),
    ("ComfyUI-VideoHelperSuite", "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite"),
    ("rgthree-comfy", "https://github.com/rgthree/rgthree-comfy"),
    # NEW: All-In-One Workflow dependencies
    ("comfyui_controlnet_aux", "https://github.com/Fannovel16/comfyui_controlnet_aux"),
    ("ComfyUI-Easy-Use", "https://github.com/yolain/ComfyUI-Easy-Use"),
    ("ComfyUI-Math", "https://github.com/evanspearman/ComfyMath"),
    ("comfyui-custom-scripts", "https://github.com/pythongosssss/ComfyUI-Custom-Scripts"),
    ("ComfyUI-MediaMixer", "https://github.com/ComfyUI-MediaMixer/ComfyUI-MediaMixer"),
]

for folder, url in repos:
    if not os.path.isdir(folder):
        print(f"  Cloning {folder}...")
        run_cmd(f"git clone {url} {folder}")
    req_file = os.path.join(folder, "requirements.txt")
    if os.path.isfile(req_file):
        run_cmd(f"pip install -q -r {req_file} || true")

print("✅ Cell 4: Custom Nodes installed.")

# ════════════════════════════════════════════════════════════════════════════════
# CELL 5: DOWNLOAD MODELS (LTX-2.5 / 5 + All-In-One Additions)
# ════════════════════════════════════════════════════════════════════════════════
import torch

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

DIT_GGUF_NAME = os.environ.get("LTX_DIT_GGUF", "LTX-2.5-Distilled-Q4_K_M.gguf")

def download_file(url: str, dest_dir: str, filename: Optional[str] = None, needs_token: bool = False) -> Optional[str]:
    try:
        Path(dest_dir).mkdir(parents=True, exist_ok=True)
        if filename is None:
            filename = url.split('/')[-1].split('?')[0]
        dest = os.path.join(dest_dir, filename)
        if os.path.exists(dest) and os.path.getsize(dest) > 1024:
            print(f"  [FOUND] {filename}")
            return filename

        cmd = ['aria2c', '--console-log-level=error', '-c', '-x', '16',
               '-s', '16', '-k', '1M', '-d', dest_dir, '-o', filename]
        if needs_token and HF_TOKEN:
            cmd.append(f'--header=Authorization: Bearer {HF_TOKEN}')
        elif needs_token and not HF_TOKEN:
            print(f"\n  ❌ ERROR: {filename} is GATED. Please set HF_TOKEN in Colab Secrets.")
            return None
        cmd.append(url)
        print(f"  ↓ Downloading {filename}...", end=' ', flush=True)
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        if os.path.exists(dest) and os.path.getsize(dest) > 1024:
            print("Done!")
            return filename
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

print("📦 Downloading LTX-2.5 Core Models (DiT, Fused Gemma, VAEs, Upscaler)...")
_LTX_ROOT = "https://huggingface.co"

# 1. Distilled DiT GGUF (Ungated Mirror)
download_file(f"{_LTX_ROOT}/FenomAI/LTX-2.5-Distilled-GGUF/resolve/main/{DIT_GGUF_NAME}",
              os.path.join(MODELS_DIR, "unet"), filename=DIT_GGUF_NAME)
link_file_safe(os.path.join(MODELS_DIR, "unet", DIT_GGUF_NAME),
               os.path.join(MODELS_DIR, "diffusion_models", DIT_GGUF_NAME))

# 2. Fused Text Encoder (GATED)
gemma_fused = "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors"
download_file(f"{_LTX_ROOT}/Lightricks/LTX-2.5/resolve/main/text_encoders/{gemma_fused}",
              "/content/ComfyUI/models/text_encoders", filename=gemma_fused, needs_token=True)
link_file_safe(f"/content/ComfyUI/models/text_encoders/{gemma_fused}",
               f"/content/ComfyUI/models/clip/{gemma_fused}")

# 3. Video VAE (GATED)
video_vae = "ltx-2.5-video-vae-bf16.safetensors"
download_file(f"{_LTX_ROOT}/Lightricks/LTX-2.5/resolve/main/vae/{video_vae}",
              "/content/ComfyUI/models/vae", filename=video_vae, needs_token=True)

# 4. Audio VAE (GATED)
audio_vae = "ltx-2.5-audio-vae-bf16.safetensors"
download_file(f"{_LTX_ROOT}/Lightricks/LTX-2.5/resolve/main/vae/{audio_vae}",
              "/content/ComfyUI/models/vae", filename=audio_vae, needs_token=True)

# 5. Tiny VAE Preview
download_file(f"{_LTX_ROOT}/Kijai/LTX2.3_comfy/resolve/main/vae/taeltx2_3.safetensors",
              "/content/ComfyUI/models/vae", filename="taeltx2_3.safetensors")
link_file_safe("/content/ComfyUI/models/vae/taeltx2_3.safetensors",
               "/content/ComfyUI/models/vae_approx/taeltx2_3.safetensors")

# 6. Spatial Upscaler (GATED)
upscaler = "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"
download_file(f"{_LTX_ROOT}/Lightricks/LTX-2.5/resolve/main/latent_upscale_models/{upscaler}",
              "/content/ComfyUI/models/latent_upscale_models", filename=upscaler, needs_token=True)
link_file_safe(f"/content/ComfyUI/models/latent_upscale_models/{upscaler}",
               f"/content/ComfyUI/models/upscale_models/{upscaler}")

# ── NEW: All-In-One Workflow Additions ──────────────────────────────────────
print("📦 Downloading Director 2.0 4-LoRA Stack + All-In-One Additions...")
lora_dir = "/content/ComfyUI/models/loras"

# Director 2.0 legacy LoRAs
download_file("https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/loras/ltx-2.3-22b-distilled-lora-dynamic_fro09_avg_rank_105_bf16.safetensors",
              lora_dir, filename="ltx-2.3-22b-distilled-lora-dynamic_fro09_avg_rank_105_bf16.safetensors")
download_file("https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/loras/LTX-2.3-OmniNFT-RL-Lora_bf16.safetensors",
              lora_dir, filename="LTX-2.3-OmniNFT-RL-Lora_bf16.safetensors")
download_file("https://huggingface.co/joyfox/LTX-2.3-Transition-LORA/resolve/main/ltx2.3-transition.safetensors",
              lora_dir, filename="ltx2.3-transition.safetensors")
download_file("https://huggingface.co/vidfom/aimusic/resolve/main/ComfyUI/models/loras/LTX2.3-MVCamera-drclips.safetensors",
              lora_dir, filename="LTX2.3-MVCamera-drclips.safetensors")

# All-In-One additions: IC-LoRA for Lipsync (Modes 05/06/08)
download_file("https://huggingface.co/Lightricks/LTX-Video/resolve/main/ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors",
              lora_dir, filename="ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors")

# All-In-One additions: Detailer LoRA for Stage 2 refinement
download_file("https://huggingface.co/Lightricks/LTX-2.3/resolve/main/ltx-2-19b-ic-lora-detailer.safetensors",
              lora_dir, filename="ltx-2-19b-ic-lora-detailer.safetensors")

# All-In-One additions: Spatial Upscaler v1.1 (Director mode compat)
download_file("https://huggingface.co/Lightricks/LTX-Video/resolve/main/ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
              os.path.join(MODELS_DIR, "latent_upscale_models"),
              filename="ltx-2.3-spatial-upscaler-x2-1.1.safetensors")

# All-In-One additions: Text projection for DualCLIPLoader path
download_file("https://huggingface.co/Lightricks/LTX-2.3/resolve/main/ltx-2.3_text_projection_bf16.safetensors",
              os.path.join(MODELS_DIR, "text_encoders"),
              filename="ltx-2.3_text_projection_bf16.safetensors", needs_token=True)

# Audio file
audio_dest_dir = "/content/ComfyUI/input/whatdreamscost"
os.makedirs(audio_dest_dir, exist_ok=True)
audio_file_target = os.path.join(audio_dest_dir, "Late night trap.mp3")
if not os.path.exists(audio_file_target) or os.path.getsize(audio_file_target) < 10000:
    download_file("https://huggingface.co/vidfom/aimusic/resolve/main/Late%20night%20trap.mp3",
                  audio_dest_dir, filename="Late night trap.mp3")

print("✅ Cell 5: Models, LoRAs and audio validated.")

# ════════════════════════════════════════════════════════════════════════════════
# CELL 6: MASTER TIMELINE "NOTES" + ALL-IN-ONE SETTINGS
# ════════════════════════════════════════════════════════════════════════════════
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

# @markdown # 💥 Cell 6: Storyboard, Resolution & LoRA Settings
# @markdown ---
# @markdown ## 🎬 Generation Mode Selection (All-In-One)
generation_mode = "09_music_video_timeline"  # @param ["01_text_to_video", "02_audio_to_video", "03_image_to_video", "04_first_last_frame_to_video", "05_lipsync", "06_first_last_frame_lipsync", "07_first_middle_last_frame_custom_audio", "08_style_transfer_video_to_video", "09_music_video_timeline"]

# @markdown ## 🖼️ Image Inputs (Modes 03-08)
image_1_path = "whatdreamscost/1.png"  # @param {type:"string"}
image_2_path = "whatdreamscost/2.png"  # @param {type:"string"}
image_3_path = "whatdreamscost/3.png"  # @param {type:"string"}

# @markdown ## 🎥 Video Input (Mode 08)
video_input_path = ""  # @param {type:"string"}

# @markdown ## 🎵 Audio Input (Modes 02, 04-08)
audio_input_path = "whatdreamscost/Late night trap.mp3"  # @param {type:"string"}
audio_start_time = "00:00"  # @param {type:"string"}
audio_clip_length_seconds = 10  # @param {type:"slider", min:1, max:30, step:1}

# @markdown ## 📐 Resolution (Modes 01-08)
output_width = 1280  # @param [608, 736, 864, 960, 1056, 1152, 1216, 1280, 1344, 1376, 1504, 1664, 1824, 1920] {type:"raw"}
output_height = 736  # @param [352, 416, 480, 544, 608, 640, 672, 736, 768, 832, 928, 1024, 1088] {type:"raw"}

# @markdown ## ⚙️ Sampling (Modes 01-08)
sampler_name_subgraph = "euler_ancestral"  # @param ["euler", "euler_ancestral", "euler_ancestral_cfg_pp", "euler_cfg_pp", "dpmpp_2m", "dpmpp_sde"]
stage1_sigmas = "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"  # @param {type:"string"}
stage2_sigmas = "0.85, 0.7250, 0.4219, 0.0"  # @param {type:"string"}
use_dual_cfg = True  # @param {type:"boolean"}
cfg_value = 1.0  # @param {type:"slider", min:1.0, max:8.0, step:0.5}

# @markdown ## 🎭 IC-LoRA Settings (Modes 05, 06, 08)
use_ic_lora = True  # @param {type:"boolean"}
ic_lora_strength = 1.0  # @param {type:"slider", min:0.0, max:2.0, step:0.1}
use_detailer_lora = True  # @param {type:"boolean"}
detailer_lora_strength = 0.3  # @param {type:"slider", min:0.0, max:1.0, step:0.05}

# @markdown ## 🎞️ Frame Settings (Modes 03-08)
img_to_video_strength_s1 = 0.7  # @param {type:"slider", min:0.0, max:1.0, step:0.05}
img_to_video_strength_s2 = 1.0  # @param {type:"slider", min:0.0, max:1.0, step:0.05}
img_compression = 18  # @param {type:"slider", min:0, max:60, step:1}

# @markdown ## 🔊 Audio Output
use_native_ltx_audio = False  # @param {type:"boolean"}
audio_output_mode = "custom_audio"  # @param ["custom_audio", "native_ltx", "model_generated"]

# @markdown ---
# @markdown ## 🎬 Generation Targets (Mode 09 - Director Timeline)
generation_width  = 832    # @param [416, 512, 640, 704, 768, 832, 960, 1024, 1280] {type:"raw"}
generation_height = 480    # @param [240, 320, 384, 448, 480, 512, 544, 576, 720] {type:"raw"}
fps               = 24     # @param {type:"integer"}
render_seconds    = 31.5   # @param {type:"slider", min:3.0, max:31.5, step:0.5}

# @markdown ## 🎞️ Timeline Authoring Canvas (LTXDirector)
custom_width      = 1280   # @param [768, 1024, 1280] {type:"raw"}
custom_height     = 720    # @param [512, 576, 720] {type:"raw"}
divisible_by      = 32     # @param {type:"raw"}
keyframe_guide_strength = 1.0  # @param {type:"slider", min:0.0, max:1.0, step:0.05}
two_stage_base_render = True  # @param {type:"boolean"}

# @markdown ## 🎛️ Director 2.0 4-LoRA Stack
use_lora_1 = True      # @param {type: "boolean"}
lora_strength_1 = 0.4  # @param {type: "slider", min:0.0, max:1.5, step:0.05}
use_lora_2 = True      # @param {type: "boolean"}
lora_strength_2 = 0.6  # @param {type: "slider", min:0.0, max:1.5, step:0.05}
use_lora_3 = True      # @param {type: "boolean"}
lora_strength_3 = 0.7  # @param {type: "slider", min:0.0, max:1.5, step:0.05}
use_lora_4 = True      # @param {type: "boolean"}
lora_strength_4 = 0.9  # @param {type: "slider", min:0.0, max:1.5, step:0.05}

# @markdown ## ⚡ Two-Stage Sampler (Mode 09)
scheduler_name = "linear_quadratic"  # @param ["linear_quadratic", "normal", "simple", "beta", "sgm_uniform", "karras"]
sampler_name   = "euler"             # @param ["euler", "euler_ancestral", "dpmpp_2m", "dpmpp_sde", "ddim"]
cfg            = 1.0                 # @param {type: "slider", min:1.0, max:8.0, step:0.5}
stage1_steps   = 8                   # @param {type: "slider", min:4, max:30, step:1}
stage1_denoise = 1.0                 # @param {type: "slider", min:0.1, max:1.0, step:0.01}
stage1_guide_strength = 0.5          # @param {type: "slider", min:0.0, max:1.0, step:0.05}
stage2_steps   = 4                   # @param {type: "slider", min:2, max:20, step:1}
stage2_denoise = 0.42                # @param {type: "slider", min:0.1, max:1.0, step:0.01}
stage2_guide_strength = 1.0          # @param {type: "slider", min:0.0, max:1.0, step:0.05}

# @markdown ## 🎵 Audio (Mode 09)
use_song_audio = True                        # @param {type:"boolean"}
audio_trim_start_frames = 446.9222739141953  # @param {type:"raw"}

# @markdown ## 🧠 Memory / Performance (free-tier T4)
VRAM_MODE = "auto"            # @param ["auto", "normalvram", "lowvram", "novram", "highvram"]
essential_loras_only = False  # @param {type: "boolean"}
vram_shield_mb       = 1200   # @param {type: "raw"}
min_ram_guard_gb     = 1.5    # @param {type: "slider", min:1.0, max:6.0, step:0.5}
auto_safe_on_t4           = True   # @param {type: "boolean"}
t4_singlepass_max_seconds = 31.5   # @param {type: "slider", min:3.0, max:31.5, step:0.5}
t4_singlepass_max_height  = 384    # @param {type: "slider", min:192, max:480, step:32}

# @markdown ## 💾 Output & Run
output_crf         = 8     # @param {type:"slider", min:0, max:30, step:1}
base_seed          = 0     # @param {type:"integer"}
resume_checkpoints = True  # @param {type:"boolean"}

try:
    _gpu_total_gb = (torch.cuda.get_device_properties(0).total_memory / 1e9) if torch.cuda.is_available() else 0.0
    _gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
except Exception:
    _gpu_total_gb, _gpu_name = 0.0, "unknown"

FULL_QUALITY_HW = _gpu_total_gb >= 20.0
print(f"  🖥️  GPU: {_gpu_name} ({_gpu_total_gb:.1f} GB VRAM) → "
      f"{'FULL-QUALITY capable' if FULL_QUALITY_HW else 'small GPU (T4-class)'}")

if not FULL_QUALITY_HW:
    if auto_safe_on_t4:
        two_stage_base_render = True
        if str(VRAM_MODE).lower() in ("auto", "novram"):
            VRAM_MODE = "normalvram"
        _cap = float(t4_singlepass_max_seconds)
        if render_seconds > _cap:
            print(f"  ✂️  render_seconds {render_seconds}s → {_cap}s cap किया (T4 पर 22B model के लिए)")
            render_seconds = _cap
        _hcap = int(t4_singlepass_max_height)
        if int(generation_height) > _hcap:
            _db = max(1, int(divisible_by))
            _ratio = _hcap / float(int(generation_height))
            _new_h = max(_db, int(round(_hcap / _db)) * _db)
            _new_w = max(_db, int(round(int(generation_width) * _ratio / _db)) * _db)
            print(f"  ✂️  Resolution {generation_width}x{generation_height} → {_new_w}x{_new_h} cap किया")
            generation_width, generation_height = _new_w, _new_h
        print(f"  ✅ FAITHFUL T4 MODE: single continuous pass · VRAM={VRAM_MODE}.")
    else:
        print("  ⚠️  auto_safe_on_t4=False (छोटा GPU) — जैसा configured है वैसा ही चलेगा")
elif not two_stage_base_render:
    print("  ✅ FAITHFUL MODE: single continuous full-timeline pass, base custom//2 → 2x → full canvas.")

_ORIG_SEGMENTS = [
    ("1785555235678s2fn3", 226.01059340956584, "whatdreamscost/1.png"),
    ("17855552413529uw9r", 161.31859976617454, "whatdreamscost/2.png"),
    ("1785555243885y3h85", 131.45629831196658, "whatdreamscost/3.png"),
    ("1785555247117rcoma", 225.5063328766255,  "whatdreamscost/4.png"),
    ("17855554543736wlrg", 83.22765271847516,  "whatdreamscost/5.3.png"),
]
_ORIG_TOTAL_FRAMES = 756
_ORIG_AUDIO_LEN = 756.5194770828076

def _snap_ltx_frames(n: float) -> int:
    n = int(max(9, round(n)))
    return ((n - 1) // 8) * 8 + 1

_total_frames = _snap_ltx_frames(render_seconds * fps)
_factor = _total_frames / _ORIG_TOTAL_FRAMES
_duration_seconds = _total_frames / float(fps)

ORIGINAL_SEGMENTS: List[Dict[str, Any]] = []
_cursor = 0.0
for _sid, _slen, _img in _ORIG_SEGMENTS:
    _L = _slen * _factor
    ORIGINAL_SEGMENTS.append({"id": _sid, "start": _cursor, "length": _L,
                              "prompt": "", "type": "image", "imageFile": _img})
    _cursor += _L

_segment_lengths_str = ", ".join(f"{s['length']}" for s in ORIGINAL_SEGMENTS)
_guide_strength_str = ", ".join(f"{keyframe_guide_strength:.2f}" for _ in ORIGINAL_SEGMENTS)

ORIGINAL_AUDIO_SEGMENTS = [{
    "id": "1785169457779kollx", "type": "audio", "start": 0.0,
    "length": _ORIG_AUDIO_LEN * _factor, "trimStart": float(audio_trim_start_frames),
    "audioDurationFrames": 2880, "audioFile": "whatdreamscost/Late night trap.mp3",
    "fileName": "Late night trap.mp3",
}]

ORIGINAL_MOTION_SEGMENTS: List[Dict[str, Any]] = []

def _snap_div(n: int, d: int) -> int:
    d = max(1, int(d))
    return max(d, int(round(n / d)) * d)

_base_w = _snap_div(int(generation_width) // 2, divisible_by)
_base_h = _snap_div(int(generation_height) // 2, divisible_by)

if two_stage_base_render:
    _director_render_w, _director_render_h = _base_w, _base_h
else:
    _director_render_w = _snap_div(int(custom_width) // 2, divisible_by)
    _director_render_h = _snap_div(int(custom_height) // 2, divisible_by)

TIMELINE_METADATA = {
    "frame_rate": float(fps),
    "duration_seconds": _duration_seconds,
    "normalDurationFrames": _total_frames,
    "start_frame": 0,
    "end_frame": _total_frames,
    "custom_width": int(_director_render_w),
    "custom_height": int(_director_render_h),
    "authoring_width": int(custom_width),
    "authoring_height": int(custom_height),
    "generation_width": int(generation_width),
    "generation_height": int(generation_height),
    "base_stage1_width": _base_w,
    "base_stage1_height": _base_h,
    "mainTrackEnabled": True,
    "audioTrackEnabled": bool(use_song_audio),
    "motionTrackEnabled": True,
    "inpaint_audio": True,
    "override_audio": False,
    "use_custom_audio": bool(use_song_audio),
    "use_custom_motion": True,
    "audio_file": "whatdreamscost/Late night trap.mp3",
    "audio_duration_frames": 2880,
    "audio_trim_start_frames": float(audio_trim_start_frames),
    "resize_method": "maintain aspect ratio",
    "divisible_by": int(divisible_by),
    "img_compression": int(img_compression),
    "epsilon": 0.001,
    "display_mode": "seconds",
    "guide_strength": _guide_strength_str,
    "segment_lengths": _segment_lengths_str,
    "local_prompts": "",
}

_ALL_LORAS = [
    (use_lora_1, "ltx-2.3-22b-distilled-lora-dynamic_fro09_avg_rank_105_bf16.safetensors", lora_strength_1),
    (use_lora_2, "LTX-2.3-OmniNFT-RL-Lora_bf16.safetensors",                                lora_strength_2),
    (use_lora_3, "ltx2.3-transition.safetensors",                                           lora_strength_3),
    (use_lora_4, "LTX2.3-MVCamera-drclips.safetensors",                                     lora_strength_4),
]
LORA_STACK = [{"on": bool(o), "lora": n, "strength": float(s)} for (o, n, s) in _ALL_LORAS if o]

STAGE1 = {"scheduler": scheduler_name, "steps": int(stage1_steps), "denoise": float(stage1_denoise),
          "cfg": float(cfg), "guide_strength": float(stage1_guide_strength)}
STAGE2 = {"scheduler": scheduler_name, "steps": int(stage2_steps), "denoise": float(stage2_denoise),
          "cfg": float(cfg), "guide_strength": float(stage2_guide_strength)}

VHS_SETTINGS = {"format": "video/h264-mp4", "pix_fmt": "yuv420p",
                "crf": int(output_crf), "filename_prefix": "LTX25_Director_Master"}

ESSENTIAL_LORAS_ONLY = bool(essential_loras_only)
VRAM_MODE = str(VRAM_MODE)

if int(vram_shield_mb) == 1200:
    if _gpu_total_gb and _gpu_total_gb < 20.0:
        VRAM_SHIELD_MB = 640
    else:
        VRAM_SHIELD_MB = 1200
else:
    VRAM_SHIELD_MB = int(vram_shield_mb)

print(f"  🛡️  VRAM shield = {VRAM_SHIELD_MB} MB (GPU {_gpu_total_gb:.1f} GB)")

BASE_SEED = int(base_seed)
OUTPUT_CRF = int(output_crf)
RESUME_CHECKPOINTS = bool(resume_checkpoints)
USE_SONG_AUDIO = bool(use_song_audio)

print("✅ Cell 6: Master Timeline notes loaded.")

# ════════════════════════════════════════════════════════════════════════════════
# CELL 7: NODE REGISTRY (EXTENDED)
# ════════════════════════════════════════════════════════════════════════════════
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
                def send_sync(self, *a, **k):
                    pass
            PromptServer.instance = MockServer()
except Exception as e:
    print(f"  ⚠️  PromptServer init notice: {e}")

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
        loop.run_until_complete(asyncio.ensure_future(_init_nodes_async()))
    else:
        loop.run_until_complete(_init_nodes_async())
except Exception:
    pass

from nodes import NODE_CLASS_MAPPINGS, LoraLoaderModelOnly

# 🔧 FIX: Split into CRITICAL (must have) and OPTIONAL (nice to have) nodes.
# This prevents the entire pipeline from crashing if one optional node is missing.
CRITICAL_NODES = [
    # Core Director pipeline
    "LTXDirector", "LTXDirectorGuide", "LTXDirectorCropGuides",
    "LTXVConditioning", "LTXVConcatAVLatent", "LTXVSeparateAVLatent",
    "LTXVLatentUpsampler", "LTXVAudioVAEDecode",
    # Model loaders
    "UnetLoaderGGUF", "CLIPLoader", "VAELoader", "LatentUpscaleModelLoader",
    # Sampling
    "SamplerCustomAdvanced", "CFGGuider", "KSamplerSelect", "BasicScheduler", "RandomNoise",
    "ConditioningZeroOut", "VAEDecode",
    # LoRA
    "Power Lora Loader (rgthree)", "LoraLoaderModelOnly",
    # Output
    "VHS_VideoCombine",
    # KJNodes essentials
    "ModelPreviewOverrideKJ", "VAELoaderKJ",
]

OPTIONAL_NODES = [
    # All-In-One subgraph nodes (Modes 01-08)
    "LTXVPreprocess", "LTXVImgToVideoInplace", "LTXVEmptyLatentAudio",
    "LTXVAudioVAEEncode", "LTXVCropGuides", "LTXVDualCFGGuider",
    "EmptyLTXVLatentVideo", "LTXVImgToVideoConditionOnly",
    "LTXAddVideoICLoRAGuide",
    "LTXVImgToVideoInplaceKJ", "VAEDecodeTiled",
    "LTXICLoRALoaderModelOnly",
    "DualCLIPLoader", "UNETLoader", "CLIPTextEncode",
    "EmptyImage", "ImageScaleBy", "GetImageSize", "SolidMask",
    "SetLatentNoiseMask", "LoadImage", "ResizeImageMaskNode",
    "ManualSigmas",
    "VHS_LoadVideo", "VHS_LoadAudioUpload",
    "DWPreprocessor", "DepthAnythingV2Preprocessor", "CannyEdgePreprocessor",
    "MathExpression|pysssss", "CM_FloatToInt", "CM_IntToFloat",
    "PrimitiveFloat", "INTConstant", "FloatConstant",
    "SimpleCalculatorKJ", "ComfySwitchNode", "FinalFrameSelector",
    "Any Switch (rgthree)", "Fast Groups Bypasser (rgthree)",
    "easy cleanGpuUsed",
    "GetNode", "SetNode",
    "ResolutionSelector",
]

def validate_original_nodes() -> bool:
    print("\n" + "=" * 70 + "\n🔍 ORIGINAL WORKFLOW NODE AUDIT\n" + "=" * 70)

    # Check CRITICAL nodes — must all be present
    missing_critical = []
    for name in CRITICAL_NODES:
        if name in NODE_CLASS_MAPPINGS:
            print(f"  ✓ {name:<36} -> {NODE_CLASS_MAPPINGS[name].__name__}")
        else:
            print(f"  ❌ MISSING (CRITICAL): {name}")
            missing_critical.append(name)

    # Check OPTIONAL nodes — warn but don't fail
    missing_optional = []
    for name in OPTIONAL_NODES:
        if name in NODE_CLASS_MAPPINGS:
            print(f"  ✓ {name:<36} -> {NODE_CLASS_MAPPINGS[name].__name__}")
        else:
            print(f"  ⚠️  MISSING (optional): {name}")
            missing_optional.append(name)

    if missing_critical:
        raise RuntimeError(
            "CRITICAL NODE VALIDATION FAILED. Missing: " + ", ".join(missing_critical)
            + "\n\nThis usually means a custom node repo failed to install. "
            "Check Cell 4 output for errors."
        )

    if missing_optional:
        print(f"\n  ⚠️  {len(missing_optional)} optional nodes missing. "
              f"Modes 01-08 may have limited functionality. "
              f"Mode 09 (Director Timeline) will still work.")

    print("✅ Cell 7: nodes verified.")
    return True

validate_original_nodes()

# ════════════════════════════════════════════════════════════════════════════════
# CELL 8: PRODUCTION MEMORY ENGINE
# ════════════════════════════════════════════════════════════════════════════════
def malloc_trim_os():
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass

def get_ram_free_gb() -> float:
    try:
        import psutil
        return psutil.virtual_memory().available / 1e9
    except Exception:
        return 99.0

def patch_comfy_memory_manager():
    try:
        import comfy.model_management as mm
        if not getattr(mm, "_ltx_patched", False):
            _orig_free = mm.free_memory
            def _safe_free(*a, **k):
                try:
                    r = _orig_free(*a, **k)
                    return r if isinstance(r, list) else []
                except Exception:
                    return []
            mm.free_memory = _safe_free

            _orig_getfree = mm.get_free_memory
            def _buffered_getfree(dev=None, torch_free_too=False):
                try:
                    free = _orig_getfree(dev, torch_free_too)
                    shield = int(globals().get("VRAM_SHIELD_MB", 256)) * 1024 * 1024
                    return max(256 * 1024 * 1024, free - shield)
                except Exception:
                    return 2 * 1024 * 1024 * 1024
            mm.get_free_memory = _buffered_getfree
            mm._ltx_patched = True
    except Exception as e:
        print(f"  [mem-patch notice] {e}")

def patch_safetensors_direct_to_gpu():
    pass

def drop_os_page_cache():
    patterns = [
        "/content/ComfyUI/models/unet/*.gguf",
        "/content/ComfyUI/models/diffusion_models/*.gguf",
        "/content/ComfyUI/models/text_encoders/*.safetensors",
        "/content/ComfyUI/models/clip/*.safetensors",
        "/content/ComfyUI/models/vae/*.safetensors",
        "/content/ComfyUI/models/latent_upscale_models/*.safetensors",
        "/content/ComfyUI/models/upscale_models/*.safetensors",
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

def nuclear_purge():
    """THE FIX: Aggressively destroys ComfyUI's model tracker so NO PyTorch items are cached in VRAM."""
    try:
        import comfy.model_management as mm
        if hasattr(mm, 'current_loaded_models'):
            mm.current_loaded_models.clear()
        if hasattr(mm, 'loaded_models'):
            mm.loaded_models.clear()
        mm.unload_all_models()
        mm.cleanup_models()
        mm.soft_empty_cache()
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()
    drop_os_page_cache()
    malloc_trim_os()

def ram_guard(min_free_gb: float = 2.0, tag: str = ""):
    if get_ram_free_gb() < min_free_gb:
        print(f"  ⚠️ [RAM GUARD] Free RAM {get_ram_free_gb():.2f} GB < {min_free_gb} GB → nuclear purge")
        nuclear_purge()

def light_clear():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def medium_clear(tag: str = ""):
    try:
        import comfy.model_management as mm
        mm.soft_empty_cache()
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    malloc_trim_os()

def install_sampling_memory_hook(clear_every: int = 4, ram_guard_gb: float = 0.0):
    try:
        import comfy.utils as cu
        state = {"n": 0, "t0": None, "total": None}
        def _hook(value, total, preview_bytes=None, *args, **kwargs):
            state["n"] += 1
            try:
                v = float(value) if value is not None else 0.0
                tot = float(total) if total else 0.0
                if state["t0"] is None or state["total"] != tot or v <= 1.0:
                    state["t0"] = time.time()
                    state["total"] = tot
                elif v > 1.0 and tot > 0.0:
                    elapsed = time.time() - state["t0"]
                    per = elapsed / max(1.0, (v - 1.0))
                    eta = per * max(0.0, tot - v)
                    log(f"step {int(v)}/{int(tot)} · {per:.1f}s/it · ETA ~{eta/60:.1f} min", "INFO")
            except Exception:
                pass
            if clear_every <= 1 or (state["n"] % clear_every == 0):
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            if ram_guard_gb > 0 and (state["n"] % 4 == 0):
                if get_ram_free_gb() < ram_guard_gb:
                    gc.collect()
                    malloc_trim_os()

        if hasattr(cu, "set_progress_bar_global_hook"):
            cu.set_progress_bar_global_hook(_hook)
            print("  ⚙️ Per-step memory-clear + ETA hook active.")
    except Exception as e:
        print(f"  [mem-hook notice] {e}")

def mem_report(phase: str = "", node: str = ""):
    ram = get_ram_free_gb()
    if torch.cuda.is_available():
        gfree = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved()) / 1e9
        galloc = torch.cuda.memory_allocated() / 1e9
    else:
        gfree = galloc = 0.0
    tail = f" | {phase}" if phase else ""
    tail += f" · {node}" if node else ""
    print(f"  📊 RAM free {ram:.2f} GB | VRAM free {gfree:.2f} GB (alloc {galloc:.2f} GB){tail}")

def configure_vram_state(mode: str = "auto"):
    try:
        import comfy.model_management as mm
        if not torch.cuda.is_available():
            return
        mode = (mode or "auto").lower()
        dev = mm.get_torch_device()
        vs = getattr(mm, "VRAMState", None)
        if vs is None:
            return
        if mode == "highvram":
            mm.vram_state = vs.HIGH_VRAM
            mm.unet_offload_device = lambda: dev
        elif mode == "lowvram":
            mm.vram_state = vs.LOW_VRAM
        elif mode == "novram":
            mm.vram_state = vs.NO_VRAM
        elif mode == "normalvram":
            mm.vram_state = vs.NORMAL_VRAM
        else:
            mm.vram_state = vs.NORMAL_VRAM
    except Exception as e:
        pass

patch_comfy_memory_manager()
patch_safetensors_direct_to_gpu()
print("✅ Cell 8: Memory engine active.")

# ════════════════════════════════════════════════════════════════════════════════
# CELL 9: UNIVERSAL NODE DISPATCHER & TENSOR UTILS
# ════════════════════════════════════════════════════════════════════════════════
import numpy as np
from PIL import Image, ImageOps, ImageDraw

PARAM_ALIASES = {
    "weight_dtype": ["weight_dtype", "dtype", "weight_type", "precision"],
    "device": ["device", "device_type", "target_device"],
    "vae_name": ["vae_name", "name", "vae"],
    "model_name": ["model_name", "unet_name", "name"],
    "unet_name": ["unet_name", "model_name", "name"],
    "clip_name": ["clip_name", "clip_name1", "name"],
    "samples": ["samples", "latent", "latents", "video_latent", "av_latent", "latent_image"],
    "latent": ["latent", "latents", "samples", "video_latent", "latent_image"],
    "av_latent": ["av_latent", "latent", "samples", "latent_image"],
    "upscale_model": ["upscale_model", "latent_upscale_model", "model"],
    "frame_rate": ["frame_rate", "fps"],
    "fps": ["fps", "frame_rate"],
}

def gv(obj: Any, index: int = 0) -> Any:
    if obj is None: return None
    if isinstance(obj, (tuple, list)): return obj[index] if len(obj) > index else None
    if isinstance(obj, dict):
        if "result" in obj and isinstance(obj["result"], (list, tuple)): return obj["result"][index] if len(obj["result"]) > index else None
        return obj.get(index, None)
    if hasattr(obj, "args") and isinstance(obj.args, (list, tuple)): return obj.args[index] if len(obj.args) > index else None
    for attr in ["output", "outputs", "result", "values", "data"]:
        if hasattr(obj, attr):
            val = getattr(obj, attr)
            if isinstance(val, (list, tuple)) and len(val) > index: return val[index]
            if index == 0: return val
    try: return obj[index]
    except Exception: pass
    return obj if index == 0 else None

def unwrap_tensor(obj: Any) -> Any:
    if obj is None or isinstance(obj, torch.Tensor): return obj
    for attr in ["output", "result"]:
        if hasattr(obj, attr) and getattr(obj, attr) is not None: return unwrap_tensor(getattr(obj, attr))
    if isinstance(obj, (tuple, list)) and obj: return unwrap_tensor(obj[0])
    if isinstance(obj, dict):
        if "samples" in obj: return unwrap_tensor(obj["samples"])
        if "result" in obj and obj["result"]: return unwrap_tensor(obj["result"][0])
        for v in obj.values():
            if isinstance(v, torch.Tensor): return v
    if hasattr(obj, "args") and obj.args: return unwrap_tensor(obj.args[0])
    return obj

def unwrap_latent(x: Any) -> Dict[str, Any]:
    if x is None: return {"samples": None}
    for attr in ["output", "result"]:
        if hasattr(x, attr) and getattr(x, attr) is not None: x = getattr(x, attr)
    while isinstance(x, (tuple, list)) and x: x = x[0]
    if isinstance(x, dict):
        cur = x
        while isinstance(cur, dict) and "samples" in cur and isinstance(cur["samples"], dict): cur = cur["samples"]
        if isinstance(cur, dict) and "samples" in cur: return cur
        for v in cur.values():
            if isinstance(v, torch.Tensor): return {"samples": v}
        return {"samples": cur}
    if isinstance(x, torch.Tensor): return {"samples": x}
    return {"samples": x}

def sync_latent_device(latent: Any, target_device: Union[str, torch.device] = "cpu") -> Dict[str, Any]:
    target = torch.device(target_device)
    d = unwrap_latent(latent)
    s = d.get("samples", None)
    if isinstance(s, torch.Tensor):
        if s.is_nested: d["samples"] = torch.nested.nested_tensor([t.to(target) for t in s.unbind()])
        else: d["samples"] = s.to(target)
    return d

def sync_cond_to_cpu(cond: Any) -> Any:
    """CRITICAL MEMORY FIX: Ensures no PyTorch CUDA tensors or ModelPatchers leak into caching dictionaries."""
    if cond is None:
        return None
    if isinstance(cond, torch.Tensor):
        return cond.detach().cpu()
    if isinstance(cond, list):
        return [sync_cond_to_cpu(c) for c in cond]
    if isinstance(cond, tuple):
        return tuple(sync_cond_to_cpu(c) for c in cond)
    if isinstance(cond, dict):
        return {k: sync_cond_to_cpu(v) for k, v in cond.items() if k not in ['model', 'control', 'gligen']}
    return cond

def call_node(node_name: str, node_instance: Optional[Any] = None, **kwargs) -> Any:
    if node_instance is None:
        if node_name not in NODE_CLASS_MAPPINGS:
            raise RuntimeError(f"FATAL: node '{node_name}' is not registered.")
        node_instance = NODE_CLASS_MAPPINGS[node_name]()

    func_name = getattr(node_instance, "FUNCTION", None)
    callables = []
    if func_name and hasattr(node_instance, func_name) and callable(getattr(node_instance, func_name)):
        callables.append(getattr(node_instance, func_name))
    if hasattr(node_instance, "execute") and callable(getattr(node_instance, "execute")):
        callables.append(node_instance.execute)
    for fb in ["direct", "get_guider", "get_noise", "get_sampler", "get_sigmas", "sample",
               "apply_guide", "crop_guides", "upsample_latent", "concat", "separate",
               "encode", "decode", "load_unet", "load_clip", "load_vae", "combine_video",
               "override", "load_lora", "process"]:
        if hasattr(node_instance, fb) and callable(getattr(node_instance, fb)):
            callables.append(getattr(node_instance, fb))

    schema_defaults = {}
    if hasattr(node_instance, "INPUT_TYPES") and callable(node_instance.INPUT_TYPES):
        try:
            it = node_instance.INPUT_TYPES()
            for grp in ["required", "optional", "hidden"]:
                for p_name, p_spec in it.get(grp, {}).items():
                    if isinstance(p_spec, tuple) and len(p_spec) > 1 and isinstance(p_spec[1], dict) and "default" in p_spec[1]:
                        schema_defaults[p_name] = p_spec[1]["default"]
                    elif isinstance(p_spec, tuple) and p_spec and isinstance(p_spec[0], list) and p_spec[0]:
                        schema_defaults[p_name] = p_spec[0][0]
        except Exception:
            pass

    last_err = None
    for func in callables:
        try:
            sig = inspect.signature(func)
            valid = {}
            has_kwargs = False
            for p_name, param in sig.parameters.items():
                if p_name in ("cls", "self"):
                    continue
                if param.kind == inspect.Parameter.VAR_POSITIONAL:
                    continue
                if param.kind == inspect.Parameter.VAR_KEYWORD:
                    has_kwargs = True
                    continue
                if p_name in kwargs:
                    valid[p_name] = kwargs[p_name]
                    continue
                matched = False
                for alias in PARAM_ALIASES.get(p_name, [p_name]):
                    if alias in kwargs:
                        valid[p_name] = kwargs[alias]
                        matched = True
                        break
                if matched:
                    continue
                if param.default is not inspect.Parameter.empty:
                    continue
                if p_name in schema_defaults:
                    valid[p_name] = schema_defaults[p_name]
                    continue
                ann = str(param.annotation)
                if "int" in ann: valid[p_name] = 0
                elif "float" in ann: valid[p_name] = 0.0
                elif "bool" in ann: valid[p_name] = False
                elif "str" in ann: valid[p_name] = ""
                else: valid[p_name] = None
            if has_kwargs:
                for k, v in kwargs.items():
                    valid.setdefault(k, v)
            return func(**valid)
        except Exception:
            last_err = traceback.format_exc()
            continue

    if last_err is not None:
        raise RuntimeError(f"Error calling node '{node_name}':\n{last_err}")
    raise AttributeError(f"No callable function found on node '{node_name}'.")

def tiled_decode_video(video_latent: Any, vae_obj: Any, tile_size: int = 128) -> torch.Tensor:
    lat = unwrap_latent(video_latent)
    if "VAEDecodeTiled" in NODE_CLASS_MAPPINGS:
        try:
            return unwrap_tensor(call_node("VAEDecodeTiled", samples=lat, vae=vae_obj, tile_size=tile_size))
        except Exception:
            pass
    return unwrap_tensor(call_node("VAEDecode", samples=lat, vae=vae_obj))

def prepare_reference_image(image_path: str, width: int, height: int) -> torch.Tensor:
    full_path = image_path
    if not os.path.isabs(image_path):
        full_path = os.path.join(INPUT_DIR, image_path)
    if os.path.exists(full_path):
        img = ImageOps.exif_transpose(Image.open(full_path).convert("RGB"))
        target_aspect = width / height
        w, h = img.size
        if (w / h) > target_aspect:
            nw = int(target_aspect * h)
            off = (w - nw) // 2
            img = img.crop((off, 0, off + nw, h))
        else:
            nh = int(w / target_aspect)
            off = (h - nh) // 2
            img = img.crop((0, off, w, off + nh))
        arr = np.array(img.resize((width, height), Image.BICUBIC)).astype(np.float32) / 255.0
        return torch.from_numpy(arr).unsqueeze(0)
    return torch.full((1, height, width, 3), 0.5)

print("✅ Cell 9: Node dispatcher ready.")

# ════════════════════════════════════════════════════════════════════════════════
# CELL 10: MASTER TIMELINE CONTROLLER
# ════════════════════════════════════════════════════════════════════════════════
class DirectorTimelineController:
    def __init__(self, global_prompt, negative_prompt, meta, segments,
                 audio_segments, motion_segments, base_input_dir=INPUT_DIR):
        self.global_prompt = global_prompt
        self.negative_prompt = negative_prompt
        self.meta = meta
        self.segments = segments
        self.audio_segments = audio_segments
        self.motion_segments = motion_segments
        self.base_input_dir = base_input_dir
        self.validate_reference_images()

    def validate_reference_images(self):
        print("\n" + "=" * 70 + "\n🔍 VALIDATING DIRECTOR KEYFRAMES\n" + "=" * 70)
        self.n_real_keyframes = 0
        self.n_placeholder_keyframes = 0
        for s in self.segments:
            full = os.path.join(self.base_input_dir, s["imageFile"])
            if not os.path.exists(full):
                if "5.3.png" in s["imageFile"]:
                    alt = full.replace("5.3.png", "5.png")
                    if os.path.exists(alt):
                        os.makedirs(os.path.dirname(full), exist_ok=True)
                        shutil.copyfile(alt, full)
                        print(f"  ✓ Alias resolved 5.png → {full}")
                        self.n_real_keyframes += 1
                        continue
                os.makedirs(os.path.dirname(full), exist_ok=True)
                ph = Image.new("RGB", (768, 512), color=(40, 30, 70))
                ImageDraw.Draw(ph).text((40, 230),
                    f"UPLOAD singer photo → {os.path.basename(s['imageFile'])}", fill=(255, 255, 255))
                ph.save(full)
                self.n_placeholder_keyframes += 1
                print(f"  ⚠️  Placeholder created: {full}")
            else:
                self.n_real_keyframes += 1
                print(f"  ✓ Keyframe OK: {full}")
        total = len(self.segments)
        print(f"  📸 Keyframes: {self.n_real_keyframes}/{total} real, "
              f"{self.n_placeholder_keyframes} placeholder.")

    def build_timeline_json_string(self) -> str:
        m = self.meta
        tl = {
            "mainTrackEnabled": m["mainTrackEnabled"],
            "audioTrackEnabled": m["audioTrackEnabled"],
            "motionTrackEnabled": m["motionTrackEnabled"],
            "propHeight": 90,
            "globalPropHeight": 470,
            "showFilenames": True,
            "overrideAudio": m["override_audio"],
            "inpaint_audio": m["inpaint_audio"],
            "global_prompt": self.global_prompt,
            "retake_global_prompt": "",
            "retakeMode": False,
            "retakeStart": 24,
            "retakeLength": 48,
            "retakePrompt": "",
            "retakeStrength": 1,
            "retakeVideo": None,
            "normalStartFrame": int(m["start_frame"]),
            "normalDurationFrames": int(m["normalDurationFrames"]),
            "segments": [
                {
                    "id": s["id"], "start": float(s["start"]), "length": float(s["length"]),
                    "prompt": s.get("prompt", ""), "type": s["type"], "imageFile": s["imageFile"],
                    "imageB64": f"/api/view?filename={os.path.basename(s['imageFile'])}"
                                f"&type=input&subfolder={os.path.dirname(s['imageFile'])}",
                    "isEndFrame": False,
                } for s in self.segments
            ],
            "motionSegments": self.motion_segments,
            "audioSegments": [
                {
                    "id": a["id"], "type": a["type"], "start": float(a["start"]),
                    "length": float(a["length"]), "trimStart": float(a["trimStart"]),
                    "audioDurationFrames": int(a["audioDurationFrames"]),
                    "audioFile": a["audioFile"], "fileName": a["fileName"],
                } for a in self.audio_segments
            ],
        }
        return json.dumps(tl)

    def configure_ltxdirector(self, node_instance: Any):
        m = self.meta
        tl_json = self.build_timeline_json_string()
        widgets_values = [
            0, float(m["duration_seconds"]), float(m["duration_seconds"]),
            int(m["start_frame"]), int(m["end_frame"]), int(m["normalDurationFrames"]),
            tl_json, m["local_prompts"] or " |  |  |  | ", str(m["segment_lengths"]),
            float(m["epsilon"]), str(m["guide_strength"]), bool(m["mainTrackEnabled"]),
            bool(m["audioTrackEnabled"]), bool(m["motionTrackEnabled"]), float(m["frame_rate"]),
            m["display_mode"], int(m["custom_width"]), int(m["custom_height"]),
            m["resize_method"], int(m["divisible_by"]), int(m["img_compression"]),
            False, "",
        ]
        props = {
            "global_prompt": self.global_prompt,
            "timeline_data": tl_json,
            "frame_rate": float(m["frame_rate"]),
            "duration_frames": int(m["normalDurationFrames"]),
            "start_frame": int(m["start_frame"]),
            "end_frame": int(m["end_frame"]),
            "custom_width": int(m["custom_width"]),
            "custom_height": int(m["custom_height"]),
            "guide_strength": str(m["guide_strength"]),
            "segment_lengths": str(m["segment_lengths"]),
            "mainTrackEnabled": bool(m["mainTrackEnabled"]),
            "audioTrackEnabled": bool(m["audioTrackEnabled"]),
            "motionTrackEnabled": bool(m["motionTrackEnabled"]),
            "inpaint_audio": bool(m["inpaint_audio"]),
            "overrideAudio": bool(m["override_audio"]),
            "has_serialized_properties": True,
        }
        if hasattr(node_instance, "properties") and isinstance(node_instance.properties, dict):
            node_instance.properties.update(props)
        else:
            setattr(node_instance, "properties", props)
        setattr(node_instance, "widgets_values", widgets_values)
        setattr(node_instance, "timeline_data", tl_json)
        setattr(node_instance, "global_prompt", self.global_prompt)
        print("  ✓ LTXDirector Master Timeline payload attached.")

print("✅ Cell 10: DirectorTimelineController ready.")

# ════════════════════════════════════════════════════════════════════════════════
# CELL 11: MODEL / LoRA LOADERS (LTX-2.5)
# ════════════════════════════════════════════════════════════════════════════════
def load_dit_and_loras(clip_obj: Any = None):
    nuclear_purge()
    mem_report("DiT load", "UnetLoaderGGUF")
    _gguf = globals().get("DIT_GGUF_NAME", "LTX-2.5-Distilled-Q4_K_M.gguf")
    model = gv(call_node("UnetLoaderGGUF", unet_name=_gguf), 0)
    print(f"  ✓ UnetLoaderGGUF loaded ({_gguf}).")

    active_stack = LORA_STACK
    if globals().get("ESSENTIAL_LORAS_ONLY", False):
        active_stack = [lc for lc in LORA_STACK if "distilled" in lc["lora"].lower()] or LORA_STACK[:1]
        print(f"  ⚙️ ESSENTIAL_LORAS_ONLY → applying {len(active_stack)} LoRA(s) to save RAM.")

    applied = False
    if active_stack and "Power Lora Loader (rgthree)" in NODE_CLASS_MAPPINGS:
        try:
            lora_kwargs = {"model": model, "clip": clip_obj}
            for i, lc in enumerate(active_stack, start=1):
                lora_kwargs[f"lora_{i}"] = {"on": lc["on"], "lora": lc["lora"],
                                            "strength": lc["strength"], "strengthTwo": None}
            res = call_node("Power Lora Loader (rgthree)", **lora_kwargs)
            new_model = gv(res, 0)
            new_clip = gv(res, 1)
            if new_model is not None: model = new_model
            if new_clip is not None: clip_obj = new_clip
            applied = True
            print(f"  ✓ Power Lora Loader (rgthree): {len(active_stack)}-LoRA stack applied.")
        except Exception as e:
            print(f"  [notice] rgthree Power Lora fallback: {e}")

    if not applied:
        for lc in active_stack:
            path = os.path.join(MODELS_DIR, "loras", lc["lora"])
            if lc["on"] and os.path.exists(path):
                try:
                    gc.collect()
                    if torch.cuda.is_available(): torch.cuda.empty_cache()
                    malloc_trim_os()
                    ll = LoraLoaderModelOnly()
                    model = gv(call_node("LoraLoaderModelOnly", node_instance=ll,
                                         model=model, lora_name=lc["lora"],
                                         strength_model=float(lc["strength"])), 0) or model
                    print(f"    + LoRA {lc['lora']} (strength {lc['strength']})")
                except Exception as e:
                    print(f"    [notice] LoRA {lc['lora']} skipped: {e}")

    if "PatchSageAttentionKJ" in NODE_CLASS_MAPPINGS:
        try:
            model = gv(call_node("PatchSageAttentionKJ", model=model, sage_attention="auto"), 0) or model
            print("  ✓ SageAttention hook applied.")
        except Exception: pass

    if "LTXVChunkFeedForward" in NODE_CLASS_MAPPINGS:
        try:
            model = gv(call_node("LTXVChunkFeedForward", model=model, chunks=8, dim_threshold=4096), 0) or model
            print("  ✓ ChunkFeedForward hook applied (chunks=8).")
        except Exception: pass

    if "ModelPreviewOverrideKJ" in NODE_CLASS_MAPPINGS:
        try:
            tiny_vae = gv(call_node("VAELoaderKJ", vae_name="taeltx2_3.safetensors",
                                    device="main_device", weight_dtype="bf16"), 0)
            model = gv(call_node("ModelPreviewOverrideKJ", model=model, vae=tiny_vae), 0) or model
            print("  ✓ ModelPreviewOverrideKJ applied.")
        except Exception: pass

    _want_compile = (globals().get("FULL_QUALITY_HW", False)
                     and os.environ.get("LTX_TORCH_COMPILE", "1") != "0")
    if _want_compile and hasattr(torch, "compile"):
        try:
            if hasattr(model, "model") and hasattr(model.model, "diffusion_model"):
                model.model.diffusion_model = torch.compile(
                    model.model.diffusion_model, mode="reduce-overhead", dynamic=True)
                print("  ✓ torch.compile applied to the DiT.")
        except Exception: pass

    medium_clear("post_dit_load")
    return model, clip_obj

print("✅ Cell 11: DiT + Loader ready.")

# ════════════════════════════════════════════════════════════════════════════════
# CELL 12: PHASE A — LTX-2.5 INGESTION (Single Fused Encoder)
# ════════════════════════════════════════════════════════════════════════════════
class PrecomputedClipProxy:
    def __init__(self, precomputed_conditioning, tokenizer=None):
        self.cond = precomputed_conditioning
        self.cond_stage_model = None
        self.patcher = None
        self.layer_idx = None
        self.tokenizer = tokenizer

    def tokenize(self, text, *a, **k):
        if self.tokenizer is not None and hasattr(self.tokenizer, "tokenize_with_weights"):
            try: return self.tokenizer.tokenize_with_weights(text)
            except Exception: pass
        return {"ltxv": [], "text": text}

    def encode_from_tokens_scheduled(self, *a, **k): return self.cond
    def encode_from_tokens(self, *a, **k): return self.cond
    def encode(self, *a, **k): return self.cond
    def load_model(self, *a, **k): return self
    def clone(self): return self
    def get_key_patches(self): return {}
    def add_patches(self, *a, **k): return []
    def __getattr__(self, name): return lambda *a, **k: self.cond

def encode_prompt_on_gpu(prompt_text: str):
    nuclear_purge()
    import comfy.model_management as mm
    _prev_state = getattr(mm, "vram_state", None)
    if hasattr(mm, "VRAMState") and not globals().get("FULL_QUALITY_HW", False):
        mm.vram_state = getattr(mm.VRAMState, "NO_VRAM", mm.vram_state)
        print("  ⚙️ Forced NO_VRAM sequential streaming for the 12.5GB Gemma-12B encoder to prevent T4 OOM.")

    clip = gv(call_node("CLIPLoader",
                        clip_name="gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors",
                        type="ltxv"), 0)
    saved_tokenizer = getattr(clip, "tokenizer", None)
    mem_report("Phase A", "CLIPLoader (Gemma4-12B streaming active)")
    print("  ⚡ Encoding global prompt on GPU (LTX-2.5 fused encoder)...")
    t0 = time.time()
    with torch.inference_mode():
        cond_raw = gv(call_node("CLIPTextEncode", text=prompt_text, clip=clip), 0)
        cond_cpu = sync_cond_to_cpu(cond_raw)
        del cond_raw
    print(f"  ✓ Prompt encoded in {time.time() - t0:.2f}s. Purging Gemma4-12B weights...")
    if _prev_state is not None:
        mm.vram_state = _prev_state
    try:
        clip.cond_stage_model = None
        clip.patcher = None
    except Exception: pass
    del clip
    nuclear_purge()
    mem_report("Phase A", "Gemma weights purged (tokenizer kept)")
    return cond_cpu, saved_tokenizer

def execute_phase_a(timeline_ctrl: "DirectorTimelineController",
                    workdir: str, resume: bool = True) -> Tuple[Dict[str, Any], Any]:
    print("\n" + "=" * 70 + "\n🎬 PHASE A: LTX-2.5 DIRECTOR TIMELINE INGESTION\n" + "=" * 70)
    os.makedirs(workdir, exist_ok=True)
    state_file = os.path.join(workdir, "director_state.pt")
    if resume and os.path.exists(state_file) and os.path.getsize(state_file) > 1024:
        print(f"  ⏭ [RESUME] Loading cached Director state from: {state_file}")
        try: return torch.load(state_file, map_location="cpu"), None
        except Exception as e: print(f"  [notice] Director state cache unreadable ({e}) — regenerating.")

    nuclear_purge()
    m = timeline_ctrl.meta
    with torch.inference_mode():
        precomputed_cond, saved_tokenizer = encode_prompt_on_gpu(timeline_ctrl.global_prompt)
        model, _ = load_dit_and_loras(clip_obj=None)
        mem_report("Phase A", "DiT ready")

        audio_vae = gv(call_node("VAELoader", vae_name="ltx-2.5-audio-vae-bf16.safetensors"), 0)
        clip_proxy = PrecomputedClipProxy(precomputed_cond, tokenizer=saved_tokenizer)
        director_node = NODE_CLASS_MAPPINGS["LTXDirector"]()
        timeline_ctrl.configure_ltxdirector(director_node)

        print("  🚀 Ingesting the full timeline via LTXDirector...")
        t_dir = time.time()
        director_out = call_node(
            "LTXDirector", node_instance=director_node,
            model=model, clip=clip_proxy, audio_vae=audio_vae, optional_latent=None,
            global_prompt=timeline_ctrl.global_prompt,
            timeline_data=timeline_ctrl.build_timeline_json_string(),
            local_prompts="", segment_lengths=str(m["segment_lengths"]),
            start_second=0.0, end_second=float(m["duration_seconds"]),
            duration_seconds=float(m["duration_seconds"]),
            start_frame=int(m["start_frame"]), end_frame=int(m["end_frame"]),
            duration_frames=int(m["normalDurationFrames"]),
            epsilon=float(m["epsilon"]), guide_strength=str(m["guide_strength"]),
            frame_rate=float(m["frame_rate"]), display_mode=m["display_mode"],
            custom_width=int(m["custom_width"]), custom_height=int(m["custom_height"]),
            resize_method=m["resize_method"], divisible_by=int(m["divisible_by"]),
            img_compression=int(m["img_compression"]),
            use_custom_audio=bool(m["use_custom_audio"]),
            inpaint_audio=bool(m["inpaint_audio"]),
            use_custom_motion=bool(m["use_custom_motion"]),
            override_audio=bool(m["override_audio"]))
        print(f"  ⚡ Timeline ingested in {time.time() - t_dir:.2f}s.")

        patched_model = gv(director_out, 0) or model
        dir_pos = sync_cond_to_cpu(gv(director_out, 1) or precomputed_cond)
        dir_vid = sync_latent_device(gv(director_out, 2), "cpu")
        dir_aud = sync_latent_device(gv(director_out, 3), "cpu")
        dir_guide = sync_cond_to_cpu(gv(director_out, 4))
        dir_motion = sync_cond_to_cpu(gv(director_out, 5))
        fps_raw = gv(director_out, 6)
        dir_fps = float(fps_raw) if fps_raw is not None else float(m["frame_rate"])
        neg_zeroed = sync_cond_to_cpu(gv(call_node("ConditioningZeroOut", conditioning=dir_pos), 0))
        cond_out = call_node("LTXVConditioning", positive=dir_pos, negative=neg_zeroed, frame_rate=dir_fps)
        final_pos = sync_cond_to_cpu(gv(cond_out, 0))
        final_neg = sync_cond_to_cpu(gv(cond_out, 1))

        state = {
            "positive": final_pos, "negative": final_neg,
            "video_latent": dir_vid, "audio_latent": dir_aud,
            "guide_data": dir_guide, "motion_guide_data": dir_motion,
            "frame_rate": dir_fps, "meta": m,
        }
        try:
            tmp = state_file + ".tmp"
            torch.save(state, tmp)
            os.replace(tmp, state_file)
        except Exception: pass

        del audio_vae, clip_proxy, director_node, dir_pos, neg_zeroed, cond_out
        nuclear_purge()
    mem_report("Phase A complete (Gemma purged, DiT retained)")
    return state, patched_model

print("✅ Cell 12: Phase A ready.")

# ════════════════════════════════════════════════════════════════════════════════
# CELL 13: PHASE B — 2-STAGE DIFFUSION (LTX-2.5 VAEs & Upscaler)
# ════════════════════════════════════════════════════════════════════════════════
def _run_stage(model, video_vae, guide_strength, base_pos, base_neg,
               video_latent, audio_latent, guide_data, motion_guide_data,
               scheduler, steps, denoise, cfg, seed, stage_name):
    g = call_node(
        "LTXDirectorGuide",
        positive=base_pos, negative=base_neg, vae=video_vae,
        latent=video_latent, guide_data=guide_data,
        motion_guide_data=motion_guide_data, model=model,
        strength=guide_strength, rescale_method="None", guide_frame=1,
        interpolation="bicubic", crop_position="center", enable_guide=True)
    g_pos = sync_cond_to_cpu(gv(g, 0) if gv(g, 0) is not None else base_pos)
    g_neg = sync_cond_to_cpu(gv(g, 1) if gv(g, 1) is not None else base_neg)
    g_vid = sync_latent_device(gv(g, 2) if gv(g, 2) is not None else video_latent, "cpu")
    g_model = gv(g, 3) if gv(g, 3) is not None else model
    light_clear()

    av = sync_latent_device(gv(call_node("LTXVConcatAVLatent",
                                         video_latent=g_vid, audio_latent=audio_latent), 0), "cpu")
    light_clear()

    noise = gv(call_node("RandomNoise", noise_seed=seed), 0)
    guider = gv(call_node("CFGGuider", cfg=cfg, model=g_model, positive=g_pos, negative=g_neg), 0)
    sampler = gv(call_node("KSamplerSelect", sampler_name="euler"), 0)
    sigmas = gv(call_node("BasicScheduler", model=g_model, scheduler=scheduler,
                          steps=steps, denoise=denoise), 0)
    print(f"  ⚡ {stage_name}: sampling {steps} steps (denoise {denoise})...")
    t0 = time.time()
    ram_guard(globals().get("min_ram_guard_gb", 1.5), stage_name)
    out = call_node("SamplerCustomAdvanced", noise=noise, guider=guider,
                    sampler=sampler, sigmas=sigmas, latent_image=av)
    sampled = sync_latent_device(gv(out, 0), "cpu")
    print(f"  ✓ {stage_name} done in {time.time() - t0:.2f}s.")

    sep = call_node("LTXVSeparateAVLatent", av_latent=sampled)
    v_lat = sync_latent_device(gv(sep, 0), "cpu")
    a_lat = sync_latent_device(gv(sep, 1), "cpu")
    crop = call_node("LTXDirectorCropGuides", positive=g_pos, negative=g_neg, latent=v_lat)
    c_pos = sync_cond_to_cpu(gv(crop, 0) if gv(crop, 0) is not None else g_pos)
    c_neg = sync_cond_to_cpu(gv(crop, 1) if gv(crop, 1) is not None else g_neg)
    c_vid = sync_latent_device(gv(crop, 2) if gv(crop, 2) is not None else v_lat, "cpu")
    del g, av, noise, guider, sampler, sigmas, out, sampled, sep
    light_clear()
    return c_pos, c_neg, c_vid, a_lat

def _save_ckpt(path: str, obj: Dict[str, Any]) -> bool:
    try:
        tmp = path + ".tmp"
        torch.save(obj, tmp)
        os.replace(tmp, path)
        return True
    except Exception: return False

def _ensure_model(model):
    if model is None: model, _ = load_dit_and_loras(clip_obj=None)
    return model

def execute_phase_b(director_state: Dict[str, Any], model: Any, seed: int,
                    workdir: str, resume: bool = True) -> str:
    latent_file = os.path.join(workdir, "final_latents.pt")
    if resume and os.path.exists(latent_file) and os.path.getsize(latent_file) > 1024:
        return latent_file
    try: return _execute_phase_b_single(director_state, model, seed, workdir, resume)
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if "out of memory" in str(e).lower(): nuclear_purge()
        raise

def _execute_phase_b_single(director_state: Dict[str, Any], model: Any, seed: int,
                            workdir: str, resume: bool = True) -> str:
    latent_file = os.path.join(workdir, "final_latents.pt")
    print("\n" + "=" * 70 + "\n🎬 PHASE B: WHOLE-TIMELINE 2-STAGE DIFFUSION\n" + "=" * 70)
    nuclear_purge()
    guide_data = director_state["guide_data"]
    motion_guide = director_state["motion_guide_data"]
    base_pos = director_state["positive"]
    base_neg = director_state["negative"]
    stage1_file = os.path.join(workdir, "stage1_latents.pt")

    with torch.inference_mode():
        if resume and os.path.exists(stage1_file) and os.path.getsize(stage1_file) > 1024:
            _pk = torch.load(stage1_file, map_location="cpu")
            s1_pos, s1_neg = _pk["s1_pos"], _pk["s1_neg"]
            v_ups = {"samples": _pk["v_ups"]}
            s1_aud = {"samples": _pk["s1_aud"]} if _pk.get("s1_aud") is not None else None
            model = _ensure_model(model)
            video_vae = gv(call_node("VAELoader", vae_name="ltx-2.5-video-vae-bf16.safetensors"), 0)
        else:
            model = _ensure_model(model)
            video_vae = gv(call_node("VAELoader", vae_name="ltx-2.5-video-vae-bf16.safetensors"), 0)
            s1_pos, s1_neg, s1_vid, s1_aud = _run_stage(
                model, video_vae, STAGE1["guide_strength"], base_pos, base_neg,
                director_state["video_latent"], director_state["audio_latent"],
                guide_data, motion_guide,
                STAGE1["scheduler"], STAGE1["steps"], STAGE1["denoise"], STAGE1["cfg"],
                seed, "STAGE 1")
            medium_clear("single_post_s1")
            up_model = gv(call_node("LatentUpscaleModelLoader",
                                    model_name="ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"), 0)
            v_ups = sync_latent_device(gv(call_node("LTXVLatentUpsampler",
                                                    samples=s1_vid, upscale_model=up_model,
                                                    vae=video_vae), 0), "cpu")
            del up_model, s1_vid
            medium_clear("single_post_upscale")
            _save_ckpt(stage1_file, {
                "s1_pos": sync_cond_to_cpu(s1_pos),
                "s1_neg": sync_cond_to_cpu(s1_neg),
                "v_ups": unwrap_tensor(v_ups).detach().cpu().half(),
                "s1_aud": (unwrap_tensor(s1_aud).detach().cpu().half() if s1_aud is not None else None),
            })

        s2_pos, s2_neg, s2_vid, s2_aud = _run_stage(
            model, video_vae, STAGE2["guide_strength"], s1_pos, s1_neg,
            v_ups, s1_aud, guide_data, motion_guide,
            STAGE2["scheduler"], STAGE2["steps"], STAGE2["denoise"], STAGE2["cfg"],
            seed, "STAGE 2")
        final_video_lat = unwrap_tensor(s2_vid).detach().cpu().half()
        final_audio_lat = unwrap_tensor(s2_aud).detach().cpu().half() if s2_aud is not None else None

        del model, video_vae, v_ups, s1_pos, s1_neg, s1_aud, s2_pos, s2_neg, s2_vid, s2_aud
        if '_' in locals(): del _
        torch.save({"video": final_video_lat, "audio": final_audio_lat,
                    "frame_rate": director_state["frame_rate"]}, latent_file + ".tmp")
        os.replace(latent_file + ".tmp", latent_file)
        del final_video_lat, final_audio_lat
        try:
            if os.path.exists(stage1_file): os.remove(stage1_file)
        except Exception: pass

    nuclear_purge()
    mem_report("Phase B complete")
    return latent_file

print("✅ Cell 13: Phase B ready.")

# ════════════════════════════════════════════════════════════════════════════════
# CELL 14: PHASE C — CHUNKED STREAMING VAE DECODE (LTX-2.5 VAEs)
# ════════════════════════════════════════════════════════════════════════════════
DECODE_CHUNK_LAT_FRAMES = 4
DECODE_CHUNK_OVERLAP = 1

def _save_audio_wav(audio_dict: Any, wav_path: str, fallback_sr: int = 48000) -> bool:
    try:
        if not isinstance(audio_dict, dict) or "waveform" not in audio_dict:
            return False
        wf = audio_dict["waveform"]
        sr = int(audio_dict.get("sample_rate", fallback_sr))
        if not isinstance(wf, torch.Tensor):
            return False
        w = wf.detach().cpu().float()
        while w.dim() > 2:
            w = w[0]
        if w.dim() == 1:
            w = w.unsqueeze(0)
        import torchaudio
        torchaudio.save(wav_path, w, sr)
        return os.path.exists(wav_path) and os.path.getsize(wav_path) > 100
    except Exception:
        return False

def execute_phase_c(latent_file: str, workdir: str, fps: int, crf: int,
                    resume: bool = True) -> Tuple[str, str]:
    raw_video = os.path.join(workdir, "raw_video_noaudio.mp4")
    audio_file = os.path.join(workdir, "decoded_audio.pt")
    chunk_dir = os.path.join(workdir, "dec_chunks")
    os.makedirs(chunk_dir, exist_ok=True)
    need_video = not (resume and os.path.exists(raw_video) and os.path.getsize(raw_video) > 1024)
    need_audio = not (resume and os.path.exists(audio_file))
    pack = torch.load(latent_file, map_location="cpu") if (need_video or need_audio) else None

    if need_video:
        print("\n" + "=" * 70 + "\n🎬 PHASE C1: PER-CHUNK RESUMABLE VIDEO DECODE\n" + "=" * 70)
        nuclear_purge()
        import comfy.model_management as mm
        import imageio
        _prev_vram_state = getattr(mm, "vram_state", None)
        if hasattr(mm, "VRAMState") and not globals().get("FULL_QUALITY_HW", False):
            mm.vram_state = getattr(mm.VRAMState, "LOW_VRAM", mm.vram_state)
            print("  ⚙️ Forced LOW_VRAM mode to guarantee strict VAE tiling for 3D T4 memory safety.")

        v_full = unwrap_latent({"samples": pack["video"]})["samples"]
        T = int(v_full.shape[2]) if (v_full is not None and v_full.dim() >= 3) else 1
        chunk = max(2, int(globals().get("DECODE_CHUNK_LAT_FRAMES", 4)))
        ov = max(0, int(globals().get("DECODE_CHUNK_OVERLAP", 1)))
        chunk_ranges, start = [], 0
        while start < T:
            chunk_ranges.append((start, min(start + chunk, T)))
            start += chunk

        video_vae = None
        for ci, (s, e) in enumerate(chunk_ranges):
            seg_mp4 = os.path.join(chunk_dir, f"vchunk_{ci:03d}.mp4")
            if resume and os.path.exists(seg_mp4) and os.path.getsize(seg_mp4) > 512:
                continue
            if video_vae is None:
                video_vae = gv(call_node("VAELoader", vae_name="ltx-2.5-video-vae-bf16.safetensors"), 0)
            with torch.inference_mode():
                ctx = max(0, s - ov) if s > 0 else 0
                sub = v_full[:, :, ctx:e].float()
                frames_tensor = unwrap_tensor(tiled_decode_video({"samples": sub}, video_vae, tile_size=128)).clamp(0, 1)
                frames_cpu = frames_tensor.detach().cpu().numpy()
                del frames_tensor, sub
                drop_px = (s - ctx) * 8 if s > 0 else 0
                fslice = frames_cpu[drop_px:] if drop_px > 0 else frames_cpu
                arr = (fslice * 255.0).astype(np.uint8)
            tmp_seg = seg_mp4 + ".tmp.mp4"
            w = imageio.get_writer(tmp_seg, fps=int(fps), codec="libx264", format="FFMPEG",
                                   macro_block_size=None,
                                   ffmpeg_params=["-crf", str(int(crf)), "-pix_fmt", "yuv420p"])
            for i in range(arr.shape[0]):
                w.append_data(arr[i])
            w.close()
            os.replace(tmp_seg, seg_mp4)
            print(f"  🎨 chunk {ci+1}/{len(chunk_ranges)} (latent {s}:{e}) → {arr.shape[0]} frames")
            del fslice, arr, frames_cpu
            light_clear()

        if video_vae is not None:
            del video_vae
        del v_full
        if _prev_vram_state is not None:
            mm.vram_state = _prev_vram_state
        nuclear_purge()

        seglist = sorted(glob.glob(os.path.join(chunk_dir, "vchunk_*.mp4")))
        listfile = os.path.join(chunk_dir, "concat_list.txt")
        with open(listfile, "w") as fh:
            for sp in seglist:
                fh.write(f"file '{sp}'\n")
        run_cmd(f'ffmpeg -y -f concat -safe 0 -i "{listfile}" -c copy "{raw_video}"', silent=False)
        if not (os.path.exists(raw_video) and os.path.getsize(raw_video) > 1024):
            run_cmd(f'ffmpeg -y -f concat -safe 0 -i "{listfile}" -c:v libx264 -crf {int(crf)} '
                    f'-pix_fmt yuv420p "{raw_video}"', silent=False)
    else:
        print(f"  ⏭ [RESUME] Raw video cached.")

    if need_audio:
        print("\n" + "=" * 70 + "\n🎬 PHASE C2: AUDIO VAE DECODE\n" + "=" * 70)
        nuclear_purge()
        with torch.inference_mode():
            a_lat = pack["audio"] if pack is not None else None
            if a_lat is not None:
                audio_vae = gv(call_node("VAELoader", vae_name="ltx-2.5-audio-vae-bf16.safetensors"), 0)
                decoded_audio = gv(call_node("LTXVAudioVAEDecode",
                                             samples={"samples": a_lat.float()}, audio_vae=audio_vae), 0)
                if isinstance(decoded_audio, dict) and "waveform" in decoded_audio:
                    decoded_audio["waveform"] = decoded_audio["waveform"].detach().cpu()
                torch.save(decoded_audio, audio_file + ".tmp")
                os.replace(audio_file + ".tmp", audio_file)
                del audio_vae, decoded_audio
            else:
                torch.save(None, audio_file)
        nuclear_purge()
    del pack
    gc.collect()
    malloc_trim_os()
    return raw_video, audio_file

print("✅ Cell 14: Phase C ready.")

# ════════════════════════════════════════════════════════════════════════════════
# CELL 15: PHASE D — FINAL AUDIO MUX
# ════════════════════════════════════════════════════════════════════════════════
def execute_phase_d(raw_video: str, audio_file: str, fps: int, crf: int,
                    outdir: str, song_path: str, trim_start_frames: float) -> str:
    os.makedirs(outdir, exist_ok=True)
    final_path = os.path.join(outdir, "LTX25_Director_Master.mp4")
    print("\n" + "=" * 70 + "\n🎬 PHASE D: FINAL AUDIO MUX\n" + "=" * 70)
    nuclear_purge()
    if not (os.path.exists(raw_video) and os.path.getsize(raw_video) > 1024):
        raise RuntimeError(f"Raw video missing: {raw_video}")
    audio_dict = torch.load(audio_file, map_location="cpu") if os.path.exists(audio_file) else None
    muxed = False
    wav_path = os.path.join(outdir, "_synced_audio.wav")

    def _mux_model_vocals() -> bool:
        if audio_dict is None or not _save_audio_wav(audio_dict, wav_path):
            return False
        print("  🎵 Muxing model-generated synced vocals...")
        cmd = (f'ffmpeg -y -i "{raw_video}" -i "{wav_path}" -map 0:v:0 -map 1:a:0 '
               f'-c:v copy -c:a aac -b:a 320k -shortest "{final_path}"')
        ok = run_cmd(cmd, silent=False) == 0 and os.path.exists(final_path)
        try: os.remove(wav_path)
        except: pass
        return ok

    def _mux_original_song() -> bool:
        if not (song_path and os.path.exists(song_path)):
            return False
        print("  🎵 Muxing original song track...")
        trim_sec = float(trim_start_frames) / float(fps)
        cmd = (f'ffmpeg -y -i "{raw_video}" -ss {trim_sec} -i "{song_path}" '
               f'-map 0:v:0 -map 1:a:0 -c:v copy -c:a aac -b:a 320k -shortest "{final_path}"')
        return run_cmd(cmd, silent=False) == 0 and os.path.exists(final_path)

    for _mux_fn in (_mux_model_vocals, _mux_original_song):
        if not muxed:
            muxed = _mux_fn()
    if not muxed:
        shutil.copyfile(raw_video, final_path)
        print("  ⚠️ No audio muxed; published video-only.")
    del audio_dict
    nuclear_purge()
    print(f"  🎉 Master MP4: {final_path}")
    return final_path

print("✅ Cell 15: Phase D ready.")

# ════════════════════════════════════════════════════════════════════════════════
# CELL 16: OUTPUT VERIFICATION
# ════════════════════════════════════════════════════════════════════════════════
def verify_output(video_path: str):
    print("\n" + "=" * 70 + "\n🔍 FINAL ARTIFACT VERIFICATION\n" + "=" * 70)
    if not os.path.exists(video_path) or os.path.getsize(video_path) < 1000:
        raise RuntimeError(f"Output '{video_path}' is missing or empty.")
    vprobe = (f'ffprobe -v error -select_streams v:0 -count_packets '
              f'-show_entries stream=nb_read_packets,r_frame_rate,duration '
              f'-of csv=p=0 "{video_path}"')
    aprobe = (f'ffprobe -v error -select_streams a:0 '
              f'-show_entries stream=codec_name,duration -of csv=p=0 "{video_path}"')
    vout = subprocess.run(vprobe, shell=True, capture_output=True, text=True).stdout.strip()
    aout = subprocess.run(aprobe, shell=True, capture_output=True, text=True).stdout.strip()
    print(f"  ✓ Path        : {video_path}")
    print(f"  ✓ Size        : {os.path.getsize(video_path)/(1024*1024):.2f} MB")
    print(f"  ✓ Video stream: {vout}")
    print(f"  ✓ Audio stream: {aout if aout else '(none)'}")
    print("=" * 70)
    print("✅ Cell 16: Verifier ready.")

print("✅ Cell 16: Verifier ready.")

# ════════════════════════════════════════════════════════════════════════════════
# CELL 17: RUNTIME CONFIG & MASTER ONE-CLICK GENERATION
# ════════════════════════════════════════════════════════════════════════════════
WORK_DIRECTORY = os.path.join(CONTENT_ROOT, "LTXDirector_Work")
OUTPUT_DIRECTORY = os.path.join(CONTENT_ROOT, "LTXStudio_Output")
SONG_PATH = os.path.join(WHATDREAMS_INPUT, "Late night trap.mp3")
MAX_OOM_RETRIES = 3

def validate_config(meta: Dict[str, Any], segments: List[Dict[str, Any]],
                    use_song_audio: bool, song_path: str) -> bool:
    print("\n🔍 CONFIG Validation...")
    problems = []
    db = int(meta.get("divisible_by", 32))
    for k in ("generation_width", "generation_height", "custom_width", "custom_height"):
        v = int(meta.get(k, 0))
        if v <= 0: problems.append(f"{k} is invalid ({v})")
        elif v % db: problems.append(f"{k}={v} is not divisible by {db}")
    if int(meta.get("normalDurationFrames", 0)) < 9:
        problems.append("Timeline too short (frames < 9)")
    if float(meta.get("frame_rate", 0)) <= 0:
        problems.append("Invalid frame_rate")
    if not segments:
        problems.append("No keyframe segments")
    if use_song_audio and song_path and not os.path.exists(song_path):
        log(f"Song file missing: {song_path} — no voice/lip-sync.", "WARN")
    if problems:
        for p in problems: log(f"   • {p}", "WARN")
        return False
    return True

def _scale_meta_resolution(meta: Dict[str, Any], scale: float) -> Dict[str, Any]:
    db = max(1, int(meta.get("divisible_by", 32)))
    def _snap(n): return max(db, int(round(float(n) * scale / db)) * db)
    m = dict(meta)
    for k in ("generation_width", "generation_height", "custom_width", "custom_height",
              "base_stage1_width", "base_stage1_height"):
        if m.get(k): m[k] = _snap(m[k])
    return m

def config_signature(meta: Dict[str, Any]) -> str:
    pipeline_rev = "v5.2-ltx25-nuclear-fix"
    return (f"rev={pipeline_rev}"
            f"|render{meta.get('custom_width')}x{meta.get('custom_height')}"
            f"|gen{meta.get('generation_width')}x{meta.get('generation_height')}"
            f"|frames{meta.get('normalDurationFrames')}|fps{meta.get('frame_rate')}"
            f"|loras{len(globals().get('LORA_STACK', []))}"
            f"|s1_{STAGE1['steps']}x{STAGE1['denoise']}|s2_{STAGE2['steps']}x{STAGE2['denoise']}")

def guard_stale_cache(workdir: str, meta: Dict[str, Any]):
    os.makedirs(workdir, exist_ok=True)
    sig = _config_signature(meta) if '_config_signature' in globals() else config_signature(meta)
    sig_file = os.path.join(workdir, "cache_sig.txt")
    existing = glob.glob(os.path.join(workdir, "*.pt"))
    old = None
    if os.path.exists(sig_file):
        try: old = open(sig_file).read().strip()
        except: pass
    stale = (old is None and len(existing) > 0) or (old is not None and old != sig)
    if stale:
        for f in existing:
            try: os.remove(f)
            except: pass
        try:
            with open(sig_file, "w") as fh: fh.write(sig)
        except: pass

def run_ltx25_director_master(global_prompt, negative_prompt, meta, segments,
                              audio_segments, motion_segments,
                              seed=0, crf=8, workdir=WORK_DIRECTORY,
                              outdir=OUTPUT_DIRECTORY, resume=True,
                              use_song_audio=True) -> str:
    t_start = time.time()
    print("\n" + "=" * 70 + "\n🎬 LTX-2.5 / LTX-5 DIRECTOR MASTER\n" + "=" * 70)
    validate_original_nodes()
    patch_comfy_memory_manager()
    patch_safetensors_direct_to_gpu()
    configure_vram_state(mode=globals().get("VRAM_MODE", "auto"))
    install_sampling_memory_hook(clear_every=4, ram_guard_gb=globals().get("min_ram_guard_gb", 1.5))
    os.makedirs(workdir, exist_ok=True)
    validate_config(meta, segments, use_song_audio, SONG_PATH if use_song_audio else "")

    base_meta = dict(meta)
    active_meta = dict(meta)
    latent_file = os.path.join(workdir, "final_latents.pt")
    attempt = 0
    while True:
        guard_stale_cache(workdir, active_meta)
        ctrl = DirectorTimelineController(
            global_prompt=global_prompt, negative_prompt=negative_prompt,
            meta=active_meta, segments=segments,
            audio_segments=audio_segments, motion_segments=motion_segments)
        if resume and os.path.exists(latent_file) and os.path.getsize(latent_file) > 1024:
            break
        try:
            director_state, patched_model = execute_phase_a(ctrl, workdir=workdir, resume=resume)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" in str(e).lower():
                raise RuntimeError("CUDA OOM during Phase A (Text Encode). Ensure Gemma memory patch logic fired.") from e
            raise
        try:
            latent_file = execute_phase_b(director_state, patched_model, seed=seed,
                                          workdir=workdir, resume=resume)
            del director_state, patched_model
            nuclear_purge()
            break
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" not in str(e).lower(): raise
            attempt += 1
            nuclear_purge()
            if attempt > MAX_OOM_RETRIES:
                raise RuntimeError("CUDA OOM limit reached on T4.") from e
            active_meta = _scale_meta_resolution(base_meta, 0.8 ** attempt)
            log(f"OOM #{attempt}: Retrying at {active_meta['generation_width']}x{active_meta['generation_height']}", "WARN")

    raw_video, audio_file = execute_phase_c(
        latent_file, workdir=workdir,
        fps=int(active_meta["frame_rate"]), crf=crf, resume=resume)
    final_video = execute_phase_d(
        raw_video, audio_file,
        fps=int(active_meta["frame_rate"]), crf=crf, outdir=outdir,
        song_path=SONG_PATH if use_song_audio else "",
        trim_start_frames=active_meta["audio_trim_start_frames"])
    verify_output(final_video)
    elapsed = time.time() - t_start
    print("\n" + "=" * 70)
    print("🎉 GENERATION COMPLETE")
    print(f"  Duration : {active_meta['duration_seconds']:.2f}s "
          f"({active_meta['normalDurationFrames']} frames @ {active_meta['frame_rate']} fps)")
    print(f"  Time     : {elapsed/60:.2f} min")
    print(f"  Output   : {final_video}")
    print(f"  RAM free : {get_ram_free_gb():.2f} GB")
    print("=" * 70 + "\n")
    return final_video

_base_input = WHATDREAMS_INPUT
os.makedirs(_base_input, exist_ok=True)
if os.path.exists(f"{_base_input}/5.png") and not os.path.exists(f"{_base_input}/5.3.png"):
    shutil.copy(f"{_base_input}/5.png", f"{_base_input}/5.3.png")

# ════════════════════════════════════════════════════════════════════════════════
# CELL 18: QUALITY SELF-CHECK
# ════════════════════════════════════════════════════════════════════════════════
def quality_self_check(video_path: str, outdir: str = "/content/LTXStudio_Output",
                       n_thumbs: int = 6) -> Dict[str, Any]:
    print("\n" + "=" * 70 + "\n🔎 QUALITY SELF-CHECK\n" + "=" * 70)
    report: Dict[str, Any] = {"video": video_path}
    if not os.path.exists(video_path) or os.path.getsize(video_path) < 1000:
        return report

    def _probe(stream, entries):
        cmd = (f'ffprobe -v error -select_streams {stream} '
               f'-show_entries {entries} -of default=nw=1 "{video_path}"')
        return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()

    vinfo = _probe("v:0", "stream=width,height,r_frame_rate,nb_read_packets,duration")
    ainfo = _probe("a:0", "stream=codec_name,duration,sample_rate,channels")
    print("  🎞️  Video:", vinfo.replace("\n", " ") or "(none)")
    print("  🔊 Audio:", ainfo.replace("\n", " ") or "(none)")

    def _get(info, key):
        for line in info.splitlines():
            if line.startswith(key + "="):
                try: return float(line.split("=", 1)[1])
                except: pass
        return None

    vd, ad = _get(vinfo, "duration"), _get(ainfo, "duration")
    if vd and ad:
        drift = abs(vd - ad)
        print(f"  ⏱️  Durations: video {vd:.2f}s | audio {ad:.2f}s | Δ {drift:.2f}s")

    thumb_dir = os.path.join(outdir, "thumbnails")
    os.makedirs(thumb_dir, exist_ok=True)
    for f in glob.glob(os.path.join(thumb_dir, "*.jpg")):
        try: os.remove(f)
        except: pass

    dur = vd or 30.0
    thumbs = []
    for i in range(n_thumbs):
        t = max(0.1, dur * (i + 0.5) / n_thumbs)
        out = os.path.join(thumb_dir, f"thumb_{i:02d}.jpg")
        run_cmd(f'ffmpeg -y -ss {t:.2f} -i "{video_path}" -frames:v 1 -q:v 2 "{out}"')
        if os.path.exists(out): thumbs.append(out)
    report["thumbnails"] = thumbs

    try:
        from PIL import Image as _Im
        means, prev, frozen = [], None, 0
        for tpath in thumbs:
            a = np.asarray(_Im.open(tpath).convert("L"), dtype=np.float32)
            means.append(a.mean())
            if prev is not None and np.abs(a - prev).mean() < 2.0: frozen += 1
            prev = a
        if means:
            print(f"  💡 Brightness: min {min(means):.0f} / max {max(means):.0f}")
    except: pass

    try:
        from IPython.display import Image as _IPyImage, display
        for t in thumbs: display(_IPyImage(filename=t, width=320))
    except: pass

    print("=" * 70)
    return report

# ════════════════════════════════════════════════════════════════════════════════
# CELL 19: SUBGRAPH MODE PIPELINE (Modes 01-08)
# ════════════════════════════════════════════════════════════════════════════════
class SubgraphModePipeline:
    """
    Executes the All-In-One workflow subgraphs for modes 01-08.
    Each mode maps to a specific subgraph from LTX_2.5_All-In-One_Workflow.json.
    """
    def __init__(self, mode: str, config: Dict[str, Any]):
        self.mode = mode
        self.config = config
        self.workdir = config.get("workdir", WORK_DIRECTORY)
        self.outdir = config.get("outdir", OUTPUT_DIRECTORY)
        os.makedirs(self.workdir, exist_ok=True)
        os.makedirs(self.outdir, exist_ok=True)
        self._parse_sigmas()

    def _parse_sigmas(self):
        try:
            self.s1_sigmas = [float(x.strip()) for x in self.config.get("stage1_sigmas",
                "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0").split(",")]
        except Exception:
            self.s1_sigmas = [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0]
        try:
            self.s2_sigmas = [float(x.strip()) for x in self.config.get("stage2_sigmas",
                "0.85, 0.7250, 0.4219, 0.0").split(",")]
        except Exception:
            self.s2_sigmas = [0.85, 0.7250, 0.4219, 0.0]

    def _load_models(self, need_ic_lora: bool = False):
        """Load DiT GGUF + base LoRAs. Optionally load IC-LoRA for lipsync modes."""
        nuclear_purge()
        mem_report("Subgraph Load", "UnetLoaderGGUF")
        _gguf = globals().get("DIT_GGUF_NAME", "LTX-2.5-Distilled-Q4_K_M.gguf")
        model = gv(call_node("UnetLoaderGGUF", unet_name=_gguf), 0)
        print(f"  ✓ UnetLoaderGGUF loaded ({_gguf}).")

        # Apply Director LoRA stack (if any enabled)
        if LORA_STACK:
            try:
                lora_kwargs = {"model": model, "clip": None}
                for i, lc in enumerate(LORA_STACK, start=1):
                    lora_kwargs[f"lora_{i}"] = {"on": lc["on"], "lora": lc["lora"],
                                                "strength": lc["strength"], "strengthTwo": None}
                res = call_node("Power Lora Loader (rgthree)", **lora_kwargs)
                new_model = gv(res, 0)
                if new_model is not None: model = new_model
                print(f"  ✓ Power Lora Loader: {len(LORA_STACK)}-LoRA stack applied.")
            except Exception as e:
                print(f"  [notice] Power Lora fallback: {e}")

        # IC-LoRA for lipsync modes (05/06/08)
        ic_lora_model = None
        ic_downscale_factor = 1.0
        if need_ic_lora and self.config.get("use_ic_lora", True):
            try:
                ic_out = call_node("LTXICLoRALoaderModelOnly",
                                   model=model,
                                   lora_name="ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors",
                                   strength=float(self.config.get("ic_lora_strength", 1.0)))
                ic_lora_model = gv(ic_out, 0) or model
                ic_downscale_factor = float(gv(ic_out, 1) or 1.0)
                print(f"  ✓ IC-LoRA loaded (strength={self.config.get('ic_lora_strength', 1.0)}, df={ic_downscale_factor:.2f})")
            except Exception as e:
                print(f"  [notice] IC-LoRA load failed: {e}")
                ic_lora_model = model

        medium_clear("post_subgraph_load")
        return model, ic_lora_model, ic_downscale_factor

    def _encode_text(self, positive: str, negative: str):
        """Encode text prompts using the fused Gemma-12B encoder, then purge CLIP."""
        nuclear_purge()
        import comfy.model_management as mm
        _prev_state = getattr(mm, "vram_state", None)
        if hasattr(mm, "VRAMState") and not globals().get("FULL_QUALITY_HW", False):
            mm.vram_state = getattr(mm.VRAMState, "NO_VRAM", mm.vram_state)

        clip = gv(call_node("CLIPLoader",
                            clip_name="gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors",
                            type="ltxv"), 0)
        print("  ⚡ Encoding prompts on GPU (LTX-2.5 fused encoder)...")
        t0 = time.time()
        with torch.inference_mode():
            pos_raw = gv(call_node("CLIPTextEncode", text=positive, clip=clip), 0)
            neg_raw = gv(call_node("CLIPTextEncode", text=negative, clip=clip), 0)
            pos_cpu = sync_cond_to_cpu(pos_raw)
            neg_cpu = sync_cond_to_cpu(neg_raw)
            del pos_raw, neg_raw
        print(f"  ✓ Prompts encoded in {time.time() - t0:.2f}s. Purging Gemma4-12B...")
        if _prev_state is not None:
            mm.vram_state = _prev_state
        try:
            clip.cond_stage_model = None
            clip.patcher = None
        except Exception: pass
        del clip
        nuclear_purge()
        mem_report("Subgraph", "Gemma purged")
        return pos_cpu, neg_cpu

    def _build_empty_latents(self, width: int, height: int, length: int, fps: int,
                              video_vae, audio_vae):
        """Create empty video + audio latents for text-to-video / audio-to-video modes."""
        base_w = max(32, (width // 2) // 32 * 32)
        base_h = max(32, (height // 2) // 32 * 32)

        empty_img = gv(call_node("EmptyImage", width=width, height=height, batch_size=1, color=0), 0)
        scaled_img = gv(call_node("ImageScaleBy", image=empty_img, upscale_method="lanczos", scale_by=0.5), 0)
        img_size = call_node("GetImageSize", image=scaled_img)
        w_base = int(gv(img_size, 0) or base_w)
        h_base = int(gv(img_size, 1) or base_h)

        vid_lat = gv(call_node("EmptyLTXVLatentVideo",
                               width=w_base, height=h_base, length=length, batch_size=1), 0)
        aud_lat = gv(call_node("LTXVEmptyLatentAudio",
                               audio_vae=audio_vae, frames_number=length, frame_rate=fps, batch_size=1), 0)
        return vid_lat, aud_lat

    def _apply_audio_mask(self, audio_input, audio_vae, vid_lat, width: int, height: int):
        """Encode audio and apply as noise mask to video latent (for audio-to-video mode)."""
        audio_encoded = gv(call_node("LTXVAudioVAEEncode",
                                     audio=audio_input, audio_vae=audio_vae), 0)
        mask = gv(call_node("SolidMask", width=width, height=height, value=0.01), 0)
        masked_audio = gv(call_node("SetLatentNoiseMask",
                                    samples=audio_encoded, mask=mask), 0)
        return masked_audio

    def _apply_image_to_video(self, image_path: str, vid_lat, video_vae,
                               strength: float, width: int, height: int):
        """Apply image conditioning for image-to-video modes (Mode 03)."""
        img_tensor = prepare_reference_image(image_path, width, height)
        preprocessed = gv(call_node("LTXVPreprocess",
                                    image=img_tensor,
                                    img_compression=int(self.config.get("img_compression", 18))), 0)
        conditioned = gv(call_node("LTXVImgToVideoInplace",
                                   vae=video_vae, image=preprocessed,
                                   latent=vid_lat, strength=float(strength)), 0)
        return conditioned

    def _apply_first_last_frame(self, image_1_path: str, image_2_path: str,
                                 vid_lat, video_vae, width: int, height: int):
        """Apply first/last frame conditioning (Mode 04)."""
        img1 = prepare_reference_image(image_1_path, width, height)
        img2 = prepare_reference_image(image_2_path, width, height)
        img1_pre = gv(call_node("LTXVPreprocess", image=img1,
                                img_compression=int(self.config.get("img_compression", 18))), 0)
        img2_pre = gv(call_node("LTXVPreprocess", image=img2,
                                img_compression=int(self.config.get("img_compression", 18))), 0)
        conditioned = gv(call_node("LTXVImgToVideoInplaceKJ",
                                   vae=video_vae, latent=vid_lat,
                                   num_images_image_1=img1_pre,
                                   num_images_image_2=img2_pre,
                                   num_images_mode="2"), 0)
        return conditioned

    def _apply_first_middle_last_frame(self, image_1_path: str, image_2_path: str,
                                        image_3_path: str, vid_lat, video_vae,
                                        width: int, height: int, length: int):
        """Apply first/middle/last frame conditioning (Mode 07)."""
        img1 = prepare_reference_image(image_1_path, width, height)
        img2 = prepare_reference_image(image_2_path, width, height)
        img3 = prepare_reference_image(image_3_path, width, height)
        img1_pre = gv(call_node("LTXVPreprocess", image=img1,
                                img_compression=int(self.config.get("img_compression", 18))), 0)
        img2_pre = gv(call_node("LTXVPreprocess", image=img2,
                                img_compression=int(self.config.get("img_compression", 18))), 0)
        img3_pre = gv(call_node("LTXVPreprocess", image=img3,
                                img_compression=int(self.config.get("img_compression", 18))), 0)
        middle_idx = max(1, int(math.ceil(length / 2)))
        conditioned = gv(call_node("LTXVImgToVideoInplaceKJ",
                                   vae=video_vae, latent=vid_lat,
                                   num_images_image_1=img1_pre,
                                   num_images_image_2=img2_pre,
                                   num_images_image_3=img3_pre,
                                   num_images_mode="3",
                                   num_images_index_2=middle_idx), 0)
        return conditioned

    def _apply_ic_lora_guide(self, model, pos, neg, video_vae, vid_lat,
                              image_path: str, downscale_factor: float, width: int, height: int):
        """Apply IC-LoRA guide for lipsync modes (Mode 05)."""
        img_tensor = prepare_reference_image(image_path, width, height)
        preprocessed = gv(call_node("LTXVPreprocess", image=img_tensor,
                                    img_compression=int(self.config.get("img_compression", 18))), 0)
        conditioned_lat = gv(call_node("LTXVImgToVideoConditionOnly",
                                       vae=video_vae, image=preprocessed,
                                       latent=vid_lat, strength=1.0), 0)
        guide_out = call_node("LTXAddVideoICLoRAGuide",
                              positive=pos, negative=neg,
                              vae=video_vae, latent=conditioned_lat,
                              image=preprocessed,
                              latent_downscale_factor=float(downscale_factor))
        g_pos = sync_cond_to_cpu(gv(guide_out, 0) or pos)
        g_neg = sync_cond_to_cpu(gv(guide_out, 1) or neg)
        g_lat = sync_latent_device(gv(guide_out, 2) or conditioned_lat, "cpu")
        return g_pos, g_neg, g_lat

    def _sample_stage(self, model, pos, neg, vid_lat, aud_lat, sigmas,
                       sampler_name: str, seed: int, stage_tag: str,
                       use_dual_cfg: bool = True):
        """Run one sampling stage using ManualSigmas."""
        av_lat = sync_latent_device(gv(call_node("LTXVConcatAVLatent",
                                                  video_latent=vid_lat,
                                                  audio_latent=aud_lat), 0), "cpu")
        light_clear()

        if use_dual_cfg and "LTXVDualCFGGuider" in NODE_CLASS_MAPPINGS:
            guider = gv(call_node("LTXVDualCFGGuider",
                                  model=model, positive=pos, negative=neg,
                                  cfg_1=float(self.config.get("cfg_value", 1.0)),
                                  cfg_2=float(self.config.get("cfg_value", 1.0))), 0)
        else:
            guider = gv(call_node("CFGGuider",
                                  model=model, positive=pos, negative=neg,
                                  cfg=float(self.config.get("cfg_value", 1.0))), 0)
        guider = sync_cond_to_cpu(guider) if not isinstance(guider, (list, tuple, dict)) else guider

        sampler = gv(call_node("KSamplerSelect", sampler_name=sampler_name), 0)
        sigmas_str = ", ".join(f"{s:.6f}" for s in sigmas)
        manual_sigmas = gv(call_node("ManualSigmas", sigmas=sigmas_str), 0)
        noise = gv(call_node("RandomNoise", noise_seed=seed), 0)

        print(f"  ⚡ {stage_tag}: sampling {len(sigmas)-1} steps (sigmas={sigmas_str[:40]}...)")
        t0 = time.time()
        ram_guard(globals().get("min_ram_guard_gb", 1.5), stage_tag)
        out = call_node("SamplerCustomAdvanced",
                        noise=noise, guider=guider, sampler=sampler,
                        sigmas=manual_sigmas, latent_image=av_lat)
        sampled = sync_latent_device(gv(out, 0), "cpu")
        print(f"  ✓ {stage_tag} done in {time.time() - t0:.2f}s.")

        sep = call_node("LTXVSeparateAVLatent", av_latent=sampled)
        v_lat = sync_latent_device(gv(sep, 0), "cpu")
        a_lat = sync_latent_device(gv(sep, 1), "cpu")

        if "easy cleanGpuUsed" in NODE_CLASS_MAPPINGS:
            try:
                _cleaned = gv(call_node("easy cleanGpuUsed", anything=sampled), 0)
                if _cleaned is not None:
                    v_lat = sync_latent_device(_cleaned, "cpu") if isinstance(_cleaned, dict) else v_lat
            except Exception:
                pass

        del av_lat, sampled, sep, noise, guider, sampler, manual_sigmas, out
        light_clear()
        return v_lat, a_lat

    def _apply_detailer_lora(self, model):
        """Apply detailer LoRA for Stage 2 refinement."""
        if not self.config.get("use_detailer_lora", True):
            return model
        try:
            detailer_out = call_node("LoraLoaderModelOnly",
                                     model=model,
                                     lora_name="ltx-2-19b-ic-lora-detailer.safetensors",
                                     strength_model=float(self.config.get("detailer_lora_strength", 0.3)))
            new_model = gv(detailer_out, 0) or model
            print(f"  ✓ Detailer LoRA applied (strength={self.config.get('detailer_lora_strength', 0.3)})")
            return new_model
        except Exception as e:
            print(f"  [notice] Detailer LoRA failed: {e}")
            return model

    def _upscale_latent(self, vid_lat, video_vae, upscale_model):
        """Spatial latent upscale (2x) between stages."""
        up = gv(call_node("LTXVLatentUpsampler",
                          samples=vid_lat, upscale_model=upscale_model, vae=video_vae), 0)
        return sync_latent_device(up, "cpu")

    def _decode_video(self, vid_lat, video_vae, fps: int, crf: int, out_path: str):
        """Decode video latent using tiled VAE decode + chunked streaming."""
        import imageio
        nuclear_purge()
        import comfy.model_management as mm
        _prev_vram_state = getattr(mm, "vram_state", None)
        if hasattr(mm, "VRAMState") and not globals().get("FULL_QUALITY_HW", False):
            mm.vram_state = getattr(mm.VRAMState, "LOW_VRAM", mm.vram_state)

        v_full = unwrap_latent(vid_lat)["samples"]
        T = int(v_full.shape[2]) if (v_full is not None and v_full.dim() >= 3) else 1
        chunk = max(2, int(globals().get("DECODE_CHUNK_LAT_FRAMES", 4)))
        ov = max(0, int(globals().get("DECODE_CHUNK_OVERLAP", 1)))
        chunk_ranges, start = [], 0
        while start < T:
            chunk_ranges.append((start, min(start + chunk, T)))
            start += chunk

        chunk_dir = os.path.join(os.path.dirname(out_path), "subgraph_chunks")
        os.makedirs(chunk_dir, exist_ok=True)
        video_vae_loader = None

        for ci, (s, e) in enumerate(chunk_ranges):
            seg_mp4 = os.path.join(chunk_dir, f"vchunk_{ci:03d}.mp4")
            if os.path.exists(seg_mp4) and os.path.getsize(seg_mp4) > 512:
                continue
            if video_vae_loader is None:
                video_vae_loader = gv(call_node("VAELoader",
                                                vae_name="ltx-2.5-video-vae-bf16.safetensors"), 0)
            with torch.inference_mode():
                ctx = max(0, s - ov) if s > 0 else 0
                sub = v_full[:, :, ctx:e].float()
                frames_tensor = unwrap_tensor(tiled_decode_video(
                    {"samples": sub}, video_vae_loader, tile_size=128)).clamp(0, 1)
                frames_cpu = frames_tensor.detach().cpu().numpy()
                del frames_tensor, sub
                drop_px = (s - ctx) * 8 if s > 0 else 0
                fslice = frames_cpu[drop_px:] if drop_px > 0 else frames_cpu
                arr = (fslice * 255.0).astype(np.uint8)
            tmp_seg = seg_mp4 + ".tmp.mp4"
            w = imageio.get_writer(tmp_seg, fps=int(fps), codec="libx264", format="FFMPEG",
                                   macro_block_size=None,
                                   ffmpeg_params=["-crf", str(int(crf)), "-pix_fmt", "yuv420p"])
            for i in range(arr.shape[0]):
                w.append_data(arr[i])
            w.close()
            os.replace(tmp_seg, seg_mp4)
            print(f"  🎨 chunk {ci+1}/{len(chunk_ranges)} (latent {s}:{e}) → {arr.shape[0]} frames")
            del fslice, arr, frames_cpu
            light_clear()

        if video_vae_loader is not None:
            del video_vae_loader
        del v_full
        if _prev_vram_state is not None:
            mm.vram_state = _prev_vram_state
        nuclear_purge()

        seglist = sorted(glob.glob(os.path.join(chunk_dir, "vchunk_*.mp4")))
        listfile = os.path.join(chunk_dir, "concat_list.txt")
        with open(listfile, "w") as fh:
            for sp in seglist:
                fh.write(f"file '{sp}'\n")
        run_cmd(f'ffmpeg -y -f concat -safe 0 -i "{listfile}" -c copy "{out_path}"', silent=False)
        if not (os.path.exists(out_path) and os.path.getsize(out_path) > 1024):
            run_cmd(f'ffmpeg -y -f concat -safe 0 -i "{listfile}" -c:v libx264 -crf {int(crf)} '
                    f'-pix_fmt yuv420p "{out_path}"', silent=False)

    def _decode_audio(self, aud_lat, audio_vae, out_path: str):
        """Decode audio latent."""
        nuclear_purge()
        with torch.inference_mode():
            decoded = gv(call_node("LTXVAudioVAEDecode",
                                   samples={"samples": unwrap_tensor(aud_lat).float()},
                                   audio_vae=audio_vae), 0)
            if isinstance(decoded, dict) and "waveform" in decoded:
                decoded["waveform"] = decoded["waveform"].detach().cpu()
            torch.save(decoded, out_path + ".tmp")
            os.replace(out_path + ".tmp", out_path)
        nuclear_purge()

    def _mux_final(self, video_path: str, audio_pt_path: str, fps: int, crf: int,
                    song_path: str, final_path: str):
        """Mux video + audio into final MP4."""
        nuclear_purge()
        audio_dict = torch.load(audio_pt_path, map_location="cpu") if os.path.exists(audio_pt_path) else None
        muxed = False
        wav_path = final_path + ".wav"

        def _mux_model_audio() -> bool:
            if audio_dict is None or not _save_audio_wav(audio_dict, wav_path):
                return False
            cmd = (f'ffmpeg -y -i "{video_path}" -i "{wav_path}" -map 0:v:0 -map 1:a:0 '
                   f'-c:v copy -c:a aac -b:a 320k -shortest "{final_path}"')
            ok = run_cmd(cmd, silent=False) == 0 and os.path.exists(final_path)
            try: os.remove(wav_path)
            except: pass
            return ok

        def _mux_song() -> bool:
            if not (song_path and os.path.exists(song_path)):
                return False
            cmd = (f'ffmpeg -y -i "{video_path}" -i "{song_path}" '
                   f'-map 0:v:0 -map 1:a:0 -c:v copy -c:a aac -b:a 320k -shortest "{final_path}"')
            return run_cmd(cmd, silent=False) == 0 and os.path.exists(final_path)

        for _mux_fn in (_mux_model_audio, _mux_song):
            if not muxed:
                muxed = _mux_fn()
        if not muxed:
            shutil.copyfile(video_path, final_path)
            print("  ⚠️ No audio muxed; published video-only.")
        del audio_dict
        nuclear_purge()

    def run(self) -> str:
        """Main orchestrator: dispatches to the correct subgraph pipeline."""
        t_start = time.time()
        mode = self.mode
        print(f"\n{'='*70}\n🎬 SUBGRAPH MODE: {mode}\n{'='*70}")

        width = int(self.config.get("output_width", 1280))
        height = int(self.config.get("output_height", 736))
        fps = int(self.config.get("fps", 24))
        length_sec = int(self.config.get("audio_clip_length_seconds", 10))
        length_frames = length_sec * fps
        seed = int(self.config.get("base_seed", 0))
        crf = int(self.config.get("output_crf", 8))
        sampler = self.config.get("sampler_name_subgraph", "euler_ancestral")
        use_dual = bool(self.config.get("use_dual_cfg", True))
        positive = self.config.get("global_prompt", GLOBAL_PROMPT)
        negative = self.config.get("negative_prompt", NEGATIVE_PROMPT)
        song_path = self.config.get("audio_input_path", SONG_PATH)

        # T4 safety caps
        if not globals().get("FULL_QUALITY_HW", False):
            if width > 1280: width = 1280
            if height > 736: height = 736
            if length_frames > 241: length_frames = 241
            length_frames = ((length_frames - 1) // 8) * 8 + 1

        width = max(32, (width // 32) * 32)
        height = max(32, (height // 32) * 32)

        print(f"  📐 Resolution: {width}x{height} @ {fps}fps, {length_frames} frames ({length_frames/fps:.2f}s)")

        need_ic = mode in ("05_lipsync", "06_first_last_frame_lipsync", "08_style_transfer_video_to_video")
        model, ic_model, ic_df = self._load_models(need_ic_lora=need_ic)

        pos, neg = self._encode_text(positive, negative)

        video_vae = gv(call_node("VAELoader", vae_name="ltx-2.5-video-vae-bf16.safetensors"), 0)
        audio_vae = gv(call_node("VAELoader", vae_name="ltx-2.5-audio-vae-bf16.safetensors"), 0)
        upscale_model = gv(call_node("LatentUpscaleModelLoader",
                                     model_name="ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"), 0)
        mem_report("Subgraph", "VAEs + Upscaler loaded")

        if mode == "01_text_to_video":
            vid_lat, aud_lat = self._build_empty_latents(width, height, length_frames, fps,
                                                          video_vae, audio_vae)

        elif mode == "02_audio_to_video":
            vid_lat, aud_lat = self._build_empty_latents(width, height, length_frames, fps,
                                                          video_vae, audio_vae)
            audio_input = gv(call_node("VHS_LoadAudioUpload",
                                       audio=os.path.basename(song_path),
                                       start_time=0, duration=float(length_sec)), 0)
            aud_lat = self._apply_audio_mask(audio_input, audio_vae, vid_lat, width//2, height//2)

        elif mode == "03_image_to_video":
            vid_lat, aud_lat = self._build_empty_latents(width, height, length_frames, fps,
                                                          video_vae, audio_vae)
            vid_lat = self._apply_image_to_video(
                self.config.get("image_1_path", "whatdreamscost/1.png"),
                vid_lat, video_vae,
                float(self.config.get("img_to_video_strength_s1", 0.7)),
                width, height)

        elif mode == "04_first_last_frame_to_video":
            vid_lat, aud_lat = self._build_empty_latents(width, height, length_frames, fps,
                                                          video_vae, audio_vae)
            vid_lat = self._apply_first_last_frame(
                self.config.get("image_1_path", "whatdreamscost/1.png"),
                self.config.get("image_2_path", "whatdreamscost/2.png"),
                vid_lat, video_vae, width, height)

        elif mode == "05_lipsync":
            vid_lat, aud_lat = self._build_empty_latents(width, height, length_frames, fps,
                                                          video_vae, audio_vae)
            pos, neg, vid_lat = self._apply_ic_lora_guide(
                ic_model or model, pos, neg, video_vae, vid_lat,
                self.config.get("image_1_path", "whatdreamscost/1.png"),
                ic_df, width, height)
            audio_input = gv(call_node("VHS_LoadAudioUpload",
                                       audio=os.path.basename(song_path),
                                       start_time=0, duration=float(length_sec)), 0)
            aud_lat = self._apply_audio_mask(audio_input, audio_vae, vid_lat, width//2, height//2)

        elif mode == "06_first_last_frame_lipsync":
            vid_lat, aud_lat = self._build_empty_latents(width, height, length_frames, fps,
                                                          video_vae, audio_vae)
            vid_lat = self._apply_first_last_frame(
                self.config.get("image_1_path", "whatdreamscost/1.png"),
                self.config.get("image_2_path", "whatdreamscost/2.png"),
                vid_lat, video_vae, width, height)
            pos, neg, vid_lat = self._apply_ic_lora_guide(
                ic_model or model, pos, neg, video_vae, vid_lat,
                self.config.get("image_1_path", "whatdreamscost/1.png"),
                ic_df, width, height)
            audio_input = gv(call_node("VHS_LoadAudioUpload",
                                       audio=os.path.basename(song_path),
                                       start_time=0, duration=float(length_sec)), 0)
            aud_lat = self._apply_audio_mask(audio_input, audio_vae, vid_lat, width//2, height//2)

        elif mode == "07_first_middle_last_frame_custom_audio":
            vid_lat, aud_lat = self._build_empty_latents(width, height, length_frames, fps,
                                                          video_vae, audio_vae)
            vid_lat = self._apply_first_middle_last_frame(
                self.config.get("image_1_path", "whatdreamscost/1.png"),
                self.config.get("image_2_path", "whatdreamscost/2.png"),
                self.config.get("image_3_path", "whatdreamscost/3.png"),
                vid_lat, video_vae, width, height, length_frames)
            audio_input = gv(call_node("VHS_LoadAudioUpload",
                                       audio=os.path.basename(song_path),
                                       start_time=0, duration=float(length_sec)), 0)
            aud_lat = self._apply_audio_mask(audio_input, audio_vae, vid_lat, width//2, height//2)

        elif mode == "08_style_transfer_video_to_video":
            vid_input_path = self.config.get("video_input_path", "")
            if not vid_input_path or not os.path.exists(vid_input_path):
                raise RuntimeError(f"Mode 08 requires video_input_path. Got: {vid_input_path}")
            vid_frames = gv(call_node("VHS_LoadVideo",
                                      video=vid_input_path, force_rate=float(fps),
                                      frame_load_cap=length_frames), 0)
            first_frame = vid_frames[0:1] if isinstance(vid_frames, torch.Tensor) else gv(vid_frames, 0)
            vid_lat, aud_lat = self._build_empty_latents(width, height, length_frames, fps,
                                                          video_vae, audio_vae)
            vid_lat = self._apply_image_to_video(
                vid_input_path, vid_lat, video_vae,
                float(self.config.get("img_to_video_strength_s1", 0.7)),
                width, height)
            pos, neg, vid_lat = self._apply_ic_lora_guide(
                ic_model or model, pos, neg, video_vae, vid_lat,
                self.config.get("image_1_path", "whatdreamscost/1.png"),
                ic_df, width, height)
            audio_input = gv(call_node("VHS_LoadAudioUpload",
                                       audio=os.path.basename(song_path),
                                       start_time=0, duration=float(length_sec)), 0)
            aud_lat = self._apply_audio_mask(audio_input, audio_vae, vid_lat, width//2, height//2)

        else:
            raise ValueError(f"Unknown mode: {mode}")

        cond_out = call_node("LTXVConditioning", positive=pos, negative=neg, frame_rate=float(fps))
        final_pos = sync_cond_to_cpu(gv(cond_out, 0))
        final_neg = sync_cond_to_cpu(gv(cond_out, 1))
        del pos, neg, cond_out
        light_clear()

        s1_vid, s1_aud = self._sample_stage(
            model, final_pos, final_neg, vid_lat, aud_lat,
            self.s1_sigmas, sampler, seed, "STAGE 1 (Low-Res)",
            use_dual_cfg=use_dual)
        medium_clear("post_s1")

        s1_vid_up = self._upscale_latent(s1_vid, video_vae, upscale_model)
        del s1_vid
        medium_clear("post_upscale")

        model_s2 = self._apply_detailer_lora(model)

        s2_vid, s2_aud = self._sample_stage(
            model_s2, final_pos, final_neg, s1_vid_up, s1_aud,
            self.s2_sigmas, sampler, seed + 1, "STAGE 2 (High-Res)",
            use_dual_cfg=use_dual)
        nuclear_purge()

        raw_video = os.path.join(self.workdir, f"subgraph_{mode}_raw.mp4")
        audio_pt = os.path.join(self.workdir, f"subgraph_{mode}_audio.pt")
        final_video = os.path.join(self.outdir, f"LTX25_{mode}.mp4")

        self._decode_video(s2_vid, video_vae, fps, crf, raw_video)
        self._decode_audio(s2_aud, audio_vae, audio_pt)
        self._mux_final(raw_video, audio_pt, fps, crf, song_path, final_video)

        del model, model_s2, video_vae, audio_vae, upscale_model
        del s2_vid, s2_aud, s1_vid_up, s1_aud, vid_lat, aud_lat
        del final_pos, final_neg
        nuclear_purge()

        verify_output(final_video)
        elapsed = time.time() - t_start
        print("\n" + "=" * 70)
        print(f"🎉 SUBGRAPH MODE {mode} COMPLETE")
        print(f"  Resolution: {width}x{height} @ {fps}fps")
        print(f"  Frames    : {length_frames} ({length_frames/fps:.2f}s)")
        print(f"  Time      : {elapsed/60:.2f} min")
        print(f"  Output    : {final_video}")
        print("=" * 70 + "\n")
        return final_video

print("✅ Cell 19: SubgraphModePipeline ready.")

# ════════════════════════════════════════════════════════════════════════════════
# CELL 20: MODE ROUTER
# ════════════════════════════════════════════════════════════════════════════════
def run_selected_mode(generation_mode: str, **kwargs) -> str:
    """
    Routes to the correct pipeline based on generation_mode:
      - 09_music_video_timeline → existing Director Phase A→D pipeline
      - 01-08 → SubgraphModePipeline (All-In-One workflow subgraphs)
    """
    if generation_mode == "09_music_video_timeline":
        return run_ltx25_director_master(
            global_prompt=kwargs.get("global_prompt", GLOBAL_PROMPT),
            negative_prompt=kwargs.get("negative_prompt", NEGATIVE_PROMPT),
            meta=kwargs.get("meta", TIMELINE_METADATA),
            segments=kwargs.get("segments", ORIGINAL_SEGMENTS),
            audio_segments=kwargs.get("audio_segments", ORIGINAL_AUDIO_SEGMENTS),
            motion_segments=kwargs.get("motion_segments", ORIGINAL_MOTION_SEGMENTS),
            seed=kwargs.get("base_seed", BASE_SEED),
            crf=kwargs.get("output_crf", OUTPUT_CRF),
            workdir=kwargs.get("workdir", WORK_DIRECTORY),
            outdir=kwargs.get("outdir", OUTPUT_DIRECTORY),
            resume=kwargs.get("resume_checkpoints", RESUME_CHECKPOINTS),
            use_song_audio=kwargs.get("use_song_audio", USE_SONG_AUDIO),
        )
    else:
        pipeline = SubgraphModePipeline(generation_mode, kwargs)
        return pipeline.run()

print("✅ Cell 20: Mode router ready.")

# ════════════════════════════════════════════════════════════════════════════════
# CELL 21: UPDATED MAIN EXECUTION BLOCK
# ════════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    validate_original_nodes()
    patch_comfy_memory_manager()
    patch_safetensors_direct_to_gpu()
    configure_vram_state(mode=VRAM_MODE)
    install_sampling_memory_hook(clear_every=4, ram_guard_gb=min_ram_guard_gb)

    config_dict = {
        "generation_mode": generation_mode,
        "global_prompt": GLOBAL_PROMPT,
        "negative_prompt": NEGATIVE_PROMPT,
        "meta": TIMELINE_METADATA,
        "segments": ORIGINAL_SEGMENTS,
        "audio_segments": ORIGINAL_AUDIO_SEGMENTS,
        "motion_segments": ORIGINAL_MOTION_SEGMENTS,
        "base_seed": BASE_SEED,
        "output_crf": OUTPUT_CRF,
        "workdir": WORK_DIRECTORY,
        "outdir": OUTPUT_DIRECTORY,
        "resume_checkpoints": RESUME_CHECKPOINTS,
        "use_song_audio": USE_SONG_AUDIO,
        "fps": fps,
        "output_width": output_width,
        "output_height": output_height,
        "sampler_name_subgraph": sampler_name_subgraph,
        "stage1_sigmas": stage1_sigmas,
        "stage2_sigmas": stage2_sigmas,
        "use_dual_cfg": use_dual_cfg,
        "cfg_value": cfg_value,
        "use_ic_lora": use_ic_lora,
        "ic_lora_strength": ic_lora_strength,
        "use_detailer_lora": use_detailer_lora,
        "detailer_lora_strength": detailer_lora_strength,
        "img_to_video_strength_s1": img_to_video_strength_s1,
        "img_to_video_strength_s2": img_to_video_strength_s2,
        "img_compression": img_compression,
        "image_1_path": image_1_path,
        "image_2_path": image_2_path,
        "image_3_path": image_3_path,
        "video_input_path": video_input_path,
        "audio_input_path": audio_input_path,
        "audio_start_time": audio_start_time,
        "audio_clip_length_seconds": audio_clip_length_seconds,
        "use_native_ltx_audio": use_native_ltx_audio,
        "audio_output_mode": audio_output_mode,
        "render_seconds": render_seconds,
    }

    final_output_file = run_selected_mode(**config_dict)
    print(f"\n🎬 Your LTX-2.5 video is ready:\n   {final_output_file}")

    if os.path.exists(final_output_file):
        quality_self_check(final_output_file, outdir=OUTPUT_DIRECTORY)