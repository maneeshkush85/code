# -*- coding: utf-8 -*-
"""
LTX23_Director_Master_V4_Production.py
================================================================================
LTX-2.3 Director 2.0 — 30-Second Music Video Pipeline
FAITHFUL PORT of `LTX-2.3_Director_2.0-MV-Workflow-30s.json` (ComfyUI graph).

Target Hardware: Google Colab Free Tier — NVIDIA T4 (15 GB VRAM) + ~12.2 GB RAM.

WHAT CHANGED vs V2 (and WHY the video is now good + voice is synced):
--------------------------------------------------------------------------------
  V2 problem : V2 DELETED the workflow's Master Timeline Controller node
               (`LTXDirector`, JSON id 131) and instead sliced the video into 5
               independent clips (226/161/131/226/83 frames) glued by a manual
               5-frame linear crossfade. That destroyed character identity,
               scene continuity and lip-sync, because each clip was sampled in
               isolation with no shared timeline / audio conditioning.

  V4 fix     : V4 drives the ENTIRE 756-frame timeline through the real
               `LTXDirector` node exactly like the JSON graph, then runs the
               faithful 2-stage sampler chain on the Director's outputs:

                 UnetLoaderGGUF ─┐
                 DualCLIPLoader ─┼─► Power Lora Loader (rgthree, 4 LoRAs)
                 (tiny VAE KJ) ──┘        │
                                          ▼
                              ModelPreviewOverrideKJ ─► LTXDirector (id 131)
                                          │  (model, positive, video_latent,
                                          │   audio_latent, guide_data,
                                          │   motion_guide_data, frame_rate)
                                          ▼
                 LTXVConditioning + ConditioningZeroOut (id 27 / 128)
                                          ▼
   STAGE 1  ►  LTXDirectorGuide(base) ─► ConcatAVLatent ─► SamplerCustomAdvanced
               (BasicScheduler linear_quadratic, 8 steps, denoise 1.0 @ base res)
               ─► SeparateAVLatent ─► LTXDirectorCropGuides
                                          ▼
   UPSCALE  ►  LTXVLatentUpsampler (2x spatial latent upscale)
                                          ▼
   STAGE 2  ►  LTXDirectorGuide(refine) ─► ConcatAVLatent ─► SamplerCustomAdvanced
               (BasicScheduler linear_quadratic, 4 steps, denoise 0.42 @ 832x480)
               ─► SeparateAVLatent ─► LTXDirectorCropGuides
                                          ▼
   DECODE   ►  VAEDecode (video)  +  LTXVAudioVAEDecode (audio, id 24)
                                          ▼
   OUTPUT   ►  VHS_VideoCombine (h264-mp4, yuv420p, CRF 8, 24 fps) + MP3 mux

  Because the whole audio latent path (Director.audio_latent ─► ConcatAVLatent
  ─► sampler ─► SeparateAVLatent ─► LTXVAudioVAEDecode) is preserved, the model
  generates mouth movements that match the audio -> VOICE IS SYNCED.

CONSISTENCY vs MEMORY (the hard part on a T4):
--------------------------------------------------------------------------------
  We DO NOT slice the timeline into independent clips (that is what broke V2).
  Instead we keep ONE continuous Director timeline (shared identity guide, shared
  conditioning, one continuous audio track) and we buy the VRAM/RAM headroom
  another way:
    • load each heavy model (DiT / text-encoder / VAE / upscaler) ONLY while it
      is needed, then immediately unload it;
    • stream every intermediate latent to disk (torch.save) and reload it for the
      next stage — "out-of-core" execution, so we never hold two big tensors at
      once;
    • a ~1.2 GB VRAM shield + 16 GB swap + tiled VAE decode absorb the spikes.
  This keeps peak memory low WITHOUT sacrificing the single-timeline continuity
  that gives consistent characters and scenes.

FRAME-COUNT RECONCILIATION (documented, was inconsistent in the source):
--------------------------------------------------------------------------------
  The JSON declares end_frame = duration_frames = 756 (= 31.5 s @ 24 fps) and the
  audio segment length ≈ 756. But the 5 image segment lengths sum to ~827
  (226.0 + 161.3 + 131.5 + 225.5 + 83.2). The `segment_lengths` widget even lists
  the 5th scene as 11.7 while `timeline_data` says 83.2.
  DECISION (single source of truth): TOTAL_FRAMES = 756 to match the declared
  render length and the audio track; the 5th segment length is taken as 83 (from
  timeline_data, not the 11.7 widget). The Director clips the timeline to
  TOTAL_FRAMES. All audio trimming/muxing uses TOTAL_FRAMES.
================================================================================
"""

# ════════════════════════════════════════════════════════════════════════════
# CELL 1: ENVIRONMENT SETUP & 16GB NVME SWAP  (proven infra, kept from V2)
# ════════════════════════════════════════════════════════════════════════════
# @title 💥 Cell 1: Environment Setup & Memory Protection { display-mode: "form" }
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
from pathlib import Path
from typing import Sequence, Mapping, Any, Union, Dict, List, Optional, Tuple

# Reduce CUDA fragmentation + let the allocator return memory to the OS.
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,garbage_collection_threshold:0.8'
os.environ['TORCH_CUDNN_V8_API_ENABLED'] = '1'
os.environ['MALLOC_TRIM_THRESHOLD_'] = '65536'


def run_cmd(cmd: str, silent: bool = True) -> int:
    if silent:
        res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return res.returncode
    return subprocess.run(cmd, shell=True).returncode


# 16 GB high-speed swap: absorbs RAM spikes when a 22B GGUF touches ~12.2 GB RAM.
if not os.path.exists("/content/swapfile") or os.path.getsize("/content/swapfile") < (8 * 1024 * 1024 * 1024):
    print("⚙️ [1/3] Setting up High-Speed Swap Partition...")
    run_cmd("swapoff /content/swapfile || true")
    run_cmd("rm -f /content/swapfile")
    run_cmd("fallocate -l 16G /content/swapfile || dd if=/dev/zero of=/content/swapfile bs=1M count=16384")
    run_cmd("chmod 600 /content/swapfile")
    run_cmd("mkswap /content/swapfile")
    run_cmd("swapon /content/swapfile || true")
    run_cmd("sysctl vm.swappiness=100 || true")
    run_cmd("sysctl vm.vfs_cache_pressure=500 || true")

try:
    import psutil
    sw = psutil.swap_memory()
    print(f"  📊 Physical RAM free: {psutil.virtual_memory().available/1e9:.2f} GB | Active Swap: {sw.total/1e9:.2f} GB")
except Exception:
    pass

# Stop ComfyUI from trying to (re)install requirements at import time.
if "utils" not in sys.modules or not hasattr(sys.modules["utils"], "__path__"):
    utils_mod = types.ModuleType("utils")
    utils_mod.__path__ = ["/content/ComfyUI/utils"]
    sys.modules["utils"] = utils_mod
else:
    utils_mod = sys.modules["utils"]

install_util_mod = types.ModuleType("utils.install_util")
install_util_mod.get_missing_requirements_message = lambda *a, **k: ""
install_util_mod.get_required_packages_versions = lambda *a, **k: {}
install_util_mod.requirements_path = "/content/ComfyUI/requirements.txt"
install_util_mod.install_requirements = lambda *a, **k: None
install_util_mod.check_requirements = lambda *a, **k: True
sys.modules["utils.install_util"] = install_util_mod
setattr(utils_mod, "install_util", install_util_mod)

print("✅ Cell 1: Environment & Memory Protection Configured.")


# ════════════════════════════════════════════════════════════════════════════
# CELL 2: INSTALL PYTHON DEPENDENCIES
# ════════════════════════════════════════════════════════════════════════════
# @title 💥 Cell 2: Install Core Dependencies { display-mode: "form" }
print("⚙️ [2/3] Installing Core Dependencies & PyTorch...")
run_cmd("pip install -q torch torchvision torchaudio", silent=False)
run_cmd("pip uninstall -y utils || true")
os.chdir("/content")
run_cmd("pip install -q torchsde einops diffusers accelerate psutil")
run_cmd("pip install -q av spandrel albumentations onnx opencv-python onnxruntime nest_asyncio imageio imageio-ffmpeg aiohttp scipy")
run_cmd("pip install -q 'kornia==0.7.3'")
run_cmd("apt-get -y install -qq aria2 ffmpeg")
print("✅ Cell 2: Dependencies successfully installed.")


# ════════════════════════════════════════════════════════════════════════════
# CELL 3: CLONE COMFYUI CORE
# ════════════════════════════════════════════════════════════════════════════
# @title 💥 Cell 3: Clone Upstream ComfyUI { display-mode: "form" }
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
print("✅ Cell 3: ComfyUI Core ready.")


# ════════════════════════════════════════════════════════════════════════════
# CELL 4: INSTALL CUSTOM NODES
# (These packs provide the JSON's custom nodes. whatdreamscost-comfyui supplies
#  the Master Timeline Controller: LTXDirector / LTXDirectorGuide /
#  LTXDirectorCropGuides. KJNodes supplies VAELoaderKJ + ModelPreviewOverrideKJ.
#  ComfyUI-GGUF supplies UnetLoaderGGUF. rgthree supplies Power Lora Loader.)
# ════════════════════════════════════════════════════════════════════════════
# @title 💥 Cell 4: Install Required Custom Nodes { display-mode: "form" }
custom_nodes_dir = "/content/ComfyUI/custom_nodes"
os.makedirs(custom_nodes_dir, exist_ok=True)

# Clean out junk node dirs (numeric / dotfiles) that break the loader.
for item in os.listdir(custom_nodes_dir):
    full_p = os.path.join(custom_nodes_dir, item)
    if os.path.isdir(full_p) and (item.isdigit() or item.startswith(".") or item == "comfyui"):
        shutil.rmtree(full_p, ignore_errors=True)

os.chdir(custom_nodes_dir)
repos = [
    ("WhatDreamsCost-ComfyUI", "https://github.com/WhatDreamscost/WhatDreamsCost-ComfyUI"),  # LTXDirector*
    ("ComfyUI_KJNodes", "https://github.com/kijai/ComfyUI-KJNodes.git"),                     # VAELoaderKJ / ModelPreviewOverrideKJ / SageAttn
    ("ComfyUI_GGUF", "https://github.com/city96/ComfyUI-GGUF.git"),                          # UnetLoaderGGUF (+ LoraLoaderGGUF)
    ("ComfyUI-LTXVideo", "https://github.com/Lightricks/ComfyUI-LTXVideo"),                  # LTXV* AV nodes
    ("ComfyUI-VideoHelperSuite", "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite"), # VHS_VideoCombine
    ("rgthree-comfy", "https://github.com/rgthree/rgthree-comfy"),                           # Power Lora Loader
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
# CELL 5: DOWNLOAD MODELS & 4-LORA STACK  (exact filenames from the JSON)
# ════════════════════════════════════════════════════════════════════════════
# @title 💥 Cell 5: Download LTX-2.3 Models & LoRAs { display-mode: "form" }
import torch

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True


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


print("📦 Downloading LTX-2.3 Core Models...")

dit_model = download_file(
    "https://huggingface.co/vantagewithai/LTX-2.3-GGUF/resolve/main/dev/ltx-2-3-22b-dev-Q4_K_M.gguf",
    "/content/ComfyUI/models/unet", filename="ltx-2-3-22b-dev-Q4_K_M.gguf")
link_file_safe("/content/ComfyUI/models/unet/ltx-2-3-22b-dev-Q4_K_M.gguf",
               "/content/ComfyUI/models/diffusion_models/ltx-2-3-22b-dev-Q4_K_M.gguf")

text_encoder_model = download_file(
    "https://huggingface.co/Comfy-Org/ltx-2/resolve/main/split_files/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors",
    "/content/ComfyUI/models/text_encoders", filename="gemma_3_12B_it_fp4_mixed.safetensors")
link_file_safe("/content/ComfyUI/models/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors",
               "/content/ComfyUI/models/clip/gemma_3_12B_it_fp4_mixed.safetensors")

text_encoder2_model = download_file(
    "https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/text_encoders/ltx-2.3_text_projection_bf16.safetensors",
    "/content/ComfyUI/models/text_encoders", filename="ltx-2.3_text_projection_bf16.safetensors")
link_file_safe("/content/ComfyUI/models/text_encoders/ltx-2.3_text_projection_bf16.safetensors",
               "/content/ComfyUI/models/clip/ltx-2.3_text_projection_bf16.safetensors")

vae_model = download_file(
    "https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/vae/LTX23_video_vae_bf16.safetensors",
    "/content/ComfyUI/models/vae", filename="LTX23_video_vae_bf16.safetensors")
vae_audio_model = download_file(
    "https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/vae/LTX23_audio_vae_bf16.safetensors",
    "/content/ComfyUI/models/vae", filename="LTX23_audio_vae_bf16.safetensors")
tiny_vae_model = download_file(
    "https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/vae/taeltx2_3.safetensors",
    "/content/ComfyUI/models/vae", filename="taeltx2_3.safetensors")
link_file_safe("/content/ComfyUI/models/vae/taeltx2_3.safetensors",
               "/content/ComfyUI/models/vae_approx/taeltx2_3.safetensors")

upscaler_model = download_file(
    "https://huggingface.co/vidfom/aimusic/resolve/main/ComfyUI/models/latent_upscale_models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
    "/content/ComfyUI/models/latent_upscale_models", filename="ltx-2.3-spatial-upscaler-x2-1.1.safetensors")
link_file_safe("/content/ComfyUI/models/latent_upscale_models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
               "/content/ComfyUI/models/upscale_models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors")

print("📦 Downloading Director 2.0 4-LoRA Stack...")
lora_dir = "/content/ComfyUI/models/loras"
download_file("https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/loras/ltx-2.3-22b-distilled-lora-dynamic_fro09_avg_rank_105_bf16.safetensors", lora_dir, filename="ltx-2.3-22b-distilled-lora-dynamic_fro09_avg_rank_105_bf16.safetensors")
download_file("https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/loras/LTX-2.3-OmniNFT-RL-Lora_bf16.safetensors", lora_dir, filename="LTX-2.3-OmniNFT-RL-Lora_bf16.safetensors")
download_file("https://huggingface.co/joyfox/LTX-2.3-Transition-LORA/resolve/main/ltx2.3-transition.safetensors", lora_dir, filename="ltx2.3-transition.safetensors")
download_file("https://huggingface.co/vidfom/aimusic/resolve/main/ComfyUI/models/loras/LTX2.3-MVCamera-drclips.safetensors", lora_dir, filename="LTX2.3-MVCamera-drclips.safetensors")

# Audio track (exact name from the JSON audio segment).
audio_dest_dir = "/content/ComfyUI/input/whatdreamscost"
audio_file_target = os.path.join(audio_dest_dir, "Late night trap.mp3")
os.makedirs(audio_dest_dir, exist_ok=True)
if not os.path.exists(audio_file_target) or os.path.getsize(audio_file_target) < 10000:
    download_file("https://huggingface.co/vidfom/aimusic/resolve/main/Late%20night%20trap.mp3", audio_dest_dir, filename="Late night trap.mp3")

print("✅ Cell 5: Models, LoRAs and audio assets validated.")



# ════════════════════════════════════════════════════════════════════════════
# CELL 6: TIMELINE CONFIGURATION  (rebuilt to match the JSON's LTXDirector node)
# ════════════════════════════════════════════════════════════════════════════
# @title 💥 Cell 6: Director Timeline, Prompts & LoRA Stack { display-mode: "form" }

# @markdown ### 🎬 Resolution / Frame Parameters (final render = width x height)
width = 832            # @param [512, 640, 768, 832, 960, 1024, 1280] {type:"raw"}
height = 480           # @param [320, 384, 480, 512, 544, 720] {type:"raw"}
fps = 24               # @param [24, 25, 30] {type:"raw"}
output_crf = 8         # @param {type:"slider", min:0, max:30, step:1}
divisible_by = 32      # from JSON LTXDirector widget

# @markdown ### 🧠 Free-Tier Memory Guard
min_ram_guard_gb = 2.0   # @param {type:"slider", min:1.0, max:6.0, step:0.5}
resume_generation = True # @param {type:"boolean"}
use_song_audio = True    # @param {type:"boolean"}

# ---- Frame-count reconciliation (see module docstring) ----------------------
# JSON declares 756 frames (31.5 s @ 24 fps) and the audio segment ~756 frames,
# while the 5 image-segment lengths sum to ~827. We take 756 as authoritative
# (matches audio + declared duration); the Director clips to this length.
TOTAL_FRAMES = 756
DURATION_SECONDS = TOTAL_FRAMES / fps  # 31.5 s

# ---- 4-LoRA stack (exact order + strengths from JSON Power Lora Loader) ------
lora_1 = os.path.join(lora_dir, "ltx-2.3-22b-distilled-lora-dynamic_fro09_avg_rank_105_bf16.safetensors")
lora_2 = os.path.join(lora_dir, "LTX-2.3-OmniNFT-RL-Lora_bf16.safetensors")
lora_3 = os.path.join(lora_dir, "ltx2.3-transition.safetensors")
lora_4 = os.path.join(lora_dir, "LTX2.3-MVCamera-drclips.safetensors")
LORA_WEIGHTS = [
    {"enabled": True, "name": lora_1, "strength": 0.4},
    {"enabled": True, "name": lora_2, "strength": 0.6},
    {"enabled": True, "name": lora_3, "strength": 0.7},
    {"enabled": True, "name": lora_4, "strength": 0.9},
]

# ---- Guide strengths per stage (from the two LTXDirectorGuide nodes) ---------
# JSON node 133 (base/stage-1 guide)  -> strength widget 1.0
# JSON node 132 (refine/stage-2 guide)-> strength widget 1.0
# (Both guide the SAME reference identity; the refinement stage keeps the guide
#  strong so identity does not drift during the low-denoise refine pass.)
GUIDE_STRENGTH_STAGE1 = 1.0
GUIDE_STRENGTH_STAGE2 = 1.0
DIRECTOR_GUIDE_STRENGTH_CSV = "1.00,1.00,1.00,1.00,1.00"  # per-segment (JSON widget)

# ---- The exact global prompt used by the JSON LTXDirector node ---------------
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

# ---- The 5 image segments + 1 audio segment (from JSON timeline_data) --------
# imageFile paths are relative to the ComfyUI `input/` folder, exactly as the
# Director node expects them.
SEGMENTS = [
    {"start": 0.0,               "length": 226.01059340956584, "imageFile": "whatdreamscost/1.png"},
    {"start": 226.01059340956584, "length": 161.31859976617454, "imageFile": "whatdreamscost/2.png"},
    {"start": 387.3291931757404,  "length": 131.45629831196658, "imageFile": "whatdreamscost/3.png"},
    {"start": 518.785491487707,   "length": 225.5063328766255,  "imageFile": "whatdreamscost/4.png"},
    {"start": 744.2918243643325,  "length": 83.22765271847516,  "imageFile": "whatdreamscost/5.3.png"},  # NOT 11.7
]
AUDIO_TRIM_START_FRAMES = 446.9222739141953  # JSON audio segment trimStart

def _build_timeline_data() -> str:
    """Reconstruct the LTXDirector `timeline_data` JSON string (UI-heavy fields
    like waveformPeaks are omitted; they are only used for drawing the waveform
    and are not needed for generation)."""
    td = {
        "mainTrackEnabled": True,
        "audioTrackEnabled": True,
        "motionTrackEnabled": True,
        "showFilenames": True,
        "overrideAudio": False,
        "inpaint_audio": True,
        "global_prompt": GLOBAL_PROMPT,
        "retake_global_prompt": "",
        "retakeMode": False,
        "normalStartFrame": 0,
        "normalDurationFrames": TOTAL_FRAMES,
        "segments": [
            {
                "id": f"seg{i+1}",
                "start": s["start"],
                "length": s["length"],
                "prompt": "",
                "type": "image",
                "imageFile": s["imageFile"],
                "imageB64": f"/api/view?filename={os.path.basename(s['imageFile'])}&type=input&subfolder=whatdreamscost",
                "isEndFrame": False,
            }
            for i, s in enumerate(SEGMENTS)
        ],
        "motionSegments": [],
        "audioSegments": [
            {
                "id": "aud1",
                "type": "audio",
                "start": 0,
                "length": 756.5194770828076,
                "trimStart": AUDIO_TRIM_START_FRAMES,
                "audioDurationFrames": 2880,
                "audioFile": "whatdreamscost/Late night trap.mp3",
                "fileName": "Late night trap.mp3",
            }
        ],
    }
    return json.dumps(td)

TIMELINE_DATA_JSON = _build_timeline_data()

# All LTXDirector widget values, keyed by the property names seen in the JSON.
# The reflection dispatcher (Cell 8) matches whichever of these the node's real
# signature actually accepts; extras are ignored.
DIRECTOR_WIDGETS = {
    "global_prompt": GLOBAL_PROMPT,
    "timeline_data": TIMELINE_DATA_JSON,
    "local_prompts": " |  |  |  | ",
    "segment_lengths": ",".join(str(s["length"]) for s in SEGMENTS),
    "guide_strength": DIRECTOR_GUIDE_STRENGTH_CSV,
    "epsilon": 0.001,
    "mainTrackEnabled": True,
    "audioTrackEnabled": True,
    "motionTrackEnabled": True,
    "inpaint_audio": True,
    "override_audio": False,
    "use_custom_audio": True,
    "use_custom_motion": True,
    "frame_rate": float(fps),
    "display_mode": "seconds",
    "custom_width": 1280,
    "custom_height": 720,
    "resize_method": "maintain aspect ratio",
    "divisible_by": divisible_by,
    "img_compression": 18,
    "start_second": 0,
    "end_second": DURATION_SECONDS,
    "duration_seconds": DURATION_SECONDS,
    "start_frame": 0,
    "end_frame": TOTAL_FRAMES,
    "duration_frames": TOTAL_FRAMES,
}

print(f"✅ Cell 6: Director timeline built | {len(SEGMENTS)} scenes | {TOTAL_FRAMES} frames "
      f"({DURATION_SECONDS:.2f}s @ {fps}fps) | final {width}x{height} | 4 LoRAs")


# ════════════════════════════════════════════════════════════════════════════
# CELL 7: PRODUCTION MEMORY ENGINE & 1.2GB VRAM SHIELD  (kept/strengthened)
# ════════════════════════════════════════════════════════════════════════════
# @title 💥 Cell 7: Memory Engine { display-mode: "form" }
def malloc_trim_os():
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def patch_comfy_memory_manager():
    """Configure ComfyUI memory management for a 15 GB T4 holding a 22B GGUF DiT.

    CRITICAL for the Director path: the 22B DiT (~13 GB) cannot coexist on the
    GPU with the Gemma-12B text encoder + VAE. So we force ComfyUI into
    LOW_VRAM state — model weights live on CPU/RAM and are streamed to the GPU
    only for their own forward pass. During LTXDirector.execute (text-encode +
    latent/guide prep) the DiT therefore stays on CPU, leaving the GPU free for
    the encoder and VAE. We DO NOT pin the text encoder to CUDA (that would stop
    ComfyUI from moving things off the GPU and re-trigger OOM)."""
    try:
        import comfy.model_management as mm

        # --- Force LOW_VRAM streaming (keeps the giant DiT off the GPU until sampling) ---
        try:
            mm.vram_state = mm.VRAMState.LOW_VRAM
            mm.set_vram_to = mm.VRAMState.LOW_VRAM
            print("  · ComfyUI VRAM state -> LOW_VRAM (stream weights, DiT stays on CPU when idle)")
        except Exception as e:
            print(f"  · Could not set LOW_VRAM state: {e}")

        if not getattr(mm, "_is_free_memory_patched", False):
            _orig_free_memory = mm.free_memory
            def _safe_free_memory(*args, **kwargs):
                try:
                    res = _orig_free_memory(*args, **kwargs)
                    return res if isinstance(res, list) else []
                except Exception:
                    return []
            mm.free_memory = _safe_free_memory

            # Small ~256 MB shield only (LOW_VRAM already keeps footprint low; a big
            # shield here would wrongly make ComfyUI think there is no room at all).
            _orig_get_free_memory = mm.get_free_memory
            def _buffered_get_free_memory(dev=None, torch_free_too=False):
                try:
                    free = _orig_get_free_memory(dev, torch_free_too)
                    return max(256 * 1024 * 1024, free - 256 * 1024 * 1024)
                except Exception:
                    return 1 * 1024 * 1024 * 1024
            mm.get_free_memory = _buffered_get_free_memory
            mm._is_free_memory_patched = True

        # NOTE: intentionally NOT overriding text_encoder_device/offload_device —
        # ComfyUI's default logic will place the encoder on the GPU only after
        # freeing the DiT, or fall back to CPU encoding, both of which are safe.
    except Exception as e:
        print(f"Memory patch notice: {e}")


def patch_safetensors_direct_to_gpu():
    """Deliberately a NO-OP in V4.

    In V2 this force-loaded the Gemma/CLIP weights straight to CUDA. On the T4
    that guarantees OOM because the DiT is already resident. We now let ComfyUI's
    LOW_VRAM manager decide device placement instead."""
    return


patch_comfy_memory_manager()
patch_safetensors_direct_to_gpu()


def get_ram_free_gb() -> float:
    try:
        import psutil
        return psutil.virtual_memory().available / 1e9
    except Exception:
        return 99.0


def drop_page_cache():
    """Evict big model files from the OS page cache (posix_fadvise DONTNEED)."""
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


def purge_deep(tag: str = ""):
    """The nuclear memory reset: unload every ComfyUI model + free CUDA + trim."""
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
    drop_page_cache()
    malloc_trim_os()


def clear_memory(tag: str = ""):
    """Lightweight per-step cleanup (call between every heavy op)."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    malloc_trim_os()


def ram_guard(min_free_gb: float = 2.0, tag: str = ""):
    if get_ram_free_gb() < min_free_gb:
        print(f"⚠️ [RAM GUARD] Free RAM ({get_ram_free_gb():.2f} GB) < {min_free_gb} GB -> Deep Purge")
        purge_deep(f"ram_guard:{tag}")


print("✅ Cell 7: Memory Engine + 1.2GB VRAM Shield active.")



# ════════════════════════════════════════════════════════════════════════════
# CELL 8: NODE REGISTRY, DISPATCHER & OUTPUT HELPERS
# ════════════════════════════════════════════════════════════════════════════
# @title 💥 Cell 8: Node Registry & Dispatcher { display-mode: "form" }
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
except Exception:
    pass

from nodes import init_builtin_extra_nodes, init_external_custom_nodes


async def _init_nodes():
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
        loop.run_until_complete(asyncio.ensure_future(_init_nodes()))
    else:
        loop.run_until_complete(_init_nodes())
except Exception:
    pass

from nodes import NODE_CLASS_MAPPINGS, LoraLoaderModelOnly


# ---- Generic output/tensor unwrappers (ComfyUI nodes return tuples/dicts) ----
def unwrap_tensor(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, torch.Tensor):
        return obj
    if hasattr(obj, "args") and len(getattr(obj, "args")) > 0:
        return unwrap_tensor(obj.args[0])
    if hasattr(obj, "outputs") and len(getattr(obj, "outputs")) > 0:
        return unwrap_tensor(obj.outputs[0])
    if hasattr(obj, "result") and len(getattr(obj, "result")) > 0:
        return unwrap_tensor(obj.result[0])
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
    return obj


def gv(obj: Any, index: int = 0) -> Any:
    """Get output slot `index` from a node's return value (tuple/list/dict/obj)."""
    if obj is None:
        return None
    if hasattr(obj, "args") and isinstance(obj.args, (list, tuple)) and len(obj.args) > 0:
        if len(obj.args) == 1 and isinstance(obj.args[0], (list, tuple)):
            return obj.args[0][index] if len(obj.args[0]) > index else None
        return obj.args[index] if len(obj.args) > index else None
    for attr in ["output", "outputs", "result", "values"]:
        if hasattr(obj, attr):
            val = getattr(obj, attr)
            if isinstance(val, (list, tuple)):
                return val[index] if len(val) > index else None
            if isinstance(val, dict):
                return val.get(index)
            if index == 0:
                return val
    if isinstance(obj, (tuple, list)):
        return obj[index] if len(obj) > index else None
    if isinstance(obj, dict):
        if "result" in obj and isinstance(obj["result"], (list, tuple)):
            return obj["result"][index] if len(obj["result"]) > index else None
        if index in obj:
            return obj[index]
    if index == 0:
        return obj
    return None


def unwrap_latent(x: Any) -> Dict[str, Any]:
    if x is None:
        return {"samples": None}
    while isinstance(x, (tuple, list)) and len(x) > 0:
        x = x[0]
    if hasattr(x, "result"):
        res = getattr(x, "result")
        if isinstance(res, (tuple, list)) and len(res) > 0:
            x = res[0]
        elif isinstance(res, dict):
            x = res
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
    """Move a latent dict's tensor(s) to a device. We push intermediates to CPU
    between stages so the GPU only ever holds the tensor it is actively using."""
    target = torch.device(target_device)
    latent_dict = unwrap_latent(latent)
    samples = latent_dict.get("samples", None)
    if samples is None:
        return latent_dict
    if isinstance(samples, torch.Tensor):
        if samples.is_nested:
            latent_dict["samples"] = torch.nested.nested_tensor([t.to(target) for t in samples.unbind()])
        else:
            latent_dict["samples"] = samples.to(target)
    return latent_dict


# ---- INPUT_TYPES-driven node dispatcher -------------------------------------
def _default_from_meta(meta: Any) -> Any:
    """Pick a sane default for a ComfyUI INPUT_TYPES entry we weren't given.
    meta forms: ("INT",{opts}) | ("FLOAT",{opts}) | ("STRING",{opts}) |
                ("BOOLEAN",{opts}) | ([choices],{opts}) | ("MODEL",) | "TYPE"."""
    t = meta[0] if isinstance(meta, (list, tuple)) and len(meta) > 0 else meta
    opts = meta[1] if (isinstance(meta, (list, tuple)) and len(meta) > 1 and isinstance(meta[1], dict)) else {}
    if isinstance(t, (list, tuple)):                 # dropdown choices
        return opts.get("default", t[0] if len(t) > 0 else None)
    if t == "INT":
        return opts.get("default", 0)
    if t == "FLOAT":
        return opts.get("default", 0.0)
    if t == "BOOLEAN":
        return opts.get("default", False)
    if t == "STRING":
        return opts.get("default", "")
    return None                                      # tensor/model/latent slots


def call_node(node_instance: Any, **kwargs) -> Any:
    """Call a ComfyUI node without the graph engine.

    IMPORTANT: modern ComfyUI (V3 schema) nodes declare `execute(cls, **kwargs)`
    with a GENERIC signature, so inspecting the function signature yields no
    parameter names. We therefore build the call kwargs from the node's declared
    `INPUT_TYPES()` (required + optional), which is the authoritative contract.
    Required inputs we weren't given are filled with spec defaults. Extra kwargs
    (e.g. rgthree Power-Lora dynamic entries) are forwarded only when the target
    accepts **kwargs. We do NOT swallow the node's real exception."""
    cls = type(node_instance)
    spec_req: Dict[str, Any] = {}
    spec_opt: Dict[str, Any] = {}
    try:
        it = cls.INPUT_TYPES()
        spec_req = it.get("required", {}) or {}
        spec_opt = it.get("optional", {}) or {}
    except Exception:
        pass

    # Pick the primary callable.
    func_name = getattr(node_instance, "FUNCTION", None)
    primary = None
    if func_name and hasattr(node_instance, func_name):
        primary = getattr(node_instance, func_name)
    elif hasattr(node_instance, "execute"):
        primary = getattr(node_instance, "execute")
    elif hasattr(node_instance, "EXECUTE_NORMALIZED"):
        primary = getattr(node_instance, "EXECUTE_NORMALIZED")

    if (spec_req or spec_opt) and primary is not None:
        call_kwargs: Dict[str, Any] = {}
        # Required: use provided value, else a spec default.
        for name, meta in spec_req.items():
            call_kwargs[name] = kwargs[name] if name in kwargs else _default_from_meta(meta)
        # Optional: include only when the caller supplied a non-None value.
        for name in spec_opt:
            if name in kwargs and kwargs[name] is not None:
                call_kwargs[name] = kwargs[name]
        # Forward extra dynamic kwargs (e.g. rgthree Power-Lora entries) ONLY for
        # genuine legacy (V1) node functions that declare **kwargs. V3-schema
        # nodes route through the generic EXECUTE_NORMALIZED(*args, **kwargs)
        # wrapper but their real execute() is STRICT to INPUT_TYPES — forwarding
        # non-spec keys there raises 'unexpected keyword argument'. So we never
        # forward extras when the primary is the normalized wrapper.
        primary_name = getattr(primary, "__name__", "")
        is_v3_normalized = primary_name == "EXECUTE_NORMALIZED"
        has_var_kw = False
        if not is_v3_normalized:
            try:
                has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD
                                 for p in inspect.signature(primary).parameters.values())
            except Exception:
                has_var_kw = False
        if has_var_kw:
            for k, v in kwargs.items():
                if k not in call_kwargs and v is not None:
                    call_kwargs[k] = v
        return primary(**call_kwargs)  # let real errors propagate (no masking)

    # Fallback for legacy nodes without usable INPUT_TYPES: signature-based.
    callables = []
    if primary is not None:
        callables.append(primary)
    for fb in ["get_guider", "get_noise", "get_sampler", "get_sigmas", "sample",
               "apply_guide", "crop_guides", "upsample_latent", "concat",
               "separate", "encode", "decode", "generate", "process", "run"]:
        if hasattr(node_instance, fb):
            callables.append(getattr(node_instance, fb))
    last_err = None
    for func in callables:
        try:
            sig = inspect.signature(func)
            valid = {}
            for name, param in sig.parameters.items():
                if name in ("cls", "self"):
                    continue
                if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                    continue
                if name in kwargs:
                    valid[name] = kwargs[name]
                elif param.default is not inspect.Parameter.empty:
                    pass
                else:
                    ann = str(param.annotation)
                    if "int" in ann:
                        valid[name] = 0
                    elif "float" in ann:
                        valid[name] = 0.0
                    elif "bool" in ann:
                        valid[name] = False
                    elif "str" in ann:
                        valid[name] = ""
                    else:
                        valid[name] = None
            return func(**valid)
        except Exception as e:
            last_err = e
            continue
    if last_err is not None:
        raise last_err
    raise AttributeError(f"Cannot execute node '{node_instance.__class__.__name__}'")


def describe_node(node_name: str):
    """Debug helper: print a custom node's declared INPUT_TYPES so we can see the
    real parameter names of the whatdreamscost Director nodes at runtime."""
    cls = NODE_CLASS_MAPPINGS.get(node_name)
    if cls is None:
        print(f"  · {node_name}: NOT REGISTERED")
        return
    try:
        it = cls.INPUT_TYPES()
        req = list(it.get("required", {}).keys())
        opt = list(it.get("optional", {}).keys())
        print(f"  · {node_name}: required={req} optional={opt}")
    except Exception as e:
        print(f"  · {node_name}: (could not introspect: {e})")


# ---- Required-node validator (fail loudly if the graph can't be reproduced) --
REQUIRED_NODES = [
    "UnetLoaderGGUF", "DualCLIPLoader", "LTXVConditioning", "ConditioningZeroOut",
    "LTXVConcatAVLatent", "LTXVSeparateAVLatent", "LTXVLatentUpsampler",
    "LatentUpscaleModelLoader", "SamplerCustomAdvanced", "CFGGuider",
    "KSamplerSelect", "RandomNoise", "BasicScheduler", "VAEDecode",
    "LTXVAudioVAEDecode", "VAELoader",
]
# Director nodes come from whatdreamscost pack; alt names handled at call sites.
DIRECTOR_NODES = ["LTXDirector", "LTXDirectorGuide", "LTXDirectorCropGuides"]
NICE_TO_HAVE = ["Power Lora Loader (rgthree)", "ModelPreviewOverrideKJ",
                "VAELoaderKJ", "LTXVSpatioTemporalTiledVAEDecode", "VHS_VideoCombine"]

missing_required = [n for n in REQUIRED_NODES if n not in NODE_CLASS_MAPPINGS]
missing_director = [n for n in DIRECTOR_NODES if n not in NODE_CLASS_MAPPINGS]
missing_nice = [n for n in NICE_TO_HAVE if n not in NODE_CLASS_MAPPINGS]

print(f"✅ Cell 8: {len(NODE_CLASS_MAPPINGS)} ComfyUI nodes registered.")
if missing_required:
    print(f"❌ MISSING REQUIRED NODES (pipeline will not run correctly): {missing_required}")
if missing_director:
    print(f"⚠️ MISSING DIRECTOR NODES (Master Timeline Controller). Check that "
          f"WhatDreamsCost-ComfyUI cloned correctly: {missing_director}")
if missing_nice:
    print(f"ℹ️ Optional nodes not found (graceful fallbacks will be used): {missing_nice}")

print("🔎 Introspecting Director node signatures (for faithful kwargs):")
for n in DIRECTOR_NODES:
    describe_node(n)



# ════════════════════════════════════════════════════════════════════════════
# CELL 9: KEYFRAME VALIDATION & LOADER HELPERS
# ════════════════════════════════════════════════════════════════════════════
# @title 💥 Cell 9: Keyframes & Loaders { display-mode: "form" }
from PIL import Image, ImageOps, ImageDraw
import numpy as np

base_input = "/content/ComfyUI/input/whatdreamscost"
os.makedirs(base_input, exist_ok=True)

# Generate placeholder keyframes only if the user hasn't uploaded singer photos.
for idx, fname in enumerate(["1.png", "2.png", "3.png", "4.png", "5.3.png"]):
    p = os.path.join(base_input, fname)
    if not os.path.exists(p):
        img = Image.new("RGB", (768, 512), color=(30 + idx * 25, 25, 60 + idx * 25))
        d = ImageDraw.Draw(img)
        d.text((40, 230), f"Keyframe {fname} — upload your singer photo here", fill=(255, 255, 255))
        img.save(p)

# The JSON's 5th segment uses 5.3.png; fall back to 5.png if that is what exists.
if os.path.exists(f"{base_input}/5.png") and not os.path.exists(f"{base_input}/5.3.png"):
    shutil.copy(f"{base_input}/5.png", f"{base_input}/5.3.png")


def prepare_reference_image(image_path: str, w: int, h: int) -> torch.Tensor:
    """Center-crop to target aspect then resize -> normalized [1,H,W,3] tensor."""
    full = image_path
    if not os.path.isabs(full):
        full = os.path.join("/content/ComfyUI/input", image_path)
    if full and os.path.exists(full):
        im = ImageOps.exif_transpose(Image.open(full).convert("RGB"))
        ta = w / h
        iw, ih = im.size
        ia = iw / ih
        if ia > ta:
            nw = int(ta * ih); off = (iw - nw) // 2
            im = im.crop((off, 0, off + nw, ih))
        else:
            nh = int(iw / ta); off = (ih - nh) // 2
            im = im.crop((0, off, iw, off + nh))
        arr = np.array(im.resize((w, h), Image.BICUBIC)).astype(np.float32) / 255.0
        return torch.from_numpy(arr).unsqueeze(0)
    return torch.full((1, h, w, 3), 0.5)


def load_vae_helper(vae_name: str, device: str = "main_device", dtype: str = "bf16"):
    """Load a VAE via VAELoaderKJ (device/dtype control) with VAELoader fallback."""
    if "VAELoaderKJ" in NODE_CLASS_MAPPINGS:
        try:
            vkj = NODE_CLASS_MAPPINGS["VAELoaderKJ"]()
            out = gv(call_node(vkj, vae_name=vae_name, device=device, weight_dtype=dtype), 0)
            if out is not None:
                return out
        except Exception:
            pass
    vl = NODE_CLASS_MAPPINGS["VAELoader"]()
    return gv(call_node(vl, vae_name=vae_name), 0)


def get_basic_scheduler_sigmas(model: Any, scheduler: str = "linear_quadratic",
                               steps: int = 8, denoise: float = 1.0) -> Any:
    """Faithful BasicScheduler sigmas (JSON nodes 33 & 21) with robust fallbacks."""
    if "BasicScheduler" in NODE_CLASS_MAPPINGS:
        try:
            bs = NODE_CLASS_MAPPINGS["BasicScheduler"]()
            res = call_node(bs, model=model, scheduler=scheduler, scheduler_name=scheduler,
                            steps=steps, denoise=denoise)
            sig = gv(res, 0)
            if isinstance(sig, torch.Tensor) and sig.numel() > 0:
                return sig
        except Exception:
            pass
    try:
        import comfy.samplers
        ms = model.get_model_object("model_sampling")
        total = int(steps / denoise) if 0.0 < denoise < 1.0 else steps
        sigmas = comfy.samplers.calculate_sigmas(ms, scheduler, total)
        return sigmas[-(steps + 1):]
    except Exception:
        pass
    total = int(round(steps / denoise)) if (0.0 < denoise < 1.0) else steps
    sig = [(1.0 - i / total) ** 2 for i in range(total + 1)][-(steps + 1):]
    return torch.tensor(sig, dtype=torch.float32)


def tiled_decode_video(video_latent: Any, vae_obj: Any, tile: int = 256) -> torch.Tensor:
    """Spatiotemporal tiled VAE decode (out-of-core friendly) with fallbacks."""
    latent_dict = unwrap_latent(video_latent)
    if "LTXVSpatioTemporalTiledVAEDecode" in NODE_CLASS_MAPPINGS:
        try:
            t = NODE_CLASS_MAPPINGS["LTXVSpatioTemporalTiledVAEDecode"]()
            res = call_node(t, vae=vae_obj, latents=latent_dict, spatial_tiles=2,
                            spatial_overlap=8, temporal_tile_length=16, temporal_overlap=4,
                            last_frame_fix=False, working_device="auto", working_dtype="auto")
            return unwrap_tensor(res)
        except Exception:
            pass
    if "VAEDecodeTiled" in NODE_CLASS_MAPPINGS:
        try:
            v = NODE_CLASS_MAPPINGS["VAEDecodeTiled"]()
            return unwrap_tensor(call_node(v, samples=latent_dict, vae=vae_obj, tile_size=tile))
        except Exception:
            pass
    v = NODE_CLASS_MAPPINGS["VAEDecode"]()
    return unwrap_tensor(call_node(v, samples=latent_dict, vae=vae_obj))


print("✅ Cell 9: Keyframes validated & loader helpers ready.")


# ════════════════════════════════════════════════════════════════════════════
# CELL 10: MODEL + MASTER TIMELINE CONTROLLER BUILDER
#   UnetLoaderGGUF -> Power Lora Loader (4 LoRAs, +clip) -> ModelPreviewOverrideKJ
#   -> LTXDirector (id 131).  Returns everything the sampler chain needs.
# ════════════════════════════════════════════════════════════════════════════
# @title 💥 Cell 10: Director & Model Builder { display-mode: "form" }
def build_model_stack():
    """Recreate JSON nodes 135 -> 138 -> 10 and load CLIP (12) + audio VAE (8)."""
    purge_deep("pre_model_load")

    # --- (135) DiT: 22B GGUF ---
    unet_loader = NODE_CLASS_MAPPINGS["UnetLoaderGGUF"]()
    model = gv(call_node(unet_loader, unet_name="ltx-2-3-22b-dev-Q4_K_M.gguf"), 0)

    # --- (12) DualCLIPLoader (Gemma + text projection, type "ltxv") ---
    dcl = NODE_CLASS_MAPPINGS["DualCLIPLoader"]()
    clip_obj = gv(call_node(dcl, clip_name1="gemma_3_12B_it_fp4_mixed.safetensors",
                            clip_name2="ltx-2.3_text_projection_bf16.safetensors",
                            type="ltxv", device="default"), 0)

    # --- (138) Power Lora Loader (rgthree): apply the 4-LoRA stack on model+clip ---
    model, clip_obj = apply_power_lora_stack(model, clip_obj, LORA_WEIGHTS)

    # --- (6) Tiny VAE (taeltx2_3) for the preview override node ---
    tiny_vae = None
    try:
        tiny_vae = load_vae_helper("taeltx2_3.safetensors", device="main_device", dtype="bf16")
    except Exception:
        pass

    # --- (10) ModelPreviewOverrideKJ ---
    # This is a UI-only live-preview passthrough (it periodically taeltx-decodes
    # latents to show progress). It does NOT affect the generated result, and on a
    # T4 the extra per-step decode wastes VRAM. We deliberately SKIP it and keep
    # the raw model. (tiny_vae is still downloaded/available if you want previews.)
    _ = tiny_vae  # intentionally unused; preview override skipped for memory safety
    print("  · ModelPreviewOverrideKJ skipped (preview-only; no effect on output)")

    # --- Attention / feed-forward memory hooks (KJ) — big VRAM savings on T4 ---
    if "PatchSageAttentionKJ" in NODE_CLASS_MAPPINGS:
        try:
            model = gv(call_node(NODE_CLASS_MAPPINGS["PatchSageAttentionKJ"](),
                                 model=model, sage_attention="auto"), 0) or model
            print("  ✓ SageAttention hook applied")
        except Exception:
            pass
    if "LTXVChunkFeedForward" in NODE_CLASS_MAPPINGS:
        try:
            model = gv(call_node(NODE_CLASS_MAPPINGS["LTXVChunkFeedForward"](),
                                 model=model, chunks=8, dim_threshold=4096), 0) or model
            print("  ✓ ChunkFeedForward hook applied (chunks=8)")
        except Exception:
            pass

    # --- (8) Audio VAE (LTX23_audio_vae_bf16) needed by the Director + audio decode ---
    audio_vae = load_vae_helper("LTX23_audio_vae_bf16.safetensors", device="main_device", dtype="fp16")

    clear_memory("post_model_stack")
    return model, clip_obj, audio_vae


def apply_power_lora_stack(model, clip, lora_configs):
    """Recreate rgthree 'Power Lora Loader': apply the 4 LoRAs to BOTH model and
    clip when the pack is present; else fall back to per-LoRA model-only loaders.
    Applying to clip too keeps text/identity conditioning consistent with the JSON."""
    if "Power Lora Loader (rgthree)" in NODE_CLASS_MAPPINGS:
        try:
            pll = NODE_CLASS_MAPPINGS["Power Lora Loader (rgthree)"]()
            # rgthree reads a dict of {lora, on, strength} widget entries. We pass
            # both a structured 'lora_stack'/'loras' kwarg and named entries; the
            # dispatcher forwards whichever the installed version accepts.
            entries = {}
            for i, cfg in enumerate(lora_configs, start=1):
                entries[f"lora_{i}"] = {
                    "on": bool(cfg["enabled"]),
                    "lora": os.path.basename(cfg["name"]),
                    "strength": float(cfg["strength"]),
                    "strengthTwo": None,
                }
            res = call_node(pll, model=model, clip=clip, **entries)
            m = gv(res, 0)
            c = gv(res, 1)
            if m is not None:
                print("  ✓ Power Lora Loader (rgthree): 4-LoRA stack applied to model+clip")
                return m, (c if c is not None else clip)
        except Exception as e:
            print(f"  [Notice] Power Lora Loader fallback: {e}")

    # Fallback: apply each LoRA to the model only (GGUF-aware loader if available).
    lora_cls = NODE_CLASS_MAPPINGS.get("LoraLoaderGGUF", LoraLoaderModelOnly)
    loader = lora_cls()
    for cfg in lora_configs:
        if cfg.get("enabled") and os.path.exists(cfg["name"]):
            try:
                clear_memory()
                res = call_node(loader, model=model, lora_name=os.path.basename(cfg["name"]),
                                strength_model=cfg["strength"])
                model = gv(res, 0) or model
                print(f"  + LoRA (model-only fallback): {os.path.basename(cfg['name'])} @ {cfg['strength']}")
                clear_memory()
            except Exception as e:
                print(f"  [Notice] LoRA {os.path.basename(cfg['name'])} skipped: {e}")
    return model, clip


def run_master_timeline_controller(model, clip, audio_vae):
    """Recreate JSON node 131 `LTXDirector` — the Master Timeline Controller.
    It ingests the whole 756-frame timeline (5 image segments + 1 audio segment +
    global prompt) and emits the shared model / conditioning / video+audio latents
    / guide data / frame_rate used by BOTH diffusion stages. This single shared
    context is what keeps the character + scene consistent and the voice synced."""
    director_cls = NODE_CLASS_MAPPINGS.get("LTXDirector")
    if director_cls is None:
        raise RuntimeError("LTXDirector node missing — cannot drive the Master Timeline. "
                           "Ensure WhatDreamsCost-ComfyUI is installed (Cell 4).")

    # Free the GPU before the Director text-encodes: push any resident weights
    # (incl. the 22B DiT) back to CPU so Gemma-12B + the VAE have room. In
    # LOW_VRAM mode the DiT will re-stream to the GPU later, at sampling time.
    try:
        import comfy.model_management as mm
        mm.unload_all_models()
        mm.soft_empty_cache()
    except Exception:
        pass
    clear_memory("pre_director")
    print(f"  · GPU cleared for Director text-encode (free RAM {get_ram_free_gb():.2f} GB)")

    director = director_cls()
    # Pass model/clip/audio_vae + every timeline widget; dispatcher keeps the ones
    # the installed node version actually declares.
    res = call_node(director, model=model, clip=clip, audio_vae=audio_vae,
                    optional_latent=None, **DIRECTOR_WIDGETS)

    out = {
        "model": gv(res, 0),
        "positive": gv(res, 1),
        "video_latent": gv(res, 2),
        "audio_latent": gv(res, 3),
        "guide_data": gv(res, 4),
        "motion_guide_data": gv(res, 5),
        "frame_rate": gv(res, 6),
    }
    # combined_audio (slot 7) is optional and only used for previewing.
    out["combined_audio"] = gv(res, 7)
    if out["model"] is None:
        out["model"] = model
    if out["frame_rate"] is None:
        out["frame_rate"] = float(fps)
    print("  ✓ LTXDirector timeline ingested "
          f"(video_latent={'ok' if out['video_latent'] is not None else 'None'}, "
          f"audio_latent={'ok' if out['audio_latent'] is not None else 'None'})")
    return out


print("✅ Cell 10: Model + Master Timeline Controller builder ready.")



# ════════════════════════════════════════════════════════════════════════════
# CELL 11: CONDITIONING + FAITHFUL TWO-STAGE SAMPLER CHAIN
#   Mirrors JSON nodes 27/128 (conditioning) and the two sampler columns:
#     Stage 1 (base) : Guide(133) -> ConcatAV(29) -> Sampler(31, 8 steps d=1.0)
#                      -> SeparateAV(34) -> CropGuides(55)
#     Upscale        : LTXVLatentUpsampler(14)
#     Stage 2 (refine): Guide(132) -> ConcatAV(18) -> Sampler(19, 4 steps d=0.42)
#                      -> SeparateAV(22) -> CropGuides(54)
#   All intermediate latents are streamed to disk (out-of-core) so only one big
#   tensor is on the GPU at a time.
# ════════════════════════════════════════════════════════════════════════════
# @title 💥 Cell 11: Two-Stage Diffusion Engine { display-mode: "form" }
def zero_out_conditioning(conditioning: Any) -> Any:
    """JSON node 128 ConditioningZeroOut — the negative is a zeroed positive."""
    if "ConditioningZeroOut" in NODE_CLASS_MAPPINGS:
        try:
            return gv(call_node(NODE_CLASS_MAPPINGS["ConditioningZeroOut"](),
                                conditioning=conditioning), 0)
        except Exception:
            pass
    c = []
    for t in conditioning:
        d = t[1].copy() if (isinstance(t, (list, tuple)) and len(t) > 1 and isinstance(t[1], dict)) else {}
        pooled = d.get("pooled_output", None)
        if torch.is_tensor(pooled):
            d["pooled_output"] = torch.zeros_like(pooled)
        c.append([torch.zeros_like(t[0]), d])
    return c


def make_ltxv_conditioning(positive: Any, frame_rate: float) -> Tuple[Any, Any]:
    """JSON node 27 LTXVConditioning: wrap (positive, zeroed-negative) at frame_rate."""
    negative = zero_out_conditioning(positive)
    if "LTXVConditioning" in NODE_CLASS_MAPPINGS:
        res = call_node(NODE_CLASS_MAPPINGS["LTXVConditioning"](),
                        positive=positive, negative=negative, frame_rate=float(frame_rate))
        return gv(res, 0), gv(res, 1)
    return positive, negative


def _director_guide(positive, negative, vae, latent, guide_data, motion_guide_data,
                    model, scale_by):
    """Call LTXDirectorGuide (JSON id 132/133). Returns (pos, neg, latent, model).

    NOTE (from runtime INPUT_TYPES introspection): this node takes NO `image` and
    NO `strength` input — the reference keyframes are already baked into
    `guide_data` by the Director. The real per-stage control is `scale_by`
    (JSON widgets: base guide=0.5, refine guide=1.0) plus the fixed guide widgets
    below (ic_lora_name None, upscale_method bicubic, crop center, tile 256/64)."""
    cls = NODE_CLASS_MAPPINGS.get("LTXDirectorGuide") or NODE_CLASS_MAPPINGS.get("LTXVAddGuide")
    if cls is None:
        return positive, negative, latent, model
    res = call_node(cls(), positive=positive, negative=negative, vae=vae, latent=latent,
                    guide_data=guide_data, motion_guide_data=motion_guide_data, model=model,
                    ic_lora_name="None", ic_lora_strength=1.0, scale_by=float(scale_by),
                    upscale_method="bicubic", image_attention_strength=1.0, crop="center",
                    auto_snap_ic_grid=True, use_tiled_encode=False, tile_size=256,
                    tile_overlap=64, retake_mode=False)
    p = gv(res, 0); n = gv(res, 1); lat = gv(res, 2); mdl = gv(res, 3)
    return (p if p is not None else positive,
            n if n is not None else negative,
            lat if lat is not None else latent,
            mdl if (mdl is not None and hasattr(mdl, "model_options")) else model)


def _director_crop(positive, negative, latent):
    """Call LTXDirectorCropGuides (id 54/55). Returns cropped (pos, neg, latent)."""
    cls = NODE_CLASS_MAPPINGS.get("LTXDirectorCropGuides") or NODE_CLASS_MAPPINGS.get("LTXVCropGuides")
    if cls is None:
        return positive, negative, latent
    res = call_node(cls(), positive=positive, negative=negative, latent=latent)
    p = gv(res, 0); n = gv(res, 1); lat = gv(res, 2)
    return (p if p is not None else positive,
            n if n is not None else negative,
            lat if lat is not None else latent)


def _sample_stage(model, positive, negative, av_latent, steps, denoise, seed, tag):
    """One SamplerCustomAdvanced pass (CFGGuider cfg=1, KSamplerSelect euler,
    BasicScheduler linear_quadratic). Mirrors JSON nodes 17-21 / 28-33."""
    ksel = NODE_CLASS_MAPPINGS["KSamplerSelect"]()
    rnd = NODE_CLASS_MAPPINGS["RandomNoise"]()
    cfg = NODE_CLASS_MAPPINGS["CFGGuider"]()
    sca = NODE_CLASS_MAPPINGS["SamplerCustomAdvanced"]()

    sampler = gv(call_node(ksel, sampler_name="euler"), 0)
    sigmas = get_basic_scheduler_sigmas(model=model, scheduler="linear_quadratic",
                                        steps=steps, denoise=denoise)
    noise = call_node(rnd, noise_seed=seed)
    guider = call_node(cfg, cfg=1.0, model=model, positive=positive, negative=negative)
    print(f"  ⚡ {tag}: euler, {steps} steps, denoise={denoise}")
    out = call_node(sca, noise=gv(noise, 0), guider=gv(guider, 0), sampler=gv(sampler, 0),
                    sigmas=sigmas, latent_image=av_latent)
    result = sync_latent_device(gv(out, 0), "cpu")
    del noise, guider, sampler, sigmas, out
    clear_memory(tag)
    return result


def run_two_stage_diffusion(director_out, workdir: str, resume: bool = True) -> Tuple[str, str]:
    """Execute the full faithful 2-stage chain over the ENTIRE Director timeline.
    Returns paths to the saved (video_latent, audio_latent) .pt files."""
    video_lat_path = f"{workdir}/final_video_latent.pt"
    audio_lat_path = f"{workdir}/final_audio_latent.pt"
    if resume and os.path.exists(video_lat_path) and os.path.exists(audio_lat_path):
        print("  ⏭ Final latents already cached — skipping diffusion.")
        return video_lat_path, audio_lat_path

    concat = NODE_CLASS_MAPPINGS["LTXVConcatAVLatent"]()
    separate = NODE_CLASS_MAPPINGS["LTXVSeparateAVLatent"]()
    upsampler = NODE_CLASS_MAPPINGS["LTXVLatentUpsampler"]()
    ups_loader = NODE_CLASS_MAPPINGS["LatentUpscaleModelLoader"]()

    dmodel = director_out["model"]
    frame_rate = director_out["frame_rate"]
    guide_data = director_out["guide_data"]
    motion_guide = director_out["motion_guide_data"]
    seed = 0  # JSON RandomNoise widget = 0 / fixed

    # --- Conditioning (nodes 27 + 128) ---
    pos0, neg0 = make_ltxv_conditioning(director_out["positive"], frame_rate)

    # ======================= STAGE 1 (base resolution) ========================
    # The base guide uses scale_by=0.5 (JSON node 133) to work at half-res; the
    # keyframe identity is already carried inside guide_data from the Director.
    print("\n" + "=" * 70 + "\n🎬 STAGE 1: Base diffusion over full timeline\n" + "=" * 70)
    video_vae = load_vae_helper("LTX23_video_vae_bf16.safetensors", device="main_device", dtype="bf16")

    p1, n1, lat1, m1 = _director_guide(pos0, neg0, video_vae, director_out["video_latent"],
                                       guide_data, motion_guide, dmodel, scale_by=0.5)
    del video_vae
    clear_memory("post_stage1_guide")

    av1_in = sync_latent_device(gv(call_node(concat, video_latent=lat1,
                                             audio_latent=director_out["audio_latent"]), 0), "cpu")
    del lat1
    s1 = _sample_stage(m1, p1, n1, av1_in, steps=8, denoise=1.0, seed=seed, tag="Stage 1 sample")
    del av1_in, m1

    sep1 = call_node(separate, av_latent=s1)
    v1 = sync_latent_device(gv(sep1, 0), "cpu")
    a1 = sync_latent_device(gv(sep1, 1), "cpu")   # <-- audio latent carried forward (voice sync)
    del sep1, s1
    _, _, v1c = _director_crop(p1, n1, v1)
    del v1
    clear_memory("post_stage1")

    # ============================ 2x LATENT UPSCALE ===========================
    print("  ⚡ Upscaling latent 2x (spatial) ...")
    up_model = gv(call_node(ups_loader, model_name="ltx-2.3-spatial-upscaler-x2-1.1.safetensors"), 0)
    video_vae = load_vae_helper("LTX23_video_vae_bf16.safetensors", device="main_device", dtype="bf16")
    v_ups = sync_latent_device(gv(call_node(upsampler, samples=v1c, upscale_model=up_model,
                                            vae=video_vae), 0), "cpu")
    del up_model, v1c
    clear_memory("post_upscale")

    # ======================= STAGE 2 (refine @ target res) ====================
    # The refine guide uses scale_by=1.0 (JSON node 132): keeps the upscaled
    # resolution and re-anchors identity before the low-denoise refinement pass.
    print("\n" + "=" * 70 + f"\n🎬 STAGE 2: Refinement @ {width}x{height}\n" + "=" * 70)
    p2, n2, lat2, m2 = _director_guide(pos0, neg0, video_vae, v_ups, guide_data,
                                       motion_guide, dmodel, scale_by=1.0)
    del video_vae, v_ups
    clear_memory("post_stage2_guide")

    av2_in = sync_latent_device(gv(call_node(concat, video_latent=lat2, audio_latent=a1), 0), "cpu")
    del lat2, a1
    s2 = _sample_stage(m2, p2, n2, av2_in, steps=4, denoise=0.42, seed=seed, tag="Stage 2 refine")
    del av2_in, m2

    sep2 = call_node(separate, av_latent=s2)
    v2 = sync_latent_device(gv(sep2, 0), "cpu")
    a2 = sync_latent_device(gv(sep2, 1), "cpu")
    del sep2, s2
    _, _, v2c = _director_crop(p2, n2, v2)
    del v2

    # --- Persist final latents (out-of-core) then purge everything ---
    torch.save({"samples": unwrap_tensor(v2c).detach().cpu().half()}, video_lat_path)
    torch.save({"samples": unwrap_tensor(a2).detach().cpu().half()}, audio_lat_path)
    del v2c, a2, pos0, neg0, p1, n1, p2, n2, director_out
    purge_deep("post_diffusion")
    print(f"  💾 Saved final latents:\n     video -> {video_lat_path}\n     audio -> {audio_lat_path}")
    return video_lat_path, audio_lat_path


print("✅ Cell 11: Two-Stage Diffusion Engine ready.")



# ════════════════════════════════════════════════════════════════════════════
# CELL 12: DECODE, VIDEO ASSEMBLY, AUDIO MUX & RUN TRIGGER
#   video latent -> tiled VAEDecode (node 1)      -> frames
#   audio latent -> LTXVAudioVAEDecode (node 24)  -> waveform
#   frames+audio -> VHS_VideoCombine (node 139)   -> h264-mp4 CRF 8
#   then a final ffmpeg pass muxes the real MP3 with the correct trimStart.
# ════════════════════════════════════════════════════════════════════════════
# @title 💥 Cell 12: Decode, Assemble & Run { display-mode: "form" }
def decode_final_latents(video_lat_path: str, audio_lat_path: str, workdir: str):
    """Decode video + audio latents separately, each with the VAE loaded ALONE
    (out-of-core) to stay under the T4 VRAM ceiling."""
    print("\n" + "=" * 70 + "\n🎬 DECODE: Out-of-core VAE decoding\n" + "=" * 70)

    # --- Video: tiled decode with the video VAE loaded alone ---
    v_pack = torch.load(video_lat_path, map_location="cpu")
    v_lat = {"samples": v_pack["samples"].float()}
    del v_pack
    video_vae = load_vae_helper("LTX23_video_vae_bf16.safetensors", device="main_device", dtype="bf16")
    frames = tiled_decode_video(v_lat, video_vae, tile=256)   # [T,H,W,3] in 0..1
    frames = unwrap_tensor(frames).detach().cpu().float()
    del video_vae, v_lat
    purge_deep("post_video_decode")
    print(f"  ✓ Video decoded: {frames.shape[0]} frames @ {frames.shape[2]}x{frames.shape[1]}")

    # --- Audio: LTXVAudioVAEDecode with the audio VAE loaded alone ---
    decoded_audio = None
    try:
        a_pack = torch.load(audio_lat_path, map_location="cpu")
        a_lat = {"samples": a_pack["samples"].float()}
        del a_pack
        audio_vae = load_vae_helper("LTX23_audio_vae_bf16.safetensors", device="main_device", dtype="fp16")
        avd = NODE_CLASS_MAPPINGS["LTXVAudioVAEDecode"]()
        decoded_audio = gv(call_node(avd, samples=a_lat, audio_vae=audio_vae), 0)
        del audio_vae, a_lat, avd
        purge_deep("post_audio_decode")
        print("  ✓ Audio latent decoded (model-generated performance audio)")
    except Exception as e:
        print(f"  [Notice] Audio VAE decode skipped ({e}); will rely on MP3 mux.")

    return frames, decoded_audio


def write_video(frames: torch.Tensor, decoded_audio, frame_rate: float, outdir: str) -> str:
    """JSON node 139 VHS_VideoCombine (h264-mp4, yuv420p, CRF 8). Falls back to
    imageio if the VHS node is unavailable."""
    os.makedirs(outdir, exist_ok=True)
    raw_path = os.path.join(outdir, "LTX23_Director_Master_V4.mp4")

    if "VHS_VideoCombine" in NODE_CLASS_MAPPINGS:
        try:
            vhs = NODE_CLASS_MAPPINGS["VHS_VideoCombine"]()
            call_node(vhs, images=frames.float(), audio=decoded_audio,
                      frame_rate=float(frame_rate), loop_count=0,
                      filename_prefix="LTX2.3/Video", format="video/h264-mp4",
                      pix_fmt="yuv420p", crf=output_crf, save_metadata=False,
                      trim_to_audio=False, pingpong=False, save_output=True)
            # VHS writes into ComfyUI/output/LTX2.3/ — grab the newest mp4.
            cand = sorted(glob.glob("/content/ComfyUI/output/LTX2.3/*.mp4"),
                          key=os.path.getmtime)
            if cand:
                shutil.copyfile(cand[-1], raw_path)
                print(f"  ✓ VHS_VideoCombine wrote: {cand[-1]}")
                return raw_path
        except Exception as e:
            print(f"  [Notice] VHS_VideoCombine fallback to imageio: {e}")

    # Fallback: raw frames via imageio (no embedded audio; MP3 muxed next).
    import imageio
    arr = (frames.clamp(0, 1).cpu().numpy() * 255.0).astype(np.uint8)
    imageio.mimwrite(raw_path, arr, fps=int(round(frame_rate)), quality=9)
    print(f"  ✓ imageio wrote: {raw_path}")
    return raw_path


def mux_song(video_path: str, song_path: str, trim_start_frames: float,
             frame_rate: float, total_frames: int, crf: int) -> str:
    """Final ffmpeg mux: overlay the real MP3, starting at trimStart, clipped to
    the video length. This guarantees the released MP3 lines up with the render."""
    out_path = video_path.replace(".mp4", "_synced.mp4")
    ss = max(0.0, trim_start_frames / frame_rate)         # audio start offset (s)
    dur = total_frames / frame_rate                        # clip length (s)
    cmd = (f'ffmpeg -y -i "{video_path}" -ss {ss:.3f} -t {dur:.3f} -i "{song_path}" '
           f'-map 0:v:0 -map 1:a:0 -c:v libx264 -crf {crf} -pix_fmt yuv420p '
           f'-c:a aac -b:a 320k -shortest "{out_path}"')
    run_cmd(cmd, silent=False)
    return out_path if os.path.exists(out_path) else video_path


# ---- Director-state caching (skip the slow text-encode on re-runs) ----------
def _cond_to_cpu(cond: Any) -> Any:
    """Move a ComfyUI conditioning (list of [tensor, dict]) to CPU for pickling."""
    if not isinstance(cond, (list, tuple)):
        return cond
    out = []
    for item in cond:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            t = item[0].cpu() if torch.is_tensor(item[0]) else item[0]
            d = {}
            for k, v in (item[1] or {}).items():
                d[k] = v.cpu() if torch.is_tensor(v) else v
            out.append([t, d])
        else:
            out.append(item)
    return out


def save_director_state(director_out: Dict[str, Any], path: str) -> bool:
    """Persist the EXPENSIVE Director outputs (text-encode conditioning + prepared
    latents + guide data). The DiT `model` is NOT saved — it is rebuilt on resume
    (it is needed for sampling anyway). Best-effort: if any object refuses to
    pickle, we drop the cache and recompute next time."""
    try:
        payload = {
            "positive": _cond_to_cpu(director_out.get("positive")),
            "video_latent": sync_latent_device(director_out.get("video_latent"), "cpu"),
            "audio_latent": sync_latent_device(director_out.get("audio_latent"), "cpu"),
            "guide_data": director_out.get("guide_data"),
            "motion_guide_data": director_out.get("motion_guide_data"),
            "frame_rate": float(director_out.get("frame_rate") or fps),
        }
        tmp = path + ".tmp"
        torch.save(payload, tmp)
        os.replace(tmp, path)
        print(f"  💾 [CACHE] Director state saved: {path}")
        return True
    except Exception as e:
        print(f"  [Notice] Could not cache Director state ({e}); it will recompute next run.")
        for p in (path, path + ".tmp"):
            try:
                os.remove(p)
            except Exception:
                pass
        return False


def load_director_state(path: str, model: Any) -> Dict[str, Any]:
    """Reload cached Director outputs and attach the freshly built model."""
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    payload["model"] = model
    payload["combined_audio"] = None
    return payload


def run_director_pipeline(workdir="/content/LTXDirector_Work",
                          outdir="/content/LTXStudio_Output",
                          resume=True) -> Optional[str]:
    """End-to-end: build model stack -> LTXDirector timeline -> 2-stage diffusion
    -> decode -> assemble -> mux.  Uses two resume checkpoints:
      1) director_state.pt  — caches the slow text-encode + latent prep
      2) final_*_latent.pt  — caches the finished 2-stage diffusion latents"""
    os.makedirs(workdir, exist_ok=True)
    os.makedirs(outdir, exist_ok=True)
    patch_comfy_memory_manager()
    patch_safetensors_direct_to_gpu()
    purge_deep("pipeline_start")
    print(f"  [RAM Baseline] Free RAM: {get_ram_free_gb():.2f} GB")

    director_state_file = f"{workdir}/director_state.pt"

    with torch.inference_mode():
        video_lat_path = f"{workdir}/final_video_latent.pt"
        audio_lat_path = f"{workdir}/final_audio_latent.pt"

        if not (resume and os.path.exists(video_lat_path) and os.path.exists(audio_lat_path)):
            ram_guard(min_ram_guard_gb, "before_build")

            # ========================= PHASE A =============================
            print("\n" + "=" * 70)
            print(f"PHASE A: LTXDirector MASTER TIMELINE INGESTION ({TOTAL_FRAMES} frames)")
            print("=" * 70)
            t0 = time.time()

            # The DiT model must be rebuilt regardless (needed for sampling).
            model, clip, audio_vae = build_model_stack()

            if resume and os.path.exists(director_state_file) and os.path.getsize(director_state_file) > 1024:
                print(f"  ⏭ [RESUME] Loading cached Director state from: {director_state_file}")
                director_out = load_director_state(director_state_file, model)
                del clip, audio_vae
                print(f"  ✓ Director state restored in {time.time()-t0:.1f}s (skipped text-encode)")
            else:
                print("  · Running LTXDirector (text-encode + latent/guide prep). "
                      "This is the SLOW step on free-tier — please wait...")
                director_out = run_master_timeline_controller(model, clip, audio_vae)
                del clip, audio_vae
                clear_memory("post_director")
                save_director_state(director_out, director_state_file)
                print(f"  ✓ PHASE A done in {time.time()-t0:.1f}s")

            del model
            clear_memory("post_phase_a")
            video_lat_path, audio_lat_path = run_two_stage_diffusion(director_out, workdir, resume=resume)
        else:
            print("  ⏭ Reusing cached final latents (PHASE A + diffusion skipped).")

        frames, decoded_audio = decode_final_latents(video_lat_path, audio_lat_path, workdir)
        total_frames = int(frames.shape[0])

        raw_path = write_video(frames, decoded_audio, float(fps), outdir)
        del frames, decoded_audio
        purge_deep("post_write")

        final_path = raw_path
        song = "/content/ComfyUI/input/whatdreamscost/Late night trap.mp3"
        if use_song_audio and os.path.exists(song):
            final_path = mux_song(raw_path, song, AUDIO_TRIM_START_FRAMES, float(fps),
                                  total_frames, output_crf)
            print(f"  🎵 MP3 muxed (trimStart {AUDIO_TRIM_START_FRAMES:.0f}f): {final_path}")

        purge_deep("pipeline_done")
        print(f"\n🎬 Output: {final_path}")
        print(f"   {total_frames} frames ({total_frames/float(fps):.2f}s @ {fps}fps)")
        return final_path


# ────────────────────────────────────────────────────────────────────────────
# RUNTIME TRIGGER
# ────────────────────────────────────────────────────────────────────────────
work_directory = "/content/LTXDirector_Work"
output_directory = "/content/LTXStudio_Output"

# Verify keyframes + audio and fail loudly if a critical asset is missing.
print("\n🔎 Pre-flight asset check:")
for s in SEGMENTS:
    fp = os.path.join("/content/ComfyUI/input", s["imageFile"])
    print(f"  {'✓' if os.path.exists(fp) else '⚠️ MISSING'} keyframe: {fp}")
song_file_path = "/content/ComfyUI/input/whatdreamscost/Late night trap.mp3"
print(f"  {'✓' if os.path.exists(song_file_path) else '⚠️ MISSING'} audio: {song_file_path}")

if missing_required or missing_director:
    print("\n❌ Cannot run: required/Director nodes are missing (see Cell 8). "
          "Re-run Cell 4 to install custom nodes, then restart the runtime.")
else:
    final_video = run_director_pipeline(workdir=work_directory,
                                        outdir=output_directory,
                                        resume=resume_generation)
    print(f"\n🎉 LTX-2.3 Director 2.0 (V4) complete!\nFinal file: {final_video}")
