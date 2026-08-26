#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diagnostics script for testing TTS engines (Edge-TTS and XTTS-v2) on the system.
Tests Polish, English, and Japanese.
"""
import os
import sys
import subprocess
import shutil

def log(msg):
    print(f"[*] {msg}")

def check_edge_tts():
    log("Checking Edge-TTS...")
    edge_path = shutil.which("edge-tts")
    if not edge_path:
        log("❌ edge-tts command not found in PATH! Make sure it is installed in the virtualenv.")
        return False
    else:
        log(f"✅ Found edge-tts at: {edge_path}")

    # Test PL
    log("Testing Edge-TTS with Polish...")
    try:
        res = subprocess.run([
            "edge-tts", "--voice", "pl-PL-MarekNeural", "--text", "Dzień dobry, to jest test lektora.", "--write-media", "test_edge_pl.mp3"
        ], capture_output=True, text=True)
        if res.returncode == 0 and os.path.exists("test_edge_pl.mp3"):
            log("✅ Polish Edge-TTS output generated successfully (test_edge_pl.mp3)")
            os.remove("test_edge_pl.mp3")
        else:
            log(f"❌ Polish Edge-TTS failed: {res.stderr}")
    except Exception as e:
        log(f"❌ Polish Edge-TTS crashed: {e}")

    # Test JA
    log("Testing Edge-TTS with Japanese...")
    try:
        res = subprocess.run([
            "edge-tts", "--voice", "ja-JP-NanamiNeural", "--text", "こんにちは、これはテストです。", "--write-media", "test_edge_ja.mp3"
        ], capture_output=True, text=True)
        if res.returncode == 0 and os.path.exists("test_edge_ja.mp3"):
            log("✅ Japanese Edge-TTS output generated successfully (test_edge_ja.mp3)")
            os.remove("test_edge_ja.mp3")
        else:
            log(f"❌ Japanese Edge-TTS failed: {res.stderr}")
    except Exception as e:
        log(f"❌ Japanese Edge-TTS crashed: {e}")
    
    # Test Polish voice with Japanese text (user's error case)
    log("Testing Edge-TTS Polish voice with Japanese text...")
    try:
        res = subprocess.run([
            "edge-tts", "--voice", "pl-PL-MarekNeural", "--text", "どちら様?", "--write-media", "test_edge_bad.mp3"
        ], capture_output=True, text=True)
        if res.returncode == 0:
            log("✅ Edge-TTS processed Japanese text with Polish voice (though it might sound weird/silent).")
            if os.path.exists("test_edge_bad.mp3"): os.remove("test_edge_bad.mp3")
        else:
            log(f"⚠️ Edge-TTS failed Polish voice + Japanese text as expected: {res.stderr.strip()}")
    except Exception as e:
        log(f"❌ Edge-TTS failed Polish voice + Japanese text: {e}")


def check_xtts():
    log("Checking XTTS-v2...")
    try:
        import torch
        log(f"✅ PyTorch version: {torch.__version__}")
        log(f"✅ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            log(f"   CUDA device: {torch.cuda.get_device_name(0)}")
    except ImportError:
        log("❌ PyTorch is not installed!")
        return

    # Patch torch.load to avoid weights_only error
    log("Applying PyTorch weights_only patch...")
    _original_torch_load = torch.load
    def _patched_torch_load(*args, **kwargs):
        kwargs['weights_only'] = False
        return _original_torch_load(*args, **kwargs)
    torch.load = _patched_torch_load

    try:
        from TTS.tts.configs.xtts_config import XttsConfig
        from torch.serialization import add_safe_globals
        add_safe_globals([XttsConfig])
        log("✅ Configured safe globals for XTTS loading")
    except Exception as e:
        log(f"⚠️ Failed to add safe globals: {e}")

    try:
        from TTS.api import TTS
        log("✅ TTS library imported successfully.")
    except ImportError as e:
        log(f"❌ Failed to import TTS library: {e}")
        return

    # Create dummy reference voice (1 second of silence or sine wave if not exists)
    ref_path = "test_ref.wav"
    if not os.path.exists(ref_path):
        try:
            from pydub import AudioSegment
            AudioSegment.silent(duration=3000).set_frame_rate(22050).set_channels(1).export(ref_path, format="wav")
            log(f"✅ Created dummy reference voice: {ref_path}")
        except Exception as e:
            log(f"⚠️ Failed to create dummy ref: {e}. Using raw system sound or ignoring.")

    try:
        log("Loading XTTS v2 model (this may download it if not cached)...")
        model = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=torch.cuda.is_available())
        log("✅ XTTS v2 model loaded successfully!")
        
        # Test PL
        log("Testing XTTS with Polish...")
        try:
            model.tts_to_file(
                text="Dzień dobry, to jest test klonowania głosu.",
                speaker_wav=ref_path,
                language="pl",
                file_path="test_xtts_pl.wav"
            )
            if os.path.exists("test_xtts_pl.wav"):
                log("✅ Polish XTTS output generated successfully (test_xtts_pl.wav)")
                os.remove("test_xtts_pl.wav")
            else:
                log("❌ Polish XTTS generated nothing.")
        except Exception as e:
            log(f"❌ Polish XTTS failed: {e}")

        # Test JA
        log("Testing XTTS with Japanese...")
        try:
            model.tts_to_file(
                text="こんにちは、これはテストです。",
                speaker_wav=ref_path,
                language="ja",
                file_path="test_xtts_ja.wav"
            )
            if os.path.exists("test_xtts_ja.wav"):
                log("✅ Japanese XTTS output generated successfully (test_xtts_ja.wav)")
                os.remove("test_xtts_ja.wav")
            else:
                log("❌ Japanese XTTS generated nothing.")
        except Exception as e:
            log(f"❌ Japanese XTTS failed: {e}")

    except Exception as e:
        log(f"❌ XTTS initialization failed: {e}")
    finally:
        if os.path.exists(ref_path):
            os.remove(ref_path)

if __name__ == "__main__":
    log("=== STARTING TTS DIAGNOSTICS ===")
    check_edge_tts()
    print("-" * 40)
    check_xtts()
    log("=== DIAGNOSTICS COMPLETE ===")
