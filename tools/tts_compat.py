#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔬 viDubb Pro — Maslo95 Edition — TTS Compatibility Tester
=========================================
Uruchom: python3 tools/tts_compat.py

Testuje każdą kombinację silnik TTS × język i zapisuje wyniki do:
  tools/compat_report.json
  tools/compat_report.html  (czytelna tabela)

Możesz też podać projekt żeby przetestować na prawdziwym wideo:
  python3 tools/tts_compat.py --project "Mój projekt"
"""

import sys
import os
import json
import argparse
import subprocess
import time
from pathlib import Path

WORKSPACE = Path(__file__).parent.parent
sys.path.insert(0, str(WORKSPACE))

# Representative synthesis text for each supported language.
TEST_SENTENCES = {
    "pl": "Dzień dobry, to jest próba syntezy mowy w języku polskim.",
    "en": "Hello, this is a text-to-speech synthesis test in English.",
    "ja": "こんにちは、これは日本語の音声合成テストです。",
    "de": "Guten Tag, das ist ein Sprachsynthese-Test auf Deutsch.",
    "fr": "Bonjour, ceci est un test de synthèse vocale en français.",
    "es": "Hola, esta es una prueba de síntesis de voz en español.",
    "ru": "Здравствуйте, это тест синтеза речи на русском языке.",
    "it": "Buongiorno, questo è un test di sintesi vocale in italiano.",
    "ko": "안녕하세요, 이것은 한국어 음성 합성 테스트입니다.",
    "zh-cn": "你好，这是一个中文语音合成测试。",
}

EDGE_VOICE_MAP = {
    "pl": "pl-PL-MarekNeural",
    "en": "en-US-ChristopherNeural",
    "ja": "ja-JP-KeitaNeural",
    "de": "de-DE-KillianNeural",
    "fr": "fr-FR-HenriNeural",
    "es": "es-ES-AlvaroNeural",
    "ru": "ru-RU-DmitryNeural",
    "it": "it-IT-DiegoNeural",
    "ko": "ko-KR-InJoonNeural",
    "zh-cn": "zh-CN-YunxiNeural",
}

XTTS_SUPPORTED_LANGS = {
    "en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru",
    "nl", "cs", "ar", "zh-cn", "ja", "ko", "hu", "hi"
}


def log(msg, color=""):
    COLORS = {"green": "\033[92m", "red": "\033[91m", "yellow": "\033[93m", "reset": "\033[0m", "bold": "\033[1m"}
    prefix = COLORS.get(color, "")
    reset = COLORS["reset"] if color else ""
    print(f"{prefix}{msg}{reset}")


def test_edge_tts(lang: str, text: str, out_path: str) -> dict:
    """Test Edge-TTS for a given language."""
    voice = EDGE_VOICE_MAP.get(lang, "en-US-ChristopherNeural")
    start = time.time()
    try:
        res = subprocess.run(
            ["edge-tts", "--voice", voice, "--text", text, "--write-media", out_path],
            capture_output=True, timeout=30
        )
        elapsed = time.time() - start
        if res.returncode == 0 and os.path.exists(out_path):
            size_kb = os.path.getsize(out_path) / 1024
            os.remove(out_path)
            return {"ok": True, "engine": "edge", "lang": lang, "voice": voice,
                    "time_s": round(elapsed, 2), "size_kb": round(size_kb, 1),
                    "error": None}
        else:
            err = res.stderr.decode(errors="replace").strip().split("\n")[-1][:120]
            return {"ok": False, "engine": "edge", "lang": lang, "voice": voice,
                    "time_s": round(elapsed, 2), "size_kb": 0, "error": err}
    except subprocess.TimeoutExpired:
        return {"ok": False, "engine": "edge", "lang": lang, "voice": voice,
                "time_s": 30, "size_kb": 0, "error": "Timeout (30s)"}
    except Exception as e:
        return {"ok": False, "engine": "edge", "lang": lang, "voice": voice,
                "time_s": 0, "size_kb": 0, "error": str(e)[:120]}


def test_xtts(lang: str, text: str, out_path: str, ref_path: str = None) -> dict:
    """Test XTTS-v2 for a given language."""
    if lang not in XTTS_SUPPORTED_LANGS:
        return {"ok": False, "engine": "xtts", "lang": lang, "voice": "N/A",
                "time_s": 0, "size_kb": 0,
                "error": f"Language '{lang}' not supported by XTTS-v2"}

    # Apply PyTorch patch
    try:
        import torch
        _orig = torch.load
        def _safe(*a, **kw):
            kw["weights_only"] = False
            return _orig(*a, **kw)
        torch.load = _safe
        from TTS.tts.configs.xtts_config import XttsConfig
        from torch.serialization import add_safe_globals
        add_safe_globals([XttsConfig])
    except Exception as e:
        return {"ok": False, "engine": "xtts", "lang": lang, "voice": "N/A",
                "time_s": 0, "size_kb": 0, "error": f"Setup error: {e}"}

    # Create dummy ref if needed
    cleanup_ref = False
    if not ref_path or not os.path.exists(ref_path):
        ref_path = str(WORKSPACE / "tools" / "_test_ref.wav")
        try:
            from pydub import AudioSegment
            AudioSegment.silent(duration=3000).set_frame_rate(22050).set_channels(1).export(ref_path, format="wav")
            cleanup_ref = True
        except Exception as e:
            return {"ok": False, "engine": "xtts", "lang": lang, "voice": "N/A",
                    "time_s": 0, "size_kb": 0, "error": f"Cannot create ref audio: {e}"}

    start = time.time()
    try:
        from TTS.api import TTS as CoquiTTS
        model = CoquiTTS("tts_models/multilingual/multi-dataset/xtts_v2",
                         gpu=torch.cuda.is_available())
        model.tts_to_file(text=text, speaker_wav=ref_path, language=lang, file_path=out_path)
        elapsed = time.time() - start

        if os.path.exists(out_path):
            size_kb = os.path.getsize(out_path) / 1024
            os.remove(out_path)
            del model
            return {"ok": True, "engine": "xtts", "lang": lang, "voice": "clone",
                    "time_s": round(elapsed, 2), "size_kb": round(size_kb, 1), "error": None}
        else:
            del model
            return {"ok": False, "engine": "xtts", "lang": lang, "voice": "clone",
                    "time_s": round(elapsed, 2), "size_kb": 0, "error": "No output file generated"}
    except Exception as e:
        return {"ok": False, "engine": "xtts", "lang": lang, "voice": "clone",
                "time_s": round(time.time() - start, 2), "size_kb": 0, "error": str(e)[:200]}
    finally:
        if cleanup_ref and os.path.exists(ref_path):
            os.remove(ref_path)


def save_html_report(results: list, path: str):
    rows_html = ""
    for r in results:
        status = "✅" if r["ok"] else "❌"
        bg = "#0f2d1a" if r["ok"] else "#2d0f0f"
        err_cell = f'<span style="color:#f87171;font-size:0.8em">{r["error"] or ""}</span>'
        rows_html += f"""
        <tr style="background:{bg}">
          <td>{status}</td>
          <td><b>{r["engine"].upper()}</b></td>
          <td>{r["lang"]}</td>
          <td style="font-size:0.85em">{r.get("voice","")}</td>
          <td>{r["time_s"]}s</td>
          <td>{r["size_kb"]} KB</td>
          <td>{err_cell}</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8">
<title>ViDubb TTS Compatibility Report</title>
<style>
  body {{font-family: 'Segoe UI', sans-serif; background:#0d0e16; color:#f1f5f9; padding:30px;}}
  h1 {{color:#6366f1; margin-bottom:20px;}}
  table {{border-collapse: collapse; width: 100%;}}
  th {{background:#1e293b; padding:10px 14px; text-align:left; color:#94a3b8; font-size:0.8em; text-transform:uppercase;}}
  td {{padding:9px 14px; border-bottom:1px solid #1e293b;}}
  .ok {{color:#10b981;}} .err {{color:#f43f5e;}}
</style>
</head><body>
<h1>🔬 ViDubb Pro — TTS Compatibility Report</h1>
<p style="color:#64748b; margin-bottom:20px">Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
<table>
<thead><tr>
  <th>Status</th><th>Engine</th><th>Language</th><th>Voice</th>
  <th>Time</th><th>File size</th><th>Error</th>
</tr></thead>
<tbody>{rows_html}</tbody>
</table>
<script>
// Auto-refresh every 5s if we detect a running test
</script>
</body></html>"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)


def main():
    parser = argparse.ArgumentParser(description="ViDubb TTS Compatibility Tester")
    parser.add_argument("--engines", nargs="+", default=["edge", "xtts"],
                        help="TTS engines to test (edge, xtts)")
    parser.add_argument("--langs", nargs="+", default=list(TEST_SENTENCES.keys()),
                        help="Languages to test")
    parser.add_argument("--ref-audio", type=str, default=None,
                        help="Path to reference audio file for XTTS voice cloning")
    parser.add_argument("--project", type=str, default=None,
                        help="Project name to extract reference audio from")
    parser.add_argument("--out", type=str,
                        default=str(WORKSPACE / "tools" / "compat_report"),
                        help="Output report filename (without extension)")
    args = parser.parse_args()

    log("╔══════════════════════════════════════════════╗", "bold")
    log("║  🔬 ViDubb TTS Compatibility Tester          ║", "bold")
    log("╚══════════════════════════════════════════════╝", "bold")
    log("")

    ref_audio = args.ref_audio

    # Extract ref audio from project if specified
    if args.project and not ref_audio:
        project_path = WORKSPACE / "projects" / f"{args.project}.json"
        if project_path.exists():
            import json as _json
            pdata = _json.loads(project_path.read_text(encoding="utf-8"))
            video_path = pdata.get("video_path", "")
            if video_path and os.path.exists(video_path):
                log(f"📹 Używam wideo z projektu '{args.project}': {video_path}")
                ref_audio = str(WORKSPACE / "tools" / "_compat_ref.wav")
                try:
                    from pydub import AudioSegment
                    clip = AudioSegment.from_file(video_path)[1000:9000]
                    clip.set_frame_rate(22050).set_channels(1).export(ref_audio, format="wav")
                    log(f"  ✅ Wyciągnięto referencję audio → {ref_audio}", "green")
                except Exception as e:
                    log(f"  ⚠️ Nie udało się wyciągnąć ref audio: {e}", "yellow")
                    ref_audio = None
        else:
            log(f"  ⚠️ Projekt '{args.project}' nie znaleziony.", "yellow")

    results = []
    out_tmp = str(WORKSPACE / "tools" / "_test_out.wav")

    for engine in args.engines:
        for lang in args.langs:
            text = TEST_SENTENCES.get(lang, "Test sentence.")
            log(f"  Testing [{engine.upper()}] lang={lang}…", "")

            if engine == "edge":
                r = test_edge_tts(lang, text, out_tmp)
            elif engine == "xtts":
                r = test_xtts(lang, text, out_tmp, ref_audio)
            else:
                r = {"ok": False, "engine": engine, "lang": lang, "voice": "N/A",
                     "time_s": 0, "size_kb": 0, "error": "Unknown engine"}

            results.append(r)
            if r["ok"]:
                log(f"    ✅ OK  ({r['time_s']}s, {r['size_kb']} KB)", "green")
            else:
                log(f"    ❌ FAIL: {r['error']}", "red")

    # Save reports
    json_path = args.out + ".json"
    html_path = args.out + ".html"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    save_html_report(results, html_path)

    ok_count = sum(1 for r in results if r["ok"])
    log("")
    log(f"📊 Wyniki: {ok_count}/{len(results)} testów przeszło", "bold")
    log(f"   📄 JSON:  {json_path}")
    log(f"   🌐 HTML:  {html_path}  ← otwórz w przeglądarce!")
    log("")

    # Cleanup temp ref
    if args.project and ref_audio and os.path.exists(ref_audio):
        os.remove(ref_audio)


if __name__ == "__main__":
    main()
