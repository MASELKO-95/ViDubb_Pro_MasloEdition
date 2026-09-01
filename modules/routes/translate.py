import gc
import re
import threading
import requests
import torch
from flask import Blueprint, jsonify, request
from modules.state import state

translate_bp = Blueprint('translate', __name__)

def _post_ai_generate(json_data: dict, custom_endpoint: str = None, timeout: int = 120):

    urls_to_try = []
    if custom_endpoint and custom_endpoint.strip():
        ep = custom_endpoint.strip().rstrip('/')
        urls_to_try.extend([
            (f"{ep}/api/generate", "ollama"),
            (f"{ep}/v1/chat/completions", "openai"),
            (f"{ep}/v1/completions", "openai_legacy"),
        ])
    if state.active_project and getattr(state.active_project, "ai_endpoint", None):
        ep = state.active_project.ai_endpoint.strip().rstrip('/')
        if (f"{ep}/api/generate", "ollama") not in urls_to_try:
            urls_to_try.extend([
                (f"{ep}/api/generate", "ollama"),
                (f"{ep}/v1/chat/completions", "openai"),
                (f"{ep}/v1/completions", "openai_legacy"),
            ])
    urls_to_try.extend([
        ("http://127.0.0.1:11434/api/generate", "ollama"),
        ("http://localhost:11434/api/generate", "ollama")
    ])

    last_err = None
    for url, proto in urls_to_try:
        try:
            if proto == "ollama":
                r = requests.post(url, json=json_data, timeout=timeout)
                if r.status_code == 200:
                    return r.json().get("response", "").strip()
            elif proto == "openai":
                payload = {
                    "model": json_data.get("model", ""),
                    "messages": [{"role": "user", "content": json_data.get("prompt", "")}],
                    "temperature": json_data.get("options", {}).get("temperature", 0.1),
                }
                r = requests.post(url, json=payload, timeout=timeout)
                if r.status_code == 200:
                    choices = r.json().get("choices", [])
                    if choices and "message" in choices[0]:
                        return choices[0]["message"].get("content", "").strip()
            elif proto == "openai_legacy":
                payload = {
                    "model": json_data.get("model", ""),
                    "prompt": json_data.get("prompt", ""),
                    "temperature": json_data.get("options", {}).get("temperature", 0.1),
                }
                r = requests.post(url, json=payload, timeout=timeout)
                if r.status_code == 200:
                    choices = r.json().get("choices", [])
                    if choices and "text" in choices[0]:
                        return choices[0].get("text", "").strip()
        except Exception as e:
            last_err = e
    if last_err:
        raise last_err
    return ""

def clean_translation_output(raw_text: str, target_lang: str = "Polish") -> str:

    if not raw_text:
        return ""


    lines = [line.strip() for line in raw_text.strip().splitlines() if line.strip()]
    text = lines[0] if lines else ""
    if not text:
        return ""


    text = re.sub(r'^\s*(?:[\*\-•]\s+|\d{1,3}[\.\)]\s+)', '', text)

    prefix_pattern = (
        r'^(?:Translation|Translated|Tłumaczenie|'
        r'Here is(?:\s+the)?\s+translation|Polish|Polski|Output|'
        r'Wersja polska|Oto tłumaczenie|Odpowiedź|Dialog|Subtitles?|Line|'
        r'翻訳|訳|翻译|翻譯|译文|譯文|答案|回答|번역|답변)'
        r'\s*[:：]\s*'
    )
    text = re.sub(prefix_pattern, '', text, flags=re.IGNORECASE)
    text = re.sub(r'^\s*(?:[\*\-•]\s+|\d{1,3}[\.\)]\s+)', '', text)
    text = re.sub(prefix_pattern, '', text, flags=re.IGNORECASE)


    text = text.strip('\"\'`„”«»『』「」《》〈〉“”‘’ \t\r\n')


    meta_words = (
        r'zakończenie|koniec|uwaga|dopisek|w domyśle|literal|literalnie|'
        r'dosłownie|męski|żeński|l\.mn|tryb|forma|wskazówka|wyjaśnienie|'
        r'odpowiedź|odp|note|explanation|meaning|context|informal|formal|'
        r'polski|polish|translation|translated|output|source|speaker|'
        r'注|説明|翻訳|訳注|备注|備註|说明|說明|번역|설명'
    )
    text = re.sub(
        r'\s*[\(\[（【](?:.*?(?:' + meta_words + r').*?)[\)\]）】]\s*',
        ' ',
        text,
        flags=re.IGNORECASE
    )

    target = str(target_lang or "").strip().lower()
    latin_targets = {
        'polish', 'polski', 'pl',
        'english', 'en',
        'german', 'de',
        'french', 'fr',
        'spanish', 'es',
        'italian', 'it',
        'dutch', 'nl',
        'czech', 'cs',
        'turkish', 'tr',
    }
    is_latin_target = target in latin_targets

    if is_latin_target:
        has_latin = bool(re.search(
            r'[A-Za-zÀ-ÖØ-öø-ÿĄĆĘŁŃÓŚŹŻąćęłńóśźżČčŠšŽžŘřĎďŤťŇňĚě]',
            text
        ))
        if has_latin:
            text = re.sub(
                r'(?:(?<=\s)|^)'
                r'[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af]'
                r'{1,3}'
                r'(?=\s|$)',
                '',
                text
            )


    text = re.sub(r'[ \t]{2,}', ' ', text).strip()
    text = text.strip('\"\'`„”«»『』「」《》〈〉“”‘’ \t\r\n')


    if not any(ch.isalnum() for ch in text):
        return ""

    if (
        is_latin_target
        and len(text) > 1
        and text[0].islower()
        and not text.startswith(('...', 'http'))
    ):
        text = text[0].upper() + text[1:]

    return text

def translate_line_with_ollama(
    line: str,
    i: int,
    total: int,
    detected_lang: str,
    tgt_lang: str,
    model_name: str,
    system_prompt: str,
    temperature: float,
    context: str,
    custom_endpoint: str = None
) -> str:
    base_prompt = system_prompt.strip() or (
        f"You are a professional movie subtitle translator.\n"
        f"Source language: {detected_lang} (auto-detected by Whisper).\n"
        f"Translate exactly one subtitle dialogue line from {detected_lang} to {tgt_lang}.\n"
        "Return ONLY the direct dialogue line translation. NEVER include notes, brackets, parentheses (e.g. no (zakończenie)), explanations, or untranslated characters. Do NOT wrap output in quotes."
    )
    base_prompt = base_prompt.replace("{source_lang}", detected_lang).replace("{target_lang}", tgt_lang)
    base_prompt = base_prompt.replace("{context}", context or "")
    base_prompt = base_prompt.replace("{detected_lang}", detected_lang)

    if "{text}" in base_prompt:
        prompt = base_prompt.replace("{text}", line)
    else:
        prompt = (
            f"{base_prompt}\n\n"
            f"Source language: {detected_lang}\n"
            f"Target language: {tgt_lang}\n"
            f"Line to translate: {line}\n\nTranslation:"
        )

    max_retries = 3
    best_translation = ""

    for attempt in range(max_retries):
        if state.cancel_flags["translate"]:
            return ""
        try:
            if attempt == 0:
                state.add_log(f"  [{i+1}/{total}] {line[:40]}...")

            raw_text = _post_ai_generate(
                json_data={
                    "model": model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": temperature + (attempt * 0.2), "num_ctx": 2048}
                },
                custom_endpoint=custom_endpoint,
                timeout=120
            )

            first_line = clean_translation_output(raw_text, target_lang=tgt_lang)

            if not first_line:
                state.add_log(f"  ⚠️ Pusta lub tylko interpunkcyjna odpowiedź AI, ponawiam (próba {attempt+1}/{max_retries})...")
                continue

            # Anti-hallucination checks
            tgt_len = len(first_line)
            src_len = len(line)
            if src_len < 10 and tgt_len > 40:
                state.add_log(f"  🔁 Wykryto halucynację AI ({src_len} vs {tgt_len} znaków). Ponawiam...")
                continue
            if tgt_len > (src_len * 5 + 30):
                state.add_log(f"  🔁 Wykryto zbyt długą frazę ({src_len} vs {tgt_len} znaków). Ponawiam...")
                continue

            best_translation = first_line
            break

        except Exception as e:
            state.add_log(f"  ❌ Linia {i+1} błąd w komunikacji z AI (próba {attempt+1}): {e}")

    return best_translation



@translate_bp.route("/api/translate", methods=["POST"])
def run_translation():

    if not state.active_project:
        return jsonify({"error": "Brak aktywnego projektu"}), 400

    df = state.get_df()
    if df.empty:
        return jsonify({"error": "Nie wczytano żadnych napisów"}), 400

    data = request.get_json() or {}
    model_name = data.get("model", "")
    system_prompt = data.get("prompt", "")
    temperature = float(data.get("temperature", 0.1))


    if not model_name:
        from modules.app import get_ollama_models_list
        models = get_ollama_models_list()
        model_name = models[0] if models else "microai/suzume-llama3"


    state.translate_done = False
    state.translate_total = len(df)
    state.translate_progress = 0
    state.cancel_flags["translate"] = False

    state.active_project.ollama_model = model_name
    state.active_project.temperature = temperature
    state.active_project.prompt = system_prompt
    state.active_project.save()

    def run_worker():
        try:
            texts = df["Original"].tolist()
            detected_lang = getattr(state.active_project, "detected_lang", "") or state.active_project.source_lang
            tgt_lang = state.active_project.target_lang
            context = state.active_project.context
            ai_endpoint = getattr(state.active_project, "ai_endpoint", None)

            state.add_log(f"🌐 Tłumaczenie {detected_lang}→{tgt_lang} | model={model_name} | {len(texts)} linii")


            subtitles = state.active_project.subtitles
            total = len(subtitles)
            for i, item in enumerate(subtitles):
                if state.cancel_flags["translate"]:
                    state.add_log("❌ Tłumaczenie zostało anulowane przez użytkownika.")
                    break

                original_line = (item.get("Original") or "").strip()
                if not original_line:
                    state.translate_progress = i + 1
                    continue


                if item.get("Ignore", False):
                    item["Translation"] = original_line
                    item["Edited"] = False
                    state.translate_progress = i + 1
                    continue

                translated = translate_line_with_ollama(
                    original_line, i, total, detected_lang, tgt_lang,
                    model_name, system_prompt, temperature, context,
                    custom_endpoint=ai_endpoint
                )

                if state.cancel_flags["translate"]:
                    break


                if translated and translated.strip():
                    item["Translation"] = translated
                    item["Edited"] = True
                    state.add_log(f"  ✅ [{i+1}/{total}] → {translated[:50]}")
                else:
                    item["Translation"] = original_line
                    item["Edited"] = False
                    state.add_log(f"  ⚠️ [{i+1}/{total}] empty translation, kept original")

                state.active_project.save()
                state.translate_progress = i + 1

            state.add_log(f"✅ Tłumaczenie zakończone — przetworzono {state.translate_progress} linii.")

            # Clean Ollama cache
            try:
                _post_ai_generate({"model": model_name, "keep_alive": 0}, custom_endpoint=ai_endpoint, timeout=5)
            except Exception:
                pass
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            state.add_log(f"❌ Błąd krytyczny podczas tłumaczenia: {e}")
        finally:
            state.translate_done = True
            state.cancel_flags["translate"] = False

    t = threading.Thread(target=run_worker, daemon=True)
    t.start()
    return jsonify({"status": "started"})


def unload_ai_model(custom_endpoint: str = None):
    """Immediately unloads models from GPU VRAM (Ollama keep_alive=0) and frees PyTorch memory"""
    model_name = getattr(state.active_project, "ollama_model", "") if state.active_project else ""
    ep = custom_endpoint or (getattr(state.active_project, "ai_endpoint", None) if state.active_project else None)

    urls = []
    if ep:
        urls.append(f"{ep.rstrip('/')}/api/generate")
    urls.extend(["http://127.0.0.1:11434/api/generate", "http://localhost:11434/api/generate"])

    for url in urls:
        try:
            if model_name:
                requests.post(url, json={"model": model_name, "keep_alive": 0}, timeout=2)
        except Exception:
            pass

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()


@translate_bp.route("/api/translate/status", methods=["GET"])
def get_translate_status():

    return jsonify({
        "done": state.translate_done,
        "progress": state.translate_progress,
        "total": state.translate_total,
    })


@translate_bp.route("/api/translate/cancel", methods=["POST"])
def cancel_translation():
    """Cancel active translation task and immediately free GPU memory"""
    state.cancel_flags["translate"] = True
    state.add_log("⏳ Żądanie anulowania tłumaczenia zostało wysłane. Zwalnianie zasobów GPU...")
    ep = getattr(state.active_project, "ai_endpoint", None) if state.active_project else None
    threading.Thread(target=unload_ai_model, args=(ep,), daemon=True).start()
    return jsonify({"success": True})



@translate_bp.route("/api/analyze_context", methods=["POST"])
def analyze_context_api():

    if not state.active_project:
        return jsonify({"error": "Brak aktywnego projektu"}), 400

    df = state.get_df()
    if df.empty:
        return jsonify({"error": "Nie wczytano żadnych napisów"}), 400

    data = request.get_json() or {}
    model_name = data.get("model", "")

    if not model_name:
        from modules.app import get_ollama_models_list
        models = get_ollama_models_list()
        model_name = models[0] if models else "microai/suzume-llama3"

    def run_worker():
        try:
            texts = df["Original"].tolist()
            src_lang = state.active_project.source_lang
            ai_endpoint = getattr(state.active_project, "ai_endpoint", None)
            state.add_log(f"🔍 Rozpoczęcie analizy kontekstu wideo ({len(texts)} linii)...")

            context_notes = []
            max_lines_per_batch = 150

            for i in range(0, len(texts), max_lines_per_batch):
                batch = texts[i:i + max_lines_per_batch]
                batch_text = "\n".join([f"[{j + 1}] {line}" for j, line in enumerate(batch)])
                prompt = (
                    f"Analyze the following movie dialogue fragment in {src_lang}.\n"
                    f"Extract character names, genders, relationships, scene context, speaking style.\n"
                    f"Fragment:\n{batch_text}\nContext notes:"
                )
                try:
                    state.add_log(f"  Analiza partii {i // max_lines_per_batch + 1} przez AI...")
                    notes = _post_ai_generate(
                        json_data={"model": model_name, "prompt": prompt, "stream": False, "options": {"temperature": 0.3}},
                        custom_endpoint=ai_endpoint,
                        timeout=180
                    )
                    context_notes.append(f"--- Part {i // max_lines_per_batch + 1} ---\n{notes}\n")
                    state.add_log(f"  Partia {i // max_lines_per_batch + 1} przeanalizowana pomyślnie.")
                except Exception as e:
                    state.add_log(f"  ❌ Błąd analizy kontekstu: {e}")
                    context_notes.append("")

            state.active_project.context = "\n".join(context_notes)
            state.active_project.save()
            state.add_log("✅ Analiza kontekstu zakończona sukcesem.")

            # Clean Ollama cache
            try:
                _post_ai_generate({"model": model_name, "keep_alive": 0}, custom_endpoint=ai_endpoint, timeout=5)
            except Exception:
                pass

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            state.add_log(f"❌ Błąd podczas analizy kontekstu: {e}")


    t = threading.Thread(target=run_worker, daemon=True)
    t.start()
    return jsonify({"status": "started"})


@translate_bp.route("/api/context", methods=["GET"])
def get_context():
    """Retrieve active project context notes"""
    if state.active_project:
        return jsonify({"context": state.active_project.context})
    return jsonify({"context": ""})
