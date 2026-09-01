

import re
from pathlib import Path


def _read_subtitle_text(path: str) -> str:

    encodings = [
        "utf-8-sig",
        "utf-8",
        "cp1250",
        "latin2",
        "iso-8859-2",
        "cp1252",
    ]

    last_error = None

    for enc in encodings:
        try:
            with open(path, "r", encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError as e:
            last_error = e

    raise ValueError(
        f"Cannot read subtitle file: {path}. "
        f"Tried encodings: {', '.join(encodings)}. "
        f"Last error: {last_error}"
    )


def parse_ass_file(ass_path: str) -> list:

    content = _read_subtitle_text(ass_path)

    content = (
        content
        .replace("\r\n", "\n")
        .replace("\r", "\n")
    )

    lines = content.split("\n")

    events = []
    in_events = False

    for raw_line in lines:
        line = raw_line.strip("\ufeff")

        if line.strip().lower() == "[events]":
            in_events = True
            continue

        if not in_events:
            continue

        if not line.lstrip().lower().startswith("dialogue:"):
            continue

        parts = line.split(",", 9)

        if len(parts) < 10:
            continue

        start = parts[1].strip()
        end = parts[2].strip()
        text = parts[9].strip()

        # Remove ASS formatting tags.
        text = re.sub(r"\{[^}]*\}", "", text)

        # ASS line-break / hard-space escapes.
        text = (
            text
            .replace("\\N", " ")
            .replace("\\n", " ")
            .replace("\\h", " ")
        )

        text = re.sub(r"\s+", " ", text).strip()

        if text:
            events.append({
                "start": start,
                "end": end,
                "text": text,
            })

    return events


def parse_srt_file(srt_path: str) -> list:

    content = _read_subtitle_text(srt_path)

    content = (
        content
        .replace("\r\n", "\n")
        .replace("\r", "\n")
        .strip()
    )

    if not content:
        return []

    blocks = re.split(
        r"\n[ \t]*\n+",
        content,
    )

    events = []

    timing_re = re.compile(
        r"^\s*"
        r"(?P<start>\d{1,2}:\d{2}:\d{2}(?:[,.]\d{1,3})?)"
        r"\s*-->\s*"
        r"(?P<end>\d{1,2}:\d{2}:\d{2}(?:[,.]\d{1,3})?)"
    )

    for block in blocks:
        lines = [
            line.strip("\ufeff")
            for line in block.split("\n")
            if line.strip()
        ]

        if not lines:
            continue

        timing_index = None
        timing_match = None

        for idx, line in enumerate(lines[:4]):
            match = timing_re.match(line)

            if match:
                timing_index = idx
                timing_match = match
                break

        if timing_index is None or timing_match is None:
            continue

        text_lines = lines[timing_index + 1:]

        if not text_lines:
            continue

        text = " ".join(text_lines).strip()


        text = re.sub(r"<[^>]+>", "", text)

        text = (
            text
            .replace("\u200b", "")
            .replace("\ufeff", "")
        )

        text = re.sub(r"\s+", " ", text).strip()

        if not text:
            continue

        events.append({
            "start": timing_match.group("start"),
            "end": timing_match.group("end"),
            "text": text,
        })

    return events


def parse_subtitle_file(path: str) -> list:

    subtitle_path = Path(path)
    suffix = subtitle_path.suffix.lower()

    if suffix == ".srt":
        return parse_srt_file(str(subtitle_path))

    if suffix in (".ass", ".ssa"):
        return parse_ass_file(str(subtitle_path))

    # Fallback content sniffing in case the uploaded filename lost
    # or has an unusual extension.
    content = _read_subtitle_text(str(subtitle_path))

    if "-->" in content:
        return parse_srt_file(str(subtitle_path))

    if "[Events]" in content and "Dialogue:" in content:
        return parse_ass_file(str(subtitle_path))

    raise ValueError(
        f"Unsupported subtitle format: {subtitle_path.name}. "
        "Expected SRT, ASS or SSA."
    )
