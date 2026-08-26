# -*- coding: utf-8 -*-
"""
File loading and parsing utilities for subtitles
"""
import re

def parse_ass_file(ass_path: str) -> list:
    encodings = ['utf-8', 'cp1250', 'latin2', 'iso-8859-2']
    content = None
    for enc in encodings:
        try:
            with open(ass_path, 'r', encoding=enc) as f:
                content = f.read()
            break
        except UnicodeDecodeError:
            continue
    if content is None:
        raise ValueError("Cannot read ASS file")
    lines = content.split('\n')
    events = []
    in_events = False
    for line in lines:
        if line.startswith('[Events]'):
            in_events = True
            continue
        if in_events and line.startswith('Dialogue:'):
            parts = line.split(',', 9)
            if len(parts) >= 10:
                start = parts[1].strip()
                end = parts[2].strip()
                text = parts[9].strip()
                text = re.sub(r'\{[^}]*\}', '', text).replace('\\N', ' ').strip()
                if text:
                    events.append({'start': start, 'end': end, 'text': text})
    return events

def parse_srt_file(srt_path: str) -> list:
    encodings = ['utf-8', 'cp1250', 'latin2', 'iso-8859-2']
    content = None
    for enc in encodings:
        try:
            with open(srt_path, 'r', encoding=enc) as f:
                content = f.read()
            break
        except UnicodeDecodeError:
            continue
    if content is None:
        raise ValueError("Cannot read SRT file")
    blocks = re.split(r'\n\s*\n', content.strip())
    events = []
    for block in blocks:
        lines = block.split('\n')
        if len(lines) >= 3:
            times = lines[1].split(' --> ')
            if len(times) == 2:
                start = times[0].strip()
                end = times[1].strip()
                text = " ".join(lines[2:]).strip()
                text = re.sub(r'<[^>]+>', '', text)
                if text:
                    events.append({'start': start, 'end': end, 'text': text})
    return events
