#!/usr/bin/env python3
"""Pobiera audio z YouTube i automatycznie buduje dataset głosu."""
import argparse, shutil, sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import yt_dlp
    from voice_dataset_builder import build_dataset, BuildConfig
except ImportError as e:
    print(f"BŁĄD: {e}. Upewnij się, że yt-dlp jest zainstalowany, a voice_dataset_builder.py istnieje.")
    sys.exit(1)

def download_youtube_audio(urls: list[str], output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    downloaded_files = []
    
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': str(output_dir / '%(id)s.%(ext)s'),
        'quiet': False,
        'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'wav', 'preferredquality': '192'}]
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        for url in urls:
            try:
                print(f"\n⬇️ Pobieranie: {url}")
                info = ydl.extract_info(url, download=True)
                filename = ydl.prepare_filename(info)
                wav_filename = Path(filename).with_suffix('.wav')
                
                if wav_filename.exists():
                    downloaded_files.append(wav_filename)
                else:
                    possible_files = list(output_dir.glob(f"{Path(filename).stem}.*"))
                    if possible_files: downloaded_files.append(possible_files[0])
            except Exception as e:
                print(f"⚠️ Błąd pobierania {url}: {e}")
    return downloaded_files

def main():
    parser = argparse.ArgumentParser(description="YouTube -> ViDubb Dataset Builder")
    parser.add_argument("--name", required=True, help="Nazwa osoby/głosu (np. Zmaslo)")
    parser.add_argument("--urls-file", required=True, help="Plik .txt z linkami YouTube")
    parser.add_argument("--reference", type=Path, help="Opcjonalny plik wzorcowy (.wav)")
    parser.add_argument("--clean", action="store_true", help="Użyj Demucs do usunięcia tła/muzyki")
    parser.add_argument("--keep-downloads", action="store_true", help="Nie usuwaj pobranych plików po zakończeniu")
    args = parser.parse_args()
    
    if not Path(args.urls_file).exists():
        print(f"BŁĄD: Nie znaleziono pliku {args.urls_file}")
        sys.exit(1)
        
    with open(args.urls_file, 'r', encoding='utf-8') as f:
        urls = [line.strip() for line in f if line.strip() and line.startswith('http')]
        
    if not urls:
        print("BŁĄD: Brak prawidłowych linków w pliku.")
        sys.exit(1)
        
    temp_dir = PROJECT_ROOT / "yt_temp_downloads"
    print(f"\n📥 Rozpoczynam pobieranie {len(urls)} plików do: {temp_dir}")
    
    downloaded_files = download_youtube_audio(urls, temp_dir)
    if not downloaded_files:
        print("❌ Nie pobrano żadnych plików. Kończę działanie.")
        sys.exit(1)
        
    print(f"\n✅ Pobrano {len(downloaded_files)} plików. Rozpoczynam budowanie datasetu...")
    
    config = BuildConfig(
        name=args.name, inputs=downloaded_files, reference=args.reference,
        language="auto", whisper_model="turbo", clean_vocals=args.clean
    )
    
    result_path = build_dataset(config, log=print)
    print(f"\n🎉 GOTOWE! Dataset zapisany w: {result_path}")
    
    if not args.keep_downloads:
        print("🧹 Czyszczenie plików tymczasowych z YouTube...")
        shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == "__main__":
    main()
