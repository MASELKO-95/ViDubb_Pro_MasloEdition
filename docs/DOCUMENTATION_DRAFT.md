# viDubb Pro — szkic dokumentacji

> Status: szkic roboczy. Dokument wyznacza strukturę przyszłej dokumentacji;
> opisy techniczne i zrzuty ekranu wymagają jeszcze uzupełnienia.

## 1. Wprowadzenie

viDubb Pro jest lokalną aplikacją do transkrypcji, tłumaczenia i dubbingu
materiałów wideo. Łączy Whisper, lokalne modele tłumaczeniowe, syntezę lub
klonowanie głosu, separację tła audio oraz opcjonalny Wav2Lip.

### Docelowy odbiorca

- osoba uruchamiająca aplikację lokalnie,
- twórca przygotowujący dubbing filmu,
- programista rozwijający backend lub interfejs.

## 2. Najważniejsze funkcje

- import lokalnego filmu albo pobieranie z YouTube,
- automatyczna transkrypcja i wykrywanie mówców,
- tłumaczenie przez lokalny endpoint AI,
- edycja, zatwierdzanie i eksport napisów,
- Edge-TTS oraz klonowanie głosu,
- zachowanie tła i poprawa jakości audio,
- opcjonalny hardsub i synchronizacja ust przez Wav2Lip,
- zapisywanie wielu projektów i lokalnej bazy głosów.

## 3. Wymagania i instalacja

Do opisania w wersji finalnej:

1. wspierane systemy i Python 3.10,
2. FFmpeg, CUDA i minimalna ilość VRAM,
3. instalacja zależności w `.venv`,
4. Ollama i rekomendowane modele,
5. opcjonalne wagi Wav2Lip,
6. pierwsze uruchomienie i pobierane modele.

## 4. Pierwszy projekt — proponowany przewodnik

1. Uruchom `python run_launcher.py`.
2. Utwórz lub otwórz projekt.
3. Wybierz plik wideo albo wklej adres YouTube.
4. Ustaw język, model Whisper i rozpocznij analizę.
   Jeśli znasz liczbę rozmówców, ustaw ją zamiast `Auto` — zwiększa to
   stabilność przypisania głosów w krótkich albo głośnych scenach.
5. Sprawdź automatycznie wczytany film oraz napisy.
6. Wybierz model AI i przetłumacz tekst.
7. Popraw oraz zatwierdź linie.
8. Wybierz silnik głosu i ustawienia audio.
9. Wygeneruj, odsłuchaj i pobierz gotowy film.

### Timeline Review przed renderem

Po wybraniu `Generuj` aplikacja najpierw tworzy osobne próbki audio dla
dialogów. W oknie Timeline Review można:

- porównać czas slotu z rzeczywistą długością TTS,
- odsłuchać audio i odpowiedni fragment filmu,
- poprawić Start/End albo wykonać automatyczny ripple edit,
- zmienić rozmówcę, głos lub tekst i ponownie wygenerować próbki,
- zaakceptować albo pominąć każdą kwestię.

Dopiero przycisk `Renderuj zatwierdzone dialogi` składa finalny film. Linie
oczekujące i odrzucone nie trafiają do finalnej ścieżki dubbingu.

## 5. Przepływ danych

```text
Film/YouTube
  -> pobranie lub upload
  -> Whisper
  -> diarization
  -> edycja i tłumaczenie
  -> TTS / klonowanie głosu
  -> miks z tłem
  -> hardsub / Wav2Lip (opcjonalnie)
  -> gotowy film
```

## 6. Konfiguracja modułów

Planowane podrozdziały:

- Whisper i dobór rozmiaru modelu,
- diarization oraz token Hugging Face,
- endpoint i modele Ollama,
- Edge-TTS, XTTS i baza głosów,
- separacja audio i DSP,
- Wav2Lip oraz wymagane checkpointy.

## 7. Dane lokalne i prywatność

Filmy, audio, projekty, modele i profile głosowe pozostają lokalnie. Katalogi
robocze są wykluczone przez `.gitignore`. Dokumentacja finalna powinna opisać
retencję danych, ręczne czyszczenie i tworzenie kopii projektów.

## 8. Rozwiązywanie problemów

Do rozwinięcia:

- brak `yt-dlp` albo FFmpeg,
- locale inne niż UTF-8,
- brak CUDA lub pamięci GPU,
- niedostępny Ollama,
- brak tokenu Hugging Face,
- brak checkpointu lub kodu Wav2Lip,
- niezgodne wersje bibliotek.

## 9. Dokumentacja dla programistów

Planowane elementy:

- drzewo katalogów i odpowiedzialność modułów,
- cykl życia projektu i stan aplikacji,
- lista endpointów Flask wraz z przykładami,
- kontrakty danych napisów i profili głosu,
- uruchamianie testów i zasady kontrybucji,
- proces wydawania wersji i aktualizator launchera.

## 10. Materiały do dodania

- zrzuty każdego panelu,
- przykładowy projekt od importu do eksportu,
- tabela modeli i zużycia VRAM,
- FAQ,
- schemat architektury,
- lista znanych ograniczeń.
