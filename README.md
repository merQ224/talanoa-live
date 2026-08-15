# talanoa-live

Real-time speech-to-text and translation for Fijian, built as a proof-of-concept.

Speaks in Fijian → transcribes it → translates it to English, with a target of near-live latency (a few seconds per chunk, not true simultaneous interpretation).

## Status

**Proof of concept — core pipeline working.** Terminal-based v1 (push-to-talk, fixed 5-second chunks) is up and running end-to-end: mic → ASR → translation → printed output.

Accuracy is mixed, as expected for a low-resource language:
- Full, grammatically complete sentences translate well
- Idiomatic/fixed phrases (e.g. greetings) are often mistranslated by NLLB even when transcribed correctly
- Some words are inconsistently transcribed by MMS (occasional word-merging or letter/vowel confusion), though often correct on a retry

See test results and findings in the project's Notion log for detailed before/after examples.

## How it works

```
Mic input
   → Voice Activity Detection (chunks audio into short segments)
   → ASR: Meta MMS (facebook/mms-1b-all, Fijian adapter) — speech → Fijian text
   → MT: NLLB-200 (facebook/nllb-200-distilled-600M) — Fijian text → English text
   → Output: terminal / subtitles / optional TTS
```

Currently implemented: mic capture → fixed-length chunking → ASR → MT → terminal output. VAD-based chunking (in place of fixed-length) is next.

## Tech stack

| Component | Tool |
|---|---|
| ASR | [Meta MMS](https://huggingface.co/facebook/mms-1b-all) (`fij` language adapter) |
| Translation | [NLLB-200](https://huggingface.co/facebook/nllb-200-distilled-600M) (`fij_Latn` → `eng_Latn`) |
| TTS (optional) | [MMS-TTS](https://huggingface.co/facebook/mms-tts-eng) |
| VAD | Silero VAD / webrtcvad |
| Backend | Python, Hugging Face Transformers |
| Frontend | Terminal first, then a lightweight web UI (WebSocket + plain HTML/JS) |

All models are free and open-source, run locally — no API costs.

## Roadmap

**Done:**
- [x] Terminal pipeline: mic capture → ASR → MT, printed to console (fixed-length chunks)
- [x] Initial accuracy testing on real Fijian speech samples

**Next up:**
- [ ] VAD-based chunking (replace fixed 5-second recording with automatic speech-start/speech-stop detection)
- [ ] Fine-tune MMS (ASR) on a specific speaker's voice/dialect using paired audio + transcript data, to improve recognition accuracy
- [ ] Fine-tune NLLB (translation) on a small set of Fijian↔English sentence pairs, targeting known weak spots like idioms and greetings
- [ ] Split `main.py` into modules (`asr.py`, `translation.py`, `audio.py`) once structure stabilizes
- [ ] Simple web frontend with live subtitles

**Later / optional:**
- [ ] TTS playback of English translation
- [ ] Reverse direction (English → Fijian)
- [ ] Manual override dictionary for common fixed phrases/greetings, as a lightweight patch alongside fine-tuning
- [ ] Confidence-based flagging — surface uncertain transcriptions instead of silently passing errors downstream

## Setup

```bash
git clone https://github.com/merQ224/talanoa-live.git
cd talanoa-live
pip install -r requirements.txt
python main.py
```

## Why

Most mainstream speech tools (Whisper, Google STT) don't support Fijian. Meta's MMS project covers 1,100+ languages including Fijian, making this newly feasible to prototype with fully open-source tools.