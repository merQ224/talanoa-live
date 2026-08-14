# talanoa-live

Real-time speech-to-text and translation for Fijian, built as a proof-of-concept.

Speaks in Fijian → transcribes it → translates it to English, with a target of near-live latency (a few seconds per chunk, not true simultaneous interpretation).

## Status

**Proof of concept.** Not production-ready. Accuracy will be rough — Fijian is a low-resource language for ASR/MT, so expect errors, especially with dialect-specific vocabulary or code-switching with English.

## How it works

```
Mic input
   → Voice Activity Detection (chunks audio into short segments)
   → ASR: Meta MMS (facebook/mms-1b-all, Fijian adapter) — speech → Fijian text
   → MT: NLLB-200 (facebook/nllb-200-distilled-600M) — Fijian text → English text
   → Output: terminal / subtitles / optional TTS
```

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

- [ ] Terminal pipeline: mic capture → VAD → ASR → MT, printed to console
- [ ] Evaluate ASR/MT accuracy on real Fijian speech samples
- [ ] Simple web frontend with live subtitles
- [ ] Optional: TTS playback of English translation
- [ ] Optional: reverse direction (English → Fijian)

## Setup

```bash
git clone https://github.com/merQ224/talanoa-live.git
cd talanoa-live
pip install -r requirements.txt
```

_(Instructions will be filled in as the pipeline comes together.)_

## Why

Most mainstream speech tools (Whisper, Google STT) don't support Fijian. Meta's MMS project covers 1,100+ languages including Fijian, making this newly feasible to prototype with fully open-source tools.