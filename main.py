"""
talanoa-live: Terminal POC

Records short chunks of audio from the microphone, transcribes them from
Fijian using Meta's MMS model, translates the result to English using
NLLB-200, and prints both to the console.

This is a v1 script: it records in fixed-length chunks (no voice activity
detection yet), which is simpler but means it'll transcribe silence too.
VAD-based chunking is a planned next step.
"""

import numpy as np
import sounddevice as sd
import torch
from transformers import (
    Wav2Vec2ForCTC,
    AutoProcessor,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)

# ---- Config ----
SAMPLE_RATE = 16000        # MMS expects 16kHz audio
CHUNK_SECONDS = 5          # length of each recorded chunk
SOURCE_LANG_MMS = "fij"    # Fijian language code for MMS (Massively Multilingual Speech)
SOURCE_LANG_NLLB = "fij_Latn"  # Fijian language code for NLLB (No Langauage Left Behind)
TARGET_LANG_NLLB = "eng_Latn"  # English language code for NLLB (No Langauage Left Behind)


def load_asr_model():
    """Load Meta MMS ASR model + processor, set to Fijian adapter."""
    print("Loading ASR model (MMS)... this may take a while the first time.")
    processor = AutoProcessor.from_pretrained("facebook/mms-1b-all")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/mms-1b-all")

    # Point the model at the Fijian language adapter
    processor.tokenizer.set_target_lang(SOURCE_LANG_MMS)
    model.load_adapter(SOURCE_LANG_MMS)

    return processor, model


def load_translation_model():
    """Load NLLB-200 translation model + tokenizer."""
    print("Loading translation model (NLLB-200)...")
    model_name = "facebook/nllb-200-distilled-600M"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model


def transcribe(audio, processor, model):
    """Run MMS ASR on a numpy audio array, return Fijian text."""
    inputs = processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs).logits
    ids = torch.argmax(outputs, dim=-1)[0]
    text = processor.decode(ids)
    return text


def translate(text, tokenizer, model):
    """Translate Fijian text to English using NLLB."""
    if not text.strip():
        return ""
    tokenizer.src_lang = SOURCE_LANG_NLLB
    inputs = tokenizer(text, return_tensors="pt")
    forced_bos_token_id = tokenizer.convert_tokens_to_ids(TARGET_LANG_NLLB)
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            forced_bos_token_id=forced_bos_token_id,
            max_new_tokens=200,
            max_length=None,
        )
    translation = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
    return translation


def record_chunk(seconds=CHUNK_SECONDS):
    """Record a fixed-length audio chunk from the default microphone."""
    audio = sd.rec(
        int(seconds * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
    )
    sd.wait()
    return audio.flatten()


def main():
    asr_processor, asr_model = load_asr_model()
    mt_tokenizer, mt_model = load_translation_model()

    print("\nReady! Speak in Fijian after each prompt.")
    print("Press Ctrl+C to stop.\n")

    try:
        while True:
            input(f"Press Enter, then speak for {CHUNK_SECONDS} seconds...")
            print("Recording...")
            audio = record_chunk()

            print("Transcribing...")
            fijian_text = transcribe(audio, asr_processor, asr_model)
            print(f"  Fijian:  {fijian_text}")

            print("Translating...")
            english_text = translate(fijian_text, mt_tokenizer, mt_model)
            print(f"  English: {english_text}\n")

    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__": #Checks if this file was run directly, or imported by something else.
    main()