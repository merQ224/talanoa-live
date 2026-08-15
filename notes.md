## What each package does:

torch — PyTorch, runs the ML models
transformers — Hugging Face library, loads MMS (ASR) and NLLB (translation)
sounddevice — captures audio from your microphone
numpy — numerical processing for audio arrays
soundfile — reading/writing audio data
sentencepiece — tokenizer NLLB depends on
accelerate — helps Transformers run models efficiently

----------

## AutoProcessor and Wav2Vec2ForCTC — what are these?

Wav2Vec2ForCTC — the actual speech recognition model (neural network). CTC (Connectionist Temporal Classification) handles aligning audio to text even when lengths don't match.
AutoProcessor — converts raw audio into the format the model needs, and converts the model's output back into text.

.from_pretrained("facebook/mms-1b-all") — downloads (or loads from cache) the pretrained model by name from Hugging Face.

set_target_lang / load_adapter — MMS is one shared model with a small language-specific "adapter" for each of its 1,100+ languages. These two lines swap in the Fijian adapter so the model recognizes Fijian instead of defaulting to another language.

-----------

https://huggingface.co/facebook/mms-1b-all#example

-----------

## Known issue: greeting translation inaccurate

Tested: "Bula vinaka" (Fijian greeting, ~"hello"/"welcome")

- ASR (MMS): correctly transcribed as "bulavinaka" ✅
- Translation (NLLB): incorrectly translated as "good luck" ❌

Likely cause: NLLB is low-resource for Fijian and doesn't have enough
training data to learn idiomatic/non-literal phrases like greetings —
it's guessing based on limited patterns rather than "knowing" the phrase.

Next steps to investigate:
- Test with longer, full sentences (MT tends to do better with more context)
- Consider a small manual override dictionary for common fixed phrases
  (greetings, idioms) that bypasses NLLB for known cases
- Keep testing more phrases to see how widespread the issue is