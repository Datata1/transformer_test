from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torchaudio

# Lade den Whisper-Processor und das Modell und stelle sicher, dass es auf der CPU läuft
processor = WhisperProcessor.from_pretrained("microsoft/Phi-4-multimodal-instruct")
model = WhisperForConditionalGeneration.from_pretrained("microsoft/Phi-4-multimodal-instruct", device_map="cpu")

# Lade die Audiodatei
waveform, sample_rate = torchaudio.load("path_to_audio_file.wav")

# Preprocessiere das Audio (z. B. Resampling)
audio_input = processor(waveform, return_tensors="pt", sampling_rate=sample_rate)

# Generiere die Transkription
predicted_ids = model.generate(audio_input["input_features"])

# Decodiere die Transkription
transcription = processor.decode(predicted_ids[0], skip_special_tokens=True)

print("Transkription:", transcription)
