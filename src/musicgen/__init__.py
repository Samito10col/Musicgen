import torch
from scipy.io.wavfile import write
from transformers import AutoProcessor, MusicgenForConditionalGeneration

MODEL_ID = "facebook/musicgen-small"
DEVICE = "cpu"
PROMPT = "Create a romantic Dominican bachata with sensual acoustic guitars, melodic requinto solos, rhythmic bongos, güira, deep bass, and a catchy danceable rhythm. Emotional, passionate, modern Latin sound with professional studio quality."

print("cargando musicgen..")

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = MusicgenForConditionalGeneration.from_pretrained(MODEL_ID)

model = model.to(DEVICE)
model.eval()
print("Preparando promt...")
inputs = processor(text=[PROMPT],
                    padding = True,
                    return_tensors = "pt"
                     ).to(DEVICE)

print("Generando musica...")

with torch.inference_mode():
    audio = model.generate(
        **inputs,
        do_sample=True,
        guidance_scale=3.0,
        max_new_tokens=1500
        )
sample_rate = model.config.audio_encoder.sampling_rate
waveform = audio[0,0].cpu().float().numpy()

write("musica.wav",sample_rate, waveform)
print("Musica Generada")
