import torch
from TTS.api import TTS

# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Rodando usando: {}".format(device))

# List available 🐸TTS models
# print(TTS().list_models())

# Init TTS
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

text = "Olá, estamos atualmente em busca de soluções de telefonia que atendam às nossas necessidades e gostaríamos de explorar as opções oferecidas pela sua empresa"
speaker = "speakers/jefao.wav"

# Run TTS
# ❗ Since this model is multi-lingual voice cloning model, we must set the target speaker_wav and language
# Text to speech list of amplitude values as output
# wav = tts.tts(text=text, speaker_wav=speaker, language="pt")
# Text to speech to a file
tts.tts_to_file(text=text, 
                # speaker_wav=speaker,
                speaker="Ana Florence", 
                language="pt", 
                top_k=30,
                top_p=0.7,
                file_path="outputs/busca-empresa8.mp3")