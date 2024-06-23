from transliterate import transliterate
import numpy as np
from pydub import AudioSegment
from transformers import VitsModel, AutoTokenizer
from IPython.display import Audio
import torch

model = VitsModel.from_pretrained("facebook/mms-tts-uzb-script_cyrillic")
tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-uzb-script_cyrillic")

def tts_full(text):
  text=transliterate(text, 'cyrillic')
  inputs = tokenizer(text, return_tensors="pt")

  with torch.no_grad():
      output = model(**inputs).waveform
  return output

data = tts_full("input file")

data_numpy = data.detach().cpu().numpy()
if data_numpy.dtype != np.int16:
    data_numpy = (data_numpy * 32767).astype(np.int16)
audio_data = data_numpy.tobytes()
audio_segment = AudioSegment(
    data=audio_data,
    frame_rate=model.config.sampling_rate,
    sample_width=2,
    channels=1
)
audio_segment.export("result_21.mp3", format="mp3")
