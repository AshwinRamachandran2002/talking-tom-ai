import os
import torch
import torchaudio
import numpy as np

from gtts import gTTS  
from pydub import AudioSegment
from scipy.io.wavfile import write as write_wav

os.environ["SUNO_USE_SMALL_MODELS"] = "1"

from encodec.utils import convert_audio
from bark.generation import load_codec_model

from hubert.bark_hubert_quantizer.hubert_manager import HuBERTManager
from hubert.bark_hubert_quantizer.pre_kmeans_hubert import CustomHubert
from hubert.bark_hubert_quantizer.customtokenizer import CustomTokenizer

from bark import SAMPLE_RATE, generate_audio, preload_models
from bark.generation import codec_decode, generate_coarse, generate_fine, generate_text_semantic


class Pitch:
    def setup_voice_clone(self, voice_name, voice_path):
        if os.path.exists(f'bark/new_voice_clone/{voice_name}.npz'):
            return

        device = 'cuda'
        model = load_codec_model(use_gpu=True if device == 'cuda' else False)

        hubert_manager = HuBERTManager()
        hubert_manager.make_sure_hubert_installed()
        hubert_manager.make_sure_tokenizer_installed()

        tokenizer = CustomTokenizer.load_from_checkpoint('data/models/hubert/tokenizer.pth').to(device)  # Automatically uses the right layers
        hubert_model = CustomHubert(checkpoint_path='data/models/hubert/hubert.pt').to(device)

        audio_filepath = voice_path
        wav, sr = torchaudio.load(audio_filepath, backend='soundfile')
        wav = convert_audio(wav, sr, model.sample_rate, model.channels)
        wav = wav.to(device)

        semantic_vectors = hubert_model.forward(wav, input_sample_hz=model.sample_rate)
        semantic_tokens = tokenizer.get_token(semantic_vectors)

        with torch.no_grad():
            encoded_frames = model.encode(wav.unsqueeze(0))
        codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1).squeeze()  # [n_q, T]
        codes = codes.cpu().numpy()

        semantic_tokens = semantic_tokens.cpu().numpy()
        output_path = 'bark/new_voice_clone/' + voice_name + '.npz'
        np.savez(output_path, fine_prompt=codes, coarse_prompt=codes[:2, :], semantic_prompt=semantic_tokens)        


    def pitch_voice(self, text_prompt, voice_name, voice_path):
        self.setup_voice_clone(voice_name, voice_path)

        preload_models()
        audio_array = generate_audio(text_prompt, history_prompt=voice_name, text_temp=0.7, waveform_temp=0.7)

        x_semantic = generate_text_semantic(
            text_prompt,
            history_prompt=voice_name,
            temp=0.7,
            top_k=50,
            top_p=0.95,
        )

        x_coarse_gen = generate_coarse(
            x_semantic,
            history_prompt=voice_name,
            temp=0.7,
            top_k=50,
            top_p=0.95,
        )

        x_fine_gen = generate_fine(
            x_coarse_gen,
            history_prompt=voice_name,
            temp=0.5,
        )

        audio_array = codec_decode(x_fine_gen)

        filepath = "output.wav"
        write_wav(filepath, SAMPLE_RATE, audio_array)
        return filepath

    def pitch_normal(self, file_name):
        sound = AudioSegment.from_file(file_name, format="wav")

        octaves = 0.3
        new_sample_rate = int(sound.frame_rate * (2.0 ** octaves))
        hipitch_sound = sound._spawn(sound.raw_data, overrides={'frame_rate': new_sample_rate})
        hipitch_sound = hipitch_sound.set_frame_rate(44100)
        hipitch_sound.export("output_pitch_normal.wav", format="wav")
        return "output.wav"

    def pitch_vanilla(self, text_prompt):
        my_obj = gTTS(text=text_prompt, lang="en") 
        file_name = "output.mp3"
        my_obj.save(file_name)
        return file_name

    def pitch(self, text_prompt, voice_name, voice_path=None):
        # The original Talking Tom
        if voice_name == "original":
            return self.pitch_normal(voice_path)
        # TTS
        elif voice_name == "Simple TTS":
            return self.pitch_vanilla(text_prompt)
        # Voice Cloned
        else:
            return self.pitch_voice(text_prompt, voice_name, voice_path)
