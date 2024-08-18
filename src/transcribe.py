import os
import json
from groq import Groq
from typing import Optional
from pydantic import BaseModel
from faster_whisper import WhisperModel


class Translation(BaseModel):
    text: str
    comments: Optional[str] = None


class Transcribe:
    def __init__(self, api_key):
        self.client = Groq(api_key=api_key)

    def whisper_transcribe(self, audio_chunk):
        model = WhisperModel("base", device="cpu", compute_type="int8", cpu_threads=int(os.cpu_count() / 2)) 
        segments, info = model.transcribe(audio_chunk, beam_size=5) 
        speech_text = " ".join([segment.text for segment in segments]) 
        return speech_text

    def groq_tinker(self, query, person):
        chat_completion = self.client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": f"You are a helpful assistant that changes text spoken by user to text that might have been spoken by a celebrity."
                            f"You will only reply with the new text and nothing else in JSON."
                            f" The JSON object must use the schema: {json.dumps(Translation.model_json_schema(), indent=2)}",
                },
                {
                    "role": "user",
                    "content": f"Change '{query}' to something spoken by {person}."
                }
            ],
            model="mixtral-8x7b-32768",
            temperature=0.2,
            max_tokens=1024,
            stream=False,
            response_format={"type": "json_object"},
        )
        return Translation.model_validate_json(chat_completion.choices[0].message.content)

    def groq_transcribe(self, filename):
        with open(filename, "rb") as file:
            transcription = self.client.audio.transcriptions.create(
              file=(filename, file.read()),
              model="whisper-large-v3",
              prompt="Specify context or spelling",  # Optional
              response_format="json",  # Optional
              language="en",  # Optional
              temperature=0.0  # Optional
            )
            return transcription.text


if __name__ == '__main__':
    transcribe = Transcribe()
    filename = os.path.dirname(__file__) + "/output.wav"
    print(transcribe.transcribe(filename))
