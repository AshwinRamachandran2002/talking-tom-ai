import os
import dotenv
import streamlit as st
from pitch import Pitch
from transcribe import Transcribe
from audio_recorder_streamlit import audio_recorder


st.set_page_config(page_title='Talking Tom AI', page_icon='🎤')
st.title('Talking Tom AI')

voices = ["Rajini", "Vanilla"]

option = st.selectbox(
   "Voice Type:",
   voices,
   index=None,
   placeholder="Select voice...",
)

dotenv.load_dotenv()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY") # Obtain from user input

API_KEY = st.text_input("Enter your GROQ API Key", GROQ_API_KEY, type="password")

audio_bytes = audio_recorder(
    energy_threshold=(-1.0, 1.0),
    pause_threshold=3.0,
)

if audio_bytes and option:
    st.audio(audio_bytes, format="audio/wav")

    with open('recorded_audio.wav', mode='wb') as f:
        f.write(audio_bytes)

    st.divider()

    with st.spinner('Transcribing...'):
        text = Transcribe(api_key=API_KEY).whisper_transcribe('recorded_audio.wav')
    st.subheader('Transcribed Text')
    st.write(text)

    audio_file = Pitch().pitch(text, option.lower())
    if ".mp3" in audio_file:
        st.audio(audio_file, format="audio/mp3")
    else:
        st.audio(audio_file, format="audio/wav") 


def vanilla():
    # Record audio
    from record import Record

    time_to_record = 3 # Obtain from user input
    record = Record(time_to_record)
    record.record() # saved as recorded_audio.wav
    print ("Recording done for ", time_to_record)
    print ("Saved as recorded_audio.wav")

    # Transcribe audio
    import os
    import dotenv

    dotenv.load_dotenv()
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY") # Obtain from user input

    from transcribe import Transcribe

    transcribe = Transcribe(api_key=GROQ_API_KEY)
    filename = "recorded_audio.wav"
    text_prompt = transcribe.transcribe(filename)
    print(text_prompt)


    # Generate audio
    from pitch import Pitch
    from playsound import playsound

    pitch = Pitch()

    file_name = "recorded_audio.wav"

    # First sound talking tom, done directly from voice recorded
    pitch.pitch_normal(file_name)
    print("Pitch normal done")
    print("Saved as output_pitch_normal.wav")
    playsound('output_pitch_normal.wav')

    # Second sound rajini, done from voice recorded and transcribed
    pitch.pitch_voice_clone(text_prompt, "rajini")
    print("Pitch rajini done")
    print("Saved as output_pitch_rajini.wav")
    playsound('output_pitch_rajini.wav')
