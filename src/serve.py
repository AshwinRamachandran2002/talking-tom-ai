import os
import dotenv
import streamlit as st
from pitch import Pitch
from transcribe import Transcribe
from audio_recorder_streamlit import audio_recorder

st.set_page_config(page_title='Talking Tom AI', page_icon='🎤', layout='wide')
st.markdown("<h1 style='text-align: center;'>Talking Tom AI</h1>", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 2, 1])

with col1:
    st.empty()

with col2:
    recording_placeholder = st.empty()

    st.image('static/talking-tom.jpeg', use_column_width=True)
    st.write('<style>img{cursor: pointer;}</style>', unsafe_allow_html=True)

    with recording_placeholder.container():
        st.markdown('<p style="text-align: center; font-size: 18px;">Press here to record:</p>', unsafe_allow_html=True)
        audio_bytes = audio_recorder(
            energy_threshold=(-1.0, 1.0),
            pause_threshold=3.0,
        )

with col3:
    st.empty()

st.sidebar.title("Settings")

voices = ["Rajini", "Vanilla"]
option = st.sidebar.selectbox(
    "Voice Type:",
    voices,
    index=0,
    placeholder="Select voice...",
)

dotenv.load_dotenv()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

API_KEY = st.sidebar.text_input("Enter your GROQ API Key", GROQ_API_KEY, type="password")

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
