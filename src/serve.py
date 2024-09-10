import av
import io
import tempfile
import numpy as np
import soundfile as sf
import streamlit as st
from pytubefix import YouTube
from audio_recorder_streamlit import audio_recorder

from pitch import Pitch
from transcribe import Transcribe

st.set_page_config(page_title='Talking Tom AI', page_icon='🎤', layout='wide')
st.markdown("<h1 style='text-align: center;'>Talking Tom AI</h1>", unsafe_allow_html=True)

# Global variable to store custom voices
if 'custom_voices' not in st.session_state:
    st.session_state.custom_voices = {}

# Two tabs
tab1, tab2 = st.tabs(["Talking Tom", "Voice Chooser"])


# Tab 1: Talking Tom with speaker
with tab1:
    left_co, cent_co,last_co = st.columns(3)
    with cent_co:
        st.image('static/talking_tom_but_as_hollywood_avatar.jpeg')

    st.write('<style>img{cursor: pointer;}</style>', unsafe_allow_html=True)

    # Initialize the audio recorder with error handling
    recording_placeholder = tab1.empty()
    with recording_placeholder.container():
        try:
            audio_bytes = audio_recorder(
                energy_threshold=(-1.0, 1.0),
                pause_threshold=3.0,
            )
            if audio_bytes is None:
                st.error("Failed to record audio. Please ensure you have granted microphone access.")
            else:
                st.success("Recording successful!")
        except Exception as e:
            st.error(f"Error initializing audio recorder: {e}")

    # If recording done
    if audio_bytes:
        st.audio(audio_bytes, format="audio/wav")

        # Save recorded audio in a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_file.write(audio_bytes)
            temp_file_path = temp_file.name

        st.divider()

        with st.spinner('Transcribing...'):
            try:
                text = Transcribe(api_key="").whisper_transcribe(temp_file_path)
                st.subheader('Transcribed Text')
                st.write(text)
            except Exception as e:
                st.error(f"Error during transcription: {e}")

        # Process and display pitch
        voices = ["Original"] + list(st.session_state.custom_voices.keys())
        option = st.selectbox("Choose a voice:", voices, index=0)

        # Process and display pitch
        if option:
            try:
                if option in st.session_state.custom_voices:
                    voice_file = st.session_state.custom_voices[option]
                else:
                    voice_file = None
                audio_file = Pitch().pitch("Hi! I am Ashwin and I am going to UC San Diego.", option.lower(), voice_file)

                if audio_file:
                    if ".mp3" in audio_file:
                        st.audio(audio_file, format="audio/mp3")
                    else:
                        st.audio(audio_file, format="audio/wav")
                else:
                    st.error("Error generating pitched audio.")
            except Exception as e:
                st.error(f"Error during pitch processing: {e}")
    else:
        st.info("Please record audio to proceed.")


# Tab 2: Voice chooser through YouTube
# Video link and time inputs
with tab2:
    st.subheader("Voice Chooser")
    video_url = st.text_input("Enter Video URL")
    start_time = st.number_input("Start Time (in seconds)", min_value=0, value=0)
    end_time = st.number_input("End Time (in seconds)", min_value=0, value=10)

    # Video processing
    if video_url:
        st.video(video_url)

        if st.button("Extract Audio from Video"):
            if end_time - start_time != 10:
                st.error("Clip duration must be exactly 10 seconds.")
            else:
                try:
                    # Stream video
                    yt = YouTube(video_url)
                    audio_stream = yt.streams.filter(only_audio=True).first()
                    audio_file = io.BytesIO()
                    audio_stream.stream_to_buffer(audio_file)
                    audio_file.seek(0)

                    container = av.open(audio_file)
                    audio_stream = next(s for s in container.streams if s.type == 'audio')
                    start_sample = int(start_time * audio_stream.sample_rate)
                    end_sample = int(end_time * audio_stream.sample_rate)

                    samples = []
                    for packet in container.demux(audio_stream):
                        for frame in packet.decode():
                            if start_sample <= frame.pts <= end_sample:
                                samples.append(frame.to_ndarray())
                            elif frame.pts > end_sample:
                                break

                    audio_data = np.concatenate(samples, axis=1).T
                    sample_rate = audio_stream.sample_rate

                    # Create a temporary file to save the extracted audio
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
                        sf.write(temp_file.name, audio_data, sample_rate, format='WAV')
                        temp_file.seek(0)
                        temp_file_path = temp_file.name
                        st.audio(temp_file.name, format="audio/wav")
                        st.success("Audio extracted and converted to WAV successfully.")

                        voice_name = "Lee"
                        st.session_state.custom_voices[voice_name] = temp_file_path
                        st.success(f"Voice '{voice_name}' saved successfully!")

                    # # Ask user for a name to save this voice
                    # voice_name = st.text_input("Enter a name to save this voice:")

                    # if st.button("Save Voice"):
                    #     if not voice_name:
                    #         st.error("Voice name cannot be empty.")
                    #     elif voice_name in st.session_state.custom_voices:
                    #         st.error("Voice name already exists. Please choose a different name.")
                    #     else:
                    #         st.session_state.custom_voices[voice_name] = temp_file_path
                    #         st.success(f"Voice '{voice_name}' saved successfully!")
                    #         st.experimental_rerun()  # Refresh the page to reflect the new option in Tab 1

                except Exception as e:
                    st.error(f"Error processing video: {e}")
            print(st.session_state)
        print(st.session_state)
    else:
        st.info("Please enter a video URL to proceed.")
