import sounddevice as sd
from scipy.io.wavfile import write


class Record:
    def __init__(self, seconds=3):
        self.fs = 44100
        self.seconds = 3

    def record(self):
        myrecording = sd.rec(int(self.seconds * self.fs), samplerate=self.fs, channels=2)
        print('Recording')
        sd.wait()
        print('Done')
        write('recorded_audio.wav', self.fs, myrecording)


if __name__ == '__main__':
    record = Record()
    record.record()
