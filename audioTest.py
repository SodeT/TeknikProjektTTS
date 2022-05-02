import pyaudio
import wave
import audioop

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
RECORD_SECONDS = 5

pyAud = pyaudio.PyAudio()
device_count = pyAud.get_device_count()

for i in range(0, device_count):
        print("Name: " + str(pyAud.get_device_info_by_index(i)["name"]))
        print("Index: " + str(pyAud.get_device_info_by_index(i)["index"]))
        print("\n")
#input_device_index = 6, output_device_index = 1,
stream = pyAud.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    rms = audioop.rms(data, 2)    # here's where you calculate the volume
    print(rms)

stream.stop_stream()
stream.close()
pyAud.terminate()