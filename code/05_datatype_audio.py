"""05_datatype_audio.py -- audio data: synthetic WAV samples + waveform/spectrum visualization."""  # module comment
import wave                                                           # standard library: write a WAV file
import numpy as np                                                    # numerical computing
import matplotlib.pyplot as plt                                       # plotting
from _common import DAT, save                                         # shared helpers

SR = 8000                                                             # sample rate 8 kHz (telephone-speech level)
T = 1.0                                                               # each clip lasts 1 second
t = np.linspace(0, T, int(SR * T), endpoint=False)                    # build the time axis (8000 sample points per second)
clips = {}                                                            # store the three audio clips in a dict

clips["tone_440"] = 0.6 * np.sin(2 * np.pi * 440 * t)                 # clip 1: a 440 Hz pure tone (single frequency)
clips["chirp_200_2000"] = 0.6 * np.sin(2 * np.pi * (200 + 900 * t) * t)  # clip 2: a chirp signal (frequency rises over time)
env = 0.5 * (1 + np.sin(2 * np.pi * 3 * t))                           # clip 3: a 3 Hz amplitude envelope (mimics speech rhythm)
clips["am_noise"] = env * np.random.default_rng(7).normal(0, 0.3, t.size)  # clip 3: amplitude-modulated noise (speech-like)

for name, x in clips.items():                                         # loop over the three clips
    pcm = np.int16(np.clip(x, -1, 1) * 32767)                         # convert floats to 16-bit integer PCM (the WAV standard)
    path = DAT / ("audio_%s.wav" % name)                              # build the output path
    with wave.open(str(path), "wb") as w:                             # open the WAV file in binary write mode
        w.setnchannels(1)                                             # mono
        w.setsampwidth(2)                                             # 2 bytes per sample = 16 bit
        w.setframerate(SR)                                            # write the sample rate
        w.writeframes(pcm.tobytes())                                  # write the raw byte stream
    print("[data]", path)                                             # print the dump location

# ---------------------------------------------------------------- visualization
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.9))                   # one row, three columns: waveform / spectrum / comparison waveform
axes[0].plot(t, clips["tone_440"], color="#8A0C3C", lw=0.8)           # left: pure-tone waveform (time domain)
axes[0].set_title("Waveform (time domain): 440 Hz pure tone")         # left title
axes[0].set_xlabel("Time / s")                                        # left x-axis meaning
axes[0].set_ylabel("Amplitude")                                       # left y-axis meaning

axes[1].specgram(clips["chirp_200_2000"], NFFT=256, Fs=SR,           # middle: spectrogram via the short-time Fourier transform (time-frequency)
                 cmap="magma", noverlap=128)                          # set the overlap and the color map
axes[1].set_title("Spectrogram (time-frequency): 200->2000 Hz chirp")  # middle title
axes[1].set_xlabel("Time / s")                                        # middle x-axis meaning
axes[1].set_ylabel("Frequency / Hz")                                  # middle y-axis meaning
axes[1].grid(False)                                                   # a spectrogram needs no grid

axes[2].plot(t, clips["am_noise"], color="#66717D", lw=0.6)           # right: amplitude-modulated noise waveform
axes[2].plot(t, env, color="#C9A227", lw=1.8, label="envelope")       # overlay the envelope to show the "rhythm"
axes[2].set_title("Amplitude-modulated noise (mimics speech rhythm)")  # right title
axes[2].set_xlabel("Time / s")                                        # right x-axis meaning
axes[2].legend(fontsize=8)                                            # show the legend
plt.tight_layout()                                                    # adjust the layout
save(fig, "fig_audio.png")                                            # save the figure
