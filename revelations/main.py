import numpy as np
import simpleaudio as sa

Frequency = 440  # Our played note will be 440 Hz
Fs = 44100  # 44100 samples per second
Seconds = 3  # Note duration of 3 seconds

# Generate array with seconds*sample_rate steps, ranging between 0 and seconds
t = np.linspace(0, Seconds, Seconds * Fs, False)

# Generate a 440 Hz sine wave
note = np.sin(Frequency * t * 2 * np.pi)

# Ensure that highest value is in 16-bit range
audio = note * (2**15 - 1) / np.max(np.abs(note))
# Convert to 16-bit data
audio = audio.astype(np.int16)

# Start playback
play_obj = sa.play_buffer(audio, 1, 2, Fs)

# Wait for playback to finish before exiting
play_obj.wait_done()
