import sounddevice as sd
import numpy as np
import time
import subprocess

# ====== SETTINGS YOU CAN TUNE ======

SAMPLE_RATE = 16000       # mic sample rate
BLOCK_DURATION = 0.1      # seconds per audio block
CLAP_THRESHOLD = 0.35     # loudness threshold (tune this)
CLAP_MIN_INTERVAL = 0.4   # minimum seconds between claps

# ===============================

last_clap_time = 0.0


def run_xdotool(args, label=""):
    cmd = ["xdotool"] + args
    print("[CMD]", " ".join(cmd), label)
    subprocess.run(cmd, check=False)


def on_clap():
    """What to do on each clap: Alt+Tab."""
    run_xdotool(["key", "Alt+Tab"], "ALT+TAB")


def audio_callback(indata, frames, time_info, status):
    global last_clap_time

    # Convert audio chunk to mono float array
    audio = indata[:, 0]

    # Simple RMS (volume)
    rms = np.sqrt(np.mean(audio**2))

    now = time.time()

    # Detect clap: volume above threshold + not too soon after previous
    if rms > CLAP_THRESHOLD and (now - last_clap_time) > CLAP_MIN_INTERVAL:
        print(f"👏 Clap detected! RMS={rms:.3f}")
        last_clap_time = now
        on_clap()


def main():
    print("🎧 Clap Alt+Tab running...")
    print("-> Every clap will send Alt+Tab to the active window.")
    print("Press Ctrl+C to stop.\n")

    try:
        with sd.InputStream(
            channels=1,
            samplerate=SAMPLE_RATE,
            blocksize=int(SAMPLE_RATE * BLOCK_DURATION),
            callback=audio_callback
        ):
            while True:
                time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nExiting clap listener.")
    except Exception as e:
        print("Audio error:", e)


if __name__ == "__main__":
    main()
