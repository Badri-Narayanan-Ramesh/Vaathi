# test_t2s.py
from voice.text_2_speech import tts_bytes

if __name__ == "__main__":
    msg = "Hello Badri! This is your AI tutor speaking."
    audio = tts_bytes(msg, rate=170)
    with open("hello.wav", "wb") as f:
        f.write(audio)
    print("✅ Saved hello.wav")

