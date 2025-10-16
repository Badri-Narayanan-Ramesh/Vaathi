from TTS.api import TTS
tts = TTS("tts_models/en/ljspeech/tacotron2-DDC")
tts.tts_to_file("Test sentence.", file_path="test_coqui.wav"