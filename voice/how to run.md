python text_2_speech_fast_v2.py --all-in-one --out all_sentences.wav --rate 170 --max 6

python text_2_speech_fast_v2.py --voice "HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Speech\\Voices\\Tokens\\TTS_MS_EN-US_ZIRA_11.0"
'

python text_2_speech_fast_v2.py --rate 170 --max 4 --wer-first-n 2    

python .\text_2_speech_fast_v2.py --all-in-one --out all_sentences.wav --max 6 --wer-first-n 6 --silence-sec 1.0

$env:KMP_DUPLICATE_LIB_OK = 'TRUE'  

----

python whisper_hf_asr.py all_sentences.wav --language en --segments

python whisper_hf_asr.py ".\all_sentences.wav" --language en --segments

----
