from paddlespeech.cli.tts.infer import TTSExecutor

tts = TTSExecutor()
tts(text="今天是monday，明天是tuesday。",
    output="out.wav",
    lang="mix",
    am="fastspeech2_mix")
