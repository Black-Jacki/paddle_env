from paddlespeech.cli.asr.infer import ASRExecutor

asr = ASRExecutor()
result = asr(audio_file="银耳浆制作.wav")
print(result)
