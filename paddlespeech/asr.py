from paddlespeech.cli.asr.infer import ASRExecutor
from paddlespeech.cli.text import TextExecutor

asr = ASRExecutor()
result = asr(
    model="conformer_talcs",
    lang="zh_en",
    codeswitch=True,
    audio_file="ch_zh_mix.wav")
print(result)

text_executor = TextExecutor()
result = text_executor(text=result)
print(result)
