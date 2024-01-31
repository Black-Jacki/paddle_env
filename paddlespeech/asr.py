from paddlespeech.cli.asr.infer import ASRExecutor
from paddlespeech.cli.text import TextExecutor

asr_executor = ASRExecutor()
text_executor = TextExecutor()

for i in range(22):
    file = f"./voice/output_{i}.wav"
    text = asr_executor(audio_file=file, lang="zh")
    result = text_executor(text=text)
    print(result)
