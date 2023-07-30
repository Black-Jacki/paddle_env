from paddlespeech.cli.asr.infer import ASRExecutor
from paddlespeech.cli.text import TextExecutor

asr_executor = ASRExecutor()
text_executor = TextExecutor()

for i in range(22):
    file = f"./voice/3/output_{i}.wav"
    if i < 10:
        file = f"./voice/3/output_0{i}.wav"
    text = asr_executor(audio_file=file, lang="zh")
    result = text_executor(text=text)
    print(result)
