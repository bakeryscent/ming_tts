#!/usr/bin/env python3
"""Build the speaker fingerprints of the premium voices (voices/fingerprints*).

A fingerprint is the speaker embedding of about a minute of the voice's own
audio. Passing it next to the built-in voice keeps the voice stable across
fragments and requests without losing clarity; embeddings from short clips make
the voice darker, so the audio has to be long. Passing half of it next to a
director instruction keeps the voice recognizable and leaves the delivery free.

    MING_DIR=/app/ming python scripts/make_fingerprints.py voices/ US_James 灵楷

A voice that was already stable on its own (US_Emma) can skip the plain
fingerprint: delete its file from voices/fingerprints/ and keep the other one.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

ENGLISH = (
    "Trying to learn from customer conversations is like excavating a delicate archaeological site. "
    "The truth is down there somewhere, but it's fragile. While each blow with your shovel gets you closer "
    "to the truth, you're liable to smash it into a million little pieces if you use too blunt an instrument. "
    "I see a lot of teams using a bulldozer and crate of dynamite for their excavation. They are, in one way "
    "or another, forcing people to say something nice about their business. It happens more than you'd ever imagine."
)
CHINESE = (
    "今晚的观测将在云层散开后开始，请耐心等待下一次通知。新图书馆明天早上开门，欢迎所有市民前来参观。"
    "大多数人低估了光线对睡眠的影响，尤其是在深夜使用手机的时候。开始之前，请确认设备已经断电，并把电池完全取出来。"
    "蜂蜜即使过了很多年也不会变质，这也是它自古以来就被人们珍视的原因之一。周末我们常常去公园散步，如果天气好，"
    "还会在湖边喝一杯咖啡，聊聊最近发生的事情。"
)
DIRECTION_SCALE = 0.5
SEED = 1895


def main() -> None:
    out_dir, voices = Path(sys.argv[1]), sys.argv[2:]
    sys.path.insert(0, os.environ.get("MING_DIR", "/app/ming"))
    import torch
    import torchaudio
    from infer import seed_everything
    from ming_tts import MingTTS

    tts = MingTTS(os.environ.get("MODEL_ID", "inclusionAI/Ming-3.0-tts-0.5B"), device="cuda:0",
                  text_normalization=True, cache_prompts=False)
    (out_dir / "fingerprints").mkdir(parents=True, exist_ok=True)
    (out_dir / "fingerprints_direction").mkdir(parents=True, exist_ok=True)
    for voice in voices:
        text = CHINESE if re.search(r"[一-鿿]", voice) else ENGLISH
        seed_everything(SEED)
        waveform = tts.synthesize("IP", text, controls={"ip": voice}, use_zero_spk_emb=True)
        audio = (waveform if waveform.dim() == 2 else waveform.unsqueeze(0)).float().cpu()
        embedding = tts.spkemb_extractor(torchaudio.functional.resample(audio, tts.sample_rate, 16000)).float()
        torch.save(embedding, out_dir / "fingerprints" / f"{voice}.pt")
        torch.save(embedding * DIRECTION_SCALE, out_dir / "fingerprints_direction" / f"{voice}.pt")
        print(voice, round(audio.shape[-1] / tts.sample_rate, 1), "s")


if __name__ == "__main__":
    main()
