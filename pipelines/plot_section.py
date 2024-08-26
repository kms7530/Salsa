import json
import os
from collections import defaultdict
from glob import glob
from typing import Dict

from openai import OpenAI
from soynlp.normalizer import repeat_normalize
from tqdm import tqdm
from glob import glob

from inferences.lang import calculate_tokens, request_to_openai
from preprocess.transcript import download_video_transcript
from preprocess.prompt import get_video_section_prompt
from preprocess.text import clean_text
from preprocess.transcript import get_video_id


def split_plot():
    with open("test_videos.txt") as f:
        video_codes = [get_video_id(code) for code in f.readlines()]
        # download_video_transcript(video_codes)

    os.makedirs("cache/prompt", exist_ok=True)

    for path in tqdm(glob("cache/transcript/*")):
        code = os.path.basename(path).split(".")[0]

        if os.path.exists(f"cache/prompt/{code}.txt"):
            print("Use cache prompt")
        else:
            with open(path, "r") as f:
                transcript = json.load(f)

            with open(f"cache/meta/{code}.meta", "r") as f:
                meta = json.load(f)

            text_ko = transcript["ko"]
            text_en = transcript["en"]
            tag = transcript["tag"]

            timeline = 0.0
            content = ""

            title = meta["video_title"]
            channel = meta["channel_name"]

            for text in text_ko:
                start = text["start"]
                duration = text["duration"]

                timeline += duration
                text = f"{start:.2f}:{start + timeline:.2f} - {repeat_normalize(clean_text(text['text']), num_repeats=3)}"
                content += text + "\n"

            prompt = get_video_section_prompt(
                path_prompt="GROUP_SECTION",
                content=content,
                title=title,
                channel=channel,
            )

            with open(f"cache/prompt/{code}.txt", "w") as f:
                f.write(prompt)

        with open(f"cache/prompt/{code}.txt", "r") as f:
            prompt = f.read()

        print(prompt)
        model = "gpt-4o"

        if os.path.exists(f"cache/completion/{code}.txt"):
            print("use cache completion")
            with open(f"cache/completion/{code}.txt") as f:
                completion = f.read()
        else:
            with open(f"cache/completion/{code}.txt", "w") as f:
                completion = request_to_openai(model, prompt)
                f.write(completion)

        with open(f"cache/tokens/{code}.txt", "w") as f:
            f.write(str(calculate_tokens(prompt, completion, model)))

    print("done!")


if __name__ == "__main__":
    split_plot()
