import bentoml
import torch
import numpy as np
import shutil
import subprocess
import tempfile
import os

from typing import Annotated, Callable, Dict, List, Any
from longva.constants import IMAGE_TOKEN_INDEX
from longva.mm_utils import process_images, tokenizer_image_token
from longva.model.builder import load_pretrained_model
from bentoml.validators import ContentType
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from PIL.Image import Image as PILImage

from config import Config
from pathlib import Path
from decord import VideoReader, cpu


def convert_video_to_mpeg4(input_path, output_path):
    """
    비디오를 MPEG-4 형식으로 변환하도록 cmd 명령어 수행.
    """
    cmd = [
        "ffmpeg",
        "-i",
        str(input_path),
        "-c:v",
        "mpeg4",
        "-c:a",
        "copy",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)


def safe_video_processing(func):
    """
    비디오 처리 함수 데코레이터
    오류 발생 시 MPEG-4로 변환 후 재시도합니다.
    """

    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"원본 비디오 처리 중 오류 발생: {e}")
            print("MPEG-4로 변환 후 재시도합니다.")

            # 임시 파일 생성
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
                temp_path = Path(temp_file.name)

            try:
                # 비디오 변환
                convert_video_to_mpeg4(args[1], temp_path)

                # 변환된 비디오로 함수 재실행
                new_args = list(args)
                new_args[1] = temp_path
                result = func(*new_args, **kwargs)

                return result
            finally:
                # 임시 파일 삭제
                os.unlink(temp_path)

    return wrapper


@bentoml.service(
    resources={"cpu": "2"},
    traffic={"timeout": 30},
)
class VisionLanguage:
    def __init__(self) -> None:
        """BentoML 서비스 구동을 위한 모델과 기타 객체 및 옵션 생성."""
        self.model_path = Config.PREF_VLM["model_name"]

        # 모델 설정이 LongVA인 경우.
        if "LongVA" in self.model_path:
            self.gen_kwargs = Config.PREF_VLM["gen_kwargs"]
            self.max_frames_num = Config.PREF_VLM["max_frames_num"]
            self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
                self.model_path, None, "longva", device_map="cuda:0"
            )
        # 모델 설정이 Qwen2-VL인 경우.
        elif "Qwen2-VL" in self.model_path:
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_path,
                device_map="auto",
                # Flash attention 2 관련 설정.
                torch_dtype=(
                    torch.bfloat16 if Config.PREF_VLM["use_flash_attn"] else "auto"
                ),
                attn_implementation=(
                    "flash_attention_2" if Config.PREF_VLM["use_flash_attn"] else ""
                ),
            )
            self.processor = AutoProcessor.from_pretrained(self.model_path)
        else:
            raise Exception(f"지원되지 않는 모델: {self.model_path}")

    def __callback_by_model(self, dict_fn_callback: Callable, **kwargs) -> str:
        """모델 설정에 맞는 함수를 호출한 후 결과를 반환하는 함수.

        Args:
            dict_fn_callback (Callable): 모델별 호출 함수를 가진 dictionary.
        """

        result = None

        if "LongVA" in self.model_path:
            result = dict_fn_callback["LongVA"](**kwargs)
        elif "Qwen2-VL" in self.model_path:
            result = dict_fn_callback["Qwen2-VL"](**kwargs)
        else:
            raise Exception(f"지원되지 않는 모델: {self.model_path}")

        return result

    def __generate_prompt_longva(self, prompt: str) -> torch.Tensor:
        """LongVA 모델을 이용해, 입력된 프롬프트를 모델에서 추론 가능하게 변환하여 Tensor로 반환하는 함수.

        Args:
            prompt (str): 모델에 추론시 이용할 프롬프트.

        Returns:
            torch.Tensor: 모델에 입력 가능한 Tensor
        """

        prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<image>\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        input_ids = (
            tokenizer_image_token(
                prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .to(self.model.device)
        )

        return input_ids

    def __run_inference_longva(
        self,
        prompt: str,
        modalities: str,
        image: PILImage = None,
        video_path: Path = "",
    ) -> str:
        """LongVA 모델을 이용해, modality에 따른 추론 후 결과를 반환하는 내부 함수.

        Args:
            prompt (str): 모델 추론시 사용될 프롬프트.
            modalities (str): 입력되는 데이터의 종류(video / image)
            image (PILImage, optional): 추론시 사용될 이미지 객체. Defaults to None.
            video_path (Path, optional): 추론시 사용될 비디오의 경로. Defaults to "".

        Returns:
            str: 추론한 결과 텍스트.
        """

        # 프롬프트 생성 및 토큰으로 변환.
        input_ids = self.__generate_prompt_longva(prompt)

        # Modality에 따른 추론 결과 가져오기.
        if modalities == "video":
            # 비디오 파일 불러오기.
            vr = VideoReader(str(video_path), ctx=cpu(0))
            total_frame_num = len(vr)
            uniform_sampled_frames = np.linspace(
                0, total_frame_num - 1, self.max_frames_num, dtype=int
            )

            frame_idx = uniform_sampled_frames.tolist()
            frames = vr.get_batch(frame_idx).asnumpy()

            video_tensor = self.image_processor.preprocess(frames, return_tensors="pt")[
                "pixel_values"
            ].to(self.model.device, dtype=torch.float16)

            with torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids,
                    images=[video_tensor],
                    modalities=[modalities],
                    **self.gen_kwargs,
                )
        elif modalities == "image":
            image = image.convert("RGB")
            images_tensor = process_images(
                [image], self.image_processor, self.model.config
            ).to(self.model.device, dtype=torch.float16)

            with torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids,
                    images=images_tensor,
                    image_sizes=[image.size],
                    modalities=[modalities],
                    **self.gen_kwargs,
                )

        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[
            0
        ].strip()

        return outputs

    def __run_inference_qwen2vl(
        self,
        prompt: str,
        modalities: str,
        image: PILImage = None,
        video_path: Path = "",
    ) -> str:
        """Qwen2-VL 모델을 이용해, modality에 따른 추론 후 결과를 반환하는 내부 함수.

        Args:
            prompt (str): 모델 추론시 사용될 프롬프트.
            modalities (str): 입력되는 데이터의 종류(video / image)
            image (PILImage, optional): 추론시 사용될 이미지 객체. Defaults to None.
            video_path (Path, optional): 추론시 사용될 비디오의 경로. Defaults to "".

        Returns:
            str: 추론한 결과 텍스트.
        """

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": modalities,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Modality에 맞는 데이터 입력.
        if modalities == "image":
            messages[0]["content"][0]["image"] = image
        elif modalities == "video":
            # 파일 확장자 추가.
            shutil.move(str(video_path), str(video_path) + ".mp4")

            messages[0]["content"][0]["video"] = str(video_path) + ".mp4"
            messages[0]["content"][0]["fps"] = 1.0
            messages[0]["content"][0]["max_pixels"] = 360 * 420

        # Prompt 양식에 맞도록 input 생성.
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        # Modality에 맞는 입력 tensor 생성.
        if modalities == "image":
            inputs = self.processor(
                text=text,
                images=image_inputs,
                padding=True,
                return_tensors="pt",
            )
        elif modalities == "video":
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
        inputs = inputs.to("cuda")

        # Output 생성.
        generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        return output[0]

    @bentoml.api(route="/video")
    def infer_with_video(
        self, prompt: str, video_path: Annotated[Path, ContentType("video/mp4")]
    ) -> str:
        """비디오 파일을 이용한 VLM 추론 함수.

        Args:
            prompt (str): 추론시 이용할 함수.
            video_path (Path): 수론시 이용할 영상 저장 경로.

        Returns:
            str: 추론 후 결과.
        """
        try:
            # 설정된 모델에 따른 결과 추론.
            outputs = self.__callback_by_model(
                {
                    "LongVA": self.__run_inference_longva,
                    "Qwen2-VL": self.__run_inference_qwen2vl,
                },
                prompt=prompt,
                modalities="video",
                video_path=video_path,
            )

            return outputs
        except Exception as e:
            print(f"원본 비디오 처리 중 오류 발생: {e}")
            print("MPEG-4로 변환 후 재시도합니다.")

            # 임시 파일 생성
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
                temp_path = Path(temp_file.name)

            try:
                # 비디오 변환
                convert_video_to_mpeg4(video_path, temp_path)

                # 변환된 비디오로 함수 재실행
                return self.infer_with_video(prompt, temp_path)
            finally:
                # 임시 파일 삭제
                os.unlink(temp_path)

    @bentoml.api(route="/image")
    def infer_with_image(self, prompt: str, image: PILImage) -> str:
        """이미지 파일을 이용한 VLM 추론 함수.

        Args:
            prompt (str): 추론시 이용할 함수.
            image (Image, optional): 추론시 이용할 PIL 이미지 객체.

        Returns:
            str: 추론 결과.
        """

        # 설정된 모델에 따른 결과 추론.
        outputs = self.__callback_by_model(
            {
                "LongVA": self.__run_inference_longva,
                "Qwen2-VL": self.__run_inference_qwen2vl,
            },
            prompt=prompt,
            modalities="image",
            image=image,
        )

        return outputs
