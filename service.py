from __future__ import annotations

from pathlib import Path

import bentoml
import numpy as np
import torch
from decord import VideoReader, cpu
from longva.constants import IMAGE_TOKEN_INDEX
from longva.mm_utils import process_images, tokenizer_image_token
from longva.model.builder import load_pretrained_model
from PIL import Image


class VisionLanguage:
    def __init__(self) -> None:
        """BentoML 서비스 구동을 위한 모델과 기타 객체 및 옵션 생성."""

        model_path = "lmms-lab/LongVA-7B-DPO"
        self.gen_kwargs = {
            "do_sample": True,
            "temperature": 0.5,
            "top_p": None,
            "num_beams": 1,
            "use_cache": True,
            "max_new_tokens": 1024,
        }
        self.max_frames_num = 16
        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
            model_path, None, "llava_qwen", device_map="cuda:0"
        )

    def __generate_prompt(self, prompt: str) -> torch.Tensor:
        """입력된 프롬프트를 모델에서 추론 가능하게 변환하여 Tensor로 반환하는 함수.

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

    def __run_inference(
        self, input_ids: torch.Tensor, input_tensor: torch.Tensor, modalities: str
    ) -> str:
        """Modality에 따른 추론 후 결과를 반환하는 내부 함수.

        Args:
            input_ids (torch.Tensor): 입력될 프롬프트의 Tensor.
            input_tensor (torch.Tensor): 입력될 영상의 Tensor.
            modalities (str): 추론 모드. (image, video 중 하나. )

        Returns:
            str: 추론한 결과 텍스트.
        """

        # Modality에 따른 추론 결과 가져오기.
        if modalities == "video":
            with torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids,
                    images=[input_tensor],
                    modalities=[modalities],
                    **self.gen_kwargs,
                )
        elif modalities == "image":
            with torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids,
                    images=input_tensor,
                    image_sizes=[input_tensor.size],
                    modalities=[modalities],
                    **self.gen_kwargs,
                )

        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[
            0
        ].strip()

        return outputs

    @bentoml.api
    def infer_with_video(self, prompt: str, video_path: Path) -> str:
        """비디오 파일을 이용한 LongVA 추론 함수.

        Args:
            prompt (str): 추론시 이용할 함수.
            video_path (str): 수론시 이용할 영상 저장 경로.

        Returns:
            str: 추론 후 결과.
        """

        # 프롬프트 생성 및 토큰으로 변환.
        input_ids = self.generate_prompt(prompt)

        # 비디오 파일 불러오기.
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frame_num = len(vr)
        uniform_sampled_frames = np.linspace(
            0, total_frame_num - 1, self.max_frames_num, dtype=int
        )

        frame_idx = uniform_sampled_frames.tolist()
        frames = vr.get_batch(frame_idx).asnumpy()

        video_tensor = self.image_processor.preprocess(frames, return_tensors="pt")[
            "pixel_values"
        ].to(self.model.device, dtype=torch.float16)

        # 추론.
        output_ids = self.__run_inference(input_ids, video_tensor, "video")
        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[
            0
        ].strip()

        return outputs

    @bentoml.api
    def infer_with_image(self, prompt: str, image: Image) -> str:
        """이미지 파일을 이용한 LongVA 추론 함수.

        Args:
            prompt (str): 추론시 이용할 함수.
            image (Image, optional): 추론시 이용할 PIL 이미지 객체.

        Returns:
            str: 추론 결과.
        """

        # 프롬프트 생성 및 토큰으로 변환.
        input_ids = self.__generate_prompt(prompt)

        image = image.convert("RGB")
        images_tensor = process_images(
            [image], self.image_processor, self.model.config
        ).to(self.model.device, dtype=torch.float16)

        output_ids = self.__run_inference(input_ids, images_tensor, "image")
        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[
            0
        ].strip()

        return outputs
