from __future__ import annotations

from pathlib import Path
from typing import Dict

import bentoml
import easyocr
import numpy as np
import torch
from decord import VideoReader, cpu
from longva.constants import IMAGE_TOKEN_INDEX
from longva.mm_utils import process_images, tokenizer_image_token
from longva.model.builder import load_pretrained_model
from PIL.Image import Image as PILImage
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from config import Config


@bentoml.service(
    resources={"cpu": "2"},
    traffic={"timeout": 300},
)
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
        self.ocr_service = OCR()

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
        self,
        input_ids: torch.Tensor,
        input_tensor: torch.Tensor,
        modalities: str,
        image: PILImage = None,
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
                    image_sizes=[image.size],
                    modalities=[modalities],
                    **self.gen_kwargs,
                )

        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[
            0
        ].strip()

        return outputs

    @bentoml.api(route="/video")
    def infer_with_video(self, prompt: str, video_path: Path) -> str:
        """비디오 파일을 이용한 LongVA 추론 함수.

        Args:
            prompt (str): 추론시 이용할 함수.
            video_path (str): 수론시 이용할 영상 저장 경로.

        Returns:
            str: 추론 후 결과.
        """

        # 프롬프트 생성 및 토큰으로 변환.
        input_ids = self.__generate_prompt(prompt)

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

        # 추론.
        outputs = self.__run_inference(input_ids, video_tensor, "video")

        return outputs

    @bentoml.api(route="/image")
    def infer_with_image(self, prompt: str, image: PILImage) -> str:
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

        outputs = self.__run_inference(input_ids, images_tensor, "image", image=image)

        return outputs

    @bentoml.api(route="/describe_image")
    def describe_image(self, prompt: str, image: PILImage) -> Dict[str, str]:
        """이미지 파일과 OCR 텍스트를 이용한 LongVA 추론 함수.

        Args:
            prompt (str): 추가적인 설명 요청 프롬프트.
            image (Image): 추론시 이용할 PIL 이미지 객체.

        Returns:
            Dict[str, str]: 추론 결과를 담은 딕셔너리.
        """
        # OCR 텍스트 추출.
        ocr_results = self.ocr_service.infer_img_to_text(image)
        ocr_text = " ".join([result[1] for result in ocr_results])

        # 프롬프트 생성.
        # TODO: 그런데 이 부분 prompt에서 한글 + 영어 섞여도 되나 의문.
        full_prompt = f"썸네일에 적혀있는 글자는 다음과 같습니다: {ocr_text}\n{prompt}\n이제 이 맥락을 고려해서 썸네일의 상황을 더 자세하게 묘사해주세요."

        # infer_with_image 함수 호출
        outputs = self.infer_with_image(full_prompt, image)

        return {"description": outputs}


@bentoml.service(
    resources={"cpu": "1"},
    traffic={"timeout": 10},
)
class DINO:
    def __init__(self) -> None:
        """Ground DINO의 serving을 위한 객체 생성 함수."""

        model_id = Config.PREF_DINO["model_name"]
        device = Config.PREF_DINO["device"]

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(
            device
        )

    @bentoml.api(route="/ground-box")
    def infer_ground_box(self, prompt: str, image: PILImage) -> Dict:
        """Ground DINO 추론을 위한 API 함수.

        Args:
            prompt (str): 추론시 이용될 프롬프트.
            image (PILImage): 추론할 이미지 객체.

        Returns:
            Dict: 결과 dict.
        """

        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(
            self.model.device
        )
        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=Config.PREF_DINO["box_threshold"],
            text_threshold=Config.PREF_DINO["text_threshold"],
            target_sizes=[image.size[::-1]],
        )

        return results


@bentoml.service(
    resources={"cpu": "1"},
    traffic={"timeout": 3},
)
class OCR:
    def __init__(self) -> None:
        """Ground DINO의 serving을 위한 객체 생성 함수."""

        self.reader = easyocr.Reader(Config.PREF_OCR["lang"])

    @bentoml.api(route="/ocr")
    def infer_img_to_text(self, image: PILImage) -> Dict:
        """Ground DINO 추론을 위한 API 함수.

        Args:
            prompt (str): 추론시 이용될 프롬프트.
            image (PILImage): 추론할 이미지 객체.

        Returns:
            Dict: 결과 dict.
        """

        results = self.reader.readtext(image)

        return results


class Bako:
    service_vlm = bentoml.depends(VisionLanguage)
    service_dino = bentoml.depends(DINO)
    service_ocr = bentoml.depends(OCR)

    def __init__(self) -> None:
        """서비스 제공중인 모델을 모두 routing하는 Bako 객체의 초기화 함수."""

        pass

    @bentoml.api(route="/video")
    async def infer_with_video(self, prompt: str, video_path: Path) -> str:
        """비디오 파일을 이용한 LongVA 추론 함수. - Bako

        Args:
            prompt (str): 추론시 이용할 함수.
            video_path (str): 수론시 이용할 영상 저장 경로.

        Returns:
            str: 추론 후 결과.
        """

        result = await self.service_vlm.to_async.infer_with_video(prompt, video_path)
        return result

    @bentoml.api(route="/image")
    async def infer_with_image(self, prompt: str, image: PILImage) -> str:
        """이미지 파일을 이용한 LongVA 추론 함수. - Bako

        Args:
            prompt (str): 추론시 이용할 함수.
            image (Image, optional): 추론시 이용할 PIL 이미지 객체.

        Returns:
            str: 추론 결과.
        """

        result = await self.service_vlm.to_async.infer_with_image(prompt, image)
        return result

    @bentoml.api(route="/ground-box")
    async def infer_ground_box(self, prompt: str, image: PILImage) -> Dict:
        """Ground DINO 추론을 위한 API 함수. - Bako

        Args:
            prompt (str): 추론시 이용될 프롬프트.
            image (PILImage): 추론할 이미지 객체.

        Returns:
            Dict: 결과 dict.
        """

        result = await self.service_dino.to_async.infer_ground_box(prompt, image)
        return result

    @bentoml.api(route="/ground-box")
    async def infer_img_to_text(self, image: PILImage) -> Dict:
        """Ground DINO 추론을 위한 API 함수. - Bako

        Args:
            prompt (str): 추론시 이용될 프롬프트.
            image (PILImage): 추론할 이미지 객체.

        Returns:
            Dict: 결과 dict.
        """

        result = await self.service_ocr.to_async.infer_img_to_text(image)
        return result

    @bentoml.api(route="/describe_image")
    async def describe_image(self, prompt: str, image: PILImage) -> Dict[str, str]:
        """이미지 파일과 OCR 텍스트를 이용한 LongVA 추론 함수. - Bako

        Args:
            prompt (str): 추가적인 설명 요청 프롬프트.
            image (Image): 추론시 이용할 PIL 이미지 객체.

        Returns:
            Dict[str, str]: 추론 결과를 담은 딕셔너리.
        """
        result = await self.service_vlm.to_async.describe_image(prompt, image)
        return result
