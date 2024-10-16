import bentoml
import os
import hashlib
from pathlib import Path

from groundingdino.util.inference import load_image, load_model, predict
from PIL.Image import Image as PILImage
from typing import Dict

from config import Config


@bentoml.service(
    resources={"cpu": "1"},
    traffic={"timeout": 10},
)
class DINO:
    def __init__(self) -> None:
        """Ground DINO의 serving을 위한 객체 생성 함수."""

        self.model = load_model(
            "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
            "GroundingDINO/weights/groundingdino_swint_ogc.pth",
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

        # 이미지 객체를 문자열로 변환하여 해시 생성
        image_hash = hashlib.md5(image.tobytes()).hexdigest()

        # .cache 디렉토리 생성
        cache_dir = Path(Config.PATH_CACHE)
        cache_dir.mkdir(parents=True, exist_ok=True)

        # 해시의 앞 5글자 추출
        hash_prefix = image_hash[:5]
        path_image = os.path.join(Config.PATH_CACHE, f"{hash_prefix}.jpg")

        image = image.convert("RGB").save(path_image)

        BOX_TRESHOLD = 0.35
        TEXT_TRESHOLD = 0.25

        _, image = load_image(path_image)

        boxes, logits, phrases = predict(
            model=self.model,
            image=image,
            caption=prompt,
            box_threshold=BOX_TRESHOLD,
            text_threshold=TEXT_TRESHOLD,
        )

        return {"boxes": boxes.tolist(), "logits": logits.tolist(), "phrases": phrases}
