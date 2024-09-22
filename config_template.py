from dataclasses import dataclass
from typing import List, Dict


@dataclass
class Config:
    # BentoML관련 설정.
    PREF_BENTOML_SERVER = {
        "protocol": "http://",
        "path_root": "127.0.0.1",
        "port": "4500",
    }

    # BentoML을 이용해 routing되고 있는 서비스의 경로.
    PATH_BENTOML_SERVER = {
        "infer_with_video": "/video",
        "infer_with_image": "/image",
        "infer_ground_box": "/ground-box",
        "infer_img_to_text": "/ocr",
    }

    PATH_BENTOML_SERVER_BASE_URL: str = (
        f"{PREF_BENTOML_SERVER['protocol']}{PREF_BENTOML_SERVER['path_root']}:{PREF_BENTOML_SERVER['port']}"
    )
    PATH_CACHE: str = ".cache"

    # DINO 설정.
    PREF_DINO = {
        "model_name": "IDEA-Research/grounding-dino-tiny",
        "device": "cuda",
        "box_threshold": 0.4,
        "text_threshold": 0.3,
    }

    # Easy OCR 설정.
    PREF_OCR = {
        # 다음의 URL 참조.
        # https://www.jaided.ai/easyocr/
        "lang": ["ko", "en"]
    }

    # 메모리 관련 설정.
    MIN_REQUIRED_MEMORY: float = 19.93  # 시스템에 필요한 최소 메모리 (GB)
    # VisionLanguage: 17.81 GB
    # DINO: 2.03 GB
    # OCR: 0.09 GB

    # 각 서비스별 필요 메모리 설정.
    MEMORY_REQUIREMENTS: Dict[str, float] = {
        "infer_with_video": 0.23,  # 비디오 추론에 필요한 메모리 (GB)
        "infer_with_image": 0.1,  # 이미지 추론에 필요한 메모리 (GB)
        "infer_ground_box": 0.64,  # DINO 추론에 필요한 메모리 (GB)
        "infer_img_to_text": 0.01,  # OCR에 필요한 메모리 (GB)
    }
