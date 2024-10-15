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

    # VLM 설정.
    PREF_VLM = {
        # 사용 가능 모델:
        #
        # Qwen2-VL 계열 모델.
        #   - Qwen/Qwen2-VL-2B-Instruct-GPTQ-Int8
        #   - Qwen/Qwen2-VL-7B-Instruct-GPTQ-Int8
        #   - and so on...
        #
        # LongVA 계열 모델.
        #   - lmms-lab/LongVA-7B-DPO
        #   - lmms-lab/LongVA-7B
        # !!! LongVA 관련 설정 !!!
        #   - LongVA 사용시 다음의 인자에 대해 정의 필요.
        #   - 주석 해제 후 사용.
        # "gen_kwargs": {
        #     "do_sample": True,
        #     "temperature": 0.5,
        #     "top_p": None,
        #     "num_beams": 1,
        #     "use_cache": True,
        #     "max_new_tokens": 1024,
        # }
        # "max_frames_num": 16
        "model_name": "Qwen/Qwen2-VL-2B-Instruct-GPTQ-Int8",
        # Flash Attention 2 사용 설정.
        # Qwen2-VL에서만 사용 가능.
        "use_flash_attn": True,
    }

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
    MEMORY_REQUIREMENTS = {
        "infer_with_video": 0.23,  # 비디오 추론에 필요한 메모리 (GB)
        "infer_with_image": 0.1,  # 이미지 추론에 필요한 메모리 (GB)
        "infer_ground_box": 0.64,  # DINO 추론에 필요한 메모리 (GB)
        "infer_img_to_text": 0.01,  # OCR에 필요한 메모리 (GB)
    }

    MODEL_SELECT_LOAD = [
        "DINO",
        "OCR",
        "VisionLanguage",
    ]  # 로드할 모델의 클래스 이름을 지정함.
