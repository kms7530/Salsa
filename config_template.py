from dataclasses import dataclass
from typing import Dict, List


@dataclass
class Config:
    # BentoML관련 설정.
    PREF_BENTOML_SERVER: Dict = {
        "protocol": "http://",
        "path_root": "127.0.0.1",
        "port": "4500",
    }

    # BentoML을 이용해 routing되고 있는 서비스의 경로.
    PATH_BENTOML_SERVER: Dict = {
        "infer_with_video": "/video",
        "infer_with_image": "/image",
    }

    PATH_BENTOML_SERVER_BASE_URL: str = f"{PREF_BENTOML_SERVER['protocol']}{PREF_BENTOML_SERVER['path_root']}:{PREF_BENTOML_SERVER['port']}"
    PATH_CACHE: str = ".cache"

    # DINO 설정.
    PREF_DINO: Dict = {
        "model_name": "IDEA-Research/grounding-dino-tiny",
        "device": "cuda",
        "box_threshold": 0.4,
        "text_threshold": 0.3,
    }
