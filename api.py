from typing import Dict, Union

from fastapi import FastAPI

from config import Config
from pipelines.video_section import get_video_description

# FastAPI 객체 생성.
app = FastAPI()
app.task_counter = 0  # 현재 동작중인 worker의 개수를 담는 변수


@app.get("/api/health")
def health_check():
    """FastAPI 서비스 상태 확인용 API.

    Returns:
        str: OK.
    """

    return "OK"


@app.get("/description")
def health_check(code: Union[str, None]) -> Dict:
    """주어진 영상 코드를 통해 영상을 받아 키워드를 도출하는 API 함수.

    Args:
        code (Union[str, None]): YouTube 영상 코드.
        service (VisionLanguage, optional): BentoML과 연결을 위한 인자. Defaults to Depends(bentoml.get_current_service).

    Returns:
        Dict: 영상의 키워드 결과값.
    """

    # TODO: 기타 pipeline 작성.

    return {
        "status": "OK",
        "result": get_video_description(code, Config.PATH_CACHE, "", 0, 60),
    }
