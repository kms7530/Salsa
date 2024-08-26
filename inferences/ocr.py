import hashlib
import os
from urllib.parse import urljoin

import requests
from PIL import Image

from config import Config


def ocr_image(
    image_path: str = "", image_PIL: Image = None, path_cache: str = ""
) -> str:
    """이미지 파일을 이용한 OCR 추론 함수.

    Args:
        image_path (str, optional): 추론시 이용할 이미지 경로. (Defaults to "";`image_path`, `image_PIL` 중 하나만 지정. )
        image_PIL (Image, optional): 추론시 이용할 PIL 이미지 객체. (Defaults to None;`image_path`, `image_PIL` 중 하나만 지정. )
        path_cache (str, optional): PIL 객체가 들어올 시 임시 저장할 캐쉬 경로. (Defaults to ""; `image_PIL` 이용 시 경로 지정 필수. )

    Raises:
        ValueError: `image_path`, `image_PIL` 중 하나도 입력되지 않는 경우 발생.
        ValueError: `image_PIL`를 이용하나, `path_cache`이 지정되지 않은 경우 발생.

    Returns:
        str: 추론 결과.
    """

    # 모든 이미지 관련 변수가 입력되지 않은 경우 오류 발생.
    if image_path == "" and image_PIL == None:
        raise ValueError("image_path or image_PIL must be provided.")

    if image_PIL != None and path_cache == "":
        raise ValueError("If you want to use image_PIL, set path_cache. ")

    # 입력된 이미지 변수에 따른 `image` 변수 생성.
    if image_PIL != None:
        # 이미지 객체를 문자열로 변환하여 해시 생성
        image_hash = hashlib.md5(image.tobytes()).hexdigest()

        # 해시의 앞 5글자 추출
        hash_prefix = image_hash[:5]
        path_image = os.path.join(path_cache, f"{hash_prefix}.jpg")

        image = image_PIL.save(path_image)

    url: str = urljoin(
        Config.PATH_BENTOML_SERVER_BASE_URL,
        Config.PATH_BENTOML_SERVER["infer_img_to_text"],
    )

    # 파일 열기
    with open(path_image, "rb") as file:
        files = {"file": file}

        # POST 요청 보내기
        response = requests.post(url, files=files)

    return response.text
