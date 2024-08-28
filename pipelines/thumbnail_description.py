from PIL import Image

from inferences.ocr import ocr_image
from inferences.vision_lang import infer_with_image
from preprocess.thumbnail import download_thumbnail

from typing import Dict


def get_image_description(
    video_id: str,
    path_cache: str,
    context: str,
) -> Dict[str, str]:
    """비디오 ID를 이용해 썸네일을 가져와 설명을 생성하는 함수.

    Args:
        video_id (str): YouTube 비디오 ID.
        path_cache (str): 썸네일 이미지를 저장할 캐시 경로.
        context (str): 추가적인 컨텍스트 정보.

    Returns:
        Dict[str, str]: 생성된 이미지 설명.
    """
    # TODO: 동작 테스트 필요.
    # 썸네일 다운로드
    thumbnail_path = download_thumbnail(video_id, path_cache)
    if not thumbnail_path:
        return "썸네일을 다운로드할 수 없습니다."

    # 이미지 로드
    with Image.open(thumbnail_path) as image:
        # inferences.OCR 서비스 이용.
        ocr_results = ocr_image(thumbnail_path)
        ocr_text = " ".join([result[1] for result in ocr_results])

        # 프롬프트 생성
        full_prompt = f"썸네일에 적혀있는 글자는 다음과 같습니다: {ocr_text}\n{context}\n이제 이 맥락을 고려해서 썸네일의 상황을 더 자세하게 묘사해주세요."

        # 이미지 설명 생성
        description = infer_with_image(full_prompt, image)

    return {"description": description}
