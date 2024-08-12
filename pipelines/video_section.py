from model.vision_lang import infer_with_video
from preprocess.video import download_video, slice_video


def get_video_description(
    video_id: str,
    path_cache: str,
    context_section: str,
    section_start: int,
    section_end: int,
) -> str:
    """비디오의 색션의 관심 섹션을 가져와 VLM에 추론 후 결과를 반환하는 함수.

    Args:
        video_id (str): YouTube video code.
        path_cache (str): 캐쉬 저장용 폴더 경로.
        context_section (str): 관심 섹션의 대사 내용.
        section_start (int): 영상의 섹션 시작 지점.
        section_end (int): 영상의 섹션 종료 시점.

    Returns:
        str: 추론 결과 텍스트.
    """
    # 프롬프트 생성.
    prompt = f"{context_section}\n\n{video_id}_{section_start}_{section_end}"

    download_video(video_id, path_cache)
    slice_video(video_id, section_start, section_end)
    result = infer_with_video(prompt, video_id, section_start, section_end)

    return result
