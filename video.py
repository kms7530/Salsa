import os

import moviepy.editor as mp
from pytube import YouTube


def download_video(video_code: str, path_cache: str):
    """지정된 YouTube 영상을 다운로드 받는 함수. (내부 라이브러리 용)

    Args:
        video_code (str): YouTube의 영상 코드.
    """
    # Cache path 생성.
    path_video: str = os.path.join(path_cache, f"{video_code}.mp4")

    # 기존의 파일이 있는 경우, 기존 파일 이용.
    if not os.path.exists(path_video):
        return

    video_url = f"https://www.youtube.com/watch?v={video_code}"
    yt = YouTube(video_url)
    yt.streams.filter(file_extension="mp4").first().download(filename=path_video)


def slice_video(video_code: str, path_cache: str, start: int, end: int):
    """입력된 비디오 코드에 해당하는 비디오를 원하는 구간만 추출하여 저장하는 함수.

    Args:
        video_code (str): YouTube 비디오 코드.
        path_cache (str): Cache 저장 폴더 경로.
        start (int): 영상의 관심 부분 시작 초.
        end (int): 영상의 관심 부분 종료 초.
    """

    # Cache path 생성.
    path_source_video: str = os.path.join(path_cache, f"{video_code}.mp4")
    path_target_video: str = os.path.join(path_cache, f"{video_code}_{start}_{end}.mp4")

    video = mp.VideoFileClip(path_source_video)
    video = video.subclip(start, end)
    video.write_videofile(path_target_video)

    video.close()
