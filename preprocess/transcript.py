import json, os

from googleapiclient.discovery import build
from tqdm import tqdm
from youtube_transcript_api import YouTubeTranscriptApi

from config import config
from typing import Dict, List


def download_video_transcript(video_codes: List) -> bool:
    os.makedirs("cache/transcript", exist_ok=True)
    os.makedirs("cache/meta", exist_ok=True)

    for video_code in tqdm(video_codes):
        meta_data = get_mata(video_code)
        ko, en, tag = get_transcript(video_code)

        with open(f"cache/transcript/{video_code}.transcript", "w") as f:
            json.dump({"ko": ko, "en": en, "tag": tag}, f, indent=4)

        with open(f"cache/meta/{video_code}.meta", "w") as f:
            json.dump(meta_data, f, sort_keys=True, indent=4)

    return True


def get_youtube_api_key() -> str:
    """YouTube API Key를 반환하는 함수.

    Returns:
        str: 현재 사용 가능한 YouTube API Key.
    """

    for api_key in config.API_KEY_YOUTUBE:
        service = build("youtube", "v3", developerKey=api_key)
        try:
            # 할당량 확인을 위해 쿼터 엔드포인트 호출
            # YouTube Data API는 명시적인 할당량 조회 엔드포인트를 제공하지 않으므로,
            # API 요청을 통해 사용량을 추적해야 합니다.

            # 현재 할당량 상태를 가져오기 위한 간단한 API 호출
            request = service.videos().list(
                part="snippet", chart="mostPopular", maxResults=1
            )
            _ = request.execute()

            # 할당량 정보 출력
            print(f">>> API call successful. {api_key}")

            return api_key

        except Exception as e:
            print(f"!!! API Key {api_key} is not avaliable. ")

    return ""


def get_video_title(api_key: str, video_id: str) -> str:
    """지정된 YouTube 영상의 제목을 가져오는 함수.

    Args:
        api_key (str): 사용자가 발급받은 API 키.
        video_id (str): YouTube 영상 코드.

    Returns:
        str: 지정된 YouTube 영상의 제목.
    """

    return __get_video_info_in_snippet(api_key, video_id, "title")


def get_video_description(api_key: str, video_id: str) -> str:
    """지정된 YouTube 영상의 요약 정보를 가져오는 함수.

    Args:
        api_key (str): 사용자가 발급받은 API 키.
        video_id (str): YouTube 영상 코드.

    Returns:
        str: 지정된 YouTube 영상의 요약 정보
    """

    return __get_video_info_in_snippet(api_key, video_id, "description")


def get_channel_title(api_key: str, video_id: str) -> str:
    """지정된 YouTube 영상의 채널명을 가져오는 함수.

    Args:
        api_key (str): 사용자가 발급받은 API 키.
        video_id (str): YouTube 영상 코드.

    Returns:
        str: 지정된 YouTube 영상의 채널명.
    """

    return __get_video_info_in_snippet(api_key, video_id, "channelTitle")


def __get_video_info_in_snippet(api_key: str, video_id: str, section: str) -> str:
    """지정된 YouTube 영상 정보 중 원하는 정보(section)를 가져오는 함수. (내부 라이브러리 용)

    Args:
        api_key (str): 사용자가 발급받은 API 키.
        video_id (str): YouTube 영상 코드.
        section (str): 요청할 정보명.

    Returns:
        str: 요청한 정보 냐용.
    """

    youtube = build("youtube", "v3", developerKey=api_key)
    request = youtube.videos().list(
        part="snippet,statistics,contentDetails", id=video_id
    )
    response = request.execute()

    information = response["items"][0]["snippet"][section]

    return information


def get_mata(video_code: str) -> Dict:
    """입력 받은 YouTube Video Code에 대한 메타데이터 정보를 가져와 반환하는 함수.
    Args:
        video_code (str): YouTube Video Code.
    Returns:
        Dict: YouTube에서 가져온 메타데이터 정보.
    """

    # YouTube API 접속을 위한 API Key 환경변수 가져오기.
    youtube_api_key = get_youtube_api_key()
    if youtube_api_key == None:
        print("YouTube API 사용을 위한 키를 환경변수 `YT_KEY`로 지정 후 다시 실행시켜 주세요. ")
        exit(1)

    video_producer = get_channel_title(youtube_api_key, video_code)
    video_name = get_video_title(youtube_api_key, video_code)
    video_body = get_video_description(youtube_api_key, video_code)

    result = {
        "channel_name": video_producer,
        "video_title": video_name,
        "video_bodytext": video_body,
    }
    return result


def get_transcript(video_id):
    """
    0 -> ko, en manually created
    1 -> ko generated, en manually created
    2 -> ko manually created, en generated
    3 -> ko, en generated
    """
    try:
        list_scripts = YouTubeTranscriptApi.list_transcripts(video_id)
        tag = 0

        if list_scripts._manually_created_transcripts.get("ko"):
            ko_script = list_scripts._manually_created_transcripts["ko"].fetch()
        elif list_scripts._generated_transcripts.get("ko"):
            ko_script = list_scripts._generated_transcripts["ko"].fetch()
            tag |= 1
        else:
            ko_script = {}

        error_in = "en"

        if list_scripts._manually_created_transcripts.get("en"):
            en_script = list_scripts._manually_created_transcripts["en"].fetch()
        elif list_scripts._generated_transcripts.get("en"):
            en_script = list_scripts._generated_transcripts["en"].fetch()
            tag |= 2
        else:
            en_script = {}

        return ko_script, en_script, tag

    except Exception as e:
        print(f"No valid transcript found in {error_in}, {e}")
        return ko_script, en_script, tag


def get_video_id(url):
    return url.split("v=")[1].strip()
