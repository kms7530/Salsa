from googleapiclient.discovery import build
import requests
import os

api_key = os.environ.get("YOUTUBE_API_KEY")
if not api_key:
    raise ValueError("YOUTUBE_API_KEY 환경 변수가 설정되지 않았습니다.")

youtube = build("youtube", "v3", developerKey=api_key)


def get_video_thumbnail(video_id):
    request = youtube.videos().list(part="snippet", id=video_id)
    response = request.execute()

    if "items" in response and len(response["items"]) > 0:
        thumbnails = response["items"][0]["snippet"]["thumbnails"]
        if "maxres" in thumbnails:
            return thumbnails["maxres"]["url"]
        elif "high" in thumbnails:
            return thumbnails["high"]["url"]
        else:
            return thumbnails["default"]["url"]
    else:
        return None


def download_thumbnail(video_id, path_cache):
    thumbnail_url = get_video_thumbnail(video_id)
    if thumbnail_url:
        filename = os.path.join(path_cache, f"{video_id}_thumbnail.jpg")
        response = requests.get(thumbnail_url)
        if response.status_code == 200:
            with open(filename, "wb") as file:
                file.write(response.content)
            return filename
    return None
