import os
import psutil
from PIL import Image
import torch
from pathlib import Path

from service import VisionLanguage, DINO, OCR, Bako


def measure_memory_usage(func, *args, **kwargs):
    """
    주어진 함수의 메모리 사용량을 측정합니다.

    Args:
        func: 측정할 함수
        *args, **kwargs: 함수에 전달할 인자들

    Returns:
        float: 사용된 메모리량 (GB)
    """
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss
    func(*args, **kwargs)
    mem_after = process.memory_info().rss
    return (mem_after - mem_before) / (1024**3)  # GB로 변환


def profile_services():
    """
    각 서비스의 메모리 사용량을 프로파일링합니다.
    """
    # 테스트용 데이터 준비
    test_image = Image.new("RGB", (100, 100))
    test_video_path = Path("./video.mp4")  # 실제 비디오 파일 경로로 변경 필요
    test_prompt = "Test prompt"

    # 각 서비스 초기화
    vision_language = VisionLanguage()
    dino = DINO()
    ocr = OCR()

    # 메모리 사용량 측정
    memory_usage = {
        "infer_with_video": measure_memory_usage(
            vision_language.infer_with_video, test_prompt, test_video_path
        ),
        "infer_with_image": measure_memory_usage(
            vision_language.infer_with_image, test_prompt, test_image
        ),
        "infer_ground_box": measure_memory_usage(
            dino.infer_ground_box, test_prompt, test_image
        ),
        "infer_img_to_text": measure_memory_usage(ocr.infer_img_to_text, test_image),
    }

    print("Measured Memory Requirements:")
    for service, usage in memory_usage.items():
        print(f"{service}: {usage:.2f} GB")

    return memory_usage


if __name__ == "__main__":
    profile_services()
