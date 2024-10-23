import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PIL import Image
import torch
from pathlib import Path
import time
from service import VisionLanguage, DINO, OCR, Bako


def measure_memory_usage_and_get_result(func, *args, **kwargs):
    """
    주어진 함수의 GPU 메모리 사용량을 측정하고 결과를 반환합니다.

    Args:
        func: 측정할 함수
        *args, **kwargs: 함수에 전달할 인자들

    Returns:
        tuple: (사용된 GPU 메모리량 (GB), 함수의 반환값)
    """
    torch.cuda.empty_cache()
    mem_before = torch.cuda.memory_allocated() / (1024**3)  # GB로 변환
    result = func(*args, **kwargs)
    mem_after = torch.cuda.memory_allocated() / (1024**3)  # GB로 변환
    memory_used = mem_after - mem_before
    return memory_used, result


def measure_load_memory(service_class):
    """
    서비스 클래스의 인스턴스를 생성할 때 사용되는 GPU 메모리 양을 측정합니다.

    Args:
        service_class: 측정할 서비스 클래스

    Returns:
        float: 사용된 GPU 메모리량 (GB)
    """
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    mem_before = torch.cuda.max_memory_allocated() / (1024**3)  # GB로 변환

    # 인스턴스 생성
    service = service_class()

    # GPU 연산 완료를 기다림
    torch.cuda.synchronize()
    time.sleep(1)  # 추가적인 지연

    mem_after = torch.cuda.max_memory_allocated() / (1024**3)  # GB로 변환
    memory_used = mem_after - mem_before
    return memory_used


def profile_services():
    """
    각 서비스의 메모리 사용량을 프로파일링합니다.
    """
    # 테스트용 데이터 준비
    test_image_path = Path(__file__).parent.parent / "docs" / "IMG_1206.png"

    # 파일이 실제로 존재하는지 확인
    if not test_image_path.exists():
        raise FileNotFoundError(f"테스트 이미지를 찾을 수 없습니다: {test_image_path}")
    # 실제 이미지 파일 경로로 변경 필요
    try:
        test_image = Image.open(test_image_path)
    except FileNotFoundError:
        print(f"Unable to open image file: {test_image_path}")
        print("Falling back to generated image.")
        test_image = Image.new("RGB", (100, 100), color=(73, 109, 137))

    test_video_path = Path("/path/to/your/video.mp4")  # 실제 비디오 파일 경로로 변경 필요
    test_prompt = "영상은 어떤 내용을 담고있죠?"
    test_prompt1 = "사진은 어떠한 모습입니까?"
    test_prompt2 = "글자에 박스를 쳐주세요"

    # 각 서비스의 로딩 메모리 측정
    load_memory = {}
    print("\nModel Loading Memory Usage:")
    for service_name, service_class in [
        ("VisionLanguage", VisionLanguage),
        ("DINO", DINO),
        ("OCR", OCR),
    ]:
        load_mem = measure_load_memory(service_class)
        load_memory[service_name] = load_mem
        print(f"{service_name}: {load_mem:.2f} GB")

    # 각 서비스 초기화
    vision_language = VisionLanguage()
    dino = DINO()
    ocr = OCR()

    # 서비스 함수와 인자 정의
    services = {
        "infer_with_video": (
            vision_language.infer_with_video,
            (test_prompt, test_video_path),
        ),
        "infer_with_image": (
            vision_language.infer_with_image,
            (test_prompt1, test_image),
        ),
        "infer_ground_box": (dino.infer_ground_box, (test_prompt2, test_image)),
        "infer_img_to_text": (ocr.infer_img_to_text, (test_image,)),
    }

    inference_memory = {}
    print("\nInference Memory Usage and Results:")
    for service_name, (func, args) in services.items():
        try:
            usage, result = measure_memory_usage_and_get_result(func, *args)
            inference_memory[service_name] = usage
            print(f"\n{service_name}:")
            print(f"Memory Usage: {usage:.2f} GB")
            print(f"Result: {result}")
        except Exception as e:
            print(f"\n{service_name}:")
            print(f"Error occurred: {e}")

    print("\nSummary of Measured Memory Requirements:")
    print("Model Loading Memory:")
    for service, usage in load_memory.items():
        print(f"{service}: {usage:.2f} GB")
    print("\nInference Memory:")
    for service, usage in inference_memory.items():
        print(f"{service}: {usage:.2f} GB")

    return load_memory, inference_memory


if __name__ == "__main__":
    profile_services()
