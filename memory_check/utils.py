import psutil
from typing import Dict
from config_template import Config
import subprocess
import pynvml


def get_available_memory() -> float:
    """
    현재 GPU에서 사용 가능한 메모리를 반환.

    Returns:
        float: 사용 가능한 GPU 메모리 (GB 단위)
    """
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # 첫 번째 GPU 사용
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        available_memory_gb = info.free / (1024**3)  # GB로 변환
        pynvml.nvmlShutdown()
        return available_memory_gb
    except pynvml.NVMLError as error:
        print(f"Error getting GPU memory: {error}")
        return 0.0


def check_system_memory() -> Dict[str, bool]:
    """
    시스템 메모리가 설정된 요구사항을 충족하는지 확인.

    Returns:
        Dict[str, bool]: 메모리 요구사항 충족 여부를 나타내는 딕셔너리
    """
    available_memory = get_available_memory()
    print("사용가능한 메모리:", available_memory, "GB")
    config = Config()
    total_required_memory = sum(config.MEMORY_REQUIREMENTS.values())
    print("사용예정 메모리:", total_required_memory, "GB")

    result = {
        # 시스템이 기본적인 작동을 위한 최소한의 메모리 요구사항을 충족하는지 확인
        "min_memory_satisfied": available_memory >= config.MIN_REQUIRED_MEMORY,
        # 모든 서비스를 동시에 실행할 수 있는 충분한 메모리가 있는지 확인
        "total_memory_satisfied": available_memory >= total_required_memory,
        # 각 서비스가 독립적으로 실행될 수 있는 충분한 메모리가 있는지 확인
        "individual_requirements_satisfied": all(
            available_memory >= mem_req
            for mem_req in config.MEMORY_REQUIREMENTS.values()
        ),
    }

    return result


def print_memory_check_result(check_result: Dict[str, bool]) -> None:
    """
    메모리 체크 결과를 출력.

    Args:
        check_result (Dict[str, bool]): check_system_memory() 함수의 반환값
    """
    print("System Memory Check Results:")
    print(
        f"Minimum Memory Requirement Satisfied: {check_result['min_memory_satisfied']}"
    )
    print(
        f"Total Memory Requirement Satisfied: {check_result['total_memory_satisfied']}"
    )
    print(
        f"Individual Memory Requirements Satisfied: {check_result['individual_requirements_satisfied']}"
    )
    config = Config()
    if not all(check_result.values()):
        print("\nWarning: Some memory requirements are not satisfied!")
        available_memory = get_available_memory()
        print(f"Available Memory: {available_memory:.2f}GB")
        print(f"Minimum Required Memory: {config.MIN_REQUIRED_MEMORY}GB")
        print(
            f"Total Required Memory: {sum(config.MEMORY_REQUIREMENTS.values()):.2f}GB"
        )
        print("\nIndividual Memory Requirements:")
        for service, req_memory in config.MEMORY_REQUIREMENTS.items():
            print(f"  {service}: {req_memory}GB")
