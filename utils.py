import psutil
from typing import Dict
from config import Config


def get_available_memory() -> float:
    """
    현재 시스템에서 사용 가능한 메모리를 반환합니다.

    Returns:
        float: 사용 가능한 메모리 (GB 단위)
    """
    memory = psutil.virtual_memory()
    available_memory_gb = memory.available / (1024**3)  # GB로 변환
    return available_memory_gb


def check_system_memory() -> Dict[str, bool]:
    """
    시스템 메모리가 설정된 요구사항을 충족하는지 확인합니다.

    Returns:
        Dict[str, bool]: 메모리 요구사항 충족 여부를 나타내는 딕셔너리
    """
    available_memory = get_available_memory()
    total_required_memory = sum(Config.MEMORY_REQUIREMENTS.values())

    result = {
        # 시스템이 기본적인 작동을 위한 최소한의 메모리 요구사항을 충족하는지 확인
        "min_memory_satisfied": available_memory >= Config.MIN_REQUIRED_MEMORY,
        # 모든 서비스를 동시에 실행할 수 있는 충분한 메모리가 있는지 확인
        "total_memory_satisfied": available_memory >= total_required_memory,
        # 각 서비스가 독립적으로 실행될 수 있는 충분한 메모리가 있는지 확인
        "individual_requirements_satisfied": all(
            available_memory >= mem_req
            for mem_req in Config.MEMORY_REQUIREMENTS.values()
        ),
    }
    print(result)

    return result


def memory_check(required_memory: float):
    """
    메모리 체크를 수행하는 데코레이터

    Args:
        required_memory (float): 필요한 메모리량 (GB)

    Returns:
        function: 데코레이터 함수
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            available_memory = get_available_memory()
            if available_memory < required_memory:
                raise MemoryError(
                    f"Not enough available memory for {func.__name__}. "
                    f"Required: {required_memory}GB, Available: {available_memory:.2f}GB"
                )
            return func(*args, **kwargs)

        return wrapper

    return decorator


def print_memory_check_result(check_result: Dict[str, bool]) -> None:
    """
    메모리 체크 결과를 출력합니다.

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

    if not all(check_result.values()):
        print("\nWarning: Some memory requirements are not satisfied!")
        available_memory = get_available_memory()
        print(f"Available Memory: {available_memory:.2f}GB")
        print(f"Minimum Required Memory: {Config.MIN_REQUIRED_MEMORY}GB")
        print(
            f"Total Required Memory: {sum(Config.MEMORY_REQUIREMENTS.values()):.2f}GB"
        )
        print("\nIndividual Memory Requirements:")
        for service, req_memory in Config.MEMORY_REQUIREMENTS.items():
            print(f"  {service}: {req_memory}GB")
