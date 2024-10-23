import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from unittest.mock import patch, MagicMock
import torch
from config_template import Config
from utils import (
    check_system_memory,
    print_memory_check_result,
    get_available_memory,
)


class TestMemoryUtils(unittest.TestCase):
    def setUp(self):
        self.config = Config()

    # 충분한 메모리가 있을떄 제대로 동작하는지 테스트
    @patch("utils.get_available_memory")
    def test_check_system_memory(self, mock_get_available_memory):
        # Mock 사용 가능한 메모리 설정
        mock_get_available_memory.return_value = 25.0  # 25GB

        result = check_system_memory()

        self.assertIn("min_memory_satisfied", result)  # 키가 있는지 확인
        self.assertIn("total_memory_satisfied", result)  # 키가 있는지 확인
        self.assertIn("individual_requirements_satisfied", result)  # 키가 있는지 확인
        self.assertTrue(all(result.values()))  # 모든 값이 True인지 확인

    # 충분한 메모리가 없을떄 제대로 동작하는지 테스트
    @patch("utils.get_available_memory")
    def test_check_system_memory_insufficient(self, mock_get_available_memory):
        mock_get_available_memory.return_value = 15.0  # 15GB

        result = check_system_memory()

        self.assertIn("min_memory_satisfied", result)  # 키가 있는지 확인
        self.assertIn("total_memory_satisfied", result)  # 키가 있는지 확인
        self.assertIn("individual_requirements_satisfied", result)  # 키가 있는지 확인
        self.assertFalse(all(result.values()))  # 모든 값이 False인지 확인

    # get_available_memory 함수가 제대로 동작하는지 테스트 (실제 GPU 없이도 테스트 가능하도록 NVML 함수 mock)
    @patch("pynvml.nvmlInit")
    @patch("pynvml.nvmlDeviceGetHandleByIndex")
    @patch("pynvml.nvmlDeviceGetMemoryInfo")
    @patch("pynvml.nvmlShutdown")
    def test_get_available_memory(
        self, mock_shutdown, mock_get_memory_info, mock_get_handle, mock_init
    ):
        mock_memory_info = MagicMock()
        mock_memory_info.free = (
            20 * 1024 * 1024 * 1024
        )  # 가상의 GPU에 20GB의 사용 가능한 메모리가 있다고 가정
        mock_get_memory_info.return_value = mock_memory_info

        available_memory = get_available_memory()

        self.assertAlmostEqual(available_memory, 20.0, places=1)  # 제대로 mock이 되었는지 확인

    # 메모리 요구사항이 최소 요구사항보다 작은지 테스트
    def test_memory_requirements(self):
        total_required = sum(self.config.MEMORY_REQUIREMENTS.values())
        self.assertLessEqual(
            total_required,
            self.config.MIN_REQUIRED_MEMORY,
            "Total required memory exceeds minimum required memory",
        )

    # print_memory_check_result 함수가 예외를 발생시키지 않고 제대로 동작하는지 테스트 (모든 메모리 요구사항이 충족된 상황)
    def test_print_memory_check_result(self):
        result = {
            "min_memory_satisfied": True,
            "total_memory_satisfied": True,
            "individual_requirements_satisfied": True,
        }
        try:
            print_memory_check_result(result)
        except Exception as e:
            self.fail(
                f"print_memory_check_result raised {type(e).__name__} unexpectedly!"
            )

    # print_memory_check_result 함수가 예외를 발생시키지 않고 제대로 동작하는지 테스트2 (메모리 요구사항이 충족되지 않은 상황)
    @patch("utils.get_available_memory")
    def test_print_memory_check_result_with_warnings(self, mock_get_available_memory):
        mock_get_available_memory.return_value = 15.0  # 15GB
        result = {
            "min_memory_satisfied": False,
            "total_memory_satisfied": False,
            "individual_requirements_satisfied": False,
        }
        try:
            print_memory_check_result(result)
        except Exception as e:
            self.fail(
                f"print_memory_check_result raised {type(e).__name__} unexpectedly!"
            )


if __name__ == "__main__":
    unittest.main()
