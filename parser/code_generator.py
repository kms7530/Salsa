from .code_parser import CodeParser
from jinja2 import Template

import argparse


class ServiceCodeGenerator:
    def __init__(self, template_path: str) -> None:
        """지정된 형식의 탬플릿 코드를 바탕으로 BentoML 서비스코드를 생성합니다.

        Args:
            template_path (str): 탬플릿코드 경로
        """
        with open(template_path) as f:
            self.template = Template(f.read())

    def generate(self, classes: list) -> str:
        """CodeParser에서 파싱한 클래스로부터 서비스 코드를 생성합니다.

        Args:
            classes (list): CodeParser에서 전달받은 클래스 리스트

        Returns:
            str: BentoML 서비스 코드
        """
        methods = [
            {"cls_name": cls["class_name"], **method}
            for cls in classes
            for method in cls["methods"]
        ]
        models = [
            {"filename": cls["filename"], "class_name": cls["class_name"]}
            for cls in classes
        ]

        return self.template.render(models=models, methods=methods)

    def save_code(self, code: str, output_file: str) -> None:
        """코드와 파일명을 전달받아 코드를 저장합니다.

        Args:
            code (str): BentoML 서비스 코드
            output_file (str): 저장할 파일명
        """
        with open(output_file, "w") as f:
            f.write(code)


def main(args: argparse.Namespace) -> None:
    """모델 디렉토리에서 BentoML 서비스 클래스를 파싱하고, 선택된 클래스에 대해 서비스 코드를 생성하여 저장합니다.

    Args:
        args (argparse.Namespace): 서비스 이름을 포함한 명령어 인수.
    """
    from glob import glob
    from config import Config

    model_service_py = glob("models/*.py")
    target_classes = []

    for filename in model_service_py:
        parser = CodeParser(filename)
        for cls in parser.bentoml_classes():
            if (
                cls["class_name"] not in Config.MODEL_SELECT_LOAD
            ):  # Config에서 선택한 모델만 로드함
                continue

            target_classes.append(cls)

    generator = ServiceCodeGenerator("./parser/CODE_TEMPLATE")
    code = generator.generate(target_classes)
    generator.save_code(code, args.service_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "service_name", type=str, help="Service name", default="service.py"
    )
    args = parser.parse_args()
    main(args)
