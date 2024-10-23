from .code_parser import CodeParser
from jinja2 import Template

import argparse


class ServiceCodeGenerator:
    def __init__(self, template_path: str):
        with open(template_path) as f:
            self.template = Template(f.read())

    def generate(self, classes: list) -> str:
        methods = [
            {"cls_name": cls["class_name"], **method}
            for cls in classes
            for method in cls["methods"]
        ]
        models = [
            {"filename": cls["filename"], "class_name": cls["class_name"]}
            for cls in classes
        ]

        print(methods)
        return self.template.render(models=models, methods=methods)

    def save_code(self, code: str, output_file: str):
        with open(output_file, "w") as f:
            f.write(code)


def main(args):
    from glob import glob
    from config import Config

    model_service_py = glob("models/*.py")
    target_classes = []

    for filename in model_service_py:
        parser = CodeParser(filename)
        for cls in parser.bentoml_classes():
            if cls["class_name"] not in Config.MODEL_SELECT_LOAD:
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
