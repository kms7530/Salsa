import ast
import os

from pprint import pprint


class CodeParser:
    def __init__(self, filename):
        self.filename = filename
        self.tree = self.__parse()
        self.class_parsed = self.__extract_class_defined()

    def __parse(self):
        with open(self.filename) as f:
            return ast.parse(f.read())

    def __parse_decorator_args(self, decorator):
        return {kw.arg: ast.unparse(kw.value) for kw in decorator.keywords}

    def __parse_function_signiture(self, func_node):
        params = [
            (arg.arg, ast.unparse(arg.annotation) if arg.annotation else "None")
            for arg in func_node.args.args
        ]
        return_type = ast.unparse(func_node.returns) if func_node.returns else "None"
        return {"params": params, "return_type": return_type}

    def __extract_decorator(self, decorator):
        return (
            {"decorator_args": self.__parse_decorator_args(decorator)}
            if (
                isinstance(decorator, ast.Call)
                and isinstance(decorator.func, ast.Attribute)
                and decorator.func.attr == "api"
            )
            else {}
        )

    def __parse_methods(self, class_node):
        methods = []
        for body_item in class_node.body:
            if isinstance(body_item, ast.FunctionDef):
                method_info = self.__parse_function_signiture(body_item)
                method_info["name"] = body_item.name

                for decorator in body_item.decorator_list:
                    if decorator_info := self.__extract_decorator(decorator):
                        method_info.update(decorator_info)
                        methods.append(method_info)

        return methods

    def __is_bentoml_service(self, class_node):
        return any(
            isinstance(decorator, ast.Call)
            and (
                isinstance(decorator.func, ast.Attribute)
                and decorator.func.attr == "service"
            )
            or (isinstance(decorator.func, ast.Name) and decorator.func.id == "service")
            for decorator in class_node.decorator_list
        )

    def __extract_class_defined(self):
        return [
            {
                "class_name": node.name,
                "filename": os.path.splitext(os.path.basename(self.filename))[0],
                "methods": self.__parse_methods(node),
            }
            for node in ast.walk(self.tree)
            if isinstance(node, ast.ClassDef) and self.__is_bentoml_service(node)
        ]

    def bentoml_classes(self):
        return self.class_parsed


if __name__ == "__main__":
    from glob import glob

    model_services_py = glob("models/*.py")

    for filename in model_services_py:
        parser = CodeParser(filename)
        print(parser.bentoml_classes())
