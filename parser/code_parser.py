import ast
import os

from typing import Dict, List
from pprint import pprint


class CodeParser:
    def __init__(self, filename):
        """
        Args:
            filename (str): BentoML 서비스 모델 코드의 경로
        """
        self.filename = filename
        self.tree = self.__parse()
        self.class_parsed = self.__extract_class_defined()

    def __parse(self) -> ast.AST:
        """주어진 소스 코드를 읽어 AST객체로 변환합니다.
        Returns:
            ast.AST: 추상 구문 트리로 파싱된 소스 코드.
        """
        with open(self.filename) as f:
            return ast.parse(f.read())

    def __parse_decorator_args(self, decorator: ast.expr) -> Dict:
        """코드에서 `decorator`의 인자를 파싱합니다.

        Args:
            decorator (ast.expr): 데코레이터 노드.

        Returns:
            Dict: 데코레이터의 인수 딕셔너리.
        """
        return {kw.arg: ast.unparse(kw.value) for kw in decorator.keywords}

    def __parse_function_signiture(self, func_node: ast.FunctionDef) -> Dict:
        """함수 파라미터와 리턴 타입을 파싱합니다.

        Args:
            func_node (ast.FunctionDef): 함수 노드 객체.

        Returns:
            Dict: 파싱된 메소드의 파라미터와 리턴 타입.
        """
        params = [
            (arg.arg, ast.unparse(arg.annotation) if arg.annotation else "None")
            for arg in func_node.args.args
        ]
        return_type = ast.unparse(func_node.returns) if func_node.returns else "None"
        return {"params": params, "return_type": return_type}

    def __extract_decorator(self, decorator: ast.expr) -> Dict:
        """BentoML API 데코레이터 패턴과 일치하는지 확인한 후, 관련 정보를 추출합니다.

        Args:
            decorator (ast.expr): 데코레이터 노드.

        Returns:
            Dict: `@api` 데코레이터와 일치할 경우, 데코레이터의 인수를 포함한 딕셔너리를 반환.
                                      일치하지 않으면 빈 딕셔너리를 반환.
        """
        return (
            {"decorator_args": self.__parse_decorator_args(decorator)}
            if (
                isinstance(decorator, ast.Call)
                and isinstance(decorator.func, ast.Attribute)  # function 객체인지 체크
                and decorator.func.attr == "api"  # bentoml."api"
            )
            else {}
        )

    def __parse_methods(self, class_node: ast.ClassDef) -> List[Dict]:
        """주어진 클래스 노드에서 메서드를 파싱하여 각 메서드의 정보(이름, 파라미터, 리턴 타입, 데코레이터)를 추출합니다.

        Args:
            class_node (ast.ClassDef): 클래스 정의 노드.

        Returns:
            List[Dict]: 각 메서드의 이름, 파라미터, 리턴 타입, 데코레이터 정보를 포함한 딕셔너리 리스트.
        """
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

    def __is_bentoml_service(self, class_node: ast.expr) -> bool:
        """주어진 클래스 노드가 BentoML 서비스 클래스인지 확인합니다.

        Args:
            class_node (ast.expr): 추상 구문 트리(AST)의 클래스 노드 객체.

        Returns:
            bool: 클래스가 BentoML 서비스 클래스인지 여부.
        """
        return any(
            isinstance(decorator, ast.Call)
            and (
                isinstance(decorator.func, ast.Attribute)
                and decorator.func.attr == "service"
            )
            or (isinstance(decorator.func, ast.Name) and decorator.func.id == "service")
            for decorator in class_node.decorator_list
        )

    def __extract_class_defined(self) -> List[Dict]:
        """BentoML 서비스 클래스를 추출하고, 해당 클래스의 이름, 파일명, 메서드를 딕셔너리 형태로 반환합니다.

        Returns:
            List[Dict]: 추출된 클래스 정보를 포함한 딕셔너리 리스트.
        """

        return [
            {
                "class_name": node.name,
                "filename": os.path.splitext(os.path.basename(self.filename))[0],
                "methods": self.__parse_methods(node),
            }
            for node in ast.walk(self.tree)
            if isinstance(node, ast.ClassDef) and self.__is_bentoml_service(node)
        ]

    def bentoml_classes(self) -> List[Dict]:
        """파싱된 BentoML 서비스 클래스 정보를 반환합니다.

        Returns:
            List[Dict]: BentoML 서비스 클래스를 담고 있는 딕셔너리 리스트.
        """
        return self.class_parsed


if __name__ == "__main__":
    from glob import glob

    model_services_py = glob("models/*.py")

    for filename in model_services_py:
        parser = CodeParser(filename)
        print(parser.bentoml_classes())
