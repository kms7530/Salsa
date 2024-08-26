import jinja2


def get_video_section_prompt(path_prompt: str = "video_section.txt", **kwargs) -> str:
    """비디오 내의 대사를 가져와 프롬프트를 반환하는 함수.

    Args:
        context_section (str): 지정된 섹션에 해당하는 대사 텍스트.
        path_prompt (str, optional): 프롬프트 파일의 경로. Defaults to "video_section.txt".

    Returns:
        str: 생성된 프롬프트.
    """

    # Jinja template 사용을 위한 설정 및 템플릿 불러오기.
    templateLoader = jinja2.FileSystemLoader(searchpath="./prompts")
    templateEnv = jinja2.Environment(loader=templateLoader)
    template = templateEnv.get_template(path_prompt)

    prompt = template.render(**kwargs)

    return prompt
