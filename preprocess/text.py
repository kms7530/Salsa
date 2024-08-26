import re
import emoji


def __remove_urls(text: str) -> str:
    """문자열 내의 URL을 모두 제거하여 반환하는 함수.

    Args:
        text (str): 원본 문자열.

    Returns:
        str: 처리된 문자열.
    """

    # URL을 정규 표현식을 사용하여 제거
    url_pattern = re.compile(r"http[s]?://\S+|www\.\S+")
    return url_pattern.sub("", text)


def __remove_special_characters(text: str) -> str:
    """문자열 내의 특수문자를 모두 제거하여 반환하는 함수.

    Args:
        text (str): 원본 문자열.

    Returns:
        str: 처리된 문자열.
    """

    # 특수문자를 정규 표현식을 사용하여 제거
    special_chars_pattern = re.compile(r"[▶•]")
    return special_chars_pattern.sub("", text)


def __remove_emojis(text: str) -> str:
    """문자열 내의 이모지를 모두 제거하여 반환하는 함수.

    Args:
        text (str): 원본 문자열.

    Returns:
        str: 처리된 문자열.
    """

    # 이모지를 제거하기 위해 emoji 라이브러리를 사용
    return emoji.replace_emoji(text, replace="")


def clean_text(text: str) -> str:
    """원본 문자열을 전처리하여 반환하는 함수.

    Args:
        text (str): 원본 문자열.

    Returns:
        str: 처리된 문자열.
    """

    text = __remove_urls(text)
    text = __remove_special_characters(text)
    text = __remove_emojis(text)
    return text
