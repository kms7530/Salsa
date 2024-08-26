from openai import OpenAI
from config import config
from typing import Tuple
from tokencost import (
    calculate_completion_cost,
    calculate_prompt_cost,
    count_string_tokens,
)


def calculate_tokens(
    prompt: str, result: str, model: str
) -> Tuple[float, float, int, int]:
    """정해진 모델에 따른 프롬프트와 완성에 따른 비용 계산 함수.

    Args:
        prompt (str): 추론에 사용된 프롬프트.
        completion (str): 프롬프트 추론 후 나온 결과.
        model (str): 추론시 사용된 모델명.

    Returns:
        tuple[float, float, int, int]: 프롬프트 가격, 결과 가격, 프롬프트 토큰수, 결과 토큰수
    """

    cost_prompt = calculate_prompt_cost(prompt, model)
    cost_response = calculate_completion_cost(result, model)
    tokens_prompt = count_string_tokens(prompt, model)
    tokens_result = count_string_tokens(result, model)

    print(f">>> Prompt Cost: ${cost_prompt} Tokens: {tokens_prompt}")
    print(f">>> completion Cost: ${cost_response} Tokens: {tokens_result}")

    return (cost_prompt, cost_response, tokens_prompt, tokens_result)


def request_to_openai(model_name: str, prompt: str) -> str:
    """OpenAI에서 GPT-3.5에게 prompt 추론 및 결과를 반환하는 함수.

    Args:
        prompt (str): 모델에 추론시킬 프롬프트.

    Returns:
        str: 프롬프트 추론 후 결과.
    """

    client = OpenAI(api_key=config.API_KEY_OPENAI)

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=4096,
        n=1,
        stop=None,
        temperature=0.7,
    )

    return response.choices[0].message.content.strip()
