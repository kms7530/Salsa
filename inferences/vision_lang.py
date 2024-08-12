import numpy as np
import torch
from decord import VideoReader, cpu
from longva.constants import IMAGE_TOKEN_INDEX
from longva.mm_utils import process_images, tokenizer_image_token
from longva.model.builder import load_pretrained_model
from PIL import Image

# 모델, 이미지 프로세서 그리고 토크나이저 불러오기.
torch.manual_seed(0)

# TODO: LongVA와 Torch Poetry에 추가 필요.
__model_path = "lmms-lab/LongVA-7B-DPO"
__gen_kwargs = {
    "do_sample": True,
    "temperature": 0.5,
    "top_p": None,
    "num_beams": 1,
    "use_cache": True,
    "max_new_tokens": 1024,
}

__tokenizer, __model, __image_processor, _ = load_pretrained_model(
    __model_path, None, "llava_qwen", device_map="cuda:0"
)


def __generate_prompt(prompt: str) -> torch.Tensor:
    """입력된 프롬프트를 모델에서 추론 가능하게 변환하여 Tensor로 반환하는 함수.

    Args:
        prompt (str): 모델에 추론시 이용할 프롬프트.

    Returns:
        torch.Tensor: 모델에 입력 가능한 Tensor
    """

    prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<image>\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    input_ids = (
        tokenizer_image_token(
            prompt, __tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        )
        .unsqueeze(0)
        .to(__model.device)
    )

    return input_ids


# TODO: 서빙을 위해 추가 작업 동기화와 같은 작업 필요.
def __run_inference(
    input_ids: torch.Tensor, input_tensor: torch.Tensor, modalities: str
) -> str:
    """Modality에 따른 추론 후 결과를 반환하는 내부 함수.

    Args:
        input_ids (torch.Tensor): 입력될 프롬프트의 Tensor.
        input_tensor (torch.Tensor): 입력될 영상의 Tensor.
        modalities (str): 추론 모드. (image, video 중 하나. )

    Returns:
        str: 추론한 결과 텍스트.
    """

    # Modality에 따른 추론 결과 가져오기.
    if modalities == "video":
        with torch.inference_mode():
            output_ids = __model.generate(
                input_ids,
                images=[input_tensor],
                modalities=[modalities],
                **__gen_kwargs,
            )
    elif modalities == "image":
        with torch.inference_mode():
            output_ids = __model.generate(
                input_ids,
                images=input_tensor,
                image_sizes=[input_tensor.size],
                modalities=[modalities],
                **__gen_kwargs,
            )

    outputs = __tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    return outputs


def infer_with_image(prompt: str, image_path: str = "", image_PIL: Image = None) -> str:
    """이미지 파일을 이용한 LongVA 추론 함수.

    Args:
        prompt (str): 추론시 이용할 함수.
        image_path (str, optional): 추론시 이용할 이미지 경로. (Defaults to "";`image_path`, `image_PIL` 중 하나만 지정. )
        image_PIL (Image, optional): 추론시 이용할 PIL 이미지 객체. (Defaults to None;`image_path`, `image_PIL` 중 하나만 지정. )

    Raises:
        ValueError: `image_path`, `image_PIL` 중 하나도 입력되지 않는 경우 발생.

    Returns:
        str: 추론 결과.
    """

    # 프롬프트 생성 및 토큰으로 변환.
    input_ids = __generate_prompt(prompt)

    # 모든 이미지 관련 변수가 입력되지 않은 경우 오류 발생.
    if image_path == "" and image_PIL == None:
        raise ValueError("image_path or image_PIL must be provided.")

    # 입력된 이미지 변수에 따른 `image` 변수 생성.
    if image_path != "":
        image = Image.open(image_path).convert("RGB")
    elif image_PIL != None:
        image = image_PIL.convert("RGB")

    images_tensor = process_images([image], __image_processor, __model.config).to(
        __model.device, dtype=torch.float16
    )

    output_ids = __run_inference(input_ids, images_tensor, "image")
    outputs = __tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    return outputs


def infer_with_video(prompt: str, video_path: str, max_frames_num: int = 16) -> str:
    """비디오 파일을 이용한 LongVA 추론 함수.

    Args:
        prompt (str): 추론시 이용할 함수.
        video_path (str): 수론시 이용할 영상 저장 경로.
        max_frames_num (int): 추론시 이용할 프레임 갯수. (GPU 환경에 따라 유동 설정. )

    Returns:
        str: 추론 후 결과.
    """

    # 프롬프트 생성 및 토큰으로 변환.
    input_ids = __generate_prompt(prompt)

    # 비디오 파일 불러오기.
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frame_num = len(vr)
    uniform_sampled_frames = np.linspace(
        0, total_frame_num - 1, max_frames_num, dtype=int
    )

    frame_idx = uniform_sampled_frames.tolist()
    frames = vr.get_batch(frame_idx).asnumpy()

    video_tensor = __image_processor.preprocess(frames, return_tensors="pt")[
        "pixel_values"
    ].to(__model.device, dtype=torch.float16)

    # 추론.
    output_ids = __run_inference(input_ids, video_tensor, "video")
    outputs = __tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    return outputs
