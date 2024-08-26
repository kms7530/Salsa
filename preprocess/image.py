from PIL import Image


def crop_image(pil_img: Image, left: int, top: int, right: int, bottom: int) -> Image:
    """입력받은 PIL 이미지를 지정된 구역만큼 crop하여 반환하는 함수.

    Args:
        pil_img (Image): 크롭할 원본 이미지(PIL 객체).
        left (int): 크롭할 영역의 왼쪽 좌표.
        top (int): 크롭할 영역의 위쪽 좌표.
        right (int): 크롭할 영역의 오른쪽 좌표.
        bottom (int): 크롭할 영역의 아래쪽 좌표.

    Returns:
        Image: 크롭된 이미지(PIL 객체).
    """

    cropped_img = pil_img.crop((left, top, right, bottom))
    return cropped_img
