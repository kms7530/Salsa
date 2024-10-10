# momory checking 기능 사용 가이드

memory check 기능을 사용하기 위해 읽어봐야할 가이드입니다.

## 목적

memory profile 기능은 본 service에서 제공되는 AI 서비스의 메모리 사용량을 모델 작동전에 미리 확인하는 과정을 통해  원활한 작동을 보장하고자 만들어졌습니다. memory profile 코드를 동작후, 개별적인 환경에서의 최대 메모리 사용 가용량과 본 서비스에 필요한 메모리 최대량을 비교하여 메모리 부족 오류를 사전에 방지할 수 있습니다. 

## Step

1. 테스트 비디오 준비
2. 메모리 프로파일러 실행 (테스트 비디오 경로 변경 필요)
3. 설정 파일(`config_template.py`) 업데이트
4. 업데이트된 설정을 애플리케이션에서 사용

### Step 1. 테스트 비디오 준비

메모리 프로파일링을 위해서는 테스트용 비디오 파일이 필요합니다. 다음 두 가지 옵션 중 하나를 선택하세요

a) 자체 테스트 비디오 사용
   - 실제로 사용하고자 하는 동영상이 있다면, 해당 동영상의 경로를 `test_video_path`에 설정하세요.
   - 이는 가장 정확한 메모리 사용량을 측정할 수 있는 방법입니다.

b) 샘플 비디오 다운로드
   - 테스트용 동영상이 없다면, 다음 명령어를 사용하여 샘플 비디오를 다운로드하세요
     ```
     wget https://github.com/EvolvingLMMs-Lab/LongVA/raw/main/local_demo/assets/dc_demo.mp4
     ```
   - 다운로드 후, `Salsa/memory_check/memory_profile.py`에서 `test_video_path`를 `다운로드한 파일의 경로`로 설정하세요.

`memory_profile.py` 파일에서 `test_video_path`를 다음과 같이 수정하세요

```python
test_video_path = Path("/path/to/your/video.mp4")  # 실제 비디오 파일 경로로 변경
```

### Step 2. 메모리 프로파일러 실행

`memory_profile.py` 스크립트를 실행합니다. 이 스크립트는 각 서비스의 메모리 사용량을 측정하고 요약을 제공합니다.

```bash
python memory_profile.py
```

예상 출력

```
측정된 메모리 요구사항 요약:
모델 로딩 메모리:
VisionLanguage: 17.81 GB
DINO: 2.03 GB
OCR: 0.09 GB

추론 메모리:
infer_with_video: 0.23 GB
infer_with_image: 0.10 GB
infer_ground_box: 0.64 GB
infer_img_to_text: 0.01 GB
```

### Step 3. 설정 파일 업데이트

메모리 프로파일러의 결과를 사용하여 설정 파일(`config_template.py`)의 `Config` 클래스를 업데이트합니다. 다음 항목을 업데이트해야 합니다

1. `MIN_REQUIRED_MEMORY`: "모델 로딩 메모리" 값들을 합산하고 소숫점은 올림하여 계산합니다.
2. `MEMORY_REQUIREMENTS`: "추론 메모리" 값들을 복사합니다.

### 예시

```python
class Config:
    # ... 다른 옵션들 ...

    MIN_REQUIRED_MEMORY: float = 19.93  # 모델 로딩 메모리의 합, 올림 처리 필요

    MEMORY_REQUIREMENTS = {
        "infer_with_video": 0.23,
        "infer_with_image": 0.10,
        "infer_ground_box": 0.64,
        "infer_img_to_text": 0.01,
    }
```

### Step 4. 업데이트된 설정을 애플리케이션에서 사용

`Config` 클래스를 업데이트했으므로, 이제 애플리케이션은 각 서비스를 실행하기 전에 충분한 메모리가 있는지 확인하기 위해 이 값들을 사용할 것입니다.

## Detail

- AI 모델이나 서비스를 업데이트한 후에는 주기적으로 메모리 프로파일러를 실행하세요.
- `MIN_REQUIRED_MEMORY` 계산 시에는 약간 올림 처리하세요.
- 가능하다면 실제 사용 환경과 유사한 비디오로 테스트하여 더 정확한 메모리 사용량을 측정하세요.

## "메모리 부족" 오류가 발생하면

1. 먼저 메모리 프로파일러를 실행하여 설정이 최신 상태인지 확인합니다.
2. 시스템이 최소 메모리 요구사항을 충족하는지 확인합니다.
3. 만약 현재 환경의 메모리 용량이 요구사항에 부족하다면, 하드웨어 메모리 증설을 고려해야 합니다.
