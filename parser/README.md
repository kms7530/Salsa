# Bako: 독립으로 생성된 모델 서비스의 선택적으로 Packing

ML Model이 디자인된 코드를 통합하여 API 엔드포인트로 제공하기 위한 서비스 생성 코드입니다.

아래와 같이 설정하여 `bako` 서비스를 생성합니다.

### 모델 서비스 폴더 구조
`models` 폴더 아래에 ML 서비스 코드를 선언합니다.

```text
Salsa
├── models
│   ├── your_bentoml_model_1.py
│   ├── your_bentoml_model_2.py
│   ...
```

### Config 세팅
`bako` 서비스에 편입시킬 서비스의 클래스명을 `MODEL_SELECT_LOAD`에 설정합니다. 
```python
class Config:
    # ... 다른 옵션들 ...

    # models 폴더 아래의 정의된 모델 클래스명을 작성한다.
    MODEL_SELECT_LOAD = [
        "DINO", 
        "OCR", 
        "VisionLanguage"
    ]

    # ... 다른 옵션들 ...
```


### 실행
위 설정이 완료되고 `generator`를 실행하여 `bako` 서비스를 생성합니다.
```shell
cd salsa # 프로젝트 폴더로 진입
python -m parser.code_generator {service 파일 이름}
```

