# 💃 Salsa
<p align="center">
  <img 
    src="https://github.com/kms7530/Salsa/blob/main/docs/IMG_1206.png?raw=true" 
    alt="사우론의 눈"
    style="width: 500px;"
  />
</p>

# 🏗️ 프로젝트 구조
```plain text
./Salsa
├── README.md
├── api.py 						-> FastAPI 배포 코드.
├── bentofile.yaml              -> BentoML 배포 코드.
├── config_template.py          -> `config.py` 예시. 
├── docs                		-> 구조 및 참조 문서. 
│   ├── call_dif_server.d2
│   ├── call_same_server.d2
│   ├── docs.d2
│   ├── docs.svg
│   ├── vigilant.d2
│   └── vigilant.svg
├── inferences          		-> VLM, OCR 및 LLM 서비스 추론용 wrapper 라이브러리.
│   ├── lang.py         		-> LLM 서비스 추론용. 
│   ├── ocr.py          		-> OCR 추론용. 
│   └── vision_lang.py  		-> VLM 추론용(Image, Video). 
├── install.sh          		-> 의존성 라이브러리 설치를 위한 shell script. 
├── main.py             		-> 실행 파일. 
├── models                  -> BentoML 모델 서비스 코드.
│   ├── your_bentoml_model.py
│   ...
├── pipelines           		-> 각 파이프라인 부분. 
│   └── video_section.py
├── parser
│   ├── code_generator.py   -> Bako 코드 생성기
│   ├── code_parser.py      -> BentoML 모델 서비스 코드 파싱
│   └── CODE_TEMPLATE       -> Bako 코드 템플릿
├── preprocess          		-> 전처리 관련 라이브러리. 
│   ├── image.py        		-> 이미지 전처리. 
│   ├── prompt.py       		-> 프롬프트 전처리. 
│   └── video.py        		-> 비디오 전처리. 
├── prompts             		-> 프롬프트 저장용 폴더. 
│   └── EXAMPLE         		-> 프롬프트 예시 파일. 
├── requirements.txt
└── service.py                  -> BentoML 배포 코드. 
```

## 규칙
- 각자 작업중인 Pipeline에 대해 `./pipelines`에 파일을 생성 후 작업하시면 됩니다. 
- Pipeline 동작 중 전처리가 필요한 부분은 `preprocess`, 모델을 통한 부분은 `inferences`에 생성 후 이용해주시기 바랍니다. 

# 🚀 실행 방법
## 의존성 라이브러리 설치
- 다음의 방법을 이용하여 의존성 파일을 설치합니다. 
```bash
# install.sh의 실행 권한 부여. 
chmod -R 666 ./install.sh

# 의존성 라이브러리 설치(LongVA, Vigilant)
bash ./install.sh
```

## 설정 파일 수정
- 다음의 명령어를 이용하여 설정 템플릿을 수정을 합니다. 
```bash
# VIM을 이용하여 설정 수정 후 저장. 
vim config.py
vim bentoml
```

## 프로젝트 실행
- 다음의 명령어를 이용하여 프로젝트를 실행합니다. 
- 최초 모델 서버 구동 시 <b>모델 다운로드로 인해 시간이 다소 소요</b>될수 있습니다. 
```bash
cd salsa # 프로젝트 폴더로 진입
python -m parser.code_generator {service 파일 이름}

# 위의 로그 확인 후 문제가 발생하지 않은 경우,
# 다음 명령어로 모델 배포. 
bentoml serve service:BakoService -p PORT_NUMBER

# 다음의 명령어로 FastAPI 서버 가동. 
uvicorn api:app
```

# 📝 프롬프트 작성
프롬프트는 `./promts/`에 저장 후 사용하시면 됩니다. 

프롬프트 파일의 양식은 Jinja template 사용방법과 같습니다. 이에 대한 예시는 다음과 같습니다. 

```plain text
입력된 고분자 이미지의 이름을 출력한 후 텍스트로 입력된 고분자와 강성 대해 비교하여 논하시오. 

{{ polymer_struct }}
```

이후 저장된 프롬프트를 이용하기 위해서는, `preprocess/prompt.py`에 다음과 같은 방식으로 함수를 제작하여 사용 가능합니다. 
```python
...
templateLoader = jinja2.FileSystemLoader(searchpath="./prompts")
templateEnv = jinja2.Environment(loader=templateLoader)
template = templateEnv.get_template(path_prompt)

prompt = template.render(template=template)
...
```
