# 💂 Vigilant
<p align="center">
  <img 
    src="http://vignette4.wikia.nocookie.net/lotr/images/9/9f/Sauron_eye_barad_dur.jpg" 
    alt="사우론의 눈"
  />
</p>

# 🏗️ 프로젝트 구조
```plain text
./vigilant
├── README.md
├── api.py 						-> FastAPI 배포 코드. 
├── config
│   └── bento					-> BentoML 배포 설정. 
│       ├── config_dino.yaml	-> DINO 배포 설정. 
│       └── config_vlm.yaml		-> LongVA 배포 설정. 
├── config_template.py          -> `config.py` 예시. 
├── docs                		-> 구조 및 참조 문서. 
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
├── pipelines           		-> 각 파이프라인 부분. 
│   └── video_section.py
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
- 생성해주신 모든 함수에 대해서는, `docstring`을 작성해주시기 바랍니다. 
  - 작성이 되어있지 않으면, 다른분들이 작업하기 힘듭니다. 😢

# 🚀 실행 방법
```bash
# install.sh의 실행 권한 부여. 
chmod -R 666 ./install.sh

# 의존성 라이브러리 설치(LongVA, Vigilant)
bash ./install.sh

# 설정 파일 복사. 
cp config_template.py config.py

# 위의 로그 확인 후 문제가 발생하지 않은 경우,
# 다음 명령어로 모델 배포. 
cd config/bento
bentoml serve service:VisionLanguage -f config_vlm.yaml
bentoml serve service:dino -f config_dino.yaml

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
