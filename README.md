# 🌙 Dacon Trade Forecast — Team *Girls_Night*

### 🏫 제3회 국민대학교 AI빅데이터 분석 경진대회  
**주제:** 품목 간 무역 연동성과 미래 예측 가능성에 대한 AI 기술의 응용  
**주최:** 국민대학교 경영대학원 · 한국기계산업진흥회(KOAMI)

---

## 👩‍💻 Team Girls_Night
| 역할 | 이름 | GitHub |
|------|------|--------|
| 팀장 | 이수민 | [@Leesoomin97](https://github.com/Leesoomin97) |
| 팀원 | 권문진 |  |
| 팀원 | 고민서 |  |
| 팀원 | 허예경 |  |

---

## 🧱 프로젝트 구조
```
dacon-trade-forecast-team_girls_night/
├── data/                   # 원본·전처리 데이터 (gitignore)
│   ├── raw/
│   ├── interim/
│   └── processed/
│
├── notebooks/              # 탐색·전처리·모델링 노트북
│   ├── 1_EDA.ipynb
│   ├── 2_Preprocessing.ipynb
│   ├── 3_FeatureEngineering.ipynb
│   ├── 4_Modeling.ipynb
│   └── 5_Evaluation.ipynb
│
├── feature_engineering/    # 피처 생성 관련 모듈
│   ├── lag_features.py
│   ├── cross_features.py
│   ├── stats_features.py
│   └── feature_selection.py
│
├── src/                    # 실행용 파이프라인 코드
│   ├── preprocess.py
│   ├── feature_engineering.py
│   ├── correlation.py
│   ├── train_model.py
│   ├── evaluate.py
│   └── utils.py
│
├── outputs/                # 결과물 저장 (gitignore)
│   ├── figures/
│   ├── logs/
│   ├── models/
│   └── submissions/
│
├── configs/                # 설정 파일
│   ├── paths.yaml
│   ├── params.yaml
│   └── features.yaml
│
├── requirements.txt        # 패키지 의존성
├── README.md
└── .gitignore
```

---

## ⚙️ 실행 순서
```bash
pip install -r requirements.txt
python src/preprocess.py
python src/feature_engineering.py
python src/train_model.py
python src/evaluate.py
```

---

## 📊 평가 지표
> **Score = 0.6 × F1 + 0.4 × (1 − NMAE)**  
- **F1:** 공행성(leading–following) 판별 정확도  
- **NMAE:** 예측 무역량 오차율  

---

## 🧩 규칙 요약
- `data/`, `outputs/` 폴더는 `.gitignore`에 등록 (업로드 금지)  
- **모델 실험은 브랜치 기반으로 진행**
  ```bash
  git checkout -b soomin-dev
  git push origin soomin-dev
  ```
- `.env` 파일에 **API key, WandB token, Kaggle key 등 민감정보 저장** (커밋 금지)  
- 제출 파일은 `outputs/submissions/` 내부에 저장  
