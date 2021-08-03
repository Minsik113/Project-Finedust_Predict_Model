# Project-Finedust_Predict_Model


2021/05/04 ~ 2021/06/04

title: Fine dust(PM10) Predict Model

tool: R Studio

----------------

### 실행방법
1. Clone the repository
```
git clone https://github.com/Minsik113/Project-Finedust_Predict_Model.git
```
2. Run R file 
```
fine_dust.r 실행
```

----------------
### 프로젝트 계획 이유
> 미세먼지 최고 농도가 2015년 66㎍/㎥ 에서 2019년 129 ㎍/㎥로 지속 상승 5년 전 대비 대기 오염 정도가 심각해졌음을 체감상 느낄 수 있다.
> 국내 다양한 기사 , 연구에서 국내 미세먼지의 원인을 분석하였지만, 그에 대한 해석이 매우 다양하다.
이에 따라, 서울 미세먼지에 영향을 주는 요인들을 찾아본 후 이를 바탕으로 예측 모델을 만들어 보고자 한다.

![image](https://user-images.githubusercontent.com/54586341/126371500-679516a6-8bf3-4d40-8374-e2d902917adb.png)

----------------
### 데이터 선정

|데이터 내용|출처|항목|수집 데이터 기간|링크| 
|------|---|---|---|---|
|서울시 교통량|TOPIS(서울 종합 교통관제 센터)|일시, 일별 교통량 합계|2015.01.01 ~ 2019.12.31|[바로가기](https://www.bigdata-transportation.kr/)|
|서울시 석유 소비량|서울 열린 데이터 광장|일시, 휘발유, 등유, 경유, 벙커C유, LPG, 기타|2010.01 ~ 2019.12 (월별 데이터)|[바로가기](http://data.seoul.go.kr/dataList/128/S/2/datasetView.do)|
|서울시 기상 정보|기상청|일시, 평균기온, 일 강수량, 평균 풍속, 풍향, 평균 상대습도, 평균 증기압, 평균 전운량, 일교차|2010.01.01 ~ 2019.12.31|[바로가기](https://data.kma.go.kr/data/grnd/selectAsosRltmList.do?pgmNo=36)|
|중국(베이징, 상하이) 미세먼지|세계 대기질 지수|일시, PM10|2014.01.01 ~ 2021.05.12|[바로가기](https://aqicn.org/data-platform/register/kr/)|
|한국 미세먼지|에어 코리아|일시, PM10, SO2, NO2, O3, CO|2015.01.01 ~ 2019.12.31|[바로가기](https://www.airkorea.or.kr/web/last_amb_hour_data?pMENU_NO=123)|

----------------
### 데이터 기술 통계량

![image](https://user-images.githubusercontent.com/54586341/126371244-29bba3d4-3808-4ee7-851d-79c3cb8cab99.png)


----------------
### 아키텍쳐, 기술정의

아키텍쳐

![image](https://user-images.githubusercontent.com/54586341/126371150-dc82a41f-a939-4627-b7fc-347016471c22.png)

기술정의

![image](https://user-images.githubusercontent.com/54586341/126371164-33eea09f-85a7-48cf-ae2d-8f6dcfeb7a10.png)


----------------
### 변수 선정 기법

> 1. 다중회귀분석을 이용한 변수 선정

변수 선정 전, 프로젝트에서 독립변수 간의 관계를 알아보기 위하여 상관관계 분석을 실시함. 

다중공선성 문제를 예방하기 위해 상관계수가 0.6 이상인 변수는 사전에 제거함

![image](https://user-images.githubusercontent.com/54586341/126375740-0e49a9ce-d318-414d-b741-6db7a227221f.png)
![image](https://user-images.githubusercontent.com/54586341/126375183-6f66a3af-79bd-46c5-9682-06365e198060.png)
![image](https://user-images.githubusercontent.com/54586341/126375203-024755b3-7985-421c-b8af-900496e8db87.png)

> 2. Boruta를 이용한 변수 선정

Boruta 알고리즘은 랜덤포레스트를 기반으로 하는 변수선택기법으로 변수 또는 조합을 이용하여 모든 변수를 대상으로 최적의 변수를 추출하는 기법.

Boruta 알고리즘을 통해  최종 변수(어제PM10, 어제중국PM10, 일교차, 일강수량, 평균풍속, 평균전운량, 교통량합산, 휘발유, 벙커C유, 석유(기타), 베이징/상하이 미세먼지 농도, CO, O3, 월 – 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 풍향 – 북동, 동북동)를 선정하였음.

![image](https://user-images.githubusercontent.com/54586341/126375216-30444c93-e656-4372-bb45-dcad3f00d060.png)


> 3. XGBOOST를 이용한 변수 선정

XGBOOST 변수선택법은 기존 GBM보다 빠른 속도와 더 높은 정확도를 가진 기법으로 교차 검증(Cross Validation)을 자동으로 수행하여 번거로움을 줄였다는 점에서 본 프로젝트의 변수 선정 기법으로 선택함.

XGBOOST 변수 선택 기법을 통해 최종 변수를 (어제PM10, 어제중국PM10, 일교차, 평균풍속, 평균기온, 평균전운량, 평균상대습도, 평균증기압, 휘발유, 등유, 상하이 미세먼지 농도, 교통량 합산, CO, O3, SO2, NO2) 선정함

![image](https://user-images.githubusercontent.com/54586341/126375265-ef636b99-b890-49f1-b361-2943d0857a96.png)

> 각 기법 별 선정된 변수

다중회귀분석, 랜덤 포레스트를 기반으로 한 Boruta, XGBOOST를 이용하여 추출한 변수는 다음과 같으며, 본 프로젝트는 서울시 미세먼지 농도 예측으로 수치형 예측 모델을 생성해야 하므로, 선정된 변수를 통해 다중회귀모형과 인공신경망을 통해 예측을 진행하였음.

![image](https://user-images.githubusercontent.com/54586341/126375413-64a6547d-3834-4db5-8d66-530a5b5662b5.png)

----------------
### 모델링 및 성능 예측
본 프로젝트는 수치 예측 모델로, 다중회귀모형과 인공신경망 기법을 선택하였고, 예측 평가는 RMSE와 실측치와 예측치 간의 Wilcox 검정을 통해 해당 집단간의 차이 유무를 확인하고자 하였음.

> 다중회귀모형을 이용한 예측 - 후진제거법

본 프로젝트에서는 다중회귀모형의 후진제거법을 이용하여 모델링과 예측을 진행함.

R-Squared는 0.6318로 해당 회귀 모형은 63%의 설명력을 가지고 있고, 회귀식의 유의확률(p-value)이 0.05보다 작으므로 적합한 모델이라 판단하였음.

추가적으로 공선성 통계량을 확인해 본 결과, VIF 값이 모두 4보다 작아 다중공선성이 존재하지 않는 것을 확인하였음.  

![image](https://user-images.githubusercontent.com/54586341/126376098-1cad32b3-519e-4018-8e82-a1a4b9edba96.png)
![image](https://user-images.githubusercontent.com/54586341/126376117-68ce37cc-c4b5-4241-aff3-4cb70f0942ad.png)

아래 그림을 통해 회귀 모형의 잔차가 등분산성 및 정규성을 따르는 것을 확인할 수 있음.

![image](https://user-images.githubusercontent.com/54586341/126376838-976f2866-bf21-41a1-b894-eb4d2b2b70b5.png)

도출된 다중회귀모형에 Test Data를 대입하여 서울시 미세먼지 농도를 예측한 후 실측치와 비교한 결과를 아래 그림으로 확인할 수 있음.

![image](https://user-images.githubusercontent.com/54586341/126376862-646ad598-56d7-4ab8-884b-06d87b77cf9d.png)


> 인공신경망을 이용한 예측 - 회귀식 변수

다중회귀식으로 도출한 변수를 인공신경망에 적용하여 미세먼지 농도 예측을 진행함.

Hidden Layer와 Node의 수를 조절하여 여러 번의 예측을 진행한 결과, 1개의 Hidden Layer에 3개의 노드를 두어 학습시킨 모델이 가장 정확도가 높다고 판단하였음.

![image](https://user-images.githubusercontent.com/54586341/126376425-707ade39-e8b9-45d1-b1c6-fb0dcf31496a.png)

도출된 인공신경망에 Test Data를 대입하여 서울시 미세먼지 농도를 예측한 후 실측치와 비교한 결과를 아래 그림으로 확인할 수 있음.

![image](https://user-images.githubusercontent.com/54586341/126376452-7bad6247-ce95-4790-b9a2-17403a7febb8.png)


> 인공신경망을 이용한 예측 - Boruta

Boruta 알고리즘으로 도출한 변수를 인공신경망에 적용하여 미세먼지 농도 예측을 진행함.

Hidden Layer와 Node의 수를 조절하여 여러 번의 예측을 진행한 결과, 1개의 Hidden Layer에 3개의 노드를 두어 학습시킨 모델이 가장 정확도가 높다고 판단하였음.

![image](https://user-images.githubusercontent.com/54586341/126376481-d35bf548-dba2-4b11-ad01-f68be484c2fa.png)

도출된 인공신경망에 Test Data를 대입하여 서울시 미세먼지 농도를 예측한 후 실측치와 비교한 결과를 아래 그림으로 확인할 수 있음.

![image](https://user-images.githubusercontent.com/54586341/126376528-1bcf3917-b589-47c2-af41-ac98ba4a7cc7.png)


> 인공신경망을 이용한 예측 - XGBOOST

XGBOOST 알고리즘으로 도출한 변수를 인공신경망에 적용하여 미세먼지 농도 예측을 진행.

Hidden Layer와 Node의 수를 조절하여 여러 번의 예측을 진행한 결과, 3개의 Hidden Layer에 3개의 노드를 두어 학습시킨 모델이 가장 정확도가 높다고 판단하였음.

![image](https://user-images.githubusercontent.com/54586341/126376562-82c6ce40-a647-4076-b101-e61681322657.png)

도출된 인공신경망에 Test Data를 대입하여 서울시 미세먼지 농도를 예측한 후 실측치와 비교한 결과를 아래 그림으로 확인할 수 있음.

![image](https://user-images.githubusercontent.com/54586341/126376604-c10ab589-73a0-4216-95ab-c9ef2c3deb85.png)
----------------
### 결론 : 최종 모델 선정

4가지 모델의 RMSE 수치로 비교한 결과, XGBOOST 변수 선별 기법을 이용한 인공신경망 모델이 가장 정확도가 높다고 판단하여 최종 예측 모델로 선정하였다.

![image](https://user-images.githubusercontent.com/54586341/126372171-4f6752fe-3e97-474e-a6c3-5b6aff251abc.png)

※ RMSE : 수치형에서 예측 값과 실제 값의 차이를 다룰 때 사용하는 측도

※ Wilcox 검정은 예측치와 실측치 간 차이 유무를 확인하기 위해서 진행함 

(0.05 이상이면 예측치와 실측치는 같은 집단이라는 뜻)

----------------
### 결론 : 기대 효과
3개의 변수 선택 방법에서 동일하게 추출된 변수
> 어제PM10, 평균풍속, 펑균 전운량, CO, O3

3개의 변수 선택 방법에서 2번 이상 추출된 변수
> 일강수량, 일교차, 교통량, 휘발유, 상하이 미세먼지 농도, 어제 중국 미세먼지 농도 

위 변수들이 미세먼지 농도에 주요한 영향을 끼치는 요소로 꼽힘.

----------------
### 결론 : 최종 모델을 통한 미세먼지 예측
> 오차가 가장 낮은 XGBOOST 변수 선택 기법을 이용한 인공신경망 모델을 이용하여 2021-05-28의 미세먼지 농도 예측을 진행함.

> 실제 해당 일의 데이터를 수집하여 조회하였으며, 해당 일자에 대한 데이터가 없는 경우 직전 5년간 월 평균으로 산출하여 예측을 진행함.

> 해당 일의 실제 미세먼지는 18.9였고, 예측된 미세먼지 농도는 16.1로 2.8의 오차(RMSE)를 가지는 것을 확인하였음.

> 환경부에 따른 미세먼지 범주 기준의 범위는 최소 30이며, 예측한 모델의 차이는 실제 농도 기준으로 2.8의 차이를 보임

> 따라서, 해당 미세먼지 예측 모형은 적합하다 할 수 있다.

![image](https://user-images.githubusercontent.com/54586341/126372712-1a71d374-03a7-4bf1-8895-186c60bdc6d3.png)

----------------
### 결론 : 한계점

![image](https://user-images.githubusercontent.com/54586341/126372804-8cea6ad0-8c4d-4dce-bed7-cd3d457e0401.png)


----------------

대표깃 주소: https://github.com/Minsik113/Project-Finedust_Predict_Model.git

회의록: https://fern-diagnostic-2b3.notion.site/065c4b36a0ab416fa83a954a5fbb1370

![3](https://user-images.githubusercontent.com/54586341/128074441-978f3125-5b7f-47a0-a1f9-a6cb5d47eba9.JPG)

개발일정

![image](https://user-images.githubusercontent.com/54586341/126370787-61ad8980-294d-43ed-a56e-2203eca986c6.png)





