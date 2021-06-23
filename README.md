# image_segmentation


## 캡스톤 프로젝트(image segmentation)



## 기존 연구 기술 조사 정리

- semantic segmentation 기본 개념 : https://medium.com/hyunjulie/1%ED%8E%B8-semantic-segmentation-%EC%B2%AB%EA%B1%B8%EC%9D%8C-4180367ec9cb

- semantic segmentation 관련 블로그: https://blog.naver.com/laonple/220873446440

- segnet 모델 개념:https://m.blog.naver.com/PostView.nhn?blogId=tory0405&logNo=221339746615&proxyReferer=https:%2F%2Fwww.google.com%2F

## 사용 환경
- google colab pro


## 사용 모델
- segnet
- fcn32
- linknet
- u-net
- FPN
- PSPNet

## backbone-model
- resnet-50
- resnet-101
- resnet-152
- se-resnext-101
- resnext-101
- vgg16
- mobilenet-ev2
- inception-resnet-v2

## result image 
![image](https://user-images.githubusercontent.com/45275607/123062249-4d533f80-d447-11eb-8249-e57fddd3ebcb.png)


## 개선 방안
- 이미지 품질 개선
1. 평탄화
2. CLAHE Algorithm
![image](https://user-images.githubusercontent.com/45275607/123062386-6e1b9500-d447-11eb-9c3d-1389be48c2e1.png)

3.FLIP (train set 증가)

## 개선 방안 적용 result image
![image](https://user-images.githubusercontent.com/45275607/123062497-8be8fa00-d447-11eb-8f50-18e90cd9e0eb.png)

