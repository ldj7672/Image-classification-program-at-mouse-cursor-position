# Image classification program at mouse cursor position

### 컴퓨터 모니터에서 마우스 커서 위치의 이미지를 분류하는 프로그램

- 목적 : 더욱 명확한 object 분류를 위해 재질 유형을 추가로 분류하여 보여줌

- Object 분류 : ImageNet으로 pre-train 된 network 사용 
- Material 분류 : minc2500 dataset으로 pre-train 된 network 사용
- Object + material 분류 : 상기 2개 모델을 모두 사용하여 분류 (e.g. 가죽 쇼파 -> object : 쇼파, material : 가죽)

- p : 마우스 커서 위치 이미지 분류, q : 종료

This is a normal paragraph:

    python image_classifier.py
    
end code block.


#### 구글 이미지 검색 후 분류 예시 
![image](https://user-images.githubusercontent.com/96943196/149810765-d2cee0e9-2827-4f87-8e25-32c8f01a072d.png)
