# Efficient-FER-proposed-method
- CSDNet (Channel-wise Split Dilated) Net: 셔플넷 기반 모델 구조의 성능 한계를 느껴 새로 고안 중인 모델
- ShuffleNet: 기존 셔플넷 기반 모델

## 데이터셋
| **데이터셋 이름** | **이미지 개수** | **공식 홈페이지** | **취득 방법** | **데이터셋 다운로드 링크** |
|:---------------:|:----------:|:---------------:|---------------|:---------------:|
| RAFDB | 15339 | [Homepage](http://www.whdeng.cn/RAF/model1.html#dataset) | [MTCNN](https://github.com/foamliu/Face-Alignment)을 이용하여 얼굴을 정렬 완료한 데이터셋을 다운로드 받아서 사용 하였음 | [Google Drive](https://drive.google.com/file/d/1GiVsA5sbhc-12brGrKdTIdrKZnXz9vtZ/view) |

<br>

### RAFDB
- RAFDB 데이터셋을 다운로드 받아 아래와 같은 형식으로 배치하여 사용하세요
```
dataset/raf-basic/
    EmoLabel/
        list_patition_label.txt
    Image/aligned/
        train_00001_aligned.jpg
        test_0001_aligned.jpg
        ...
```
