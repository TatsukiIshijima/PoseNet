# PoseNet

### 概要
Tensorflow Lite の [Pose estimation](https://www.tensorflow.org/lite/models/pose_estimation/overview) を iOS と Python で試したもの。iOS では Firebase の MLKit を用いてモデルデータの読み込み、推論を行っている。

### 開発環境
#### Python
- Python 3.6
##### ライブラリ
| 名前 | バージョン | 用途 |
|:-----------|:------------|:------------|
| Pillow | 6.1.0 | |
| Tensorflow | 1.14.0 | |
| Numpy | 1.16.4 | |

#### iOS
- macOS Mojave
- Xcode 10.3
##### ライブラリ
| 名前 | バージョン | 用途 |
|:-----------|:------------|:------------|
| [Firebase Analytics](https://firebase.google.com/docs/ios/setup) |  |  |
| [Firebase MLModelInterpreter](https://firebase.google.com/docs/ml-kit/ios/use-custom-models) |  | TFLiteモデル読み込み |

### 画像
#### Python
<div align="center">
<img src="https://user-images.githubusercontent.com/17661705/64624006-f1135d00-d424-11e9-8365-7f119971ddf5.png" alt="python_img" title="screenshot">
</div>

#### iOS
<div align="center">
<img src="https://user-images.githubusercontent.com/17661705/64624003-ef499980-d424-11e9-915c-8d5fecdf1a38.png" alt="ios_img" title="screenshot">
</div>
