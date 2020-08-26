# Quoridor

このリポジトリは，QuoridorというボードゲームのAIのプログラムを公開するためのものである．このプログラムは，AlphaGo Zeroの論文等を参考に個人的に制作中のものである．

## 環境

kivy, tensorflow, graphvizを導入する必要がある。tensorflowのバージョンは1.8.0、その他のライブラリの必要バージョンは未調査。

## 実行方法

GUIで学習済みのAIと対戦：

```sh
$ python GUI.py
```

学習を走らせる：

```sh
$ python main.py train
```
