# Project Description

This repository is for an AI program for the board game Quoridor based on AlphaGo Zero and KataGo papers.

## Environment

Install pipenv and try below:

```sh
$ pipenv shell
$ pipenv install
```

If you want to train AI, then you need tensorflow-gpu.

## Usage

Play quoridor in GUI:

```sh
$ python GUI.py
```

Train AI:

```sh
$ python main.py train
```

## Make exe

```sh
$ pyinstaller --log-level=INFO xxx.spec
```

