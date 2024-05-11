# Project Description

This repository is for an AI program for the board game Quoridor based on AlphaGo Zero and KataGo papers.

## Environment

Install pipenv and try below:

```sh
pipenv shell
pipenv install
```


If you want to use a pre-trained model, please download the zip file from the following link, unzip it, and place the application_data directly under this repository.  
https://drive.google.com/drive/u/1/folders/10ZZLK9tDxJCG-0eV6wxiKF5g0RubCsxo

If you want to train AI, then you need tensorflow-gpu.

## Usage

Play quoridor in GUI:

```sh
python GUI.py
```

Analyze quoridor games:

```sh
python game_analyzer.py
```

Train AI:

```sh
python main.py train
```

If you turn off your computer and want to restart training:

```sh
python main.py retrain
```

## Make exe

```sh
pyinstaller --log-level=INFO xxx.spec
```

## Cython

```sh
python setup.py build_ext --inplace
```

## Compile C++ program

windows:

```sh
g++ -fPIC -shared -o State_util.dll State_util.cpp
```

linux:

```sh
g++ -fPIC -shared -o State_util.so State_util.cpp
```


