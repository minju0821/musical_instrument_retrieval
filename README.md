# Show Me the Instruments: Musical Instrument Retrieval from Mixture Audio
This repository contains the code and the [dataset](https://github.com/minju0821/musical_instrument_retrieval/blob/main/dataset/README.md) for our paper,

* Kyungsu Kim*, Minju Park*, Haesun Joung*, Yunkee Chae, Yeongbeom Hong, Seonghyeon Go, Kyogu Lee. _“Show Me the Instruments: Musical Instrument Retrieval from Mixture Audio”._ 2022.

For audio samples and demo, visit [our website](https://dour-stretch-5d5.notion.site/Show-me-the-instrument-Musical-Instrument-Retrieval--cb016a6c63514eee8c30c442b37e8f6e).


## Quickstart

- Clone the repository
  ```
  git clone https://github.com/minju0821/musical_instrument_retrieval.git
  ```
- Install requirements
  ```
  pip3 install -r requirements.txt
  ```
 - Install [Nlakh dataset](https://github.com/minju0821/musical_instrument_retrieval/blob/main/dataset/README.md)


## Single-Instrument Encoder


### Pre-trained model
| Model | EER |
|------|------|
| [Single-Intrument Encoder](https://github.com/minju0821/musical_instrument_retrieval/raw/main/models/pretrained_single_inst_enc) | 0.026 |


### Train on your dataset
```
python Single_Instrument_Encoder/train.py
```

## Multi-Instrument Encoder


### Pre-trained models

| Model | Encoder Architecture | Train Dataset | F1 (macro) | F1 (weighted) | mAP (macro) | mAP (weighted) |
|-------|-------|-------|-------|-------|-------|-------|
| [Small-Nlakh](https://drive.google.com/file/d/1ADSuXwR8C06-kqOZbjH9jAWDcCsl5l8A/view?usp=share_link) | DeepCNN | Nlakh | 0.482 | 0.524 | 0.553 | 0.597 |
| [Large-Nlakh](https://drive.google.com/file/d/16u1QAfQOWe0pNc63gAuqRW1fQAaFGKeW/view?usp=share_link) | ConvNeXT | Nlakh | 0.533 | 0.578 | 0.635 | 0.666 |
| [Small-Random](https://drive.google.com/file/d/1ML1MzHKjtrwF6J5SGlAhw8Ej5EO8aC_r/view?usp=share_link) | DeepCNN | Randomly mixed | 0.528 | 0.543 | 0.598 | 0.615 |
| [Large-Random](https://drive.google.com/file/d/1zm2h-SheNREfIphcB9n9CgzqJ5wHkVc9/view?usp=share_link) | ConvNeXT | Randomly mixed | **0.694** | **0.712** | **0.752** | **0.760** |


### Train on your dataset
- Customize arguments in parse_args function in train.py before training.
```
python Multi_Instrument_Encoder/train.py
```
