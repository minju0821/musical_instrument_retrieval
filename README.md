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
| [Small-Nlakh]() | DeepCNN | Nlakh | 0.482 | 0.524 | 0.553 | 0.597 |
| [Large-Nlakh]() | ConvNeXT | Nlakh | 0.533 | 0.578 | 0.635 | 0.666 |
| [Small-Random]() | DeepCNN | Randomly mixed | 0.528 | 0.543 | 0.598 | 0.615 |
| [Large-Random]() | ConvNeXT | Randomly mixed | **0.694** | **0.712** | **0.752** | **0.760** |


### Train on your dataset
```
python Multi_Instrument_Encoder/train.py
```
