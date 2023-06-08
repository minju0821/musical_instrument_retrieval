# Nlakh

## Description

- Nlakh consists of *Nlakh-single* that contains single-track audio and *Nlakh-multi* that contains mixture audio with separate tracks (stem) of each instrument.

- The process of rendering a sample of Nlakh-single and Nlakh-multi is illustrated as below, and more details are in our paper.

<img src="./nlakh_figure.png" width="700" />


- Some statistics of Nlakh dataset

| Dataset | Size (Hours) | Number of Samples | Number of Instruments |
|------|------|------|------|
| Nlakh-single (train) | 1,323 | 953,000 | 953 | 
| Nlakh-single (valid) | 74 | 53,000 | 53 |
| Nlakh-multi (train) | 144 | 100,000 | 953 |
| Nlakh-multi (valid) | 9 | 10,000 | 53 |


## Set up dataset
- Nlakh can be downloaded at:
    - [download Nlakh-single](https://www.dropbox.com/s/x4ssq5nlhmvp4k7/nlakh-single.tar.gz?dl=0)
    - [download Nlakh-multi](https://www.dropbox.com/s/4sl6enmslzq7uob/nlakh-multi.tar.gz?dl=0)
- The folder is structured as follows
```
 Nlakh-single
  ├── train
  │  │── 001
  │      │── 0001.wav 
  │           ...
  │      │── 1000.wav
  │       ...
  │  │── 953
  ├── valid
  
  
 Nlakh-multi
  ├── train
  │  │── 000001
  │      │── 124.wav 
  │          ...
  │      │── 943.wav
  │      │── mix.wav
  │       ...
  │  │── 100000
  ├── valid
  
  ```
