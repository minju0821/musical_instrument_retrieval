# Nlakh

## Description
- Nlakh consists of *Nlakh-single* that contains single-track audio and *Nlakh-multi* that contains mixture audio with separate tracks (stem) of each instrument.

| Dataset | Size (Hours) | Number of Samples | Number of Instruments |
|------|------|------|------|
| Nlakh-single (train) | 1,397 | 953,000 | 953 | 
| Nlakh-single (valid) | ??? | 53,000 | 53 |
| Nlakh-multi (train) | 153 | 100,000 | 953 |
| Nlakh-multi (valid) | ??? | 10,000 | 53 |


## Set up dataset
- Nlakh can be downloaded at ~~
- The folder is structured as follows
```
  Nlakh
  ├── Nlakh_single
  │   ├── train
  │      │── 001
  │          │── 0001.wav 
  │              ...
  │          │── 1000.wav
  │          ...
  │      │── 953
  │   ├── valid
  ├── Nlakh_multi
  │   ├── train
  │      │── 000001
  │          │── 124.wav 
  │              ...
  │          │── 943.wav
  │          │── mix.wav
  │   ├── valid
  
  ```
