# IPDM
Fusing Infrared Polarization Images for Road Detection Via Denoising Diffusion Probabilistic Models

## Installation
[Python 3.8]

[Pytorch 1.9.0 ]

[cuda 11.4]

[cudnn 8.4.0]

## Dataset：LDDRS
We use the [ LWIR DoFP Dataset of Road Scene (LDDRS)](https://github.com/polwork/LDDRS) as our experimental dataset.
Download LDDRS dataset from https: //github.com/polwork/LDDRS.
You can randomly assign infrared intensity and polarized images for training and testing in the following directories
```
|-- dataset_all
  |-- train
    |-- S0
       |-- 0000.png
       |-- ....
    |-- dolp
    |-- img
       |-- 0000
          |-- 000.png
          |-- 045.png
          |-- 090.png
          |-- 135.png
       |-- ....
     
    |-- label
  |-- test
    |-- S0
       |-- 0000.png
       |-- ....
    |-- dolp
    |-- img
       |-- 0000
          |-- 000.png
          |-- 045.png
          |-- 090.png
          |-- 135.png
       |-- ....
    |-- label
```    

## Train & Test
* After loading the data according to the above directory, the diffusion model is trained by 'ddpm_train.py'. The road detection part is performed using 'ddpm_cd.py' after loading the pre-trained diffusion model.

We will add information about the paper later.

## Contact

[Kunyuan Li](mailto:kunyuan@mail.hfut.edu.cn)
