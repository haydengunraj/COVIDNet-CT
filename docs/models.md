# Pretrained Models

The naming convention for models uses the following format:
```
COVID-Net CT-<dataset version> <architecture version> (<minor dataset version if applicable>)
```
For example, a COVID-Net CT model using the Large (L) architecture which was trained on the COVIDx CT-2 dataset's "A" variant would be called "COVID-Net CT-2 L (2A)".

## COVID-Net CT-1 Models
These models are trained and tested on the COVID CT-1 dataset.

|        Model        | Type | Input Resolution | COVID-19 Sensitivity (%) | Accuracy (%) | # Params (K) | FLOPs (G) |
|:-------------------:|:----:|:----------------:|:------------------------:|:------------:|:------------:|:---------:|
|[COVID-Net CT-1 L](https://bit.ly/2BAPyvM)| ckpt |     512 x 512    |           97.3           |     99.1     |    1399.38   |    4.18   |
|[COVID-Net CT-1 S](https://bit.ly/3irnITl)| ckpt |     512 x 512    |           94.7           |     98.5     |     447.57   |    1.94   |

## COVID-Net CT-2 Models
These models are trained and tested on the COVIDx CT-2 dataset. Notably, COVID-Net CT-2 L (2A RAD) is a special version of the model trained exclusively on cases where slice selection or segmentation was performed manually by a radiologist.

|        Model        | Type | Input Resolution | COVID-19 Sensitivity (%) | Accuracy (%) | # Params (K) | FLOPs (G) |
|:-------------------:|:----:|:----------------:|:------------------------:|:------------:|:------------:|:---------:|
|[COVID-Net CT-2 L (2A)](https://bit.ly/3oC84ar)| ckpt |     512 x 512    |            96.2           |      98.1     |    1399.38   |    4.18   |
|[COVID-Net CT-2 S (2A)](https://bit.ly/3oCH6PJ)| ckpt |     512 x 512    |            95.7           |      97.9     |     447.57   |    1.94   |
|[COVID-Net CT-2 L (2A RAD)](https://bit.ly/3sejRhl)| ckpt |     512 x 512    |            96.4           |      98.3     |    1399.38   |    4.18   |
