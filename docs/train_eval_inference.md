# Training, Evaluation and Inference

The networks take as input an image of shape (N, 512, 512, 3) and output the softmax probabilities as (N, 3), where N is the number of images. For the TensorFlow checkpoints, here are some useful tensors:
* input tensor: `Placeholder:0`
* label tensor: `Placeholder_1:0`
* logit tensor: `resnet_model/final_dense:0`
* output confidence tensor: `softmax_tensor:0`
* output prediction tensor: `ArgMax:0`
* loss tensor: `add:0`
* training tensor: `is_training:0`

## Steps for training
1. We provide you with the TensorFlow training script, [run_covidnet_ct.py](../run_covidnet_ct.py)
2. Locate the TensorFlow checkpoint files (location of pretrained model)
3. To train from a pretrained model:
```
python run_covidnet_ct.py train \
    --model_dir models/COVIDNet-CT-A \
    --meta_name model.meta \
    --ckpt_name model
```
4. For more options and information, `python run_covid_ct.py train --help`

## Steps for testing
1. We provide you with the TensorFlow testing script, [run_covidnet_ct.py](../run_covidnet_ct.py)
2. Locate the TensorFlow checkpoint files
3. To evaluate a TensorFlow checkpoint:
```
python run_covidnet_ct.py val \
    --model_dir models/COVIDNet-CT-A \
    --meta_name model.meta \
    --ckpt_name model \
    --plot_confusion
```
4. For more options and information, `python run_covid_ct.py val --help`

## Steps for inference
**DISCLAIMER: Do not use this prediction for self-diagnosis. You should check with
your local authorities for the latest advice on seeking medical assistance.**

1. Download a model from the [pretrained models section](models.md)
2. Locate models and CT image to be inferenced
3. To run inference,
```
python run_covidnet_ct.py infer \
    --model_dir models/COVIDNet-CT-A \
    --meta_name model.meta \
    --ckpt_name model \
    --image_file assets/ex-covid-ct.png
```
4. For more options and information, `python run_covid_ct.py infer --help`
