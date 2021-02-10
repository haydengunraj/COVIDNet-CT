# Training, Evaluation and Inference

The networks take as input an image of shape (N, 512, 512, 3) and output the softmax probabilities as (N, 3), where N is the number of images. For the TensorFlow checkpoints, here are some useful tensors:
* input tensor: `Placeholder:0`
* label tensor: `Placeholder_1:0`
* logits tensor: `resnet_model/final_dense:0`
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
    --model_dir models/COVID-Net_CT-2_L \
    --meta_name model.meta \
    --ckpt_name model
```
For more options and information, `python run_covid_ct.py train --help`

## Steps for testing
1. We provide you with the TensorFlow testing script, [run_covidnet_ct.py](../run_covidnet_ct.py)
2. Locate the TensorFlow checkpoint files
3. To evaluate a TensorFlow checkpoint:
```
python run_covidnet_ct.py val \
    --model_dir models/COVID-Net_CT-2_L \
    --meta_name model.meta \
    --ckpt_name model \
    --plot_confusion
```
For more options and information, `python run_covid_ct.py val --help`

## Steps for inference
**DISCLAIMER: Do not use this prediction for self-diagnosis. You should check with
your local authorities for the latest advice on seeking medical assistance.**

A special [inference notebook](../inference_grad_cam.ipynb) is included which provides inference code and Grad-CAM visualizations. This is the easiest way to run inference and see visual results.

Inference may also be run using the main script via the following steps:
1. Download a model from the [pretrained models section](models.md)
2. Locate models and CT image to be tested
3. To run inference,
```
python run_covidnet_ct.py infer \
    --model_dir models/COVID-Net_CT-2_L \
    --meta_name model.meta \
    --ckpt_name model \
    --image_file assets/ex-covid-ct.png
```
For more options and information, `python run_covid_ct.py infer --help`
