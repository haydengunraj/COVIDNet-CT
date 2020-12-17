# COVID-Net Open Source Initiative - COVIDNet-CT

**Note: The COVIDNet-CT models provided here are intended to be used as reference models that can be built upon and enhanced as new data becomes available. They are currently at a research stage and not yet intended as production-ready models (not meant for direct clinical diagnosis), and we are working continuously to improve them as new data becomes available. Please do not use COVIDNet-CT for self-diagnosis and seek help from your local health authorities.**

**Update 2020-12-XX:** We released the [COVIDx-CT v2]() dataset on Kaggle.

**Update 2020-12-03:** We released the [COVIDx-CT v1](https://www.kaggle.com/dataset/c395fb339f210700ba392d81bf200f766418238c2734e5237b5dd0b6fc724fcb/version/1) dataset on Kaggle.

**Update 2020-09-13:** We released the [COVIDNet-CT paper](https://arxiv.org/abs/2009.05383).

<p align="center">
	<img src="assets/exp-covidnet-ct-b.png" alt="photo not available" width="90%" height="40%">
	<br>
	<em>Example CT scans of COVID-19 cases and their associated critical factors (highlighted in red) as identified by GSInquire.</em>
</p>

The coronavirus disease 2019 (COVID-19) pandemic continues to have a tremendous impact on patients and healthcare systems around the world. In the fight against this novel disease, there is a pressing need for rapid and effective screening tools to identify patients infected with COVID-19, and to this end CT imaging has been proposed as one of the key screening methods which may be used as a complement to RT-PCR testing, particularly in situations where patients undergo routine CT scans for non-COVID-19 related reasons, patients have worsening respiratory status or developing complications that require expedited care, or patients are suspected to be COVID-19-positive but have negative RT-PCR test results. Early studies on CT-based screening have reported abnormalities in chest CT images which are characteristic of COVID-19 infection, but these abnormalities may be difficult to distinguish from abnormalities caused by other lung conditions. Motivated by this, in this study we introduce COVIDNet-CT, a deep convolutional neural network architecture that is tailored for detection of COVID-19 cases from chest CT images via a machine-driven design exploration approach. Additionally, we introduce COVIDx-CT, a benchmark CT image dataset derived from a variety of sources of CT imaging data comprising 126,191 images across 2,308 patient cases. Furthermore, in the interest of reliability and transparency, we leverage an explainability-driven performance validation strategy to investigate the decision-making behaviour of COVIDNet-CT, and in doing so ensure that COVIDNet-CT makes predictions based on relevant indicators in CT images. Both COVIDNet-CT and the COVIDx-CT dataset are available to the general public in an open-source and open access manner as part of the COVID-Net initiative. While COVIDNet-CT is **not yet a production-ready screening solution**, we hope that releasing the model and dataset will encourage researchers, clinicians, and citizen data scientists alike to leverage and build upon them.

For a detailed description of the methodology behind COVIDNet-CT and a full description of the COVIDx-CT dataset, please click [here](https://arxiv.org/abs/2009.05383).

Our desire is to encourage broad adoption and contribution to this project. Accordingly this project has been licensed under the GNU Affero General Public License 3.0. Please see [license file](LICENSE.md) for terms. If you would like to discuss alternative licensing models, please reach out to us at haydengunraj@gmail.com and a28wong@uwaterloo.ca or alex@darwinai.ca.

For COVIDNet-CXR models and the COVIDx dataset for COVID-19 detection and severity assessment from chest X-ray images, please go to the [main COVID-Net repository](https://github.com/lindawangg/COVID-Net).

If you are a researcher or healthcare worker and you would like access to the **GSInquire tool to use to interpret COVIDNet-CT results** on your data or existing data, please reach out to a28wong@uwaterloo.ca or alex@darwinai.ca.

If there are any technical questions after the README, FAQ, and past/current issues have been read, please post an issue or contact:
* haydengunraj@gmail.com
* linda.wang513@gmail.com
* jamesrenhoulee@gmail.com

If you find our work useful, you can cite our paper using:

```
@Article{Gunraj2020,
    author={Gunraj, Hayden and Wang, Linda and Wong, Alexander},
    title={{COVIDNet-CT}: A Tailored Deep Convolutional Neural Network Design for Detection of {COVID}-19 Cases from Chest {CT} Images},
    journal={Frontiers in Medicine},
    year={forthcoming},
    doi={10.3389/fmed.2020.608525},
    url={https://doi.org/10.3389/fmed.2020.608525}
}
```

## Core COVID-Net Team
* DarwinAI Corp., Canada and Vision and Image Processing Research Group, University of Waterloo, Canada
  * Linda Wang
  * Alexander Wong
  * Zhong Qiu Lin
  * Paul McInnis
  * Audrey Chung
  * Melissa Rinch
  * Maya Pavlova
  * Naomi Terhljan
  * Hayden Gunraj, [COVIDNet for CT](https://github.com/haydengunraj/COVIDNet-CT)
  * Jeffer Peng, [COVIDNet UI](https://github.com/darwinai/covidnet_ui)
* Vision and Image Processing Research Group, University of Waterloo, Canada
  * James Lee
  * Hossain Aboutaleb 
* Ashkan Ebadi and Pengcheng Xi (National Research Council Canada)
* Kim-Ann Git (Selayang Hospital)
* Abdul Al-Haimi, [COVID-19 ShuffleNet Chest X-Ray Model](https://github.com/aalhaimi/covid-net-cxr-shuffle)

## Table of Contents
1. [Requirements](#requirements) to install on your system
2. How to [download and prepare COVIDx-CT dataset](docs/dataset.md)
3. Steps for [training, evaluation and inference](docs/train_eval_inference.md)
4. [Results](#results)
5. [Links to pretrained models](docs/models.md)

## Requirements
The main requirements are listed below:

* Tested with Tensorflow 1.15
* OpenCV 4.2.0
* Python 3.7
* Numpy
* Scikit-Learn
* Matplotlib

## Results
These are the final test results for each COVIDNet-CT model on the COVIDx-CT dataset.

### COVIDNet-CT-A
<p>
	<img src="assets/cm-covidnet-ct-a.png" alt="photo not available" width="50%" height="50%">
	<br>
	<em>Confusion matrix for COVIDNet-CT-A on the COVIDx-CT test dataset.</em>
</p>

<div class="tg-wrap"><table class="tg">
  <tr>
    <th class="tg-7btt" colspan="3">Sensitivity (%)</th>
  </tr>
  <tr>
    <td class="tg-7btt">Normal</td>
    <td class="tg-7btt">Pneumonia</td>
    <td class="tg-7btt">COVID-19</td>
  </tr>
  <tr>
    <td class="tg-c3ow">100</td>
    <td class="tg-c3ow">99.0</td>
    <td class="tg-c3ow">97.3</td>
  </tr>
</table></div>

<div class="tg-wrap"><table class="tg">
  <tr>
    <th class="tg-7btt" colspan="3">Positive Predictive Value (%)</th>
  </tr>
  <tr>
    <td class="tg-7btt">Normal</td>
    <td class="tg-7btt">Pneumonia</td>
    <td class="tg-7btt">COVID-19</td>
  </tr>
  <tr>
    <td class="tg-c3ow">99.4</td>
    <td class="tg-c3ow">98.4</td>
    <td class="tg-c3ow">99.7</td>
  </tr>
</table></div>

### COVIDNet-CT-B
<p>
	<img src="assets/cm-covidnet-ct-b.png" alt="photo not available" width="50%" height="50%">
	<br>
	<em>Confusion matrix for COVIDNet-CT-B on the COVIDx-CT test dataset.</em>
</p>

<div class="tg-wrap"><table class="tg">
  <tr>
    <th class="tg-7btt" colspan="3">Sensitivity (%)</th>
  </tr>
  <tr>
    <td class="tg-7btt">Normal</td>
    <td class="tg-7btt">Pneumonia</td>
    <td class="tg-7btt">COVID-19</td>
  </tr>
  <tr>
    <td class="tg-c3ow">99.8</td>
    <td class="tg-c3ow">99.2</td>
    <td class="tg-c3ow">94.7</td>
  </tr>
</table></div>

<div class="tg-wrap"><table class="tg">
  <tr>
    <th class="tg-7btt" colspan="3">Positive Predictive Value (%)</th>
  </tr>
  <tr>
    <td class="tg-7btt">Normal</td>
    <td class="tg-7btt">Pneumonia</td>
    <td class="tg-7btt">COVID-19</td>
  </tr>
  <tr>
    <td class="tg-c3ow">99.4</td>
    <td class="tg-c3ow">96.8</td>
    <td class="tg-c3ow">99.8</td>
  </tr>
</table></div>
