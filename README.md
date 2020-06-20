# COVID-Net Open Source Initiative - COVIDNet-CT

**Note: The COVIDNet-CT models provided here are intended to be used as reference models that can be built upon and enhanced as new data becomes available. They are currently at a research stage and not yet intended as production-ready models (not meant for direct clinical diagnosis), and we are working continuously to improve them as new data becomes available. Please do not use COVIDNet-CT for self-diagnosis and seek help from your local health authorities.**

<p align="center">
	<img src="assets/exp-covidnet-ct-small.png" alt="photo not available" width="40%" height="40%">
	<br>
	<em>Example chest radiography image of a COVID-19 case and its associated critical factors (highlighted in red) as identified by GSInquire.</em>
</p>

If you are a researcher or healthcare worker and you would like access to the **GSInquire tool to use to interpret COVID-Net results** on your data or existing data, please reach out to a28wong@uwaterloo.ca or alex@darwinai.ca

Our desire is to encourage broad adoption and contribution to this project. Accordingly this project has been licensed under the GNU Affero General Public License 3.0. Please see [license file](LICENSE.md) for terms. If you would like to discuss alternative licensing models, please reach out to us at haydengunraj@gmail.com and a28wong@uwaterloo.ca or alex@darwinai.ca

If there are any technical questions after the README, FAQ, and past/current issues have been read, please post an issue or contact:
* haydengunraj@gmail.com
* linda.wang513@gmail.com
* jamesrenhoulee@gmail.com

## Core COVID-Net Team
* DarwinAI Corp., Canada and Vision and Image Processing Research Group, University of Waterloo, Canada
	* Linda Wang
	* Alexander Wong
	* Zhong Qiu Lin
	* James Lee
	* Paul McInnis
	* Audrey Chung
	* Hayden Gunraj
* Matt Ross and Blake VanBerlo (City of London), COVID-19 Chest X-Ray Model: https://github.com/aildnont/covid-cxr
* Ashkan Ebadi (National Research Council Canada)
* Kim-Ann Git (Selayang Hospital)
* Abdul Al-Haimi

## Table of Contents
1. [Requirements](#requirements) to install on your system
2. How to [download COVIDx-CT dataset](docs/COVIDx-CT.md)
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
These are the final results for COVIDNet-CT-Small.

### COVIDNet-CT-Small
<p>
	<img src="assets/cm-covidnet-ct-small.png" alt="photo not available" width="50%" height="50%">
	<br>
	<em>Confusion matrix for COVIDNet-CT-Small on the COVIDx-CT test dataset.</em>
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
    <td class="tg-c3ow">99.4</td>
    <td class="tg-c3ow">82.3</td>
    <td class="tg-c3ow">97.5</td>
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
    <td class="tg-c3ow">100</td>
    <td class="tg-c3ow">93.0</td>
    <td class="tg-c3ow">92.5</td>
  </tr>
</table></div>

### COVIDNet-CT-Large
<p>
	<img src="assets/cm-covidnet-ct-large.png" alt="photo not available" width="50%" height="50%">
	<br>
	<em>Confusion matrix for COVIDNet-CT-Large on the COVIDx-CT test dataset.</em>
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
    <td class="tg-c3ow">99.4</td>
    <td class="tg-c3ow">82.3</td>
    <td class="tg-c3ow">97.5</td>
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
    <td class="tg-c3ow">100</td>
    <td class="tg-c3ow">93.0</td>
    <td class="tg-c3ow">92.5</td>
  </tr>
</table></div>