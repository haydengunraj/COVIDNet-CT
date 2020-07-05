# COVIDx-CT Dataset
COVIDx-CT, an open access benchmark dataset that we generated from open source datasets, currently comprises 104,009 CT slices from 1,489 patients. We will be adding to COVIDx-CT over time to improve the dataset.

## Preparing the Data
We construct the COVIDx-CT dataset from publicly available data provided by the China National Center for Bioinformation (CNCB):
* Kang Zhang, Xiaohong Liu, Jun Shen, et al. Jianxing He, Tianxin Lin, Weimin Li, Guangyu Wang. (2020). Clinically Applicable AI System for Accurate Diagnosis, Quantitative Measurements and Prognosis of COVID-19 Pneumonia Using Computed Tomography. **Cell**, DOI: 10.1016/j.cell.2020.04.045

The CNCB dataset may be downloaded [here](http://ncov-ai.big.ac.cn/download?). All CP, NCP, and Normal zip files should be downloaded, as well as `unzip_filenames.csv` and `lesion_slices.csv`. Additionally, the file `exclude_list.txt` should be copied from this repo into the CNCB root directory. Once downloaded and extracted, the data should have the following structure:
* `<CNCB root>/`
    * `CP/`
    * `NCP/`
    * `Normal/`
    * `unzip_filenames.csv`
    * `lesions_slices.csv`
    * `exclude_list.txt`

To prepare the dataset for training/validation/testing, run:
```
python prepare_data.py <CNCB root> -o <output directory>
```
This will construct the COVIDx-CT dataset in `<output directory>`. Data split files are provided in this repo, namely `train_COVIDx-CT.txt`, `val_COVIDx-CT.txt`, and `test_COVIDx-CT.txt`.

After preparing the data, see [this document](train_eval_inference.md) for details on how to run the models.

## Data Distribution
Chest CT image distribution

|  Type | Normal | Pneumonia | COVID-19 | Total |
|:-----:|:------:|:---------:|:--------:|:-----:|
| train |  27201 |   22061   |   12520  | 61782 |
|   val |   9107 |    7400   |    4529  | 21036 |
|  test |   9450 |    7395   |    4346  | 21191 |

Patient distribution

|  Type | Normal | Pneumonia | COVID-19 |  Total |
|:-----:|:------:|:---------:|:--------:|:------:|
| train |   144  |     420   |    300   |   864  |
|   val |    47  |     190   |     95   |   332  |
|  test |    52  |     125   |    116   |   293  |
