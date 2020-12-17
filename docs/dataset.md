# COVIDx-CT Dataset
COVIDx-CT, an open access benchmark dataset that we generated from open source datasets, currently comprises 126,191 CT slices from 2,308 patients. We will be adding to COVIDx-CT over time to improve the dataset.

## Downloading the Dataset
The easiest way to use the COVIDx-CT dataset is by downloading it directly from [Kaggle](https://www.kaggle.com/hgunraj/covidxct). Links to different versions are provided below:
* [COVIDx-CT v1](https://www.kaggle.com/dataset/c395fb339f210700ba392d81bf200f766418238c2734e5237b5dd0b6fc724fcb/version/1)
* [COVIDx-CT v2]()

## Creating the Dataset from Scratch
If you wish to construct the dataset from scratch, instructions for doing so are provided below.

### Step 1: Downloading the Raw Data
We construct the COVIDx-CT dataset from the following publicly available data sources:
* [CNCB 2019 Novel Coronavirus Resource (2019nCoVR) AI Diagnosis Dataset](http://ncov-ai.big.ac.cn/download?lang=en)
* [COVID-19 Lung CT Lesion Segmentation Challenge (COVID-19-20)](https://covid-segmentation.grand-challenge.org/)
* [COVID-19 CT Lung and Infection Segmentation Dataset](https://zenodo.org/record/3757476#.X62Iw2hKiUk)
* [LIDC-IDRI](https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI)
* [COVID-CTSet](https://www.kaggle.com/mohammadrahimzadeh/covidctset-a-large-covid19-ct-scans-dataset)

To build COVIDx-CT, the datasets above must first be downloaded. Additional instructions for particular data sources are listed below.

#### CNCB Data
The CNCB data may be downloaded [here](http://ncov-ai.big.ac.cn/download?lang=en). All CP, NCP, and Normal zip files should be downloaded, as well as `unzip_filenames.csv` and `lesion_slices.csv`. Once downloaded and extracted, the data should have the following structure:
* `<CNCB root>/`
    * `CP/`
    * `NCP/`
    * `Normal/`
    * `unzip_filenames.csv`
    * `lesions_slices.csv`

#### LIDC-IDRI Data
The LIDC-IDRI challenge data may be downloaded [here](https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI). Once downloaded, a `.pylidcrc` file must be created so that the `pylidc` package can locate the data. The `.pylidcrc` file must be located in the user's home directory, and has the following format:
```
[dicom]
path = /path/to/LIDC-IDRI
warn = True
```

### Step 2: Constructing the Dataset
Before constructing the dataset, additional packages must be installed:
* [tqdm](https://pypi.org/project/tqdm/)
* [pylidc](https://pypi.org/project/pylidc/)

The dataset is constructed from the downloaded sources using `create_COVIDx-CT.ipynb`. Several variables in this notebook must be set to reflect the locations of the raw data sources:
* `CNCB_EXCLUDE_FILE`: this points to the file `cncb_exclude_list.txt`, and should not be modified unless this file is moved from its default location.
* `CNCB_DIR`: this should be set to the location of the prepared CNCB dataset, with Normal, CP, and NCP directories as well as metadata files.
* `RADIOPAEDIA_CORONACASES_CT_DIR`: this should be set to the location of the `COVID-19-CT-Seg_20cases` directory from the COVID-19 CT Lung and Infection Segmentation Dataset.
* `RADIOPAEDIA_CORONACASES_SEG_DIR`: this should be set to the location of the `Infection_Mask` directory from the COVID-19 CT Lung and Infection Segmentation Dataset.
* `LIDC_META_CSV`: this points to the file `lidc_idri_metadata.csv`, and should not be modified unless this file is moved from its default location.
* `COVID_19_20_DIR`: this should be set to the location of the `Train` directory from the COVID-19-20 challenge dataset.
* `COVID_CTSET_META_CSV`: this should be set to the location of the file `Labels&Detailes/Patient_details.csv` from COVID-CTSet.
* `COVID_CTSET_DIR`: this should be set to the location of the `Train&Validation` directory from COVID-CTSet.
* `OUTPUT_DIR`: this should be set to the directory in which the final dataset will be created.

Notably, `create_COVIDx-CT.ipynb` will not re-create files which were previously created, allowing for the dataset construction to be interrupted without having to restart it entirely.

After running the construction cells, there is an optional check at the end of the notebook to ensure that all files were created successfully.

## Data Distribution
Chest CT image distribution

|  Type | Normal | Pneumonia | COVID-19 | Total |
|:-----:|:------:|:---------:|:--------:|:-----:|
| train |  33066 |   22044   |   19923  | 75033 |
|   val |  11064 |    7400   |    7022  | 25486 |
|  test |  11404 |    7395   |    6873  | 25672 |

Patient distribution

|  Type | Normal | Pneumonia | COVID-19 | Total |
|:-----:|:------:|:---------:|:--------:|:-----:|
| train |   380  |     419   |    477   | 1276  |
|   val |   133  |     190   |    224   |  547  |
|  test |   129  |     125   |    231   |  485  |
