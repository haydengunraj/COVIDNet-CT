# COVIDx-CT Dataset
COVIDx-CT, an open access benchmark dataset that we generated from open source datasets, currently comprises 126,191 CT slices from 2,116 patients. We will be adding to COVIDx-CT over time to improve the dataset.

COVIDx-CT is divided into "A" and "B" variants, the details of which are given below.

#### COVIDx-CT A
The "A" variant consists of cases with confirmed diagnoses (i.e., RT-PCR, radiologist-confirmed, etc.). For non-COVID-19 pneumonia and COVID-19 cases, CT slices containing abnormalities have been identified by radiologists.

COVIDNet-CT A currently comprises comprises 126,191 CT slices from 2,116 patients.

#### COVIDx-CT B
The "B" variant contains all of the "A" variant and adds some cases which are assumed to be correctly diagnosed but could not be verified. Moreover, the "B" variant also contains some cases where the identification of abnormal slices was not performed by a radiologist. Notably, the additional images included in this variant are only added to the training set, and as such **the validation and testing sets are identical to those of the "A" variant.**

COVIDNet-CT B currently comprises comprises 133,920 CT slices from 2,518 patients.

## Downloading the Dataset
The easiest way to use the COVIDx-CT dataset is by downloading it directly from [Kaggle](https://www.kaggle.com/hgunraj/covidxct). Links to different versions are provided below:
* [COVIDx-CT v1](https://www.kaggle.com/dataset/c395fb339f210700ba392d81bf200f766418238c2734e5237b5dd0b6fc724fcb/version/1)
* [COVIDx-CT v2]()

## Creating the Dataset from Scratch
If you wish to construct the dataset from scratch, instructions for doing so are provided below.

### Step 1: Downloading the Raw Data
We construct the "A" variant of the COVIDx-CT dataset from the following publicly available data sources:
* [CNCB 2019 Novel Coronavirus Resource (2019nCoVR) AI Diagnosis Dataset](http://ncov-ai.big.ac.cn/download?lang=en)
* [COVID-19 Lung CT Lesion Segmentation Challenge (COVID-19-20)](https://covid-segmentation.grand-challenge.org/)
* [COVID-19 CT Lung and Infection Segmentation Dataset](https://zenodo.org/record/3757476#.X62Iw2hKiUk)
* [LIDC-IDRI](https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI)
* [COVID-CTSet](https://www.kaggle.com/mohammadrahimzadeh/covidctset-a-large-covid19-ct-scans-dataset)

The following additional data sources are leveraged to construct the "B" variant of the dataset:
* [Radiopaedia.org](https://radiopaedia.org/)
* [TCIA CT Images in COVID-19](https://wiki.cancerimagingarchive.net/display/Public/CT+Images+in+COVID-19)
* [MosMedData](https://mosmed.ai/)

To build COVIDx-CT, the datasets above must first be downloaded. Additional instructions for some of the data sources are listed below.

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

### Step 2: Setting up the Notebook
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

To create the experimental "B" variant of the dataset, additional paths must be set:
* `MOSMED_CT_DIR`: this should be set to the location of the `COVID19_1110/studies/CT-1` directory from MosMedData.
* `MOSMED_SEG_DIR`: this should be set to the location of the `COVID19_1110/masks` directory from MosMedData.
* `RADIOPAEDIA_META_CSV`: this points to the file `radiopaedia_metadata.csv`, and should not be modified unless this file is moved from its default location.
* `RADIOPAEDIA_EXCLUDE_FILE`: this points to the file `radiopaedia_exclude_list.txt`, and should not be modified unless this file is moved from its default location.
* `TCIA_COVID_META_CSV`: this points to the file `tcia_covid_metadata.csv`, and should not be modified unless this file is moved from its default location.
* `TCIA_COVID_EXCLUDE_FILE`: this points to the file `tcia_covid_exclude_list.txt`, and should not be modified unless this file is moved from its default location.

### Step 3: Running the Notebook
Once the notebook is prepared, the dataset is constructed by simply running all cells in the notebook. Notably, `create_COVIDx-CT.ipynb` will not re-create files which were previously created, allowing for the dataset construction to be interrupted without having to recreate all the files.

After running the construction cells, there is an optional check at the end of the notebook to ensure that all files were created successfully.

## Data Distribution

### COVIDNet-CT v2A
Chest CT image distribution

|  Type | Normal | Pneumonia | COVID-19 | Total |
|:-----:|:------:|:---------:|:--------:|:-----:|
| train |  35446 |   22044   |   17557  | 75047 |
|   val |  11842 |    7400   |    6244  | 25486 |
|  test |  12245 |    7395   |    6018  | 25658 |

Patient distribution

|  Type | Normal | Pneumonia | COVID-19 | Total |
|:-----:|:------:|:---------:|:--------:|:-----:|
| train |   312  |     419   |    477   | 1208  |
|   val |   126  |     190   |    166   |  482  |
|  test |   126  |     125   |    175   |  426  |

### COVIDNet-CT v2B
Chest CT image distribution

|  Type | Normal | Pneumonia | COVID-19 | Total |
|:-----:|:------:|:---------:|:--------:|:-----:|
| train |  35996 |   23326   |   23454  | 82776 |
|   val |  11842 |    7400   |    6244  | 25486 |
|  test |  12245 |    7395   |    6018  | 25658 |

Patient distribution

|  Type | Normal | Pneumonia | COVID-19 | Total |
|:-----:|:------:|:---------:|:--------:|:-----:|
| train |   321  |     457   |    832   | 1610  |
|   val |   126  |     190   |    166   |  482  |
|  test |   126  |     125   |    175   |  426  |
