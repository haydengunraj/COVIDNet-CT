# COVIDx CT Dataset

## Quick Links
1. [Description](#description)
2. [Metadata](#metadata)
3. [Downloading the dataset](#downloading-the-dataset)
4. [Creating the dataset from scratch](#creating-the-dataset-from-scratch)
5. [Data distribution](#data-distribution)
6. [Licenses and acknowledgements for the datasets used](licenses_acknowledgements.md)

## Description
COVIDx CT, an open access benchmark dataset that we generated from open source datasets, currently comprises 201,103 CT slices from 4,501 patients. We will be adding to COVIDx CT over time to improve the dataset.

Labels for the images are obtained in one of three ways:
1. Manual labelling or segmentation by radiologists (all validation and test images are labelled this way)
2. Manual labelling by non-radiologists (these images are only included in the training set)
3. Automatic labelling using a previous COVID-Net CT model (these images are only included in the training set)

[metadata.csv](../metadata.csv) indicates the labelling method used for each patient.

COVIDx CT is divided into "A" and "B" variants, the details of which are given below.

#### COVIDx CT-A
The "A" variant consists of cases with confirmed diagnoses (i.e., RT-PCR, radiologist-confirmed, etc.). COVIDx CT-3A comprises 425,024 CT slices from 5,312 patients.

#### COVIDx CT-B
The "B" variant contains all of the "A" variant and adds some cases which are assumed to be correctly diagnosed but could not be verified. COVIDx CT-2B comprises comprises 431,205 CT slices from 6,068 patients. Notably, the additional images included in this variant are only added to the training set, and as such **the validation and testing sets are identical to those of the "A" variant.**

## Metadata
Metadata for each patient is included in [metadata.csv](../metadata.csv). The metadata includes:
* Patient ID
* Data source
* Country (if available)
* Age & sex (if available)
* Finding (Normal, Pneumonia, or COVID-19)
* Verified finding, which indicates whether the finding is confirmed (Yes or No)
* Slice selection, which indicates how slice selection was performed (either Expert, Non-expert, or Automatic)
* View and modality (all are axial CT)

Some basic patient information from the dataset:
* **Countries:** China, France, Russia, Iran, USA, Australia, Algeria, Italy, Scotland, Peru, Lebanon, England, Turkey, Belgium, Azerbaijan, Afghanistan, Ukraine
* **Age range (for cases which have age information):** 0-94
* **Sexes:** 3091 Unknown (50.9%), 1639 Male (27.0%), 1338 Female (22.1%)

## Downloading the Dataset
The easiest way to use the COVIDx CT dataset is by downloading it directly from [Kaggle](https://www.kaggle.com/hgunraj/covidxct). Links to different versions are provided below:
* [COVIDx CT-1](https://www.kaggle.com/dataset/c395fb339f210700ba392d81bf200f766418238c2734e5237b5dd0b6fc724fcb/version/1)
* [COVIDx CT-2](https://www.kaggle.com/dataset/c395fb339f210700ba392d81bf200f766418238c2734e5237b5dd0b6fc724fcb/version/4)
* [COVIDx CT-3](link-tbd)

Note that COVIDx CT B variants are not available on Kaggle, and must be generated from scratch.

## Creating the Dataset from Scratch
If you wish to construct the dataset from scratch, instructions for doing so are provided below.

### Step 1: Downloading the Raw Data
We construct the "A" variant of the COVIDx CT dataset from the following publicly available data sources:
* [CNCB 2019 Novel Coronavirus Resource (2019nCoVR) AI Diagnosis Dataset](http://ncov-ai.big.ac.cn/download?lang=en)
* [COVID-19 Lung CT Lesion Segmentation Challenge (COVID-19-20)](https://covid-segmentation.grand-challenge.org/)
* [TCIA CT Images in COVID-19](https://wiki.cancerimagingarchive.net/display/Public/CT+Images+in+COVID-19)
* [COVID-19 CT Lung and Infection Segmentation Dataset](https://zenodo.org/record/3757476#.X62Iw2hKiUk)
* [COVID-CTSet](https://www.kaggle.com/mohammadrahimzadeh/covidctset-a-large-covid19-ct-scans-dataset)
* [Radiopaedia.org](https://radiopaedia.org/)
* [LIDC-IDRI](https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI)
* [Integrative CT Images and Clinical Features for COVID-19 (iCTCF)](http://ictcf.biocuckoo.cn/index.php)
* [COVID-CT-MD](https://github.com/ShahinSHH/COVID-CT-MD)
* [Stony Brook University COVID-19 Positive Cases (COVID-19-NY-SBU)](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=89096912)
* [Study of thoracic CT in COVID-19 (STOIC)](https://stoic2021.grand-challenge.org/)

The following additional data source is leveraged to construct the "B" variant of the dataset:
* [MosMedData](https://mosmed.ai/)

To build COVIDx CT, the datasets above must first be downloaded. Additional instructions for some of the data sources are listed below.

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

#### iCTCF Data
The iCTCF data may be downloaded [here](http://ictcf.biocuckoo.cn/index.php). All patient cases listed in [ictcf_metadata.csv](../dataset_construction/metadata/ictcf_metadata.csv) should be downloaded in JPEG form. Once downloaded and extracted, the data should have the following structure:
* `<iCTCF root>/`
    * `Patient 1/`
        * `CT/`
            * `IMG-0001-00001.jpg`
            * `...`
    * `Patient 2/`
        * `CT/`
            * `IMG-0001-00001.jpg`
            * `...`
    * `...`

### Step 2: Setting up the Notebook
Before constructing the dataset, additional packages must be installed:
* [tqdm](https://pypi.org/project/tqdm/)
* [pylidc](https://pypi.org/project/pylidc/)
* [nibabel](https://pypi.org/project/nibabel/)

The dataset is constructed from the downloaded sources using [create_COVIDx_CT.ipynb](../create_COVIDx_CT.ipynb). Several variables in this notebook must be set to reflect the locations of the raw data sources:
* `CNCB_EXCLUDE_FILE`: this points to the file `cncb_exclude_list.txt`, and should not be modified unless this file is moved from its default location.
* `CNCB_EXTRA_LESION_FILE` this points to the file `cncb_extra_lesions_slices.csv`, and should not be modified unless this file is moved from its default location.
* `CNCB_DIR`: this should be set to the location of the prepared CNCB dataset, with Normal, CP, and NCP directories as well as metadata files.
* `RADIOPAEDIA_CORONACASES_CT_DIR`: this should be set to the location of the `COVID-19-CT-Seg_20cases` directory from the COVID-19 CT Lung and Infection Segmentation Dataset.
* `RADIOPAEDIA_CORONACASES_SEG_DIR`: this should be set to the location of the `Infection_Mask` directory from the COVID-19 CT Lung and Infection Segmentation Dataset.
* `RADIOPAEDIA_META_CSV`: this points to the file `radiopaedia_metadata.csv`, and should not be modified unless this file is moved from its default location.
* `RADIOPAEDIA_EXCLUDE_FILE`: this points to the file `radiopaedia_exclude_list.txt`, and should not be modified unless this file is moved from its default location.
* `LIDC_META_CSV`: this points to the file `lidc_idri_metadata.csv`, and should not be modified unless this file is moved from its default location.
* `COVID_19_20_DIR`: this should be set to the location of the `Train` directory from the COVID-19-20 challenge dataset.
* `TCIA_COVID_META_CSV`: this points to the file `tcia_covid_metadata.csv`, and should not be modified unless this file is moved from its default location.
* `TCIA_DIR`: this should be set to the location of the `CT_Images_in_COVID-19_August_2020` directory from the NIH TCIA dataset.
* `COVID_CTSET_META_CSV`: this should be set to the location of the file `Labels&Detailes/Patient_details.csv` from COVID-CTSet.
* `COVID_CTSET_DIR`: this should be set to the location of the `Train&Validation` directory from COVID-CTSet.
* `ICTCF_META_CSV`: this points to the file `ictcf_metadata.csv`, and should not be modified unless this file is moved from its default location.
* `ICTCF_DIR`: this should be set to the location of the directory containing iCTCF patient cases in JPEG form (i.e., case directories for `Patient 1`, `Patient 2`, etc.).
* `COVID_CT_MD_DIR`: this should be set to the root directory containing the `{COVID-19, Cap, Normal} Cases` directories from COVID-CT-MD.
* `COVID_CT_MD_INDEX_CSV`: this should point to COVID-CT-MD's `Index.csv` file.
* `COVID_CT_MD_META_CSV`: this points to the file `covid_ct_md_metadata.csv`, and should not be modified unless this file is moved from its default location.
* `COVID_CT_MD_LABEL_CSV`: this should be set to the location of the file `Slice-level-labels.npy` from COVID-CT-MD.
* `STOIC_META_CSV`: this points to the file `stoic_metadata.csv`, and should not be modified unless this file is moved from its default location.
* `STOIC_DIR`: this should be set to the location of the `data/mha` directory of the STOIC data.
* `STONYBROOK_META_CSV`: this points to the file `stonybrook_metadata.csv`, and should not be modified unless this file is moved from its default location.
* `STONYBROOK_DIR`: this should be set to the location of the `COVID-19-NY-SBU` directory of the Stony Brook data (the inner directory containing patient cases).
* `OUTPUT_DIR`: this should be set to the directory in which the final dataset will be created.

To create the experimental "B" variant of the dataset, additional paths must be set:
* `MOSMED_CT_DIR`: this should be set to the location of the `COVID19_1110/studies` directory from MosMedData.
* `MOSMED_SEG_DIR`: this should be set to the location of the `COVID19_1110/masks` directory from MosMedData.
* `MOSMED_META_CSV`: this points to the file `mosmed_metadata.csv`, and should not be modified unless this file is moved from its default location.

### Step 3: Running the Notebook
Once the notebook is prepared, the dataset is constructed by simply running all cells in the notebook. Notably, `create_COVIDx_CT.ipynb` will not re-create files which were previously created, allowing for the dataset construction to be interrupted without having to recreate all the files.

After running the construction cells, there is an optional check at the end of the notebook to ensure that all files were created successfully.

## Data Distribution

### COVIDx CT-3A
Chest CT image distribution

|  Type | Normal | Pneumonia | COVID-19 | Total |
|:-----:|:------:|:---------:|:--------:|:-----:|
| train |  35996 |   26970   |  294552  |357518 |
|   val |  17570 |    8008   |    8147  | 33725 |
|  test |  17922 |    7965   |    7894  | 33781 |

Patient distribution

|  Type | Normal | Pneumonia | COVID-19 | Total |
|:-----:|:------:|:---------:|:--------:|:-----:|
| train |   321  |     592   |   3336   | 4249  |
|   val |   164  |     202   |    194   |  560  |
|  test |   164  |     138   |    201   |  503  |

### COVIDx CT-3B
Chest CT image distribution

|  Type | Normal | Pneumonia | COVID-19 | Total |
|:-----:|:------:|:---------:|:--------:|:-----:|
| train |  35996 |   26970   |  300733  |363699 |
|   val |  17570 |    8008   |    8147  | 33725 |
|  test |  17922 |    7965   |    7894  | 33781 |

Patient distribution

|  Type | Normal | Pneumonia | COVID-19 | Total |
|:-----:|:------:|:---------:|:--------:|:-----:|
| train |   321  |     592   |   4092   | 5005  |
|   val |   164  |     202   |    194   |  560  |
|  test |   164  |     138   |    201   |  503  |
