# ProposedNet

## Run

create the environment:
```
conda env create --file environment.yaml
```

Execute the following commands:
```
python main.py --dataset ckplus --data_path ../datasets
```

## Folder Structure
```bash
./
├── datasets/             # Directory for storing datasets
└── src/                  # Directory containing the main source code of the project
    ├── models/           # Directory containing deep learning model architecture code
    │   ├── __init__.py
    │   └── model.py      # File defining the model structure
    ├── dataset.py        # File defining dataset classes and data loading functions
    ├── main.py           # Main script file for running experiments
    └── utils.py          # File containing auxiliary functions such as folder creation and performance measurement
```

## Datasets

| **Dataset Name** | **Number of Images** | **Official Homepage** | **Acquisition Method** | **Dataset Download Link** |
|:---------------:|:----------:|:---------------:|---------------|:---------------:|
| RAFDB | 15339 | [Homepage](http://www.whdeng.cn/RAF/model1.html#dataset) | Downloaded face-aligned dataset using [MTCNN](https://github.com/foamliu/Face-Alignment) | [Google Drive](https://drive.google.com/file/d/1GiVsA5sbhc-12brGrKdTIdrKZnXz9vtZ/view) |
| FER2013 | 35887 | [Kaggle](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data) | Downloaded `icml_face_data.csv` from official Kaggle link and created `fer2013_modified` by keeping only "emotion" and "pixel" columns | [Google Drive](https://drive.google.com/drive/folders/1-mGIAchWBUEhgmIKT36PrvQ1-LXl3Y5n?usp=sharing) |
| FERPlus | 35711 | [Github](https://github.com/Microsoft/FERPlus) | Downloaded `fer2013new.csv` from official Github link and created `FERPlus_Label_modified.csv` using self-defined [label generation code](https://github.com/StevenHSKim/FERPlus_Vote_To_Label). Generated `FERPlus_Image` by converting pixel values from [FER2013](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data) to png using `generate_training_data.py` from official Github | [Google Drive](https://drive.google.com/drive/folders/1n73_68Zq4aa0KBImIANHhiSJMg6j2zVV?usp=sharing) |
| ExpW | 90560 | [Homepage](https://mmlab.ie.cuhk.edu.hk/projects/socialrelation/index.html) | Downloaded from homepage and manually aligned faces using [MTCNN](https://github.com/foamliu/Face-Alignment) | [Google Drive](https://drive.google.com/drive/folders/1jNmC5RWqyBFvFsTHnWpg-cti0kpWxEi0?usp=sharing) |
| SFEW2.0 | 1634 | [Homepage](https://users.cecs.anu.edu.au/~few_group/AFEW.html) | Requested dataset from authors via homepage, downloaded `_Aligned_Face` dataset and manually removed non-face images | [Google Drive](https://drive.google.com/drive/folders/1FuhcMW5LXaaWe8sKoGizQW78s04VcOHW?usp=sharing) |
| CK+ | 981 | [Homepage](https://www.jeffcohn.net/Resources/), [Kaggle](https://www.kaggle.com/datasets/shuvoalok/ck-dataset) | Downloaded dataset containing the last 3 frames captured from each video | [Google Drive](https://drive.google.com/drive/folders/1kuT6zQhZtyBPgTB4UqNJWq0ZBLAooTfA?usp=sharing) |

### RAFDB
Download the RAFDB dataset and organize it in the following format:
```
dataset/raf-basic/
    EmoLabel/
        list_patition_label.txt
    Image/aligned/
        train_00001_aligned.jpg
        test_0001_aligned.jpg
        ...
```

### FER2013
Download the FER2013 dataset and organize it in the following format:
```
dataset/FER2013/
    fer2013_modified.csv
```

### FERPlus
Download the FERPlus dataset and organize it in the following format:
```
dataset/FERPlus/
    FERPlus_Label_modified.csv
    FERPlus_Image/
        fer0000000.png
        fer0000001.png
        ...
```

### ExpW
Download the ExpW dataset and organize it in the following format:
```
dataset/ExpW/
    label/
        label.lst
    aligned_image/
        afraid_African_214.jpg
        afraid_american_190.jpg
        ...
```

### SFEW2.0
Download the SFEW2.0 dataset and organize it in the following format:
```
dataset/SFEW2.0/
    sfew_2.0_labels.csv
    sfew2.0_images/
        image_000000.png
        image_000001.png
        ...
```

### CK+
Download the CK+ dataset and organize it in the following format:
```
dataset/CKPlus/
    ckplus_labels.csv
    ckplus_images/
        image_000000.png
        image_000001.png
        ...
```
