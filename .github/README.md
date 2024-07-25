# Fair-Robustness
Measuring AI Fairness in a Continuum Maintaining Nuances: A Robustness Case Study

## Short Description
As machine learning is increasingly making decisions about hiring or healthcare, we want AI to treat ethnic and socioeconomic groups fairly. Fairness is currently measured by comparing the average accuracy of reasoning across groups. We argue that improved measurement is possible on a continuum and without averaging, with the advantage that nuances could be observed within groups. Through the example of skin cancer diagnosis, we illustrate a new statistical method that treats fairness in a continuum. We outline this new approach and focus on its robustness against three distinct types of adversarial attacks. Indeed, such attacks can influence data in ways that may cause different levels of misdiagnosis for different skin tones, thereby distorting fairness. Our results reveal nuances that would not be evident in a strictly categorial approach.

## Datasets
| Name | Link |
|------|------|
| HAM10000 | [HAM10000 Dataset](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) |
| HAM10000 Segmentation | [HAM10000 Segmentation Dataset](https://www.kaggle.com/datasets/tschandl/ham10000-lesion-segmentations) |
| Fitzpatrick17K | [Fitzpatrick17K Dataset](https://www.kaggle.com/datasets/nazmussadat013/fitzpatrick-17k-dataset) |

## Preprocessing
Preprocessing consists of skin detection, data balancing, splitting datasets, and measuring the skin nuance process.

| Dataset | Link |
|---------|------|
| HAM10000 Preprocessing | [HAM10000 Preprocessing Notebook](https://github.com/Kuniko925/FairRobustness/blob/main/src/Preprocessing%20HAM10000.ipynb) |
| Fitzpatrick17K Preprocessing | [Fitzpatrick17K Preprocessing Notebook](https://github.com/Kuniko925/FairRobustness/blob/main/src/Preprocessing%20Fitzpatrick17K.ipynb) |

## Training -- Adversarial Attacks

### HAM10000
| Model | Link |
|-------|------|
| CNN | [HAM10000 CNN Experiments](https://github.com/Kuniko925/FairRobustness/blob/main/src/Experiments%20HAM10000.ipynb) |
| ResNet50 | [HAM10000 ResNet50 Experiments](https://github.com/Kuniko925/FairRobustness/blob/main/src/Experiments%20HAM10000%20ResNet.ipynb) |

### Fitzpatrick17K
| Model | Link |
|-------|------|
| CNN | [Fitzpatrick17K CNN Experiments](https://github.com/Kuniko925/FairRobustness/blob/main/src/Experiments%20Fitzpatrick17K.ipynb) |
| ResNet50 | [Fitzpatrick17K ResNet50 Experiments](https://github.com/Kuniko925/FairRobustness/blob/main/src/Experiments%20Fitzpatrick17K%20ResNet.ipynb) |
