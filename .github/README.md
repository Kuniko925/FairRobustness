<p align="left"> </p>

 <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
 <a href="https://standardjs.com"><img src="https://img.shields.io/badge/code_style-standard-brightgreen.svg" alt="Standard - \Python Style Guide"></a>

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

## Training -- Adversarial Attacks

| Dataset      | Model    | Link |
|--------------|----------|------|
| HAM10000     | CNN      | [HAM10000 CNN Experiments](https://github.com/Kuniko925/FairRobustness/blob/main/src/Experiments%20HAM10000.ipynb) |
| HAM10000     | ResNet50 | [HAM10000 ResNet50 Experiments](https://github.com/Kuniko925/FairRobustness/blob/main/src/Experiments%20HAM10000%20ResNet.ipynb) |
| Fitzpatrick17K | CNN    | [Fitzpatrick17K CNN Experiments](https://github.com/Kuniko925/FairRobustness/blob/main/src/Experiments%20Fitzpatrick17K.ipynb) |
| Fitzpatrick17K | ResNet50 | [Fitzpatrick17K ResNet50 Experiments](https://github.com/Kuniko925/FairRobustness/blob/main/src/Experiments%20Fitzpatrick17K%20ResNet.ipynb) |

## License
This framework is available under the MIT License. 
 
## Acknowledgement
The authors would like to thank the Dependable Intelligence Systems Lab, the Responsible AI Hull Research Group, and the Data Science, Artificial Intelligence, and Modelling (DAIM) Institute at the University of Hull for their support. Furthermore, the author extends heartfelt gratitude to Professor Balaraman Ravindran of the Indian Institute of Technology Madras, whose invaluable provision of the initial research idea has been the cornerstone of this study.
