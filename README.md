# A Local Spatial-Temporal Synchronous Network to Dynamic Gesture Recognition

## Introduction

We proposea local spatial-temporal synchronous network (LSTSN) for skeletonbased dynamic gestures recognition, which can simultaneously
deal with spatial and temporal information in gestures.
 The code of training our approach for skeleton-based hand gesture recognition on the [DHG-14/28 Dataset](http://www-rech.telecom-lille.fr/DHGdataset/) 
and the [SHREC’17 Track Dataset](http://www-rech.telecom-lille.fr/shrec2017-hand/) is provided in this repository.


### Prerequisites

This package has the following requirements:

* `Python 3.6`
* `Pytorch v1.6`

### Training
1. Download the [DHG-14/28 Dataset](http://www-rech.telecom-lille.fr/DHGdataset/) or the [SHREC’17 Track Dataset](http://www-rech.telecom-lille.fr/shrec2017-hand/).

2. Set the path to your downloaded dataset folder in the ```/util/DHG_parse_data.py ``` or ```the /util/SHREC_parse_data.py ```.

3. Set the path for saving your trained models in the ```train_on_DHG.py`` or the ```train_on_SHREC.py ```.

4. Show obfuscation matrix in ```SHREC-user.py ``` .

5. Run one of following commands.
```
python train_on_SHREC.py       # on SHREC’17 Track Dataset
python train_on_DHC.py         # on DHG-14/28 Dataset
```