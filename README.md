# CNN-to-automatically-recognize-mesh-texture

In this project a CNN is used to automatically recognize texture in 3D meshes

## Installation

1. Clone this repository with:
```sh
git clone https://github.com/stefanoagostini/CNN-to-automatically-recognize-mesh-texture.git
```
2. Prerequisites for run this code:
    * scipy    
    * numpy 
    * cv2
    * glob
    * keras
    * scikit-learn
    * imutils
    * argparse
    
   You can install these using pip or Anaconda

## Usage for the user

1. Run the script 'GenerateImages.py' to generete dataset of images

2. Run the script 'train.py' ( default values have been set ) 

    - For change this values run the script from the command line:
        ```sh
        python train.py -s <split-policy> -v <validation-split> -k <k-fold> -b <batch-size> -e <epochs>

        ```
3. Run the script 'evaluate_folds.py' (if you have not changed the parameter split-policy into 'train.py')
    
    - If you have set <split-policy> to 'single': run the script 'evaluate_single.py'
    
  
## Requirements

The use of a GPU is recommended to train the CNN

## Contributors
Created by [Stefano Agostini](https://github.com/stefanoagostini) & [Giovanna Scaramuzzino](https://github.com/ScaramuzzinoGiovanna)


_You can't use this code for commercial purposes_