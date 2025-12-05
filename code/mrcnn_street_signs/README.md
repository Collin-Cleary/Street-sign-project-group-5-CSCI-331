Uses the matterporn mrcnn implementation from: https://github.com/matterport/Mask_RCNN, more specifically the updated fork that works with tensforflow >= 2.0 from: https://github.com/ahmedfgad/Mask-RCNN-TF2

## Setup
(I used Python 3.7.11, but any version that supports the package versions in requirements.txt should word)
--pip install -r requirements.txt
--py setup.pu

## Using the model
signs/street_signs.py contains the config file needed to run the model on the dataset
signs/street_sign_visualization.ipynb contains code for training and evaluating the model