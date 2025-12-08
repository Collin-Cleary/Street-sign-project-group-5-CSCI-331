# CSCI-331-04-Group-5
### Traffic Sign Detection

### Abstract:
To build our Traffic Sign Detection System, we chose to implement the U-Net and Mask R-CNN image segmentation models. Based on the individual strengths of these models, we created two hypotheses: U-Net will perform 
better with semantic segmentation, and Mask R-CNN will perform better with instance segmentation. The initial dataset provided to us was unsuitable for U-Net, so a new dataset that used precise pixel maps had to be
found. After the models were implemented and trained, our hypotheses were confirmed, though the models clearly needed improvement, as accuracy was much lower than desired. The models will likely perform to our
standards with proper hyperparameter tuning.


### Members: 
Ben Simonds - research, planning, and documentation\
Collin Cleary - dataset preprocessing and U-Net implementation\
Logan Costa - dataset preprocessing and Mask R-CNN implmementation


### How to run:
to run the U-Net_trainer, ensure you have the kaggle dataset from this link: https://www.kaggle.com/datasets/viacheslavshalamov/russian-road-signs-segmentation-dataset in your data folder. The file architecture should look like data/archive/sign_dataset/-----train/
                                                                                                                                |---val/
Then cd into the code folder and run the command: "python U-Net_trainer.py"

to run the mrcnn model, navigate to code/mrcnn_street_signs/signs/street_sign_visualization.ipynb and run each cell. Ensure requirements are installed
