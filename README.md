# UNet for Image Segmentation

Image Segmentation performed on GTA 5 Games Dataset using UNet Architecture

<b>Dataset URL:</b> http://download.visinf.tu-darmstadt.de/data/from_games<br>
<b>UNet Paper:</b> https://arxiv.org/abs/1505.04597

<b>Required Libraries:</b> `torch, numpy, PIL, glob, torchsummary, argparse, os, cv2`

`datagenerator.py` : To create custom data generation that we can use in PyTorch Code.<br>
`model.py`  : Implemented U-Net architecture here.<br>
`main.py`   : Contains train, validation functions with metrics used for segmentation.<br>
`test.py`   : Test on given images and save the predicted output as images.<br>


Run `main.py` to start training the model.

# Commands to run:
python main.py -i `image_directory` -l `label_directory` -lr `learning_rate` -e `epochs` -b `batch_size` -cp `checkpoint_saved`
<br><br>
For testing:<br>
python test.py -i `image_directory` -l `label_directory` -s `save_predicted_directory` -cp `checkpoint_saved`

<b>Output:</b> <br>
Here are some of my predictions from UNet model<br>
![Prediction](https://github.com/bharath3794/UNet-Image-Segmentation/blob/main/My%20Predictions.png?raw=true)
