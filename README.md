# UNet for Image Segmentation
Basic template for Pytorch based DL training.

`datagenerator.py` : To create custom data generation that we can use in PyTorch.
`model.py`  : Implemented U-Net architecture here.
`main.py`   : Contains train, validation functions with metrics used for segmentation.

`test.py`   : Test on given images and save the predicted output as images.


Run `main.py` to start training the model.

# Commands to run:
python main.py -i `image_directory` -l `label_directory` -lr `learning_rate` -e `epochs` -b `batch_size` -cp `checkpoint_saved`

For testing:
python test.py -i `image_directory` -l `label_directory` -s `save_predicted_directory` -cp `checkpoint_saved`
