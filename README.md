## To start training
1. Unzip and put data at `/car-segmentation`
2. Run
`python train.py --dirPath='/car-segmentation' --epochs=20 --pretrained_model_name='resnet34' --model_save_path='/trained_model'`
3. Trained model is saved at `/trained_model`

## For inference
1. Put images at `/car-segmentation/test_images`
2. Run
`python inference.py --model_path='/trained_model' --dirPath='/car-segmentation' --image_dir='test_images'`
3. Results stored at `/car-segmentation/results`