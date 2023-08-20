Car Object Detection Using Tensorflow
References
Nicholas Renotte Course and Code: https://www.youtube.com/watch?v=yqkISICHH-U (https://github.com/nicknochnack/TFODCourse) 
Jett Heaton Troubleshooting TF, CUDA, cuDNN Setup: https://www.youtube.com/watch?v=OEFKlRSd8Ic

Setup
Create Virtual Environment
Link Virtual Environment to Jupyter Notebook (see video 2)
Visual Studio
Cuda, cuDNN (to use GPU to train models, much faster than CPU)
Add to Environment Variables -> Path
 
In order to use your GPU to train, you need to downgrade Python and Tensorflow (https://www.tensorflow.org/install/pip#windows-native). I used Python 3.9.17 and Tensorflow 2.10.0. You may also need to download Tensorflow-gpu 2.10.0.

Image Collection
Use Nicholas’s file 1 or alternatively screenshot or download images separately. Make sure to have different positions/colors/time/etc.

Image Labeling
>>>cd Tensorflow/labelimg
>>>python labelImg.py
Create tight boxes around your target

Training
Pick a pretrained model from Detection Model Zoo (https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md). Models vary based on time to train and accuracy.
Make sure to confirm the GPU is showing and will be used for training.

Evaluation
Run the model on the Test group
Use Tensorboard to evaluate performance
Input specific images or feed live webcam video to see how the model actually works

Improvements
Add more training/testing data
Use a different pre-trained model
Modify classes

Coding Tips
Install packages				Python -m pip install ipykernel
Install packages				conda install ipykernel
Check package version			pip freeze | findstr tensorflow
Clear Terminal Output			cls
Quit					Ctrl + C
Monitor GPU usage			nvidia-smi --query-gpu=utilization.gpu --format=csv --loop=1      
nvidia-smi


To Access Terminal
run miniconda3 as admin

cd C:\Users\Matt Matsuo\Documents\tf_final
conda activate tf_final
jupyter notebook

TEST IF GPU IS CORRECTLY HOOKED UP AND USABLE WITH CUDA
python
import tensorflow as tf
print(tf.__version__)
len(tf.config.list_physical_devices('GPU'))>0



Car Project Specifics
Documents/tf_final
Other Train and Test – used this data when attempted multiple classes. Raw video here.
Tensorflow/workspace/annotations – label_map shows categories, test and train records are the compiled images and these files will be used for train/test
Tensorflow/workspace/images – images with their corresponding label files
Tensorflow/workspace/models
	my_ssd_mobnet
	my_ssd_mobnet_v2
	ssd_resnet50_v1_fpn_640_v1
	efficientdet_d1_car_only_v1
Tensorflow/workspace/pre-trained-models – downloaded models from the Zoo


Results
The best model was efficientdet_d1_car_only_v1
Example results are in Final Model Live Webcam Footage.mp4
This model only attempted to classify cars. I dropped the trailer, truck, bus, and van tags.
 
  
  
 

Future Steps
Train for trailers, trucks, buses, and vans
Collect data on motorcycles, bicycles, and pedestrians
Classify and train a model on vehicles travelling from right to left and left to right
Try other models in Zoo as well as other methods such as YOLO (you only look once)
Collect data on vehicle speed
Collect data on traffic (volume) by time of day and day of the week
