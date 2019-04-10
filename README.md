# Face Recognition System powered by Inception Network

An Introduction to **one-shot learning**

Implementation of Face-recognition system using [FaceNet](https://arxiv.org/pdf/1503.03832.pdf) network.

This is based on learning a **Euclidean em-bedding** per image using a deep convolutional network. The network  is  trained  such  that  the  squared  L2  distances  in the embedding space directly correspond to face similarity.

FaceNet is a combination of Siamese Network at the end of Inception Network.

**FaceNet Architecture:**
      
      Image(96×96×3) -> InceptionNetwork -> SiameseNetwork -> Output

More info about [InceptionNetwork](https://www.cs.unc.edu/~wliu/papers/GoogLeNet.pdf) and [SiameseNetwork](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf) is available in the later sections of this documentation.   

we feed frames from the webcam to the network to determine whether or not the frame conatins an individual we recognise.

## Working:

Firstly, we make our database of the faces which we want to recognise. This will be a directory named **images**. To do this, different functions are defined based on the users requirements. Input image to the network must of shape **96×96×3**. A pre-processing pipeline is involved befor saving the image to database. While recognising faces, a frame (which contains a face) is taken from webcam and fed into our network. The network takes in the camera frame and database, compares the similarities and differences between each set of frame and database image. The output will be a string which is the name of the most likely similar image in the database. If the face is not found in the database, the output will be a zero. The essence of each file in this repo is each given below.

 
 - [face function.py](https://github.com/madhavambati/Face-Recognition-powered-by-FaceNet/blob/master/face_functions.py) contains a preprocessing pipeline and some other essential functions.
 - [add_to_database.py](https://github.com/madhavambati/Face-Recognition/blob/master/add_to_database.py) takes a frame from the webcam and saves in the images directory (database)
 - [face_cutter.py](https://github.com/madhavambati/Face-Recognition/blob/master/face_cutter.py) extracts a face from an image and saves in database.
 - [face_recogniser.py](https://github.com/madhavambati/Face-Recognition/blob/master/face_recogniser.py) main file which recognises faces.
 - [fr_utils.py](https://github.com/madhavambati/Face-Recognition/blob/master/fr_utils.py) contains some important functions for Inception network.
 - [inception_network.py](https://github.com/madhavambati/Face-Recognition/blob/master/inception_network.py) contains Inception network blocks
 - [haarcascade_frontalface_default.xml](https://github.com/madhavambati/Face-Recognition/blob/master/haarcascade_frontalface_default.xml) for detecting faces.

## One-shot Learning:
Normally, in deep learning, we need a large amount of data and the more we have, the better the results get. However, it will be more convenient to learn only from few data because not all of us are rich in terms of how much data we have. The idea here is that we need to learn an object class from only a few data and that’s what One-shot learning algorithm is.

## Problems with CNN's in Face-Recognition:
In face recognition systems, we want to be able to recognize a person’s identity by just feeding one picture of that person’s face to the system i.e **one-shot learning** should be implemented. And, in case, it fails to recognize the picture, it means that this person’s image is not stored in the system’s database.

To solve this problem, we cannot use only a convolutional neural network for two reasons: 
1) CNN doesn’t work on a small training set. 
2) It is not convenient to retrain the model every time we add a picture of a new person to the system.

However, we can use Siamese neural network for face recognition.


