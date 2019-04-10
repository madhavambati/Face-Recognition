# Face Recognition System powered by Inception Network

An Introduction to **one-shot learning**

Implementation of Face-recognition system using [FaceNet](https://arxiv.org/pdf/1503.03832.pdf).

This is based on learning a **Euclidean em-bedding** per image using a deep convolutional network. The network  is  trained  such  that  the  squared  L2  distances  in the embedding space directly correspond to face similarity.

FaceNet is a combination of Siamese Network at the end of Inception Network.

**FaceNet Architecture:**
      
      Image(96×96×3) -> InceptionNetwork -> SiameseNetwork -> Output

More info about [InceptionNetwork](https://www.cs.unc.edu/~wliu/papers/GoogLeNet.pdf) and [SiameseNetwork](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf) is available in the later sections of this documentation.   

we feed frames from the webcam to the network to determine whether or not the frame conatins an individual we recognise.

## Working:

Firstly, we make our database of the faces which we want to recognise. This will be a directory named **images**. To do this, different functions are defined based on the users requirements. Input image to the network must of shape **96×96×3**. A pre-processing pipeline is involved befor saving the image to database. While recognising faces, a frame (which contains a face) is taken from webcam and fed into our network. The network takes in the camera frame and database, compares the similarities and differences between each set of frame and database image. The output will be a string which is the name of the most likely similar image in the database. If the face is not found in the database, the output will be a zero. The essence of each file in this repo is each given below.

 
 - [face functions.py](https://github.com/madhavambati/Face-Recognition-powered-by-FaceNet/blob/master/face_functions.py) contains a preprocessing pipeline and some other essential functions.
 - [add_to_database.py](https://github.com/madhavambati/Face-Recognition/blob/master/add_to_database.py) takes a frame from the webcam and saves in the images directory (database)
 - [face_cutter.py](https://github.com/madhavambati/Face-Recognition/blob/master/face_cutter.py) extracts a face from an image and saves in database.
 - [face_recogniser.py](https://github.com/madhavambati/Face-Recognition/blob/master/face_recogniser.py) main file which recognises faces.
 - [fr_utils.py](https://github.com/madhavambati/Face-Recognition/blob/master/fr_utils.py) contains some important functions for Inception network.
 - [inception_network.py](https://github.com/madhavambati/Face-Recognition/blob/master/inception_network.py) contains Inception network blocks
 - [weights](https://github.com/madhavambati/Face-Recognition/tree/master/weights) cointains weights of pre-trained Inception network
 - [haarcascade_frontalface_default.xml](https://github.com/madhavambati/Face-Recognition/blob/master/haarcascade_frontalface_default.xml) for detecting faces.

## One-shot Learning:
Normally, in deep learning, we need a large amount of data and the more we have, the better the results get. However, it will be more convenient to learn only from few data because not all of us are rich in terms of how much data we have. The idea here is that we need to learn an object class from only a few data and that’s what One-shot learning algorithm is.

## Problems with CNN's in Face-Recognition:
In face recognition systems, we want to be able to recognize a person’s identity by just feeding one picture of that person’s face to the system i.e **one-shot learning** should be implemented. And, in case, it fails to recognize the picture, it means that this person’s image is not stored in the system’s database.

To solve this problem, we cannot use only a convolutional neural network for two reasons: 
1) CNN doesn’t work on a small training set. 
2) It is not convenient to retrain the model every time we add a picture of a new person to the system.

However, we can use Siamese neural network for face recognition.

## Siamese Network:

Siamese network is an artificial neural network that use the same weights while working in tandem on two different input vectors to compute comparable output vectors. Often one of the output vectors are precomputed, thus forming a baseline the other output vector are compared against. This is similar to a comparing fingerprints or more technical as a distance function for Locality-sensitive hashing. 

<img src="https://user-images.githubusercontent.com/27866638/55897417-18e46780-5bde-11e9-8b05-932523cd5388.png" width="880">

The first subnetwork’s input is an image, followed by a sequence of convolutional, pooling, fully connected layers and finally a feature vector (We are not going to use a softmax function for classification). The last vector f(x1) is the encoding of the input x1. Then, we do the same thing for the image x2, by feeding it to the second subnetwork which is totally identical to the first one to get a different encoding f(x2) of the input x2.

To compare the two images x1 and x2, we compute the distance d between their encoding f(x1) and f(x2). If it is less than a threshold (a hyperparameter), it means that the two pictures are the same person, if not, they are two different persons.
<img src="https://user-images.githubusercontent.com/27866638/55897610-80021c00-5bde-11e9-9a8a-31bc6ac9a348.png" width="500"><img src="https://user-images.githubusercontent.com/27866638/55897695-9dcf8100-5bde-11e9-8207-cad6b9ee937f.png" width="500">

In order to learn parameters to get good encodding for the images we use [Triplet loss function](https://www.youtube.com/watch?v=d2XB5-tuCWU).
## Triplet Loss function:

<img src="https://user-images.githubusercontent.com/27866638/55898109-8a70e580-5bdf-11e9-978f-16b630594158.png" width = "800">

In [Triplet loss function](https://www.youtube.com/watch?v=d2XB5-tuCWU), we use three images:

- Anchor image **A**
- Positive image **P**
- Negative image **N**

So, we want the distance d(A, P) between the encoding of the anchor and the encoding of the positive example to be less than or equal to the distance d(A, N) between the encoding of the anchor and the encoding of the negative example.

The problem here is that the model can learn to make the same encoding for different images. For this reason, we are adding a margin alpha (hyperparameter), to prevent this from happening, and to always have a gap between A and P versus A and N.

<img src="https://user-images.githubusercontent.com/27866638/55898430-2f8bbe00-5be0-11e9-9d16-484ede4df38b.png" width="500"><img src="https://user-images.githubusercontent.com/27866638/55898614-9f01ad80-5be0-11e9-97d2-b8ff4f8aa038.png" width="500">

As mentioned in earlier sections we use an Inception Network which is then connected to a siamese network to get different image encoddings.

## Inception Network/ GoogLeNet: 

[Inception Network](https://www.cs.unc.edu/~wliu/papers/GoogLeNet.pdf) is a one big mess of a neural net with a lot of hidden layers. Inception network architecture is given below. 

<img src="https://user-images.githubusercontent.com/27866638/55899664-01f44400-5be3-11e9-8ff5-cacba912bd76.png" width="880">

It contains 1×1 Convolution at the middle of the network. And global average pooling is used at the end of the network instead of using fully connected layers. These two techniques are from another paper “[Network In Network](https://arxiv.org/pdf/1312.4400.pdf)”. Another technique, called inception module, is to have different sizes/types of convolutions for the same input and stacking all the outputs.

Inception blocks with 1×1 Convolution

<img src="https://user-images.githubusercontent.com/27866638/55900263-69f75a00-5be4-11e9-8bd9-ec2f210dafd6.png" height="300">

The computational costs for training GoogLeNet are very high so I used a pre-trained model in my Face-recognition system.
If you want to go deep into the study of Inception network, refer to the links below or at the end of the documentation.

### Important links for Inception Network:

- [Research paper](https://www.cs.unc.edu/~wliu/papers/GoogLeNet.pdf)
- [Explaination of the Network](https://medium.com/coinmonks/paper-review-of-googlenet-inception-v1-winner-of-ilsvlc-2014-image-classification-c2b3565a64e7)
- [Explanation Video](https://www.youtube.com/watch?v=KfV8CJh7hE0)
- [Different versions of GoogleNet](https://towardsdatascience.com/a-simple-guide-to-the-versions-of-the-inception-network-7fc52b863202)
- [Overview](https://www.cs.colostate.edu/~dwhite54/InceptionNetworkOverview.pdf)

## Installation:

Use the following commands to install the model in your machine. 

- Clone the repository 

      git clone https://github.com/madhavambati/Face-Recognition.git
      
 - Move to directory Face-Recognition
 
       cd Face-Recognition
       
 - Install all the dependencies
 
       pip install requirements.txt
       
 - To add a face from webcam to database 
       
       python add_to_database.py
       
 - To add a face from image to database, first extract the face from image and then add the face.
       
        python face_cutter.py
        python add_to _database.py
        
 - To run the Face-recognition system
 
        python face_recogniser.py
        
## References:

- https://arxiv.org/pdf/1503.03832.pdf
- https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf
- https://www.cs.unc.edu/~wliu/papers/GoogLeNet.pdf
- https://arxiv.org/abs/1503.03832
- https://www.youtube.com/watch?v=d2XB5-tuCWU
- https://research.fb.com/publications/deepface-closing-the-gap-to-human-level-performance-in-face-verification/
- https://www.youtube.com/watch?v=-FfMVnwXrZ0
- https://medium.freecodecamp.org/making-your-own-face-recognition-system-29a8e728107c
