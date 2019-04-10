Firstly, we make our database of the faces which we want to recognise. This will be a directory named **images**. To do this, different functions are defined based on the users requirements. Input image to the network must of shape **96×96×3**. A pre-processing pipeline is involved befor saving the image to database. While recognising faces, a frame (which contains a face) is taken from webcam and fed into our network. The network takes in the camera frame and database, compares the similarities and differences between each set of frame and database image. The output will be a string which is the name of the most likely similar image in the database. If the face is not found in the database, the output will be a zero. The essence of each file in this repo is each given below.

 
 - [face function.py](https://github.com/madhavambati/Face-Recognition-powered-by-FaceNet/blob/master/face_functions.py) contains a preprocessing pipeline and some other essential functions.
 - [add_to_database.py](https://github.com/madhavambati/Face-Recognition/blob/master/add_to_database.py) takes a frame from the webcam and saves in the images directory(database)
 - [face_cutter.py](https://github.com/madhavambati/Face-Recognition/blob/master/face_cutter.py) extracts a face from an image and saves in database.
 - [face_recogniser.py](https://github.com/madhavambati/Face-Recognition/blob/master/face_recogniser.py) main file which recognises faces.
 - [fr_utils.py](https://github.com/madhavambati/Face-Recognition/blob/master/fr_utils.py) contains some important functions for Inception network.
 - [inception_network.py](https://github.com/madhavambati/Face-Recognition/blob/master/inception_network.py) contains Inception network blocks
 - [haarcascade_frontalface_default.xml](https://github.com/madhavambati/Face-Recognition/blob/master/haarcascade_frontalface_default.xml) for detecting faces.
 