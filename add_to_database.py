import cv2
import numpy as np
import matplotlib.pyplot as plt
from face_functions import speak, add_to_database
 

name = input("Enter your Name: ")
speak('saving '+ name +'to database', 2)
add_to_database(name)
