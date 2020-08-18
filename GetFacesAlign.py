from os import listdir,mkdir
from os.path import isdir, join, isfile, splitext
import os
import re
import cv2
import face_recognition
from face_recognition import face_locations

old_folder = 'data5'
new_folder = 'data55'

size_face =[160,160]
#how data! images ! olla!
"""
old_folder\
...........|negatives\
.....................|
.....................| #is Not smiler
.....................|
...........|positives\
.....................|
.....................| #is smiler!
.....................|
.....................|
...../
"""


def image_files_in_folder(folder):
    return [os.path.join(folder, f) for f in os.listdir(folder) if re.match(r'.*\.(jpg|jpeg|png|bmp)', f, flags=re.I)]

def train(train_dir ,data_save ,verbose=False):
    mm=0
    for class_dir in listdir(train_dir):
        if not isdir(join(train_dir, class_dir)):
            continue
        Name = class_dir
        path = data_save + '/' + Name
        print("data_Now!:",Name)
        if not isdir(path):
           mkdir(path)
        mm=0
        for img_path in image_files_in_folder(join(train_dir, class_dir)):
            image = cv2.imread(img_path) 
            faces_bboxes = face_locations(image)
            
            if len(faces_bboxes) != 1:
                continue

            mm+=1
            if faces_bboxes is not None:
                   for (top, right, bottom, left) in faces_bboxes:

                       cut_face = image[top:bottom, left:right]
                       gray = cv2.cvtColor(cut_face, cv2.COLOR_BGR2GRAY)
                       cut_face2 = cv2.resize(cut_face,(size_face[0],size_face[1]))
                       cv2.imwrite("{}/{}-{}.jpg".format(path,Name,mm),cut_face2)
                       print("save image:{}-{}.jpg in path! {}".format(Name,mm,path))
                       
        print("saved person: {}. and images num: {}".format(Name,mm))
    

if __name__ == "__main__":
  if not isdir(old_folder):
     print("Error! no images on dir!- '{}'! plz chk your images".format(old_folder))
  if not isdir(new_folder):
     mkdir(new_folder)
  train(old_folder,new_folder)
  print("done")#..
