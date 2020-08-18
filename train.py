from os import listdir
from os.path import isdir, join, isfile, splitext
import os
import re
import pickle
import cv2
import face_recognition


def image_files_in_folder(folder):
    return [os.path.join(folder, f) for f in os.listdir(folder) if re.match(r'.*\.(jpg|jpeg|png|bmp)', f, flags=re.I)]

def train_data(train_dir ,train_facesDB ,add_persons = "no"):

    stack_train_old = [] ,[]
    stack_train = [] ,[] # name ,face

    if os.path.isfile(train_facesDB) and  add_persons == "yes":
       with open(train_facesDB, 'rb') as infile: #load old data from file
            (stack_train_old) = pickle.load(infile)
            print("old names", list(set(stack_train_old[0])))
            stack_train = stack_train_old
    for class_name in listdir(train_dir):
        if not isdir(join(train_dir, class_name)):
            continue
        Name = class_name
        print("person_Now!:",Name)
        if os.path.isfile(train_facesDB) and add_persons == "yes":

           '''compare names'''
           if Name in stack_train_old[0]:
                  print("This person:::{}::: already exists".format(Name))
                  print("You are now in a gradual training process")
                  print("But unfortunately there is a problem !!")
                  print("The name of the person you just entered conflicts with the name of someone we already have ")
                  print("If you are sure that this is a new person and have not been trained before")
                  print("Please try to change the name by another name and then try again")
                  print("However, this person will be bypassed without training thank you.")
                  continue

              

        for img_path in image_files_in_folder(join(train_dir, class_name)):
            image = cv2.imread(img_path) # test image not other use 
            if image is None:
               print("continue_image_an_Error")
               continue
            image = face_recognition.load_image_file(img_path)
            faces_bboxes = face_recognition.face_locations(image)
            if len(faces_bboxes) != 1:
                
                print("image {} not fit for training: {}".format(img_path, \
                      "didn't find a face" if len(faces_bboxes) < 1 else "found more than one face"))
                continue
            
            if faces_bboxes is not None:
                  face_encoding = face_recognition.face_encodings(image)[0]

                  stack_train[0].append(Name)
                  stack_train[1].append(face_encoding)
                  
    if len(stack_train[0]) > 0: # save data train
       with open(train_facesDB, 'wb') as outfile:
          pickle.dump((stack_train), outfile, pickle.HIGHEST_PROTOCOL)

    if add_persons == "yes":
       print("add persons to old persons to  done run")
    if add_persons == "no" or len(stack_train_old[0]) == 0:
       print("train new not added")

if __name__ == "__main__":
  data_input = "data_align/"
  train_facesDB = "train_faces_data.pkl"
  if not isdir(data_input):
     print("Error! no images on dir!- '{}'! plz chk your data".format(data_input))

  train_data(data_input ,train_facesDB ,add_persons = "yes")
  print("done")

