import pickle
import cv2
import dlib
from imutils import paths
from face_recognition import face_locations, face_encodings, compare_faces

knownObjProperties = {}

trainSet = {}
testSet = {}

from os import listdir

def build_dataset_from_folder(curset,type,DATASET):
    for folder in listdir(f"{DATASET}/{type}/"):
        curset[folder] = []
        for file in listdir(f"{DATASET}/{type}/{folder}"):
            curset[folder].append(file)


def get_face_encodings(DATASET):
    # training set
    build_dataset_from_folder(trainSet,'training',DATASET)    
    # test set
    build_dataset_from_folder(testSet,'testing',DATASET)  

    for faces in trainSet:
        knownObjProperties[faces] = []
        for face in trainSet[faces]:
            image = cv2.imread(f"{DATASET}/training/{faces}/{face}",cv2.CV_8UC3)
            rgb = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
            boxes = face_locations(rgb,model='cnn')
            encodings = face_encodings(rgb,boxes)

            knownObjProperties[faces].append(encodings)  
        print(f"[INFO] {faces} encoded...")
    
    return knownObjProperties, trainSet, testSet


def get_face_encoding(DATASET_LOC,person,input,unknownEncodings):

    input_encodings = []

    
    for face in input:
        image = cv2.imread(f"{DATASET_LOC}/testing/{person}/{face}",cv2.CV_8UC3)
        rgb = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)

        print("[INFO] Recognizing faces...")
        boxes = face_locations(rgb,model='cnn')
        knownEncoding = face_encodings(rgb, boxes)

        for unknown in unknownEncodings:
            j = 0
            for e in unknownEncodings[unknown]:
                j += 1                
                matches = compare_faces(e,knownEncoding[0],tolerance=0.5)
                if True in matches:
                    print(person + ' se parece com ' + unknown + ' imagem ' + str(j))