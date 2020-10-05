from face_encodings import get_face_encodings, get_face_encoding
from metrics import calc_threshold

DATASET_LOC = './dataset/att_faces'
PERSON = 's17'

encodings, trainset, testset = get_face_encodings(DATASET_LOC)
threshold = calc_threshold(encodings)

# input = testset[PERSON]
# test = get_face_encoding(DATASET_LOC,PERSON,input,encodings)



# print(1)