from sklearn.metrics import f1_score, accuracy_score
from face_recognition import face_distance
import numpy as np

distances = [] # squared L2 distance between pairs
identical = [] # 1 if same identity, 0 otherwise

thresholds = np.arange(0.3, 1.0, 0.01)

def calc_threshold(encodings):

    num = len(encodings)
    _keys = list(encodings.keys())

    for i in range(num - 1):
        for j in range(i + 1, num):
            a = encodings[_keys[i]]
            b = encodings[_keys[j]][0]
            distances.append(face_distance(encodings[_keys[i]], b))
            identical.append(1 if _keys[i] == _keys[j] else 0)

    
    f1_scores = [f1_score(identical, distances < t) for t in thresholds]
    
    print(1)