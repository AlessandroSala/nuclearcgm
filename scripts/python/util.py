import numpy as np
import matplotlib.pyplot as plt
def normalize(v):
    norm = np.dot(v,v)
    if(norm==0):
        return v
    return v/norm**0.5

def positive_vector(v):
    sum = 0
    for i in range(len(v)):
        sum = sum + v[i]
    return v if sum > 0 else -v


def ps(v):
    return normalize(positive_vector(v))