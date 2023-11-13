import numpy as np
cimport numpy as np
import cv2
cimport cython
import tensorflow as tf
from icecream import ic

ctypedef np.uint8_t DTYPE_UINT8_t

model = tf.keras.models.load_model('models/centernet_hg104_512x512_coco17_tpu-8\centernet_hg104_512x512_coco17_tpu-8\saved_model')
width, height = 512,512


def detect_obj(np.ndarray[DTYPE_UINT8_t, ndim=3] frame):
    # cdef np.ndarray[DTYPE_UINT8_t, ndim=3] resize_frame
    # resize_frame = cv2.resize(frame.copy(), (width, height))
    
    cdef np.ndarray[np.uint8_t, ndim=4] tf_frame
    tf_frame = np.expand_dims(frame, axis=0)
    
    cdef dict detections
    detections = model(tf_frame)
    return detections
