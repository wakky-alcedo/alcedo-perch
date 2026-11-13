"""Common Kingfisher model -> v4 body frame (x lateral right, y cranial, z dorsal), before scaling/translation.
Bird's right side is model -z (bird faces -x with y up). Cranial axis = 1st PCA axis of the torso part."""
import numpy as np
YAX = np.array([-0.665, 0.747, 0.0]); YAX /= np.linalg.norm(YAX)
XAX = np.array([0, 0, -1.0])
ZAX = np.cross(XAX, YAX)
R_CK = np.vstack([XAX, YAX, ZAX])        # rows: body axes in model coords;  p_body = R_CK @ p_model


def to_body(V, origin=np.zeros(3), scale=1.0):
    return (np.asarray(V) - origin) @ R_CK.T * scale
