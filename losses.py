from keras import backend as K


def jaccard_loss(y_true, y_pred):
    """Jaccard (IoU) loss function for use with Keras.
    """

    intersection = K.sum(K.abs(y_true * y_pred))
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred))
    jaccard = intersection / (sum_ - intersection)

    return 1-jaccard


def hybrid_jaccard(y_true, y_pred, jac_frac=0.25):
    """Hybrid binary cross-entropy and Jaccard loss function.
    """
    jac_loss = jaccard_loss(y_true, y_pred)
    bce = K.binary_crossentropy(y_true, y_pred)

    return jac_frac*jac_loss + (1-jac_frac)*bce

smooth = 1e-12
def jaccard_coef(y_true, y_pred):
    intersection = K.sum(y_true * y_pred, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred, axis=[0, -1, -2])
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return K.mean(jac)


def jaccard_coef_int(y_true, y_pred):
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))

    intersection = K.sum(y_true * y_pred_pos, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred_pos, axis=[0, -1, -2])

    jac = (intersection + smooth) / (sum_ - intersection + smooth)

    return K.mean(jac)


def jaccard_coef_loss(y_true, y_pred):
    # return binary_crossentropy(y_pred, y_true)

    #return 0.2 * (1 - jaccard_coef(y_true, y_pred)) + 0.8 * binary_crossentropy(y_pred, y_true)
    return -K.log(jaccard_coef(y_true, y_pred)) + K.binary_crossentropy(y_pred, y_true)

