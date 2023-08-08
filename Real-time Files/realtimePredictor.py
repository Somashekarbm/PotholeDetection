import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import tensorflow as tf
from keras.layers import Flatten
from keras.models import Sequential, Model, load_model
from keras.callbacks import EarlyStopping, Callback
from keras.layers import (
    Dense,
    Dropout,
    Activation,
    Flatten,
    Lambda,
    ELU,
    GlobalAveragePooling2D,
)
from keras.layers import Flatten, MaxPooling2D, Conv2D
from sklearn.utils import shuffle
from keras.utils import to_categorical
import time, cv2, glob
from sklearn.metrics import accuracy_score

global loadedModel
size = 300


# Resize the frame to required dimensions and predict
def predict_pothole(currentFrame):
    currentFrame = cv2.resize(currentFrame, (size, size))
    print("Resized frame shape:", currentFrame.shape)  # Debug print
    currentFrame = currentFrame.reshape(1, size, size, 1).astype("float")
    currentFrame = currentFrame / 255
    prob = loadedModel.predict(currentFrame)
    max_prob = max(prob[0])
    predicted_class = np.argmax(prob[0])  # Get the predicted class index
    if max_prob > 0.90:
        return predicted_class, max_prob
    return "none", 0


if __name__ == "__main__":
    loadedModel = load_model(
        r"C:\Users\Somashekar\OneDrive\Desktop\ideation\pothole-detection-system-using-convolution-neural-networks\Real-time Files\full_model.h5"
    )

    input_mode = input("Choose input mode (1 for image, 2 for video): ")

    if input_mode == "1":
        image_path = input("Enter image file path:")
        input_frame = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if input_frame is None:
            print("Error: Unable to load image from the provided path.")
        else:
            pothole, prob = predict_pothole(input_frame)
            print("Prediction:", pothole, "with probability:", prob)
    elif input_mode == "2":
        # ... (rest of the video feed code)

        camera = cv2.VideoCapture(0)
        show_pred = False
        while True:
            (grabbed, frame) = camera.read()
            frame = cv2.flip(frame, 1)

            clone = frame.copy()

            grayClone = cv2.cvtColor(clone, cv2.COLOR_BGR2GRAY)

            pothole, prob = predict_pothole(grayClone)

            keypress_toshow = cv2.waitKey(1)

            if keypress_toshow == ord("e"):
                show_pred = not show_pred

            if show_pred:
                cv2.putText(
                    clone,
                    f"Prediction: {pothole} {prob*100:.2f}%",
                    (30, 30),
                    cv2.FONT_HERSHEY_DUPLEX,
                    1,
                    (0, 255, 0),
                    1,
                )

            cv2.imshow("GrayClone", grayClone)
            cv2.imshow("Video Feed", clone)

            keypress = cv2.waitKey(1) & 0xFF
            if keypress == ord("q"):
                break

        camera.release()
        cv2.destroyAllWindows()
        # for testing accuracy
