import cv2
import numpy as np
from keras.models import load_model

global loadedModel
size = 300


# Resize the frame to required dimensions and predict
def predict_pothole(currentFrame):
    currentFrame = cv2.resize(currentFrame, (size, size))
    currentFrame = currentFrame.reshape(1, size, size, 1).astype("float") / 255.0
    prob = loadedModel.predict(currentFrame)
    max_prob = np.max(prob)
    predicted_class = np.argmax(prob)
    return predicted_class, max_prob


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
            if pothole == 1:
                print("Potholes detected with probability:", prob)
            else:
                print("No potholes detected with probability", prob)
    elif input_mode == "2":
        video_option = input(
            "Choose video input option (1 for webcam feed, 2 for video path): "
        )

        if video_option == "1":
            camera = cv2.VideoCapture(0)
        elif video_option == "2":
            video_source = input("Enter video file path: ")
            camera = cv2.VideoCapture(video_source)
            if not camera.isOpened():
                print("Error: Unable to open video file.")
                exit()
        show_pred = False
        while True:
            (grabbed, frame) = camera.read()
            if not grabbed:
                break

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
