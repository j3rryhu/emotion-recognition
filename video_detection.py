import numpy as np
import cv2
from keras.models import load_model
from statistics import mode

def start_recognition():
    # parameters for loading data and images
    face_path = './OpenCV_xml/haarcascade_frontalface_default.xml'
    emotion_model_path = ' ' # fill in the saved model name
    emotion_labels = {
                        '0': 'anger',
                        '1': 'disgust',
                        '2': 'fear',
                        '3': 'happy',
                        '4': 'sad',
                        '5': 'surprised',
                        '6': 'normal',
                    }

    # hyper-parameters for bounding boxes shape
    frame_window = 10

    # loading models
    face_detection = cv2.CascadeClassifier(face_path)
    emotion_classifier = load_model(emotion_model_path, compile=False)

    # getting input model shapes for inference
    emotion_target_size = emotion_classifier.input_shape[1:3]

    # starting lists for calculating modes
    emotion_window = []

    # starting video streaming
    cv2.namedWindow('window_frame')
    video_capture = cv2.VideoCapture(0)
    while True:
        bgr_image = video_capture.read()[1]
        gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        faces = face_detection.detectMultiScale(gray_image, 1.3, 5)

        for coordinates in faces:
            x1, y1, w, h = coordinates
            x1 -= 20
            y1 -= 40
            x2 = x1+w+20
            y2 = y1+h+40
            gray_face = gray_image[y1:y2, x1:x2]
            try:
                gray_face = cv2.resize(gray_face, (emotion_target_size))
            except:
                continue

            gray_face = np.expand_dims(gray_face, 0)
            gray_face = np.expand_dims(gray_face, -1)
            emotion_prediction = emotion_classifier.predict(gray_face)
            emotion_probability = np.max(emotion_prediction)
            emotion_label_arg = np.argmax(emotion_prediction)
            emotion_text = emotion_labels[emotion_label_arg]
            emotion_window.append(emotion_text)

            if len(emotion_window) > frame_window:
                emotion_window.pop(0)
            try:
                emotion_mode = mode(emotion_window)
            except:
                continue

            if emotion_text == 'angry':
                color = emotion_probability * np.asarray((255, 0, 0))
            elif emotion_text == 'sad':
                color = emotion_probability * np.asarray((0, 0, 255))
            elif emotion_text == 'happy':
                color = emotion_probability * np.asarray((255, 255, 0))
            elif emotion_text == 'surprise':
                color = emotion_probability * np.asarray((0, 255, 255))
            else:
                color = emotion_probability * np.asarray((0, 255, 0))

            color = color.astype(int)
            color = color.tolist()

            cv2.rectangle(rgb_image, (x1, y1), (x2, y2), color)
            cv2.putText(rgb_image, emotion_text, (x1, y1-40), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=2,
                        color=color, thickness=1)

        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        cv2.imshow('window_frame', bgr_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video_capture.release()


if __name__ == '__main__':
    start_recognition()