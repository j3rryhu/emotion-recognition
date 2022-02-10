import pandas as pd
import numpy as np
import scipy.misc as sm
import os

emotions = {
    '0': 'anger',
    '1': 'disgust',
    '2': 'fear',
    '3': 'happy',
    '4': 'sad',
    '5': 'surprised',
    '6': 'normal',
}


def createDir(dir):
    if os.path.exists(dir) is False:
        os.makedirs(dir)


def saveImageFromFer2013(file):
    faces_data = pd.read_csv(file)
    imageCount = 0
    for index in range(len(faces_data)):
        emotion_data = faces_data.loc[index][0]
        image_data = faces_data.loc[index][1]
        usage_data = faces_data.loc[index][2]
        data_array = list(map(float, image_data.split()))
        data_array = np.asarray(data_array)
        image = data_array.reshape(48, 48)

        dirName = os.path.join('dataset', usage_data)
        emotionName = emotions[str(emotion_data)]
        imagePath = os.path.join(dirName, emotionName)

        createDir(dirName)
        createDir(imagePath)

        imageName = os.path.join(imagePath, str(index) + '.jpg')

        sm.toimage(image).save(imageName)
        imageCount = index
    print('Data saving is done, there are {} images'.format(imageCount))


if __name__ == '__main__':
    saveImageFromFer2013('./fer2013/fer2013.csv')