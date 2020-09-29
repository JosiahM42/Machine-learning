import cv2
import tensorflow as tf

CATAGORIES = ["happy", "sad"]


def preparation(emotion_path):
    image_size = 95
    # converts the image into greyscale
    image_array = cv2.imread(emotion_path, cv2.IMREAD_GRAYSCALE)
    # resizes the new grey image to 95 by 95
    new_array = cv2.resize(image_array, (image_size, image_size))
    # returns the image with the specific shaping that tensorflow requires
    return new_array.reshape(-1, image_size, image_size, 1)


emotion_model = tf.keras.models.load_model("emotions-CNN.model")

prediction = emotion_model.predict([preparation('happy.jpg')])
print(prediction)
print(CATAGORIES[int(prediction[0][0])])

prediction2 = emotion_model.predict([preparation('sad.jpg')])
print(prediction2)
print(CATAGORIES[int(prediction2[0][0])])
