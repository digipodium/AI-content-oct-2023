from keras.models import load_model  # TensorFlow is required for Keras to work
import cv2  # Install opencv-python
import numpy as np

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("keras_Model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

# CAMERA can be 0 or 1 based on default camera of your computer
camera = cv2.VideoCapture(0)

colors = {
    'red':(0,0,255),
    'blue': (255,0,0),
    'green': (0,255,0)
}
font = cv2.FONT_HERSHEY_PLAIN

while True:
    # Grab the webcamera's image.
    ret, image = camera.read()
    full_image = image

    # Resize the raw image into (224-height,224-width) pixels
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)


    # Make the image a numpy array and reshape it to the models input shape.
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

    # Normalize the image array
    image = (image / 127.5) - 1

    # Predicts the model
    prediction = model.predict(image)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]
    
    # result variables
    result = class_name[2:].strip()
    score = str(np.round(confidence_score * 100))[:-2]
   
    fc = (255,255,255)
    match result:
        case 'Mobile': fc = colors.get('red') 
        case 'File': fc = colors.get('green')
        case _: fc: colors.get('blue')
    # add text on video
    cv2.putText(full_image, f'Class:{result}', (10,30), font , 2, fc, 2)
    cv2.putText(full_image, f'{score}%', (10, 60), font, 1.5, fc)
    
    # Show the image in a window
    cv2.imshow("Webcam Image", full_image)
    
    # Listen to the keyboard for presses.
    keyboard_input = cv2.waitKey(1)

    # 27 is the ASCII for the esc key on your keyboard.
    if keyboard_input == 27:
        break

camera.release()
cv2.destroyAllWindows()

