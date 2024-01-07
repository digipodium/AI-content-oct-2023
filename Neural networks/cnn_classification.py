#%%
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import datasets ,layers ,models
#%%
(train_images,train_labels ), (test_images, test_labels) =  datasets.cifar10.load_data()
print(f"train labels {train_labels.shape}")
print(f"train images {train_images.shape}")
#%%
class_names= ["airplane",'automobile','bird','cat','deer','dog','frog','horse','ship','truck']
#%%
train_labels[5000][0] # 1st image label index
class_names[train_labels[5000][0]] # actual label
x = np.array([
              [0,0,0,0,0,0,0],
              [0,1,0,0,0,1,0],
              [0,0,0,1,0,0,0],
              [0,1,0,0,0,1,0],
              [0,0,1,1,1,0,0],
              [0,0,0,0,0,0,0],
])
plt.imshow(x, cmap='gray')
#%%
plt.imshow(train_images[5000])
#%%
plt.figure(figsize=(10,10))
for i in range(1, 26):
    plt.subplot(5,5,i)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i])
    plt.xlabel(class_names[train_labels[i][0]])
plt.show()
#%%
"""# CNN architecture"""

model = models.Sequential()
model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(32,32,3)))
model.add(layers.MaxPooling2D())
model.add(layers.Dropout(.2))
model.add(layers.Conv2D(16,(3,3),activation='relu'))
model.add(layers.MaxPooling2D())
model.add(layers.Dropout(.2))
model.add(layers.Conv2D(16,(3,3),activation='relu'))
model.add(layers.MaxPooling2D())
model.add(layers.Dropout(.2))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))
model.summary()
#%%

def vgg17_model(input_shape=(32, 32, 3), num_classes=1000):
    model = tf.keras.Sequential([
        # Block 1
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2)),

        # Block 2
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2)),

        # Block 3
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2)),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(4096, activation='relu'),
        tf.keras.layers.Dense(4096, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    return model
#%%
# Create an instance of the VGG-17 model
vgg17 = vgg17_model(num_classes=10)
vgg17.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])
#%%
# Display model summary
vgg17.summary()
#%%
# train_logs = 'logs'
# tensorflow_callback = tf.keras.callbacks.TensorBoard(train_logs,histogram_freq=1)

vgg17.fit(train_images,
          train_labels,
          epochs= 5,
          validation_data = (test_images, test_labels),
)
#%%
test_loss, test_acc = vgg17.evaluate(test_images, test_labels)
vgg17.save('cifar_10_Model_60_percent.keras')
#%%
vgg17.predict(np.array([test_images[0]]))
#%%
result = vgg17.predict(np.array([test_images[0]]))
result
#%%
result[0].argmax()
#%%
class_names[result[0].argmax()]
#%%
out= class_names[result[0].argmax()]
#%%
plt.figure(figsize=(1,1))
plt.xticks([])
plt.yticks([])
plt.imshow(test_images[0])
plt.title(out)
plt.show()
#%%
plt.figure(figsize=(10,10))
for i in range(1, 26):
    result = model.predict(np.array([test_images[i]]), verbose=0)
    prediction = result[0].argmax()
    plt.subplot(5,5,i)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(test_images[i])
    orig = class_names[test_labels[i][0]]
    pred = class_names[prediction]
    label = f'{orig} is {pred}'
    plt.xlabel(label)
plt.show()
#%%

img = Image.open('ship.jpg')
img.size
#%%
test_img = img.resize((32,32),resample=Image.LANCZOS)
test_img_array = np.array(test_img)
#%%
np.array([test_img_array]).shape
#%%
result = model.predict(np.array([test_img_array]))
plt.figure(figsize=(1,1))
plt.xticks([])
plt.yticks([])
plt.imshow(test_img_array)
plt.title(class_names[result[0].argmax()])
plt.show()
#%%
saved_model = tf.keras.models.load_model('/content/cifar_10_Model_60_percent.keras')
result = saved_model.predict(np.array([test_img_array]))
plt.figure(figsize=(1,1))
plt.xticks([])
plt.yticks([])
plt.imshow(test_img_array)
plt.title(class_names[result[0].argmax()])
plt.show()
#%%