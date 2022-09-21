
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

tf.__version__

# Preprocessing the Training set
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range= 0.2,
        zoom_range=0.2,
        horizontal_flip= True
)
training_set2 = train_datagen.flow_from_directory(
        'dataset/training_set2', 
        target_size=(64,64),
        batch_size=32,
        class_mode= 'binary'
)

# Preprocessing the Test set
test_datagen = ImageDataGenerator(rescale=1./255)
test_set2= test_datagen.flow_from_directory(
    'dataset/test_set2', target_size=(64,64),
    batch_size= 32,
    class_mode= 'binary'
)


# Building the CNN
cnn= tf.keras.models.Sequential()

# Step 1 - Convolution
cnn.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3,activation='relu', input_shape=(64,64,3) ))
# Step 2 - Pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Adding a second convolutional layer
cnn.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3,activation='relu' ))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Step 3 - Flattening
cnn.add(tf.keras.layers.Flatten())

# Step 4 - Full Connection
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

# Step 5 - Output Layer
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))



#Training the CNN
# Compiling the CNN
cnn.compile(optimizer= 'adam',  loss = 'binary_crossentropy', metrics=['accuracy'])
# Training the CNN on the Training set and evaluating it on the Test set
cnn.fit(x=training_set2, validation_data=test_set2, epochs=10)


#Making a single prediction
import numpy as np
from keras.preprocessing import image
test_image = tf.keras.preprocessing.image.load_img('dataset/single_prediction2/happy_or_sad2.jpg',target_size=(64,64))
test_image= tf.keras.preprocessing.image.img_to_array(test_image)
test_image= np.expand_dims(test_image, axis=0)
result = cnn.predict(test_image/255.0)
training_set2.class_indices
if result[0][0]> 0.5:
    prediction = 'happy'
else :
    prediction='sad'

print(prediction)

