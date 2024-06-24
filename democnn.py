import demo
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Image data generators
train_datagen = ImageDataGenerator(rescale=1./255)
valid_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(demo.train_dir,
                                                    target_size=(224, 224),
                                                    batch_size=32,
                                                    class_mode='binary')
validation_generator = valid_datagen.flow_from_directory(demo.valid_dir,
                                                         target_size=(224, 224),
                                                         batch_size=32,
                                                         class_mode='binary')
test_generator = test_datagen.flow_from_directory(demo.test_dir,
                                                  target_size=(224, 224),
                                                  batch_size=32,
                                                  class_mode='binary')

# Build the CNN model
def build_cnn_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    return model

cnn_model = build_cnn_model()
cnn_model.summary()

# Train the model
cnn_history = cnn_model.fit(train_generator,
                            epochs=10,
                            validation_data=validation_generator)

# Evaluate the model
test_loss, test_accuracy = cnn_model.evaluate(test_generator)
print(f'Test accuracy: {test_accuracy}')

# Save the model
cnn_model.save('cnn_deepfake_model.h5')
