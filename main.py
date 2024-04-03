from keras.applications import VGG16
from keras.layers import Dense, Flatten
from keras.models import Model
from keras.src.legacy.preprocessing.image import ImageDataGenerator

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
base_model.trainable = False

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'dataset',
    target_size=(150, 150),
    batch_size=20,
    class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
    'dataset',
    target_size=(150, 150),
    batch_size=20,
    class_mode='categorical')

x = Flatten()(base_model.output)
x = Dense(128, activation='relu')(x)
output = Dense(4, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=output)

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

steps_per_epoch = train_generator.samples // train_generator.batch_size
validation_steps = test_generator.samples // test_generator.batch_size

model.fit(train_generator,
          epochs=10,
          steps_per_epoch=steps_per_epoch,
          validation_data=test_generator,
          validation_steps=validation_steps)

loss, accuracy = model.evaluate(test_generator)
print(f'Test loss: {loss} \n Test accuracy: {accuracy}')


