from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from models.cnn import cnn
from utils.datasets import DatasetLoader
from keras.preprocessing.image import ImageDataGenerator


# parameters
batch_size = 128
epochs = 10000
input_shape = (64, 64, 1)
validation_split = .2
verbose = 1
num_classes = 7
patience = 50
base_path = './trained_models/'


# data generator
data_generator = ImageDataGenerator(
                        featurewise_center=False,
                        featurewise_std_normalization=False,
                        rotation_range=10,
                        width_shift_range=0.1,
                        height_shift_range=0.1,
                        zoom_range=.1,
                        horizontal_flip=True)

model = cnn(input_shape, num_classes)

model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()


dataset_name = 'fer2013'
log_file_path = base_path + dataset_name + 'training.log'
csv_logger = CSVLogger(log_file_path, append=False)
early_stop = EarlyStopping('val_loss', patience=patience)
reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1,
                                  patience=int(patience/4), verbose=1)
trained_models_path = base_path + dataset_name + 'cnn'
model_names = trained_models_path + '.{epoch:02d}-{val_acc:.2f}.hdf5'
model_checkpoint = ModelCheckpoint(model_names, 'val_loss', verbose=1,
                                                save_best_only=True)
callbacks = [model_checkpoint, csv_logger, early_stop, reduce_lr]

data_loader = DatasetLoader(dataset_path='./dataset', image_size=input_shape[:2])
train_data = data_loader.load(usage='train')
val_data = data_loader.load(usage='test')
faces, labels = train_data
num_samples, num_classes = labels.shape
datagen = ImageDataGenerator(featurewise_center=True,
                             featurewise_std_normalization=True,
                             rotation_range=20,
                             width_shift_range=0.2,
                             height_shift_range=0.2,
                             horizontal_flip=True
                             )

model.fit_generator(datagen.flow(faces, labels, batch_size),
                    steps_per_epoch=len(faces) / batch_size,
                    epochs=epochs, verbose=1, callbacks=callbacks,
                    validation_data=val_data)
