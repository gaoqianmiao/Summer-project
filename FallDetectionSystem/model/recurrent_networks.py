
#---------------------------------------Creating attention model-----------------------------------------------------------------
from keras import backend as K
from keras.engine.topology import Layer
import numpy as np
from keras import regularizers, initializers, constraints

class AttentionWithContext(Layer):

    """

        Attention operation, with a context/query vector, for temporal data.

        Supports Masking.

        Follows the work of Yang et al. [https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf]

        "Hierarchical Attention Networks for Document Classification"

        by using a context vector to assist the attention

        # Input shape

            3D tensor with shape: `(samples, steps, features)`.

        # Output shape

            2D tensor with shape: `(samples, features)`.

        :param kwargs:

        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.

        The dimensions are inferred based on the output shape of the RNN.

        Example:

            model.add(LSTM(64, return_sequences=True))

            model.add(AttentionWithContext())

        """



    def __init__(self, init='glorot_uniform', kernel_regularizer=None, bias_regularizer=None, kernel_constraint=None, bias_constraint=None,  **kwargs):

        self.supports_masking = True

        self.init = initializers.get(init)

        self.kernel_initializer = initializers.get('glorot_uniform')



        self.kernel_regularizer = regularizers.get(kernel_regularizer)

        self.bias_regularizer = regularizers.get(bias_regularizer)



        self.kernel_constraint = constraints.get(kernel_constraint)

        self.bias_constraint = constraints.get(bias_constraint)



        super(AttentionWithContext, self).__init__(**kwargs)



    def build(self, input_shape):

        self.kernel = self.add_weight((input_shape[-1], 1),

                                 initializer=self.kernel_initializer,

                                 name='{}_W'.format(self.name),

                                 regularizer=self.kernel_regularizer,

                                 constraint=self.kernel_constraint)

        self.b = self.add_weight((input_shape[1],),

                                 initializer='zero',

                                 name='{}_b'.format(self.name),

                                 regularizer=self.bias_regularizer,

                                 constraint=self.bias_constraint)



        self.u = self.add_weight((input_shape[1],),

                                 initializer=self.kernel_initializer,

                                 name='{}_u'.format(self.name),

                                 regularizer=self.kernel_regularizer,

                                 constraint=self.kernel_constraint)

        self.built = True



    def compute_mask(self, input, mask):

        return None



    def call(self, x, mask=None):

        # (x, 40, 300) x (300, 1)

        multData =  K.dot(x, self.kernel) # (x, 40, 1)

        multData = K.squeeze(multData, -1) # (x, 40)

        multData = multData + self.b # (x, 40) + (40,)



        multData = K.tanh(multData) # (x, 40)



        multData = multData * self.u # (x, 40) * (40, 1) => (x, 1)

        multData = K.exp(multData) # (X, 1)



        # apply mask after the exp. will be re-normalized next

        if mask is not None:

            mask = K.cast(mask, K.floatx()) #(x, 40)

            multData = mask*multData #(x, 40) * (x, 40, )



        # in some cases especially in the early stages of training the sum may be almost zero

        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.

        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())

        multData /= K.cast(K.sum(multData, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        multData = K.expand_dims(multData)

        weighted_input = x * multData

        return K.sum(weighted_input, axis=1)





    def compute_output_shape(self, input_shape):

        return (input_shape[0], input_shape[-1],)
#---------------------------------------------------------LSTM----------------------------------------------------------------
from keras.layers import Dense, Activation, Dropout, Bidirectional
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.applications.vgg16 import VGG16
from keras.optimizers import SGD
from keras import backend as K
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
import os
import numpy as np

from utility.frame_extractor.vgg16_feature_extractor import extract_vgg16_features_live, \
    scan_and_extract_vgg16_features

BATCH_SIZE = 64
NUM_EPOCHS = 20
VERBOSE = 1
HIDDEN_UNITS = 512
MAX_ALLOWED_FRAMES = 20
EMBEDDING_SIZE = 100

K.set_image_dim_ordering('tf')


def generate_batch(x_samples, y_samples):
    num_batches = len(x_samples) // BATCH_SIZE

    while True:
        for batchIdx in range(0, num_batches):
            start = batchIdx * BATCH_SIZE
            end = (batchIdx + 1) * BATCH_SIZE
            yield np.array(x_samples[start:end]), y_samples[start:end]


class VGG16BidirectionalLSTMVideoClassifier(object):
    model_name = 'vgg16-bidirectional-lstm'

    def __init__(self):
        self.num_input_tokens = None
        self.nb_classes = None
        self.labels = None
        self.labels_idx2word = None
        self.model = None
        self.vgg16_model = None
        self.expected_frames = None
        self.vgg16_include_top = True
        self.config = None

    def create_model(self):
        model = Sequential()
        model.add(Bidirectional(LSTM(units=HIDDEN_UNITS, return_sequences=True),
                                input_shape=(self.expected_frames, self.num_input_tokens)))
        model.add(Bidirectional(LSTM(10)))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))

        model.add(Dense(self.nb_classes))

        model.add(Activation('softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

        return model

    @staticmethod
    def get_config_file_path(model_dir_path, vgg16_include_top=None):
        if vgg16_include_top is None:
            vgg16_include_top = True
        if vgg16_include_top:
            return model_dir_path + '/' + VGG16BidirectionalLSTMVideoClassifier.model_name + '-config.npy'
        else:
            return model_dir_path + '/' + VGG16BidirectionalLSTMVideoClassifier.model_name + '-hi-dim-config.npy'

    @staticmethod
    def get_weight_file_path(model_dir_path, vgg16_include_top=None):
        if vgg16_include_top is None:
            vgg16_include_top = True
        if vgg16_include_top:
            return model_dir_path + '/' + VGG16BidirectionalLSTMVideoClassifier.model_name + '-weights.h5'
        else:
            return model_dir_path + '/' + VGG16BidirectionalLSTMVideoClassifier.model_name + '-hi-dim-weights.h5'

    @staticmethod
    def get_architecture_file_path(model_dir_path, vgg16_include_top=None):
        if vgg16_include_top is None:
            vgg16_include_top = True
        if vgg16_include_top:
            return model_dir_path + '/' + VGG16BidirectionalLSTMVideoClassifier.model_name + '-architecture.json'
        else:
            return model_dir_path + '/' + VGG16BidirectionalLSTMVideoClassifier.model_name + '-hi-dim-architecture.json'

    def load_model(self, config_file_path, weight_file_path):
        if os.path.exists(config_file_path):
            print('loading configuration from ', config_file_path)
        else:
            raise ValueError('cannot locate config file {}'.format(config_file_path))

        config = np.load(config_file_path).item()
        self.num_input_tokens = config['num_input_tokens']
        self.nb_classes = config['nb_classes']
        self.labels = config['labels']
        self.expected_frames = config['expected_frames']
        self.vgg16_include_top = config['vgg16_include_top']
        self.labels_idx2word = dict([(idx, word) for word, idx in self.labels.items()])
        self.config = config

        self.model = self.create_model()
        if os.path.exists(weight_file_path):
            print('loading network weights from ', weight_file_path)
        else:
            raise ValueError('cannot local weight file {}'.format(weight_file_path))

        self.model.load_weights(weight_file_path)

        print('build vgg16 with pre-trained model')
        vgg16_model = VGG16(include_top=self.vgg16_include_top, weights='imagenet')
        vgg16_model.compile(optimizer=SGD(), loss='categorical_crossentropy', metrics=['accuracy'])
        self.vgg16_model = vgg16_model

    def predict(self, video_file_path):
        x = extract_vgg16_features_live(self.vgg16_model, video_file_path)
        frames = x.shape[0]
        if frames > self.expected_frames:
            x = x[0:self.expected_frames, :]
        elif frames < self.expected_frames:
            temp = np.zeros(shape=(self.expected_frames, x.shape[1]))
            temp[0:frames, :] = x
            x = temp
        predicted_class = np.argmax(self.model.predict(np.array([x]))[0])
        predicted_label = self.labels_idx2word[predicted_class]
        return predicted_label

    def fit(self, data_dir_path, model_dir_path, vgg16_include_top=True, data_set_name='UCF-101', test_size=0.3,
            random_state=42):

        self.vgg16_include_top = vgg16_include_top

        config_file_path = self.get_config_file_path(model_dir_path, vgg16_include_top)
        weight_file_path = self.get_weight_file_path(model_dir_path, vgg16_include_top)
        architecture_file_path = self.get_architecture_file_path(model_dir_path, vgg16_include_top)

        self.vgg16_model = VGG16(include_top=self.vgg16_include_top, weights='imagenet')
        self.vgg16_model.compile(optimizer=SGD(), loss='categorical_crossentropy', metrics=['accuracy'])

        feature_dir_name = data_set_name + '-VGG16-Features'
        if not vgg16_include_top:
            feature_dir_name = data_set_name + '-VGG16-HiDimFeatures'
        max_frames = 0
        self.labels = dict()
        x_samples, y_samples = scan_and_extract_vgg16_features(data_dir_path,
                                                               output_dir_path=feature_dir_name,
                                                               model=self.vgg16_model,
                                                               data_set_name=data_set_name)
        self.num_input_tokens = x_samples[0].shape[1]
        frames_list = []
        for x in x_samples:
            frames = x.shape[0]
            frames_list.append(frames)
            max_frames = max(frames, max_frames)
        self.expected_frames = int(np.mean(frames_list))
        print('max frames: ', max_frames)
        print('expected frames: ', self.expected_frames)
        for i in range(len(x_samples)):
            x = x_samples[i]
            frames = x.shape[0]
            if frames > self.expected_frames:
                x = x[0:self.expected_frames, :]
                x_samples[i] = x
            elif frames < self.expected_frames:
                temp = np.zeros(shape=(self.expected_frames, x.shape[1]))
                temp[0:frames, :] = x
                x_samples[i] = temp
        for y in y_samples:
            if y not in self.labels:
                self.labels[y] = len(self.labels)
        print(self.labels)
        for i in range(len(y_samples)):
            y_samples[i] = self.labels[y_samples[i]]

        self.nb_classes = len(self.labels)

        y_samples = np_utils.to_categorical(y_samples, self.nb_classes)

        config = dict()
        config['labels'] = self.labels
        config['nb_classes'] = self.nb_classes
        config['num_input_tokens'] = self.num_input_tokens
        config['expected_frames'] = self.expected_frames
        config['vgg16_include_top'] = self.vgg16_include_top

        self.config = config

        np.save(config_file_path, config)

        model = self.create_model()
        open(architecture_file_path, 'w').write(model.to_json())

        Xtrain, Xtest, Ytrain, Ytest = train_test_split(x_samples, y_samples, test_size=test_size,
                                                        random_state=random_state)

        train_gen = generate_batch(Xtrain, Ytrain)
        test_gen = generate_batch(Xtest, Ytest)

        train_num_batches = len(Xtrain) // BATCH_SIZE
        test_num_batches = len(Xtest) // BATCH_SIZE

        checkpoint = ModelCheckpoint(filepath=weight_file_path, save_best_only=True)
        history = model.fit_generator(generator=train_gen, steps_per_epoch=train_num_batches,
                                      epochs=NUM_EPOCHS,
                                      verbose=1, validation_data=test_gen, validation_steps=test_num_batches,
                                      callbacks=[checkpoint])
        model.save_weights(weight_file_path)

        return history


class VGG16LSTMVideoClassifier(object):
    model_name = 'vgg16-lstm'
    
    def __init__(self):
        self.num_input_tokens = None
        self.nb_classes = None
        self.labels = None
        self.labels_idx2word = None
        self.model = None
        self.vgg16_model = None
        self.expected_frames = None
        self.vgg16_include_top = None
        self.config = None
        
    @staticmethod
    def get_config_file_path(model_dir_path, vgg16_include_top=None):
        if vgg16_include_top is None:
            vgg16_include_top = True
        if vgg16_include_top:
            return model_dir_path + '/' + VGG16LSTMVideoClassifier.model_name + '-config.npy'
        else:
            return model_dir_path + '/' + VGG16LSTMVideoClassifier.model_name + '-hi-dim-config.npy'

    @staticmethod
    def get_weight_file_path(model_dir_path, vgg16_include_top=None):
        if vgg16_include_top is None:
            vgg16_include_top = True
        if vgg16_include_top:
            return model_dir_path + '/' + VGG16LSTMVideoClassifier.model_name + '-weights.h5'
        else:
            return model_dir_path + '/' + VGG16LSTMVideoClassifier.model_name + '-hi-dim-weights.h5'

    @staticmethod
    def get_architecture_file_path(model_dir_path, vgg16_include_top=None):
        if vgg16_include_top is None:
            vgg16_include_top = True
        if vgg16_include_top:
            return model_dir_path + '/' + VGG16LSTMVideoClassifier.model_name + '-architecture.json'
        else:
            return model_dir_path + '/' + VGG16LSTMVideoClassifier.model_name + '-hi-dim-architecture.json'

    def create_model(self):
        model = Sequential()

        model.add(LSTM(units=HIDDEN_UNITS, input_shape=(None, self.num_input_tokens), return_sequences=False, dropout=0.5))
        model.add(AttentionWithContext())
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.nb_classes))
        model.add(Activation('softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def load_model(self, config_file_path, weight_file_path):

        config = np.load(config_file_path).item()
        self.num_input_tokens = config['num_input_tokens']
        self.nb_classes = config['nb_classes']
        self.labels = config['labels']
        self.expected_frames = config['expected_frames']
        self.vgg16_include_top = config['vgg16_include_top']
        self.labels_idx2word = dict([(idx, word) for word, idx in self.labels.items()])

        self.model = self.create_model()
        self.model.load_weights(weight_file_path)

        vgg16_model = VGG16(include_top=self.vgg16_include_top, weights='imagenet')
        vgg16_model.compile(optimizer=SGD(), loss='categorical_crossentropy', metrics=['accuracy'])
        self.vgg16_model = vgg16_model

    def predict(self, video_file_path):
        x = extract_vgg16_features_live(self.vgg16_model, video_file_path)
        frames = x.shape[0]
        if frames > self.expected_frames:
            x = x[0:self.expected_frames, :]
        elif frames < self.expected_frames:
            temp = np.zeros(shape=(self.expected_frames, x.shape[1]))
            temp[0:frames, :] = x
            x = temp
        predicted_class = np.argmax(self.model.predict(np.array([x]))[0])
        predicted_label = self.labels_idx2word[predicted_class]
        return predicted_label

    def fit(self, data_dir_path, model_dir_path, vgg16_include_top=True, data_set_name='UCF-101', test_size=0.3, random_state=42):
        self.vgg16_include_top = vgg16_include_top

        config_file_path = self.get_config_file_path(model_dir_path, vgg16_include_top)
        weight_file_path = self.get_weight_file_path(model_dir_path, vgg16_include_top)
        architecture_file_path = self.get_architecture_file_path(model_dir_path, vgg16_include_top)

        vgg16_model = VGG16(include_top=self.vgg16_include_top, weights='imagenet')
        vgg16_model.compile(optimizer=SGD(), loss='categorical_crossentropy', metrics=['accuracy'])
        self.vgg16_model = vgg16_model

        feature_dir_name = data_set_name + '-VGG16-Features'
        if not vgg16_include_top:
            feature_dir_name = data_set_name + '-VGG16-HiDimFeatures'
        max_frames = 0
        self.labels = dict()
        x_samples, y_samples = scan_and_extract_vgg16_features(data_dir_path,
                                                               output_dir_path=feature_dir_name,
                                                               model=self.vgg16_model,
                                                               data_set_name=data_set_name)
        self.num_input_tokens = x_samples[0].shape[1]
        frames_list = []
        for x in x_samples:
            frames = x.shape[0]
            frames_list.append(frames)
            max_frames = max(frames, max_frames)
            self.expected_frames = int(np.mean(frames_list))
        print('max frames: ', max_frames)
        print('expected frames: ', self.expected_frames)
        for i in range(len(x_samples)):
            x = x_samples[i]
            frames = x.shape[0]
            print(x.shape)
            if frames > self.expected_frames:
                x = x[0:self.expected_frames, :]
                x_samples[i] = x
            elif frames < self.expected_frames:
                temp = np.zeros(shape=(self.expected_frames, x.shape[1]))
                temp[0:frames, :] = x
                x_samples[i] = temp
        for y in y_samples:
            if y not in self.labels:
                self.labels[y] = len(self.labels)
        print(self.labels)
        for i in range(len(y_samples)):
            y_samples[i] = self.labels[y_samples[i]]

        self.nb_classes = len(self.labels)

        y_samples = np_utils.to_categorical(y_samples, self.nb_classes)

        config = dict()
        config['labels'] = self.labels
        config['nb_classes'] = self.nb_classes
        config['num_input_tokens'] = self.num_input_tokens
        config['expected_frames'] = self.expected_frames
        config['vgg16_include_top'] = self.vgg16_include_top
        self.config = config

        np.save(config_file_path, config)

        model = self.create_model()
        open(architecture_file_path, 'w').write(model.to_json())

        Xtrain, Xtest, Ytrain, Ytest = train_test_split(x_samples, y_samples, test_size=test_size,
                                                        random_state=random_state)

        train_gen = generate_batch(Xtrain, Ytrain)
        test_gen = generate_batch(Xtest, Ytest)

        train_num_batches = len(Xtrain) // BATCH_SIZE
        test_num_batches = len(Xtest) // BATCH_SIZE

        checkpoint = ModelCheckpoint(filepath=weight_file_path, save_best_only=True)
        history = model.fit_generator(generator=train_gen, steps_per_epoch=train_num_batches,
                                      epochs=NUM_EPOCHS,
                                      verbose=1, validation_data=test_gen, validation_steps=test_num_batches,
                                      callbacks=[checkpoint])
        model.save_weights(weight_file_path)

        return history
