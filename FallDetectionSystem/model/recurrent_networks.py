"""
A keras attention layer that wraps RNN layers.

Based on tensorflows [attention_decoder](https://github.com/tensorflow/tensorflow/blob/c8a45a8e236776bed1d14fd71f3b6755bd63cc58/tensorflow/python/ops/seq2seq.py#L506)
and [Grammar as a Foreign Language](https://arxiv.org/abs/1412.7449).

date: 20161101
author: wassname
url:
"""
from __future__ import absolute_import

from keras import backend as K
from keras.engine import InputSpec
from keras.layers import LSTM, activations, Wrapper, Recurrent

class Attention(Wrapper):
    """
    This wrapper will provide an attention layer to a recurrent layer.

    # Arguments:
        layer: `Recurrent` instance with consume_less='gpu' or 'mem'

    # Examples:

    ```python
    model = Sequential()
    model.add(LSTM(10, return_sequences=True), batch_input_shape=(4, 5, 10))
    model.add(TFAttentionRNNWrapper(LSTM(10, return_sequences=True, consume_less='gpu')))
    model.add(Dense(5))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    ```

    # References
    - [Grammar as a Foreign Language](https://arxiv.org/abs/1412.7449)


    """
    def __init__(self, layer, **kwargs):
        assert isinstance(layer, Recurrent)
        
        if layer.get_config()['consume_less']=='cpu':
            raise Exception("AttentionLSTMWrapper doesn't support RNN's with consume_less='cpu'")
        
        self.supports_masking = True
        super(Attention, self).__init__(layer, **kwargs)

    def build(self, input_shape):
        assert len(input_shape) >= 3
        self.input_spec = [InputSpec(shape=input_shape)]
        nb_samples, nb_time, input_dim = input_shape

        if not self.layer.built:
            self.layer.build(input_shape)
            self.layer.built = True

        super(Attention, self).build()

        self.W1 = self.layer.init((input_dim, input_dim, 1, 1), name='{}_W1'.format(self.name))
        self.W2 = self.layer.init((self.layer.output_dim, input_dim), name='{}_W2'.format(self.name))
        self.b2 = K.zeros((input_dim,), name='{}_b2'.format(self.name))
        self.W3 = self.layer.init((input_dim*2, input_dim), name='{}_W3'.format(self.name))
        self.b3 = K.zeros((input_dim,), name='{}_b3'.format(self.name))
        self.V = self.layer.init((input_dim,), name='{}_V'.format(self.name))

        self.trainable_weights = [self.W1, self.W2, self.W3, self.V, self.b2, self.b3]

    def get_output_shape_for(self, input_shape):
        return self.layer.get_output_shape_for(input_shape)

    def step(self, x, states):
        # This is based on [tensorflows implementation](https://github.com/tensorflow/tensorflow/blob/c8a45a8e236776bed1d14fd71f3b6755bd63cc58/tensorflow/python/ops/seq2seq.py#L506).
        # First, we calculate new attention masks:
        #   attn = softmax(V^T * tanh(W2 * X +b2 + W1 * h))
        # and we make the input as a concatenation of the input and weighted inputs which is then
        # transformed back to the shape x of using W3
        #   x = W3*(x+X*attn)+b3
        # Then, we run the cell on a combination of the input and previous attention masks:
        #   h, state = cell(x, h).

        nb_samples, nb_time, input_dim = self.input_spec[0].shape
        h = states[0]
        X = states[-1]
        xW1 = states[-2]

        Xr = K.reshape(X,(-1,nb_time,1,input_dim))
        hW2 = K.dot(h,self.W2)+self.b2
        hW2 = K.reshape(hW2,(-1,1,1,input_dim))
        u = K.tanh(xW1+hW2)
        a = K.sum(self.V*u,[2,3])
        a = K.softmax(a)
        a = K.reshape(a,(-1, nb_time, 1, 1))

        # Weight attention vector by attention
        Xa = K.sum(a*Xr,[1,2])
        Xa = K.reshape(Xa,(-1,input_dim))

        # Merge input and attention weighted inputs into one vector of the right size.
        x = K.dot(K.concatenate([x,Xa],1),self.W3)+self.b3

        h, new_states = self.layer.step(x, states)
        return h, new_states

    def get_constants(self, x):
        constants = self.layer.get_constants(x)

        # Calculate K.dot(x, W2) only once per sequence by making it a constant
        nb_samples, nb_time, input_dim = self.input_spec[0].shape
        Xr = K.reshape(x,(-1,nb_time,input_dim,1))
        Xrt = K.permute_dimensions(Xr, (0, 2, 1, 3))
        xW1t = K.conv2d(Xrt,self.W1,border_mode='same')
        xW1 = K.permute_dimensions(xW1t, (0, 2, 3, 1))
        constants.append(xW1)

        # we need to supply the full sequence of inputs to step (as the attention_vector)
        constants.append(x)

        return constants

    def call(self, x, mask=None):
        # input shape: (nb_samples, time (padded with zeros), input_dim)
        input_shape = self.input_spec[0].shape
        if K._BACKEND == 'tensorflow':
            if not input_shape[1]:
                raise Exception('When using TensorFlow, you should define '
                                'explicitly the number of timesteps of '
                                'your sequences.\n'
                                'If your first layer is an Embedding, '
                                'make sure to pass it an "input_length" '
                                'argument. Otherwise, make sure '
                                'the first layer has '
                                'an "input_shape" or "batch_input_shape" '
                                'argument, including the time axis. '
                                'Found input shape at layer ' + self.name +
                                ': ' + str(input_shape))

        if self.layer.stateful:
            initial_states = self.layer.states
        else:
            initial_states = self.layer.get_initial_states(x)
        constants = self.get_constants(x)
        preprocessed_input = self.layer.preprocess_input(x)


        last_output, outputs, states = K.rnn(self.step, preprocessed_input,
                                             initial_states,
                                             go_backwards=self.layer.go_backwards,
                                             mask=mask,
                                             constants=constants,
                                             unroll=self.layer.unroll,
                                             input_length=input_shape[1])
        if self.layer.stateful:
            self.updates = []
            for i in range(len(states)):
                self.updates.append((self.layer.states[i], states[i]))

        if self.layer.return_sequences:
            return outputs
        else:
            return last_output




# this is a copy of tensorflow with simplified matrix algebra, need to check I didn't make a mistkae
class SimplifiedAttention(Wrapper):
    def __init__(self, layer, attn_activation='tanh', **kwargs):
        assert isinstance(layer, Recurrent)
        if not layer.return_sequences:
            raise Exception("AttentionLSTMWrapper doesn't support RNN's with return_sequences=False")

        self.supports_masking = True
        super(SimplifiedAttention, self).__init__(layer, **kwargs)

    def build(self, input_shape):
        assert len(input_shape) >= 3
        self.input_spec = [InputSpec(shape=input_shape)]
        nb_samples, nb_time, input_dim = input_shape

        if not self.layer.built:
            self.layer.build(input_shape)
            self.layer.built = True

        super(SimplifiedAttention, self).build()

#         self.W1 = self.layer.init((input_dim, input_dim, 1, 1), name='{}_W1'.format(self.name))
        self.W1 = self.layer.init((input_dim, input_dim), name='{}_W1'.format(self.name))
#         self.W2 = self.layer.init((input_dim,nb_time), name='{}_W2'.format(self.name))
        self.W2 = self.layer.init((nb_time,input_dim,input_dim), name='{}_W2'.format(self.name))
        self.b2 = K.zeros((input_dim,), name='{}_b2'.format(self.name))
        self.W3 = self.layer.init((input_dim*2, input_dim), name='{}_W3'.format(self.name))
        self.b3 = K.zeros((input_dim,), name='{}_b3'.format(self.name))
        self.V = self.layer.init((input_dim,), name='{}_V'.format(self.name))

        self.trainable_weights = [self.W1, self.W2, self.W3, self.V, self.b2, self.b3]

    def get_output_shape_for(self, input_shape):
        return self.layer.get_output_shape_for(input_shape)

    def step(self, x, states):

        # First, we calculate new attention masks:
        #   attn = softmax(V^T * tanh(W2 * inputs + W1 * prev_h))
        # and then weight the previous state by the attention
        #   prev_h = prev_h * attn
        # and we make the input as a concatenation of the input and weighted inputs which is then
        # transformed back to the shape x of using W3
        #   x = W3*(x+X*attn)+b3
        # Then, we run the cell on a combination of the input and previous attention masks:
        #   h, state = cell(x, h).

        nb_samples, nb_time, input_dim = self.input_spec[0].shape
        h,c,B_U,B_W,xW2,X = states

#         # as in tensorflow
#         Xr = K.reshape(X,(-1,nb_time,input_dim,1))
#         Xrt = K.permute_dimensions(Xr, (0, 2, 1, 3))
#         xW1t = K.conv2d(Xrt,self.W1,border_mode='same') # could be cached
#         xW1 = K.permute_dimensions(xW1t, (0, 2, 1, 3))

        # or (input_dim,input_dim)x(nb_samples, nb_time, input_dim)=>(nb_samples, nb_time, input_dim)
        # same value once reshaped, need to take away the extra dims
        xW1 = K.dot(X,self.W1)

        # assert hW1.shape == Xr.shape
        hW2 = K.dot(h,self.W2)+self.b2
#         xW2 = K.reshape(xW2,(-1,1,input_dim,1))
        u = K.tanh(xW1+hW2)
        a = K.sum(self.V*u,-1)
        # assert a.shape==(nb_samples,nb_time)
        a = K.softmax(a)
        a = K.reshape(a,(-1, nb_time, 1))
        Xa = K.sum(a*X,1)
        Xa = K.reshape(Xa,(-1,input_dim))

        # Merge input and previous attentions into one vector of the right size.
        # TODO, deal with the consume_less='cpu' flag which reshapes x
        x = K.dot(K.concatenate([x,Xa],1),self.W3)+self.b3
        # assert x.shape == (nb_samples,input_dim)




        h, new_states = self.layer.step(x, [h,c,B_U,B_W])
#         new_states.append(a)
#         Tracer()()
        return h, new_states

    def get_constants(self, x):
        constants = self.layer.get_constants(x)
        # Calculate K.dot(x, W2) only once per sequence by making it a constant
#         # as in tensorflow
#         self.W1 = self.layer.init((input_dim, input_dim, 1, 1), name='{}_W1'.format(self.name))
#         Xr = K.reshape(x,(-1,nb_time,input_dim,1))
#         Xrt = K.permute_dimensions(Xr, (0, 2, 1, 3))
#         xW1t = K.conv2d(Xrt,self.W1,border_mode='same') # could be cached
#         xW1 = K.permute_dimensions(xW1t, (0, 2, 1, 3))

        # or just
#         self.W1 = self.layer.init((input_dim, input_dim), name='{}_W1'.format(self.name))
        xW1 = K.dot(x,self.W1)

        constants.append(xW1)
        # the need to provide X to the step function too so it can be weighted to produce the inputs
        constants.append(x)
        return constants

    def call(self, x, mask=None):
        # input shape: (nb_samples, time (padded with zeros), input_dim)
        input_shape = self.input_spec[0].shape
        if K._BACKEND == 'tensorflow':
            if not input_shape[1]:
                raise Exception('When using TensorFlow, you should define '
                                'explicitly the number of timesteps of '
                                'your sequences.\n'
                                'If your first layer is an Embedding, '
                                'make sure to pass it an "input_length" '
                                'argument. Otherwise, make sure '
                                'the first layer has '
                                'an "input_shape" or "batch_input_shape" '
                                'argument, including the time axis. '
                                'Found input shape at layer ' + self.name +
                                ': ' + str(input_shape))

        if self.layer.stateful:
            initial_states = self.layer.states
        else:
            initial_states = self.layer.get_initial_states(x)#+[K.ones((input_shape[0],input_shape[1]))]
        constants = self.get_constants(x)
        preprocessed_input = self.layer.preprocess_input(x)

        last_output, outputs, states = K.rnn(self.step, preprocessed_input,
                                             initial_states,
                                             go_backwards=self.layer.go_backwards,
                                             mask=mask,
                                             constants=constants,
                                             unroll=self.layer.unroll,
                                             input_length=input_shape[1])
        if self.layer.stateful:
            self.updates = []
            for i in range(len(states)):
                self.updates.append((self.layer.states[i], states[i]))

        if self.layer.return_sequences:
            return outputs
        else:
            return last_output


#---------------------------------------------------------LSTM----------------------------------------------------------------
from keras.layers import Dense, Activation, Dropout, Bidirectional, Flatten, RepeatVector, Permute, Lambda, merge, TimeDistributed, recurrent
from keras.models import Model
import keras
from keras import Input
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

from utility.vgg16_feature_extractor import extract_vgg16_features_live, \
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
        model.add(Dense(512, activation='relu',input_shape=(None, 1, self.num_input_tokens)))
        model.add(Attention(LSTM(units=HIDDEN_UNITS, return_sequences=False, consume_less='mem')))
        model.add(Dropout(0.5))
        model.add(Dense(self.nb_classes))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
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
