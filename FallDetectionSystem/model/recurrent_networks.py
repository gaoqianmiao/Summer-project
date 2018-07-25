class AttentionLSTM(LSTM):

    """LSTM with attention mechanism



    This is an LSTM incorporating an attention mechanism into its hidden states.

    Currently, the context vector calculated from the attended vector is fed

    into the model's internal states, closely following the model by Xu et al.

    (2016, Sec. 3.1.2), using a soft attention model following

    Bahdanau et al. (2014).



    The layer expects two inputs instead of the usual one:

        1. the "normal" layer input; and

        2. a 3D vector to attend.



    Args:

        attn_activation: Activation function for attentional components

        attn_init: Initialization function for attention weights

        output_alpha (boolean): If true, outputs the alpha values, i.e.,

            what parts of the attention vector the layer attends to at each

            timestep.



    References:

        * Bahdanau, Cho & Bengio (2014), "Neural Machine Translation by Jointly

          Learning to Align and Translate", <https://arxiv.org/pdf/1409.0473.pdf>

        * Xu, Ba, Kiros, Cho, Courville, Salakhutdinov, Zemel & Bengio (2016),

          "Show, Attend and Tell: Neural Image Caption Generation with Visual

          Attention", <http://arxiv.org/pdf/1502.03044.pdf>



    See Also:

        `LSTM`_ in the Keras documentation.



        .. _LSTM: http://keras.io/layers/recurrent/#lstm

    """

    def __init__(self, *args, attn_activation='tanh', attn_init='orthogonal',

                 output_alpha=False, **kwargs):

        self.attn_activation = activations.get(attn_activation)

        self.attn_init = initializations.get(attn_init)

        self.output_alpha = output_alpha

        super().__init__(*args, **kwargs)



    def build(self, input_shape):

        if not (isinstance(input_shape, list) and len(input_shape) == 2):

            raise Exception('Input to AttentionLSTM must be a list of '

                            'two tensors [lstm_input, attn_input].')



        input_shape, attn_input_shape = input_shape

        super().build(input_shape)

        self.input_spec.append(InputSpec(shape=attn_input_shape))



        # weights for attention model

        self.U_att = self.inner_init((self.output_dim, self.output_dim),

                                     name='{}_U_att'.format(self.name))

        self.W_att = self.attn_init((attn_input_shape[-1], self.output_dim),

                                    name='{}_W_att'.format(self.name))

        self.v_att = self.init((self.output_dim, 1),

                               name='{}_v_att'.format(self.name))

        self.b_att = K.zeros((self.output_dim,), name='{}_b_att'.format(self.name))

        self.trainable_weights += [self.U_att, self.W_att, self.v_att, self.b_att]



        # weights for incorporating attention into hidden states

        if self.consume_less == 'gpu':

            self.Z = self.init((attn_input_shape[-1], 4 * self.output_dim),

                               name='{}_Z'.format(self.name))

            self.trainable_weights += [self.Z]

        else:

            self.Z_i = self.attn_init((attn_input_shape[-1], self.output_dim),

                                      name='{}_Z_i'.format(self.name))

            self.Z_f = self.attn_init((attn_input_shape[-1], self.output_dim),

                                      name='{}_Z_f'.format(self.name))

            self.Z_c = self.attn_init((attn_input_shape[-1], self.output_dim),

                                      name='{}_Z_c'.format(self.name))

            self.Z_o = self.attn_init((attn_input_shape[-1], self.output_dim),

                                      name='{}_Z_o'.format(self.name))

            self.trainable_weights += [self.Z_i, self.Z_f, self.Z_c, self.Z_o]

            self.Z = K.concatenate([self.Z_i, self.Z_f, self.Z_c, self.Z_o])



        # weights for initializing states based on attention vector

        if not self.stateful:

            self.W_init_c = self.attn_init((attn_input_shape[-1], self.output_dim),

                                           name='{}_W_init_c'.format(self.name))

            self.W_init_h = self.attn_init((attn_input_shape[-1], self.output_dim),

                                           name='{}_W_init_h'.format(self.name))

            self.b_init_c = K.zeros((self.output_dim,),

                                    name='{}_b_init_c'.format(self.name))

            self.b_init_h = K.zeros((self.output_dim,),

                                    name='{}_b_init_h'.format(self.name))

            self.trainable_weights += [self.W_init_c, self.b_init_c,

                                       self.W_init_h, self.b_init_h]



        if self.initial_weights is not None:

            self.set_weights(self.initial_weights)

            del self.initial_weights



    def get_output_shape_for(self, input_shape):

        # output shape is not affected by the attention component

        return super().get_output_shape_for(input_shape[0])



    def compute_mask(self, input, input_mask=None):

        if input_mask is not None:

            input_mask = input_mask[0]

        return super().compute_mask(input, input_mask=input_mask)



    def get_initial_states(self, x_input, x_attn, mask_attn):

        # set initial states from mean attention vector fed through a dense

        # activation

        mean_attn = K.mean(x_attn * K.expand_dims(mask_attn), axis=1)

        h0 = K.dot(mean_attn, self.W_init_h) + self.b_init_h

        c0 = K.dot(mean_attn, self.W_init_c) + self.b_init_c

        return [self.attn_activation(h0), self.attn_activation(c0)]



    def call(self, x, mask=None):

        assert isinstance(x, list) and len(x) == 2

        x_input, x_attn = x

        if mask is not None:

            mask_input, mask_attn = mask

        else:

            mask_input, mask_attn = None, None

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

        if self.stateful:

            initial_states = self.states

        else:

            initial_states = self.get_initial_states(x_input, x_attn, mask_attn)

        constants = self.get_constants(x_input, x_attn, mask_attn)

        preprocessed_input = self.preprocess_input(x_input)



        last_output, outputs, states = K.rnn(self.step, preprocessed_input,

                                             initial_states,

                                             go_backwards=self.go_backwards,

                                             mask=mask_input,

                                             constants=constants,

                                             unroll=self.unroll,

                                             input_length=input_shape[1])

        if self.stateful:

            self.updates = []

            for i in range(len(states)):

                self.updates.append((self.states[i], states[i]))



        if self.return_sequences:

            return outputs

        else:

            return last_output



    def step(self, x, states):

        h_tm1 = states[0]

        c_tm1 = states[1]

        B_U = states[2]

        B_W = states[3]

        x_attn = states[4]

        mask_attn = states[5]

        attn_shape = self.input_spec[1].shape



        #### attentional component

        # alignment model

        # -- keeping weight matrices for x_attn and h_s separate has the advantage

        # that the feature dimensions of the vectors can be different

        h_att = K.repeat(h_tm1, attn_shape[1])

        att = time_distributed_dense(x_attn, self.W_att, self.b_att)

        energy = self.attn_activation(K.dot(h_att, self.U_att) + att)

        energy = K.squeeze(K.dot(energy, self.v_att), 2)

        # make probability tensor

        alpha = K.exp(energy)

        if mask_attn is not None:

            alpha *= mask_attn

        alpha /= K.sum(alpha, axis=1, keepdims=True)

        alpha_r = K.repeat(alpha, attn_shape[2])

        alpha_r = K.permute_dimensions(alpha_r, (0, 2, 1))

        # make context vector -- soft attention after Bahdanau et al.

        z_hat = x_attn * alpha_r

        z_hat = K.sum(z_hat, axis=1)



        if self.consume_less == 'gpu':

            z = K.dot(x * B_W[0], self.W) + K.dot(h_tm1 * B_U[0], self.U) + K.dot(z_hat, self.Z) + self.b



            z0 = z[:, :self.output_dim]

            z1 = z[:, self.output_dim: 2 * self.output_dim]

            z2 = z[:, 2 * self.output_dim: 3 * self.output_dim]

            z3 = z[:, 3 * self.output_dim:]

        else:

            if self.consume_less == 'cpu':

                x_i = x[:, :self.output_dim]

                x_f = x[:, self.output_dim: 2 * self.output_dim]

                x_c = x[:, 2 * self.output_dim: 3 * self.output_dim]

                x_o = x[:, 3 * self.output_dim:]

            elif self.consume_less == 'mem':

                x_i = K.dot(x * B_W[0], self.W_i) + self.b_i

                x_f = K.dot(x * B_W[1], self.W_f) + self.b_f

                x_c = K.dot(x * B_W[2], self.W_c) + self.b_c

                x_o = K.dot(x * B_W[3], self.W_o) + self.b_o

            else:

                raise Exception('Unknown `consume_less` mode.')



            z0 = x_i + K.dot(h_tm1 * B_U[0], self.U_i) + K.dot(z_hat, self.Z_i)

            z1 = x_f + K.dot(h_tm1 * B_U[1], self.U_f) + K.dot(z_hat, self.Z_f)

            z2 = x_c + K.dot(h_tm1 * B_U[2], self.U_c) + K.dot(z_hat, self.Z_c)

            z3 = x_o + K.dot(h_tm1 * B_U[3], self.U_o) + K.dot(z_hat, self.Z_o)



        i = self.inner_activation(z0)

        f = self.inner_activation(z1)

        c = f * c_tm1 + i * self.activation(z2)

        o = self.inner_activation(z3)



        h = o * self.activation(c)

        if self.output_alpha:

            return alpha, [h, c]

        else:

            return h, [h, c]



    def get_constants(self, x_input, x_attn, mask_attn):

        constants = super().get_constants(x_input)

        attn_shape = self.input_spec[1].shape

        if mask_attn is not None:

            if K.ndim(mask_attn) == 3:

                mask_attn = K.all(mask_attn, axis=-1)

        constants.append(x_attn)

        constants.append(mask_attn)

        return constants



    def get_config(self):

        cfg = super().get_config()

        cfg['output_alpha'] = self.output_alpha

        cfg['attn_activation'] = self.attn_activation.__name__

        return cfg



    @classmethod

    def from_config(cls, config):

        instance = super(AttentionLSTM, cls).from_config(config)

        if 'output_alpha' in config:

            instance.output_alpha = config['output_alpha']

        if 'attn_activation' in config:

            instance.attn_activation = activations.get(config['attn_activation'])

        return instance
#---------------------------------------------------------LSTM----------------------------------------------------------------
from keras.layers import Dense, Activation, Dropout, Bidirectional, Flatten, RepeatVector, Permute, Lambda, merge, TimeDistributed, recurrent, InputLayer
from keras.models import Model
import keras
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
NUM_EPOCHS = 100
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
        model.add(InputLayer(input_shape=(None, self.num_input_tokens)))
        model.add(AttentionLSTM(units=HIDDEN_UNITS, return_sequences=False, dropout=0.5))
        model.add(Dense(512, activation='relu'))
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
