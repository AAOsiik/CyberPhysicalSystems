from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten # layers
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
from collections import deque
import time
import numpy as np


REPLAY_MEMORY_SIZE = 50_000
MODEL_NAME = "256x2"


# Own Tensorboard class
class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.FileWriter(self.log_dir)

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)


class DQNAgent:
    def __init__(self):

        # Main Model, fitting/training every step of 1 value
        self.model = self.create_model()
        # Target Model, this is what we .predict every step 
        # overwrite every episode
        # initialized randomly, will go crazy with epsilon = 1
        # Fitting random is useless tho
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())
        
        # Replay memory, random sampling is the batch to train our NN
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/{MODEL_NAME}-{int(time.time())}")
        # counts how many times we updated our model
        self.target_update_counter = 0


    def create_model(self): # try to apply it to something else
        model = Sequential()
        model.add(Conv2D(256, (3, 3), input_shape=env.OBSERVATION_SPACE_VALUES))
        model.add(Activation("relu")) # rectified linear unit
        model.add(MaxPooling2D(2, 2))
        model.add(Dropout(0.2))

        model.add(Conv2D(256, (3, 3)))
        model.add(Activation("relu")) # rectified linear unit
        model.add(MaxPooling2D(2, 2))
        model.add(Dropout(0.2))

        model.add(Flatten())
        model.add(Dense(64))

        model.add(Dense(env.ACTION_SPACE_SIZE, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy']) # mean square error
        return model

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def get_qs(self, state, step):
        return self.model.predict(np.array(state).reshape(-1, *state.shape)/255)[0]
