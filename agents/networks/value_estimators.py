"""Neural networks that output value estimates for actions, given a state."""

import numpy as np
import tensorflow as tf


class PlayerRelativeMovementCNNStackedFrames(object):
    """Uses feature_screen.player_relative to assign q value to movements."""

    def __init__(self,
                 spatial_dimensions,
                 learning_rate,
                 sequence_length,
                 save_path=None,
                 summary_path=None,
                 name="DQNStackedFrames"):
        """Initialize instance-specific hyperparameters, build tf graph."""
        self.spatial_dimensions = spatial_dimensions
        self.learning_rate = learning_rate
        self.name = name
        self.save_path = save_path
        self.sequence_length = sequence_length

        # build graph
        self._build()

        # setup summary writer
        if summary_path:
            self.writer = tf.summary.FileWriter(summary_path)
            tf.summary.scalar("Loss", self.loss)
            tf.summary.scalar("Score", self.score)
            tf.summary.scalar("Batch_Max_Q", self.max_q)
            tf.summary.scalar("Batch_Mean_Q", self.mean_q)
            self.write_op = tf.summary.merge_all()

        # setup model saver
        if self.save_path:
            self.saver = tf.train.Saver()

    def save_model(self, sess):
        """Write tensorflow ckpt."""
        self.saver.save(sess, self.save_path)

    def load(self, sess):
        """Restore from ckpt."""
        self.saver.restore(sess, self.save_path)

    def write_summary(self, sess, states, actions, targets, score):
        """Write summary to Tensorboard."""
        global_episode = self.global_episode.eval(session=sess)
        summary = sess.run(
            self.write_op,
            feed_dict={self.inputs: states,
                       self.actions: actions,
                       self.targets: targets,
                       self.score: score})
        self.writer.add_summary(summary, global_episode - 1)
        self.writer.flush

    def optimizer_op(self, sess, states, actions, targets):
        """Perform one iteration of gradient updates."""
        loss, _ = sess.run(
            [self.loss, self.optimizer],
            feed_dict={self.inputs: states,
                       self.actions: actions,
                       self.targets: targets})

    def increment_global_episode_op(self, sess):
        """Increment the global episode tracker."""
        sess.run(self.increment_global_episode)

    def _build(self):
        """Construct graph."""
        with tf.variable_scope(self.name):
            # score tracker
            self.score = tf.placeholder(
                tf.int32,
                [],
                name="score")

            # global step trackers for multiple runs restoring from ckpt
            self.global_step = tf.Variable(
                0,
                trainable=False,
                name="global_step")

            self.global_episode = tf.Variable(
                0,
                trainable=False,
                name="global_episode")

            # placeholders
            self.inputs = tf.placeholder(
                tf.int32,
                [None, self.sequence_length, *self.spatial_dimensions],
                name="inputs")

            self.actions = tf.placeholder(
                tf.float32,
                [None, np.prod(self.spatial_dimensions)],
                name="actions")

            self.targets = tf.placeholder(
                tf.float32,
                [None],
                name="targets")

            self.increment_global_episode = tf.assign(
                self.global_episode,
                self.global_episode + 1,
                name="increment_global_episode")

            # spatial coordinates are given in y-major screen coordinate space
            # transpose them to (x, y) space before beginning
            self.transposed = tf.transpose(
                self.inputs,
                perm=[0, 1, 3, 2],
                name="transpose")

            # embed layer (one-hot in channel dimension, 1x1 convolution)
            # the player_relative feature layer has 5 categorical values
            self.one_hot = tf.one_hot(
                self.transposed,
                depth=5,
                axis=-1,
                name="one_hot")

            self.transposed2 = tf.transpose(
                self.one_hot,
                perm=[0, 2, 3, 4, 1],
                name="transpose2")

            self.seq_removed = tf.reshape(
                self.transposed2,
                [-1, 84, 84, 5 * self.sequence_length],
                name="remove_seq_dim")


            self.embed = tf.layers.conv2d(
                inputs=self.seq_removed,
                filters=self.sequence_length,
                kernel_size=[1, 1],
                strides=[1, 1],
                padding="SAME",
                name="embed")

            # convolutional layer 1
            self.conv1 = tf.layers.conv2d(
                inputs=self.embed,
                filters=16,
                kernel_size=[5, 5],
                strides=[1, 1],
                padding="SAME",
                name="conv1")

            self.conv1_activation = tf.nn.relu(
                self.conv1,
                name="conv1_activation")

            # convolutional layer 2
            self.conv2 = tf.layers.conv2d(
                inputs=self.conv1_activation,
                filters=32,
                kernel_size=[3, 3],
                strides=[1, 1],
                padding="SAME",
                name="conv2")

            self.conv2_activation = tf.nn.relu(
                self.conv2,
                name="conv2_activation")

            # spatial output layer
            self.output = tf.layers.conv2d(
                inputs=self.conv2_activation,
                filters=1,
                kernel_size=[1, 1],
                strides=[1, 1],
                padding="SAME",
                name="output")

            self.flatten = tf.layers.flatten(self.output, name="flat")

            # value estimate trackers for summaries
            self.max_q = tf.reduce_max(self.flatten, name="max")
            self.mean_q = tf.reduce_mean(self.flatten, name="mean")

            # optimization: MSE between state predicted Q and target Q
            self.prediction = tf.reduce_sum(
                tf.multiply(self.flatten, tf.to_float(self.actions)),
                axis=1,
                name="prediction")

            self.loss = tf.reduce_mean(
                tf.square(self.targets - self.prediction),
                name="loss")

            self.optimizer = tf.train.RMSPropOptimizer(
                self.learning_rate).minimize(self.loss,
                                             global_step=self.global_step)

class PlayerRelativeMovementCNN(object):
    """Uses feature_screen.player_relative to assign q value to movements."""

    def __init__(self,
                 spatial_dimensions,
                 learning_rate,
                 save_path=None,
                 summary_path=None,
                 name="DQN"):
        """Initialize instance-specific hyperparameters, build tf graph."""
        self.spatial_dimensions = spatial_dimensions
        self.learning_rate = learning_rate
        self.name = name
        self.save_path = save_path

        # build graph
        self._build()

        # setup summary writer
        if summary_path:
            self.writer = tf.summary.FileWriter(summary_path)
            tf.summary.scalar("Loss", self.loss)
            tf.summary.scalar("Score", self.score)
            tf.summary.scalar("Batch_Max_Q", self.max_q)
            tf.summary.scalar("Batch_Mean_Q", self.mean_q)
            self.write_op = tf.summary.merge_all()

        # setup model saver
        if self.save_path:
            self.saver = tf.train.Saver()

    def save_model(self, sess):
        """Write tensorflow ckpt."""
        self.saver.save(sess, self.save_path)

    def load(self, sess):
        """Restore from ckpt."""
        self.saver.restore(sess, self.save_path)

    def write_summary(self, sess, states, actions, targets, score):
        """Write summary to Tensorboard."""
        global_episode = self.global_episode.eval(session=sess)
        summary = sess.run(
            self.write_op,
            feed_dict={self.inputs: states,
                       self.actions: actions,
                       self.targets: targets,
                       self.score: score})
        self.writer.add_summary(summary, global_episode - 1)
        self.writer.flush

    def optimizer_op(self, sess, states, actions, targets):
        """Perform one iteration of gradient updates."""
        loss, _ = sess.run(
            [self.loss, self.optimizer],
            feed_dict={self.inputs: states,
                       self.actions: actions,
                       self.targets: targets})

    def increment_global_episode_op(self, sess):
        """Increment the global episode tracker."""
        sess.run(self.increment_global_episode)

    def _build(self):
        """Construct graph."""
        with tf.variable_scope(self.name):
            # score tracker
            self.score = tf.placeholder(
                tf.int32,
                [],
                name="score")

            # global step trackers for multiple runs restoring from ckpt
            self.global_step = tf.Variable(
                0,
                trainable=False,
                name="global_step")

            self.global_episode = tf.Variable(
                0,
                trainable=False,
                name="global_episode")

            # placeholders
            self.inputs = tf.placeholder(
                tf.int32,
                [None, *self.spatial_dimensions],
                name="inputs")

            self.actions = tf.placeholder(
                tf.float32,
                [None, np.prod(self.spatial_dimensions)],
                name="actions")

            self.targets = tf.placeholder(
                tf.float32,
                [None],
                name="targets")

            self.increment_global_episode = tf.assign(
                self.global_episode,
                self.global_episode + 1,
                name="increment_global_episode")

            # spatial coordinates are given in y-major screen coordinate space
            # transpose them to (x, y) space before beginning
            self.transposed = tf.transpose(
                self.inputs,
                perm=[0, 2, 1],
                name="transpose")

            # embed layer (one-hot in channel dimension, 1x1 convolution)
            # the player_relative feature layer has 5 categorical values
            self.one_hot = tf.one_hot(
                self.transposed,
                depth=5,
                axis=-1,
                name="one_hot")

            self.embed = tf.layers.conv2d(
                inputs=self.one_hot,
                filters=1,
                kernel_size=[1, 1],
                strides=[1, 1],
                padding="SAME",
                name="embed")

            # convolutional layer
            self.conv1 = tf.layers.conv2d(
                inputs=self.embed,
                filters=16,
                kernel_size=[3, 3],
                strides=[1, 1],
                padding="SAME",
                name="conv1")

            self.conv1_activation = tf.nn.relu(
                self.conv1,
                name="conv1_activation")

            # spatial output layer
            self.output = tf.layers.conv2d(
                inputs=self.conv1_activation,
                filters=1,
                kernel_size=[1, 1],
                strides=[1, 1],
                padding="SAME",
                name="output")

            self.flatten = tf.layers.flatten(self.output, name="flat")

            # value estimate trackers for summaries
            self.max_q = tf.reduce_max(self.flatten, name="max")
            self.mean_q = tf.reduce_mean(self.flatten, name="mean")

            # optimization: MSE between state predicted Q and target Q
            self.prediction = tf.reduce_sum(
                tf.multiply(self.flatten, self.actions),
                axis=1,
                name="prediction")

            self.loss = tf.reduce_mean(
                tf.square(self.targets - self.prediction),
                name="loss")

            self.optimizer = tf.train.RMSPropOptimizer(
                self.learning_rate).minimize(self.loss,
                                             global_step=self.global_step)


class FullyConvNet(object):
    """A fully convolutional model."""

    def __init__(self,
                 spatial_dimensions,
                 learning_rate,
                 save_path=None,
                 summary_path=None,
                 name="fullyconv"):

        """Initialize instance-specific hyperparameters, build tf graph."""
        self.spatial_dimensions = spatial_dimensions
        self.learning_rate = learning_rate
        self.name = name
        self.save_path = save_path

        # build graph
        self._build()

        # setup summary writer
        if summary_path:
            self.writer = tf.summary.FileWriter(summary_path)
            tf.summary.scalar("Loss", self.loss)
            tf.summary.scalar("Score", self.score)
            tf.summary.scalar("Batch_Max_Q", self.max_q)
            tf.summary.scalar("Batch_Mean_Q", self.mean_q)
            self.write_op = tf.summary.merge_all()

        # setup model saver
        if self.save_path:
            self.saver = tf.train.Saver()

    def save_model(self, sess):
        """Write tensorflow ckpt."""
        self.saver.save(sess, self.save_path)

    def load(self, sess):
        """Restore from ckpt."""
        self.saver.restore(sess, self.save_path)

    def write_summary(self, sess, states, actions, targets, score):
        """Write summary to Tensorboard."""
        global_episode = self.global_episode.eval(session=sess)
        summary = sess.run(
            self.write_op,
            feed_dict={self.inputs: states,
                       self.actions: actions,
                       self.targets: targets,
                       self.score: score})
        self.writer.add_summary(summary, global_episode - 1)
        self.writer.flush

    def optimizer_op(self, sess, states, actions, targets):
        """Perform one iteration of gradient updates."""
        loss, _ = sess.run(
            [self.loss, self.optimizer],
            feed_dict={self.inputs: states,
                       self.actions: actions,
                       self.targets: targets})


    def increment_global_episode_op(self, sess):
        """Increment the global episode tracker."""
        sess.run(self.increment_global_episode)

    def _build(self):
        """Construct graph."""
        with tf.variable_scope(self.name):
            # score tracker
            self.score = tf.placeholder(
                tf.int32,
                [],
                name="score")

            # global step trackers for multiple runs restoring from ckpt
            self.global_step = tf.Variable(
                0,
                trainable=False,
                name="global_step")

            self.global_episode = tf.Variable(
                0,
                trainable=False,
                name="global_episode")

            # placeholders
            self.inputs = tf.placeholder(
                tf.int32,
                [None, *self.spatial_dimensions],
                name="screen")

            self.actions = tf.placeholder(
                tf.float32,
                [None, np.prod(self.spatial_dimensions)],
                name="actions")

            self.targets = tf.placeholder(
                tf.float32,
                [None],
                name="targets")            

            self.increment_global_episode = tf.assign(
                self.global_episode,
                self.global_episode + 1,
                name="increment_global_episode")

            # spatial coordinates are given in y-major screen coordinate space
            # transpose them to (x, y) space before beginning
            self.transposed = tf.transpose(
                self.inputs,
                perm=[0, 2, 1],
                name="transpose")

            # embed layer (one-hot in channel dimension, 1x1 convolution)
            # the player_relative feature layer has 5 categorical values
            self.one_hot = tf.one_hot(
                self.transposed,
                depth=5,
                axis=-1,
                name="one_hot")

            self.embed = tf.layers.conv2d(
                inputs=self.one_hot,
                filters=1,
                kernel_size=[1, 1],
                strides=[1, 1],
                padding="SAME",
                name="embed")

            # convolutional layer 1
            self.conv1 = tf.layers.conv2d(
                inputs=self.embed,
                filters=16,
                kernel_size=[5, 5],
                strides=[1, 1],
                padding="SAME",
                name="conv1")

            self.conv1_activation = tf.nn.relu(
                self.conv1,
                name="conv1_activation")

            # convolutional layer 2
            self.conv2 = tf.layers.conv2d(
                inputs=self.conv1_activation,
                filters=32,
                kernel_size=[3, 3],
                strides=[1, 1],
                padding="SAME",
                name="conv2")

            self.conv2_activation = tf.nn.relu(
                self.conv2,
                name="conv2_activation")

            self.flattened_conv_output = tf.layers.flatten(self.conv2_activation, name="lstm_input")


            # spatial output layer
            self.spatial_output = tf.layers.conv2d(
                inputs=self.conv2_activation,
                filters=1,
                kernel_size=[1, 1],
                strides=[1, 1],
                padding="SAME",
                name="spatial_output")

            self.spatial_flatten = tf.layers.flatten(self.spatial_output, name="flat")


            # value estimate trackers for summaries
            self.max_q = tf.reduce_max(self.spatial_flatten, name="max")
            self.mean_q = tf.reduce_mean(self.spatial_flatten, name="mean")

            # optimization: MSE between state predicted Q and target Q
            self.prediction = tf.reduce_sum(
                tf.multiply(self.spatial_flatten, self.actions),
                axis=1,
                name="prediction")

            self.loss = tf.reduce_mean(
                tf.square(self.targets - self.prediction),
                name="loss")

            self.optimizer = tf.train.RMSPropOptimizer(
                self.learning_rate).minimize(self.loss,
                                             global_step=self.global_step)


class ConvLstmNet(object):
    """A fully convolutional model."""

    def __init__(self,
                 spatial_dimensions,
                 learning_rate,
                 seq_length,
                 batch_size,
                 save_path=None,
                 summary_path=None,
                 name="fullyconv"):

        """Initialize instance-specific hyperparameters, build tf graph."""
        self.spatial_dimensions = spatial_dimensions
        self.learning_rate = learning_rate
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.name = name
        self.save_path = save_path

        # build graph
        self._build()

        # setup summary writer
        if summary_path:
            self.writer = tf.summary.FileWriter(summary_path)
            tf.summary.scalar("Loss", self.loss)
            tf.summary.scalar("Score", self.score)
            tf.summary.scalar("Batch_Max_Q", self.max_q)
            tf.summary.scalar("Batch_Mean_Q", self.mean_q)
            self.write_op = tf.summary.merge_all()

        # setup model saver
        if self.save_path:
            self.saver = tf.train.Saver()

    def save_model(self, sess):
        """Write tensorflow ckpt."""
        self.saver.save(sess, self.save_path)

    def load(self, sess):
        """Restore from ckpt."""
        self.saver.restore(sess, self.save_path)

    def write_summary(self, sess, states, actions, targets, score):
        """Write summary to Tensorboard."""
        global_episode = self.global_episode.eval(session=sess)
        summary = sess.run(
            self.write_op,
            feed_dict={self.inputs: states,
                       self.actions: actions,
                       self.targets: targets,
                       self.score: score})
        self.writer.add_summary(summary, global_episode - 1)
        self.writer.flush

    def optimizer_op(self, sess, states, actions, targets):
        """Perform one iteration of gradient updates."""
        loss, _ = sess.run(
            [self.loss, self.optimizer],
            feed_dict={self.inputs: states,
                       self.actions: actions,
                       self.targets: targets})


    def increment_global_episode_op(self, sess):
        """Increment the global episode tracker."""
        sess.run(self.increment_global_episode)

    def _build(self):
        """Construct graph."""
        with tf.variable_scope(self.name):
            # score tracker
            self.score = tf.placeholder(
                tf.int32,
                [],
                name="score")

            # global step trackers for multiple runs restoring from ckpt
            self.global_step = tf.Variable(
                0,
                trainable=False,
                name="global_step")

            self.global_episode = tf.Variable(
                0,
                trainable=False,
                name="global_episode")

            # placeholders
            # inputs is batch size * sequence length * spatial dimensions -> reshape to pass in all experiences in parallel
            self.inputs = tf.placeholder(
                tf.int32,
                [None, self.seq_length, *self.spatial_dimensions],
                name="inputs")

            self.reshaped_inputs = tf.reshape(
                self.inputs,
                [-1, *self.spatial_dimensions],
                name="reshaped_inputs")

            self.actions = tf.placeholder(
                tf.float32,
                [None, np.prod(self.spatial_dimensions)],
                name="actions")

            self.targets = tf.placeholder(
                tf.float32,
                [None],
                name="targets")            

            self.increment_global_episode = tf.assign(
                self.global_episode,
                self.global_episode + 1,
                name="increment_global_episode")

            # spatial coordinates are given in y-major screen coordinate space
            # transpose them to (x, y) space before beginning
            self.transposed = tf.transpose(
                self.reshaped_inputs,
                perm=[0, 2, 1],
                name="transpose")

            # embed layer (one-hot in channel dimension, 1x1 convolution)
            # the player_relative feature layer has 5 categorical values
            self.one_hot = tf.one_hot(
                self.transposed,
                depth=5,
                axis=-1,
                name="one_hot")

            self.embed = tf.layers.conv2d(
                inputs=self.one_hot,
                filters=1,
                kernel_size=[1, 1],
                strides=[1, 1],
                padding="SAME",
                name="embed")

            # convolutional layer 1
            self.conv1 = tf.layers.conv2d(
                inputs=self.embed,
                filters=16,
                kernel_size=[5, 5],
                strides=[1, 1],
                padding="VALID",
                name="conv1")

            self.conv1_activation = tf.nn.relu(
                self.conv1,
                name="conv1_activation")

            # convolutional layer 2
            self.conv2 = tf.layers.conv2d(
                inputs=self.conv1_activation,
                filters=32,
                kernel_size=[3, 3],
                strides=[1, 1],
                padding="VALID",
                name="conv2")

            self.conv2_activation = tf.nn.relu(
                self.conv2,
                name="conv2_activation")

            # convolve down so # channels = num available actions
            self.spatial_output = tf.layers.conv2d(
                inputs=self.conv2_activation,
                filters=1,
                kernel_size=[1, 1],
                strides=[1, 1],
                padding="SAME",
                name="action_channels")

            self.flatten = tf.layers.flatten(self.spatial_output, name="flat_action_channels")

            self.fc1 = tf.layers.dense(
                inputs=self.flatten,
                units=256,
                name="fc1")

            # THIS MUST BE ADJUSTED FOR SIZE OF CONV OUTPUT
            self.reshaped_flatten = tf.reshape(
                self.fc1,
                [self.batch_size, self.seq_length, 256],
                name="add_seq_dimension")


            # lstm cell. TODO: change to cudunn lstm cell for better gpu perf
            self.lstm_cell = tf.nn.rnn_cell.LSTMCell(self.reshaped_flatten.shape[2])
            self.lstm_output, self.lstm_state = tf.nn.dynamic_rnn(
                cell=self.lstm_cell, 
                inputs=self.reshaped_flatten,
                dtype=tf.float32)

            #self.outputs = self.flatten
            self.fc_x = tf.layers.dense(
                inputs=self.lstm_output,
                units=1)
            self.fc_y = tf.layers.dense(
                inputs=self.lstm_output,
                units=1)

            self.outputs = self.fc_x, self.fc_y

            # value estimate trackers for summaries
            self.max_q = tf.reduce_max(self.outputs, name="max")
            self.mean_q = tf.reduce_mean(self.outputs, name="mean")

            # optimization: MSE between state predicted Q and target Q
            #self.prediction = tf.reduce_sum(
            #    tf.multiply(self.outputs, self.actions),
            #    axis=1,
            #    name="prediction")
            self.prediction = self.outputs

            self.loss = tf.reduce_mean(
                tf.square(self.targets - self.prediction),
                name="loss")

            self.optimizer = tf.train.RMSPropOptimizer(
                self.learning_rate).minimize(self.loss,
                                             global_step=self.global_step)


class FullyConvNetExpandedActionSpace(object):
    """A fully convolutional model."""

    def __init__(self,
                 spatial_dimensions,
                 learning_rate,
                 save_path=None,
                 summary_path=None,
                 name="fullyconvexpanded"):

        """Initialize instance-specific hyperparameters, build tf graph."""
        self.spatial_dimensions = spatial_dimensions
        self.learning_rate = learning_rate
        self.name = name
        self.save_path = save_path

        # build graph
        self._build()

        # setup summary writer
        if summary_path:
            self.writer = tf.summary.FileWriter(summary_path)
            tf.summary.scalar("Loss", self.loss)
            tf.summary.scalar("Score", self.score)
            tf.summary.scalar("Batch_Max_Q", self.max_q)
            tf.summary.scalar("Batch_Mean_Q", self.mean_q)
            self.write_op = tf.summary.merge_all()

        # setup model saver
        if self.save_path:
            self.saver = tf.train.Saver()

    def save_model(self, sess):
        """Write tensorflow ckpt."""
        self.saver.save(sess, self.save_path)

    def load(self, sess):
        """Restore from ckpt."""
        self.saver.restore(sess, self.save_path)

    def write_summary(self, sess, states, actions, action_types, targets, score):
        """Write summary to Tensorboard."""
        global_episode = self.global_episode.eval(session=sess)
        summary = sess.run(
            self.write_op,
            feed_dict={self.inputs: states,
                       self.actions: actions,
                       self.action_type : action_types,
                       self.targets: targets,
                       self.score: score})
        self.writer.add_summary(summary, global_episode - 1)
        self.writer.flush

    def optimizer_op(self, sess, states, actions, action_types, targets):
        """Perform one iteration of gradient updates."""
        loss, _ = sess.run(
            [self.loss, self.optimizer],
            feed_dict={self.inputs: states,
                       self.actions: actions,
                       self.action_type: action_types,
                       self.targets: targets})


    def increment_global_episode_op(self, sess):
        """Increment the global episode tracker."""
        sess.run(self.increment_global_episode)

    def _build(self):
        """Construct graph."""
        with tf.variable_scope(self.name):
            # score tracker
            self.score = tf.placeholder(
                tf.int32,
                [],
                name="score")

            # global step trackers for multiple runs restoring from ckpt
            self.global_step = tf.Variable(
                0,
                trainable=False,
                name="global_step")

            self.global_episode = tf.Variable(
                0,
                trainable=False,
                name="global_episode")

            # placeholders
            self.inputs = tf.placeholder(
                tf.int32,
                [None, *self.spatial_dimensions],
                name="inputs")

            self.actions = tf.placeholder(
                tf.int32,
                [None, np.prod(self.spatial_dimensions)],
                name="actions")

            self.action_type = tf.placeholder(
                tf.int32,
                [None],
                name="action_types")


            self.targets = tf.placeholder(
                tf.float32,
                [None],
                name="targets")            

            self.increment_global_episode = tf.assign(
                self.global_episode,
                self.global_episode + 1,
                name="increment_global_episode")

            # spatial coordinates are given in y-major screen coordinate space
            # transpose them to (x, y) space before beginning
            self.transposed = tf.transpose(
                self.inputs,
                perm=[0, 2, 1],
                name="transpose")

            # embed layer (one-hot in channel dimension, 1x1 convolution)
            # the player_relative feature layer has 5 categorical values
            self.one_hot = tf.one_hot(
                self.transposed,
                depth=5,
                axis=-1,
                name="one_hot")

            self.embed = tf.layers.conv2d(
                inputs=self.one_hot,
                filters=1,
                kernel_size=[1, 1],
                strides=[1, 1],
                padding="SAME",
                name="embed")

            # convolutional layer 1
            self.conv1 = tf.layers.conv2d(
                inputs=self.embed,
                filters=16,
                kernel_size=[5, 5],
                strides=[1, 1],
                padding="SAME",
                name="conv1")

            self.conv1_activation = tf.nn.relu(
                self.conv1,
                name="conv1_activation")

            # convolutional layer 2
            self.conv2 = tf.layers.conv2d(
                inputs=self.conv1_activation,
                filters=32,
                kernel_size=[3, 3],
                strides=[1, 1],
                padding="SAME",
                name="conv2")

            self.conv2_activation = tf.nn.relu(
                self.conv2,
                name="conv2_activation")

            #self.flattened_conv_output = tf.layers.flatten(self.conv2_activation, name="lstm_input")


            # attack_screen spatial output
            self.screen_out = tf.layers.conv2d(
                inputs=self.conv2_activation,
                filters=1,
                kernel_size=[1, 1],
                strides=[1, 1],
                padding="SAME",
                name="screen_spatial_output")

            # rectangle spatial output
            #self.screen2_out = tf.layers.conv2d(
            #    inputs=self.conv2_activation,
            #    filters=1,
            #    kernel_size=[1, 1],
            #    strides=[1, 1],
            #    padding="SAME",
            #    name="screen2_spatial_output")

            self.spatial_flatten = tf.layers.flatten(self.conv2_activation, name="flat")

            # function selection
            self.function_fc = tf.layers.dense(
                inputs=self.spatial_flatten,
                units=2,
                name="action_fc")

            self.fc_box_size = tf.layers.dense(
                inputs=self.spatial_flatten,
                units=1,
                name="box_size")

            #self.screen2_flatten = tf.layers.flatten(self.screen2_out, name="screen2_flatten")

            self.screen_flatten = tf.layers.flatten(self.screen_out, name="screen_flatten")


            self.best_action_type = tf.argmax(self.function_fc, axis = 1)
            self.best_action = tf.argmax(self.screen_flatten, axis = 1)
            self.best_values = tf.multiply(tf.reduce_max(self.function_fc, axis=1), tf.reduce_max(self.screen_flatten, axis = 1))


            # value estimate trackers for summaries
            #self.max_screen2_q = tf.reduce_max(self.screen2_flatten) * self.function_fc[1]
            #self.max_screen_q = tf.reduce_max(self.screen_flatten) * self.function_fc[0]

            self.max_q = 0# tf.reduce_max([self.max_screen_q, self.max_screen2_q], name="max")
            self.mean_q = 0#tf.reduce_mean(self.spatial_flatten, name="mean")

            action_type_oh = tf.one_hot(self.action_type, 2, name="action_type_one_hot")
            self.function_vals = tf.reduce_sum(self.function_fc * action_type_oh, axis=1, name="action_extraction")

            self.prediction = self.function_vals * tf.reduce_sum(tf.multiply(tf.to_float(self.actions), self.screen_flatten, name="mul"), axis = 1)


            # optimization: MSE between state predicted Q and target Q
            #self.prediction = tf.reduce_sum(
            #    tf.multiply(self.spatial_flatten, self.actions),
            #    axis=1,
            #    name="prediction")
            #print(self.targets.shape)
            #print(self.prediction.shape)

            self.loss = tf.reduce_mean(
                tf.square(self.targets - self.prediction),
                name="loss")

            self.optimizer = tf.train.RMSPropOptimizer(
                self.learning_rate).minimize(self.loss,
                                             global_step=self.global_step)



