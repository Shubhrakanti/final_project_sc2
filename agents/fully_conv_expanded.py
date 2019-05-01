"""Deep Q-learning agents."""
import numpy as np
import os
import tensorflow as tf

# local submodule
import agents.networks.value_estimators as nets

from absl import flags

from collections import deque

from pysc2.agents import base_agent
from pysc2.lib import actions as sc2_actions

FLAGS = flags.FLAGS

# agent interface settings (defaults specified in pysc2.bin.agent)
feature_screen_size = FLAGS.feature_screen_size
feature_minimap_size = FLAGS.feature_minimap_size

# pysc2 convenience
FUNCTIONS = sc2_actions.FUNCTIONS


class Memory(object):
    """Memory buffer for Experience Replay."""

    def __init__(self, max_size):
        """Initialize a buffer containing max_size experiences."""
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        """Add an experience to the buffer."""
        self.buffer.append(experience)

    def sample(self, batch_size):
        """Sample a batch of experiences from the buffer."""
        buffer_size = len(self.buffer)
        index = np.random.choice(
            np.arange(buffer_size),
            size=batch_size,
            replace=False)

        return [self.buffer[i] for i in index]

    def __len__(self):
        """Interface to access buffer length."""
        return len(self.buffer)

class FullyConvExpandedAgent(base_agent.BaseAgent):
    """A fully convolutional DQN that receives `player_relative` features and takes movements."""

    def __init__(self,
                 learning_rate=FLAGS.learning_rate,
                 discount_factor=FLAGS.discount_factor,
                 epsilon_max=FLAGS.epsilon_max,
                 epsilon_min=FLAGS.epsilon_min,
                 epsilon_decay_steps=FLAGS.epsilon_decay_steps,
                 train_frequency=FLAGS.train_frequency,
                 target_update_frequency=FLAGS.target_update_frequency,
                 max_memory=FLAGS.max_memory,
                 batch_size=FLAGS.batch_size,
                 training=FLAGS.training,
                 indicate_nonrandom_action=FLAGS.indicate_nonrandom_action,
                 save_dir="./checkpoints/",
                 ckpt_name="DQNFullyConvExpandedActionSpace",
                 summary_path="./tensorboard/DQNfullyconvexpanded"):
        """Initialize rewards/episodes/steps, build network."""
        super(FullyConvExpandedAgent, self).__init__()

        # saving and summary writing
        if FLAGS.save_dir:
            save_dir = FLAGS.save_dir
        if FLAGS.ckpt_name:
            ckpt_name = FLAGS.ckpt_name
        if FLAGS.summary_path:
            summary_path = FLAGS.summary_path

        # neural net hyperparameters
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

        # agent hyperparameters
        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min
        self.epsilon_decay_steps = epsilon_decay_steps
        self.train_frequency = train_frequency
        self.target_update_frequency = target_update_frequency

        # other parameters
        self.training = training
        self.indicate_nonrandom_action = indicate_nonrandom_action

        # build network
        self.save_path = save_dir + ckpt_name + ".ckpt"

        print("Building models...")
        tf.reset_default_graph()
        self.network = nets.FullyConvNetExpandedActionSpace(
            spatial_dimensions=feature_screen_size,
            learning_rate=self.learning_rate,
            save_path=self.save_path,
            summary_path=summary_path)

        if self.training:
            self.target_net = nets.FullyConvNetExpandedActionSpace(
                spatial_dimensions=feature_screen_size,
                learning_rate=self.learning_rate,
                name="DQNFullyConvExpandedTarget")

            # initialize Experience Replay memory buffer
            self.memory = Memory(max_memory)
            self.batch_size = batch_size

        print("Done.")

        self.last_state = None
        self.last_action = None
        self.last_action_type = None

        # initialize session
        self.sess = tf.Session()
        if os.path.isfile(self.save_path + ".index"):
            self.network.load(self.sess)
            if self.training:
                self._update_target_network()
        else:
            self._tf_init_op()

    def reset(self):
        """Handle the beginning of new episodes."""
        self.episodes += 1
        self.reward = 0

        if self.training:
            self.last_state = None
            self.last_action = None
            episode = self.network.global_episode.eval(session=self.sess)
            print("Global training episode:", episode + 1)

    def step(self, obs):
        """If no units selected, selects army, otherwise move."""
        self.steps += 1
        self.reward += obs.reward

        # handle end of episode if terminal step
        if self.training and obs.step_type == 2:
            self._handle_episode_end()

        screen = obs.observation.feature_screen.player_relative

        if self.training:
            # predict an action to take and take it
            x, y, size, action = self._epsilon_greedy_action_selection(screen)

            # update online DQN
            if (self.steps % self.train_frequency == 0 and
                    len(self.memory) > self.batch_size):
                self._train_network()

            # update network used to estimate TD targets
            if self.steps % self.target_update_frequency == 0:
                self._update_target_network()
                print("Target network updated.")

            # add experience to memory
            if self.last_state is not None:
                self.memory.add(
                        (self.last_state,
                         self.last_action,
                         self.last_action_type,
                         obs.reward,
                         screen))

            self.last_state = screen
            self.last_action = np.ravel_multi_index(
                (x, y),
                feature_screen_size)
            self.last_action_type = action

        else:
            x, y, size, action = self._epsilon_greedy_action_selection(
                    screen,
                    self.epsilon_min)

        if action == 0 and 12 in obs.observation.available_actions:
            return FUNCTIONS.Attack_screen("now", (x, y))
        else:
            #h = 30
            #w = 30
            return FUNCTIONS.select_rect("select", (x, y), (min(x+size, 83), min(y+size, 83)))


    def _handle_episode_end(self):
        """Save weights and write summaries."""
        # increment global training episode
        self.network.increment_global_episode_op(self.sess)

        # save current model
        self.network.save_model(self.sess)
        print("Model Saved")

        # write summaries from last episode
        states, actions, action_types, targets = self._get_batch()
        self.network.write_summary(
            self.sess, states, actions, action_types, targets, self.reward)
        print("Summary Written")

    def _tf_init_op(self):
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

    def _update_target_network(self):
        online_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, "fullyconvexpanded")
        target_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, "DQNFullyConvExpandedTarget")

        update_op = []
        for online_var, target_var in zip(online_vars, target_vars):
            update_op.append(target_var.assign(online_var))

        self.sess.run(update_op)

    def _epsilon_greedy_action_selection(self, state, epsilon=None):
        """Choose action from state with epsilon greedy strategy."""
        step = self.network.global_step.eval(session=self.sess)

        if epsilon is None:
            epsilon = max(
                self.epsilon_min,
                (self.epsilon_max - ((self.epsilon_max - self.epsilon_min) *
                                     step / self.epsilon_decay_steps)))

        if epsilon > np.random.rand():
            x = np.random.randint(0, feature_screen_size[0])
            y = np.random.randint(0, feature_screen_size[1])
            action = np.random.randint(2)

            return x, y, action

        else:
            inputs = np.expand_dims(state, 0)

            action_type, max_index, size = self.sess.run(
                    [self.network.best_action_type, self.network.best_action, self.box_size],
                    feed_dict={self.network.inputs: inputs})

            x, y = np.unravel_index(max_index, feature_screen_size)
            return x, y, size, action_type

    def _train_network(self):
        states, actions, action_types, targets = self._get_batch()
        self.network.optimizer_op(self.sess, states, actions, action_types, targets)

    def _get_batch(self):
        batch = self.memory.sample(self.batch_size)
        states = np.array([each[0] for each in batch])
        actions = np.array([int(each[1]) for each in batch])
        action_types = np.array([each[2] for each in batch])
        rewards = np.array([each[3] for each in batch])
        next_states = np.array([each[4] for each in batch])

        # one-hot encode actions
        actions = np.eye(np.prod(feature_screen_size))[actions]

        # get targets
        next_outputs = self.sess.run(
            self.target_net.best_values,
            feed_dict={self.target_net.inputs: next_states,
                        self.target_net.action_type : action_types})

        targets = [rewards[i] + self.discount_factor * next_outputs[i]#np.max(next_outputs[i])
                   for i in range(self.batch_size)]

        return states, actions, action_types, targets
