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

class Episode(object):
    """Wrapper for one episode. Stores state/action/next_state/reward to be sampled as sequences."""
    def __init__(self):
        """Store one list of states and then state indices to avoid storing states twice."""
        self.states = []
        self.experiences = []
        self.length = 0

    def add_experience(state, action, next_state, reward):
        """Add experience to episode."""
        self.experiences.append((self.length - 1, action, reward, self.length))
        self.states.append(next_state)
        self.length += 1

    def sample_sequence(seq_length):
        """Returns array of (state, action, next_state, reward) tuples."""
        start_idx = np.randint(0, self.length - seq_length)
        res = []
        for i in range(start_idx, start_idx + seq_length):
            exp = self.experiences[i]
            state = self.states[exp[0]]
            action = exp[1]
            reward = exp[2]
            next_state = self.states[exp[3]]
            res.append((state, action, reward, next_state))
        return np.array(res)

class SequenceMemory(object):
    """Memory buffer for Experience Replay."""

    def __init__(self, max_size, seq_length):
        """Initialize a buffer containing max_size experiences."""
        self.buffer = deque(maxlen=max_size)
        self.seq_length = seq_length

    def add(self, episode):
        """Add an experience to the buffer."""
        self.buffer.append(episode)

    def sample(self, batch_size):
        """Sample a batch of experiences from the buffer."""
        buffer_size = len(self.buffer)
        index = np.random.choice(
            np.arange(buffer_size),
            size=batch_size,
            replace=True)

        return [self.buffer[i].sample_sequence(self.seq_length) for i in index]

    def __len__(self):
        """Interface to access buffer length."""
        return len(self.buffer)


class ConvLstmAgent(base_agent.BaseAgent):
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
                 seq_length=FLAGS.seq_length,
                 training=FLAGS.training,
                 indicate_nonrandom_action=FLAGS.indicate_nonrandom_action,
                 save_dir="./checkpoints/",
                 ckpt_name="DQNConvLstmMoveOnly",
                 summary_path="./tensorboard/DQNConvLstm"):
        """Initialize rewards/episodes/steps, build network."""
        super(ConvLstmAgent, self).__init__()

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
        self.seq_length = seq_length
        self.batch_size = batch_size

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
        self.network = nets.ConvLstmNet(
            spatial_dimensions=feature_screen_size,
            learning_rate=self.learning_rate,
            seq_length=self.seq_length,
            batch_size=self.batch_size,
            save_path=self.save_path,
            summary_path=summary_path)

        if self.training:
            self.target_net = nets.ConvLstmNet(
                spatial_dimensions=feature_screen_size,
                learning_rate=self.learning_rate,
                seq_length=self.seq_length,
                batch_size=self.batch_size,
                name="DQNConvLstmTarget")

            # initialize Experience Replay memory buffer
            self.memory = SequenceMemory(max_memory, self.seq_length)
            self.batch_size = batch_size

        print("Done.")

        self.last_state = None
        self.last_action = None

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
            self.current_episode = Episode()

            episode = self.network.global_episode.eval(session=self.sess)
            print("Global training episode:", episode + 1)

    def step(self, obs):
        """If no units selected, selects army, otherwise move."""
        self.steps += 1
        self.reward += obs.reward

        # handle end of episode if terminal step
        if self.training and obs.step_type == 2:
            self._handle_episode_end()

        if FUNCTIONS.Move_screen.id in obs.observation.available_actions:
            state = obs.observation.feature_screen.player_relative

            if self.training:
                # predict an action to take and take it
                x, y, action = self._epsilon_greedy_action_selection(state)

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
                    self.current_episode.add(
                         self.last_state,
                         self.last_action,
                         obs.reward,
                         state)

                self.last_state = state
                self.last_action = np.ravel_multi_index(
                    (x, y),
                    feature_screen_size)

            else:
                x, y, action = self._epsilon_greedy_action_selection(
                    screen,
                    self.epsilon_min)

            if self.indicate_nonrandom_action and action == "nonrandom":
                # cosmetic difference between random and Q based actions
                return FUNCTIONS.Attack_screen("now", (x, y))
            else:
                return FUNCTIONS.Move_screen("now", (x, y))
        else:
            return FUNCTIONS.select_army("select")

    def _handle_episode_end(self):
        """Save weights and write summaries."""
        # increment global training episode
        self.network.increment_global_episode_op(self.sess)

        # save current model
        self.network.save_model(self.sess)
        print("Model Saved")

        # write summaries from last episode
        states, actions, targets = self._get_batch()
        self.network.write_summary(
            self.sess, states, actions, targets, self.reward)
        print("Summary Written")

    def _tf_init_op(self):
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

    def _update_target_network(self):
        online_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, "ConvLstm")
        target_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, "DQNConvLstmTarget")

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

            return x, y, "random"

        else:
            inputs = np.expand_dims(state, 0)
            q_values = self.sess.run(
                    self.network.outputs,
                    feed_dict={self.network.inputs: inputs})

            max_index = np.argmax(q_values)
            x, y = np.unravel_index(max_index, feature_screen_size)
            return x, y, "nonrandom"

    def _train_network(self):
        states, actions, targets = self._get_batch()
        self.network.optimizer_op(self.sess, states, actions, targets)

    def _get_batch(self):
        batch = self.memory.sample(self.batch_size)
        states = np.array([each[0] for each in batch])
        actions = np.array([each[1] for each in batch])
        rewards = np.array([each[2] for each in batch])
        next_states = np.array([each[3] for each in batch])

        # one-hot encode actions
        actions = np.eye(np.prod(feature_screen_size))[actions]

        # get targets
        next_outputs = self.sess.run(
            self.target_net.outputs,
            feed_dict={self.target_net.inputs: next_states})

        targets = [rewards[i] + self.discount_factor * np.max(next_outputs[i])
                   for i in range(self.batch_size)]

        return states, actions, targets
