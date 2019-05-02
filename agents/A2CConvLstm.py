"""Actor-critic agents."""
import numpy as np
import os
import tensorflow as tf

# local submodule
import agents.networks.policy_value_estimators as nets

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
functions_mask = np.zeros(len(FUNCTIONS))
allowed_funcs = [12]
functions_mask[allowed_funcs] = 1


FUNCTION_TYPES = sc2_actions.FUNCTION_TYPES
FunctionCall = sc2_actions.FunctionCall

# manually state the argument types which take points on screen/minimap
SCREEN_TYPES = [sc2_actions.TYPES[0], sc2_actions.TYPES[2]]
MINIMAP_TYPES = [sc2_actions.TYPES[1]]


class A2CConvLstmAgent(base_agent.BaseAgent):
    """Synchronous version of DeepMind baseline Advantage actor-critic."""

    def __init__(self,
                 learning_rate=FLAGS.learning_rate,
                 value_gradient_strength=FLAGS.value_gradient_strength,
                 regularization_strength=FLAGS.regularization_strength,
                 discount_factor=FLAGS.discount_factor,
                 trajectory_training_steps=FLAGS.trajectory_training_steps,
                 training=FLAGS.training,
                 save_dir="./checkpoints/",
                 ckpt_name="A2CConvLstm",
                 summary_path="./tensorboard/A2CConvLstm"):
        """Initialize rewards/episodes/steps, build network."""
        super(A2CConvLstmAgent, self).__init__()

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
        self.trajectory_training_steps = trajectory_training_steps

        # other parameters
        self.training = training

        # build network
        self.save_path = save_dir + ckpt_name + ".ckpt"
        print("Building models...")
        tf.reset_default_graph()
        self.network = nets.ConvLstmNet(
            screen_dimensions=feature_screen_size,
            learning_rate=learning_rate,
            value_gradient_strength=value_gradient_strength,
            save_path=self.save_path,
            summary_path=summary_path)

        print("Done.")

        # initialize session
        self.sess = tf.Session()
        if os.path.isfile(self.save_path + ".index"):
            self.network.load(self.sess)
        else:
            self._tf_init_op()

    def reset(self):
        """Handle the beginning of new episodes."""
        self.episodes += 1
        self.steps = 0
        self.reward = 0
        self.lstm_state = (np.zeros((1, 256)), np.zeros((1, 256)))

        if self.training:
            self.last_action = None
            self.state_buffer = deque(maxlen=self.trajectory_training_steps)
            self.action_buffer = deque(maxlen=self.trajectory_training_steps)
            self.reward_buffer = deque(maxlen=self.trajectory_training_steps)
            self.glob_ep = self.network.global_episode.eval(session=self.sess)
            self.lstm_state_buffer = deque(maxlen=self.trajectory_training_steps)
            print("Global training episode:", self.glob_ep + 1)

    def step(self, obs):
        """If no units selected, selects army, otherwise move."""
        self.steps += 1
        self.reward += obs.reward

        # handle end of episode if terminal step
        if self.training and obs.step_type == 2:
            self._handle_episode_end()

        # get observations of state
        observation = obs.observation
        # expand so they form a batch of 1
        screen_features = observation.feature_screen.player_relative
        available_actions = observation.available_actions


        # sample action (function identifier and arguments) from policy
        args, lstm_next_state = self._sample_action(
            screen_features,
            available_actions)


        if (self.steps == 1):
            return FUNCTIONS.select_army("select")

        # train model
        if self.training:
            if self.last_action:
                # most recent steps on the left of the deques
                self.state_buffer.appendleft((screen_features))
                self.action_buffer.appendleft(self.last_action)
                self.reward_buffer.appendleft(obs.reward)
                self.lstm_state_buffer.appendleft(lstm_next_state)

            # cut trajectory and train model
            if self.steps % self.trajectory_training_steps == 0:
                self._train_network()

            self.last_action = args
            self.lstm_state = lstm_next_state

        return FUNCTIONS.Attack_screen("now", args)

    def _sample_action(self,
                       screen_features,
                       available_actions):

        """Sample actions and arguments from policy output layers."""
        screen_features = np.expand_dims(screen_features, 0)

        action_mask = np.zeros(len(FUNCTIONS), dtype=np.int32)
        for i in range(len(available_actions)):
            if (available_actions[i] and functions_mask[i]):
                action_mask[i] = 1
            else:
                action_mask[i] = 0

        feed_dict = {self.network.screen_features: screen_features,
                    self.network.c : self.lstm_state[0],
                     self.network.h : self.lstm_state[1]}

        # sample function identifier
        action_id = 12

        # sample function arguments

        x_policy, y_policy, lstm_next_state = self.sess.run(
                    [self.network.x, self.network.y, self.network.lstm_next_state],
                    feed_dict=feed_dict)

        x_policy = np.squeeze(x_policy)
        x_ids = np.arange(len(x_policy))
        x = np.random.choice(x_ids, p=x_policy)

        y_policy = np.squeeze(y_policy)
        y_ids = np.arange(len(y_policy))
        y = np.random.choice(y_ids, p=y_policy)
        args = (x, y)



        #print("CHOSEN ACTION: ", str(action_id), str(args))
        return args, lstm_next_state

    def _handle_episode_end(self):
        """Save weights and write summaries."""
        # train network
        feed_dict = self._train_network(terminal=True)

        # increment global training episode
        self.network.increment_global_episode_op(self.sess)

        # save current model
        self.network.save_model(self.sess)
        print("Model Saved")

        # write summaries from last episode
        self.network.write_summary(
            self.sess, self.glob_ep, self.reward, feed_dict)
        print("Summary Written")

    def _tf_init_op(self):
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

    def _train_network(self, terminal=False):
        feed_dict = self._get_batch(terminal)
        self.network.optimizer_op(self.sess, feed_dict)
        return feed_dict

    def _get_batch(self, terminal):
        # state
        screen = [each for each in self.state_buffer]

        # actions and arguments
        actions = [each[0] for each in self.action_buffer]
        actions = np.eye(len(FUNCTIONS))[actions]  # one-hot encode actions

        lstm_states = [i for i in self.lstm_state_buffer]

        args = self.action_buffer
        x = np.eye(84)[[a[0] for a in args]]
        y = np.eye(84)[[a[1] for a in args]]

        # calculate discounted rewards
        raw_rewards = list(self.reward_buffer)
        if terminal:
            value = 0
        else:
            value = np.squeeze(self.sess.run(
                self.network.value_estimate,
                feed_dict={self.network.screen_features: screen[-1:],
                self.network.c : lstm_states[-1][0],
                self.network.h : lstm_states[-1][1]}))

        returns = []
        # n-step discounted rewards from 1 < n < trajectory_training_steps
        for i, reward in enumerate(raw_rewards):
            value = reward + self.discount_factor * value
            returns.append(value)

        feed_dict = {self.network.screen_features: screen,
                     self.network.actions: actions,
                     self.network.returns: returns,
                     self.network.arg_placeholder_x : x,
                     self.network.arg_placeholder_y : y,
                     self.network.c : [i[0].reshape((256))  for i in lstm_states],
                     self.network.h : [i[1].reshape((256)) for i in lstm_states]}

        return feed_dict
