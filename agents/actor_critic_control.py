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

_PLAYER_SELF = features.PlayerRelative.SELF
_PLAYER_NEUTRAL = features.PlayerRelative.NEUTRAL  # beacon/minerals
_PLAYER_ENEMY = features.PlayerRelative.ENEMY

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


class A2CAtariControl(base_agent.BaseAgent):
    """Synchronous version of DeepMind baseline Advantage actor-critic."""

    def __init__(self,
                 learning_rate=FLAGS.learning_rate,
                 value_gradient_strength=FLAGS.value_gradient_strength,
                 discount_factor=FLAGS.discount_factor,
                 trajectory_training_steps=FLAGS.trajectory_training_steps,
                 training=FLAGS.training,
                 save_dir="./checkpoints/",
                 ckpt_name="A2CAtariControl",
                 summary_path="./tensorboard/A2CAtariControl"):
        """Initialize rewards/episodes/steps, build network."""
        super(A2CAtariControl, self).__init__()

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
        self.network = nets.AtariControlNet(
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
        self.count = 0
        self._marine_selected = False
        self.args_to_use = 0

        if self.training:
            self.last_action = None
            self.state_buffer = deque(maxlen=self.trajectory_training_steps)
            self.action_buffer = deque(maxlen=self.trajectory_training_steps)
            self.reward_buffer = deque(maxlen=self.trajectory_training_steps)
            self.glob_ep = self.network.global_episode.eval(session=self.sess)
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
        args1, args2 = self._sample_action(
            screen_features,
            available_actions)

        # train model
        if self.training:
            if self.last_action:
                # most recent steps on the left of the deques
                self.state_buffer.appendleft((screen_features))
                self.action_buffer.appendleft(self.last_action)
                self.reward_buffer.appendleft(obs.reward)

            # cut trajectory and train model
            if self.steps % self.trajectory_training_steps == 0:
                self._train_network()

            self.last_action = (args1, args2)


        marines = [unit for unit in obs.observation.feature_units
               if unit.alliance == _PLAYER_SELF]
        if not marines:
            return FUNCTIONS.no_op()
        marine_unit = next((m for m in marines if m.is_selected == self._marine_selected), marines[0])
        marine_xy = [marine_unit.x, marine_unit.y]

        if not marine_unit.is_selected:
        # Nothing selected or the wrong marine is selected.
            self._marine_selected = True
            return FUNCTIONS.select_point("select", marine_xy)

        self._marine_selected = False
        if (self.args_to_use == 1):
            self.args_to_use = 1 - self.args_to_use
            return FUNCTIONS.Move_screen("now", args1)
        else:
            self.args_to_use = 1 - self.args_to_use
            return FUNCTIONS.Move_screen("now", args2)

    def _sample_action(self,
                       screen_features,
                       available_actions):

        """Sample actions and arguments from policy output layers."""
        screen_features = np.expand_dims(screen_features, 0)

        feed_dict = {self.network.screen_features: screen_features}

        # sample function arguments

        x1_policy = self.sess.run(
                    self.network.x1,
                    feed_dict=feed_dict)

        y1_policy = self.sess.run(
                    self.network.y1,
                    feed_dict=feed_dict)

        x1_policy = np.squeeze(x1_policy)
        x1_ids = np.arange(len(x1_policy))
        x1 = np.random.choice(x1_ids, p=x1_policy)

        y1_policy = np.squeeze(y1_policy)
        y1_ids = np.arange(len(y1_policy))
        y1 = np.random.choice(y1_ids, p=y1_policy)
        args1 = (x1, y1)



        x2_policy = self.sess.run(
                    self.network.x2,
                    feed_dict=feed_dict)

        y2_policy = self.sess.run(
                    self.network.y2,
                    feed_dict=feed_dict)

        x2_policy = np.squeeze(x2_policy)
        x2_ids = np.arange(len(x2_policy))
        x2 = np.random.choice(x2_ids, p=x2_policy)

        y2_policy = np.squeeze(y2_policy)
        y2_ids = np.arange(len(y2_policy))
        y2 = np.random.choice(y2_ids, p=y2_policy)
        args2 = (x2, y2)


        #print("CHOSEN ACTION: ", str(action_id), str(args))
        return args1, args2

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

        args1 = [each[0] for each in self.action_buffer]
        args2 = [each[1] for each in self.action_buffer]
        x1 = np.eye(84)[[a[0] for a in args1]]
        y1 = np.eye(84)[[a[1] for a in args1]]

        x2 = np.eye(84)[[a[0] for a in args2]]
        y2 = np.eye(84)[[a[1] for a in args2]]

        # calculate discounted rewards
        raw_rewards = list(self.reward_buffer)
        if terminal:
            value = 0
        else:
            value = np.squeeze(self.sess.run(
                self.network.value_estimate,
                feed_dict={self.network.screen_features: screen[-1:]}))

        returns = []
        # n-step discounted rewards from 1 < n < trajectory_training_steps
        for i, reward in enumerate(raw_rewards):
            value = reward + self.discount_factor * value
            returns.append(value)

        feed_dict = {self.network.screen_features: screen,
                     self.network.returns: returns,
                     self.network.arg_placeholder_x1 : x1,
                     self.network.arg_placeholder_y1 : y1,
                     self.network.arg_placeholder_x2 : x2,
                     self.network.arg_placeholder_y2 : y2}

        return feed_dict
