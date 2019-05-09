"""Actor-critic agents."""
import numpy as np
import os
import tensorflow as tf

# local submodule
import agents.networks.value_estimators as nets

from absl import flags

from collections import deque

from pysc2.agents import base_agent
from pysc2.lib import actions as sc2_actions
from pysc2.lib import features

FLAGS = flags.FLAGS

# agent interface settings (defaults specified in pysc2.bin.agent)
feature_screen_size = FLAGS.feature_screen_size
feature_minimap_size = FLAGS.feature_minimap_size

_PLAYER_SELF = features.PlayerRelative.SELF
_PLAYER_NEUTRAL = features.PlayerRelative.NEUTRAL  # beacon/minerals
_PLAYER_ENEMY = features.PlayerRelative.ENEMY

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


class DQNSeperate(base_agent.BaseAgent):
    """Synchronous version of DeepMind baseline Advantage actor-critic."""

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
                 ckpt_name="dqn_seperate",
                 summary_path="./tensorboard/dqnsep"):
        """Initialize rewards/episodes/steps, build network."""
        super(DQNSeperate, self).__init__()

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
        self.network = nets.PlayerRelativeMovementCNNSeparateOutputs(
            spatial_dimensions=feature_screen_size,
            learning_rate=self.learning_rate,
            save_path=self.save_path,
            summary_path=summary_path)

        if self.training:
            self.target_net = nets.PlayerRelativeMovementCNNSeparateOutputs(
                spatial_dimensions=feature_screen_size,
                learning_rate=self.learning_rate,
                name="DQNSepTarget")

            # initialize Experience Replay memory buffer
            self.memory = Memory(max_memory)
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
        self.steps = 0
        self.reward1 = 0
        self.reward2 = 0
        self.count = 0
        self._marine_selected = False
        self.args_to_use = 0
        self.minerals = None

        if self.training:
            self.last_state = None
            self.last_action = None

            episode = self.network.global_episode.eval(session=self.sess)
            print("Global training episode:", episode + 1)

    def attribute_rewards(obs):
        new_minerals = [[unit.x, unit.y] for unit in obs.observation.feature_units
                  if unit.alliance == _PLAYER_NEUTRAL]
        if (not self.minerals):
            self.minerals = new_minerals
            return

        marines = [unit for unit in obs.observation.feature_units
               if unit.alliance == _PLAYER_SELF]
        if not marines:
            return
        
        for min in self.minerals:
            if min not in new_minerals:
                # det which marine to give reward to
                x, y = min
                distance1 = np.linalg.norm(
            np.array(min) - np.array([marines[0].x, marines[0].y]))
            distance2 = np.linalg.norm(
            np.array(min) - np.array([marines[1].x, marines[1].y]))

            if (obs.reward > 1):
                print("REWARD WAS GREATER THAN ONE: ", obs.reward)
            if (distance1 < distance2):
                self.reward1 += obs.reward
            else:
                self.reward2 += obs.reward
        
        

    def step(self, obs):
        """If no units selected, selects army, otherwise move."""
        self.steps += 1

        if self.minerals:
            # Find the closest.
            attribute_rewards(obs)

        # handle end of episode if terminal step
        if self.training and obs.step_type == 2:
            self._handle_episode_end()

        # get observations of state
        state = obs.observation.feature_screen.player_relative
        # expand so they form a batch of 1

        if self.training:
            # predict an action to take and take it
            x, y, action = self._epsilon_greedy_action_selection(state, self.args_to_use)

            # update online DQN
            if (self.steps % self.train_frequency == 0 and
                        len(self.memory) > self.batch_size):
                self._train_network()

            # update network used to estimate TD targets
            if self.steps % self.target_update_frequency == 0:
                self._update_target_network()
                print("Target network updated.")
            
            if (self.args_to_use == 0):
                reward = self.reward1
                self.reward1 = 0
            else:
                reward = self.reward2
                self.reward2 = 0

            # add experience to memory
            if self.last_state is not None:
                self.memory.add(
                        (self.last_state,
                         self.last_action,
                         reward,
                         state,
                         self.args_to_use))

            self.last_state = state
            self.last_action = np.ravel_multi_index(
                    (x, y),
                    feature_screen_size)

        else:
            x, y, action = self._epsilon_greedy_action_selection(
                    state,
                    self.epsilon_min)

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
        self.args_to_use = 1 - self.args_to_use
        return FUNCTIONS.Move_screen("now", (x, y))



    def _epsilon_greedy_action_selection(self, state, marine, epsilon=None):
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
                self.network.output,
                feed_dict={self.network.inputs: inputs})

            max_index = np.argmax(q_values, axis = 1)
            print(max_index.shape)
            if (marine == 0):
                max_index = max_index[0]
            else:
                max_index = max_index[1]
            x, y = np.unravel_index(max_index, feature_screen_size)
            return x, y, "nonrandom"

    def _handle_episode_end(self):
        """Save weights and write summaries."""
        self.network.increment_global_episode_op(self.sess)

        # save current model
        self.network.save_model(self.sess)
        print("Model Saved")

        # write summaries from last episode
        states, actions, targets = self._get_batch()
        self.network.write_summary(
            self.sess, states, actions, targets, self.reward1 + self.reward2)
        print("Summary Written")

    def _tf_init_op(self):
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

    def _update_target_network(self):
        online_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, "DQN")
        target_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, "DQNSepTarget")

        update_op = []
        for online_var, target_var in zip(online_vars, target_vars):
            update_op.append(target_var.assign(online_var))

        self.sess.run(update_op)
    def _train_network(self):
        states, actions, targets, marine = self._get_batch()
        self.network.optimizer_op(self.sess, states, actions, targets, marine)

    def _get_batch(self):
        batch = self.memory.sample(self.batch_size)
        states = np.array([each[0] for each in batch])
        actions = np.array([each[1] for each in batch])
        marines = np.array([each[4] for each in batch])
        rewards = np.array([each[2] for each in batch])
        next_states = np.array([each[3] for each in batch])


        # one-hot encode actions
        actions = np.eye(np.prod(feature_screen_size))[actions]

        rewards = rewards[marines]

        # get targets
        next_outputs = self.sess.run(
            self.target_net.result,
            feed_dict={self.target_net.inputs: next_states,
            self.target_net.marines: 1 - marines})

        targets = [rewards[i] + self.discount_factor * np.max(next_outputs[i])
                   for i in range(self.batch_size)]

        return states, actions, targets, marines