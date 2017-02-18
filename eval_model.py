import os.path
import argparse
import numpy as np
import random
import pickle
import tensorflow as tf
from flat_game import carmunk

# Constant passed to functions to indicate whether call is for red team or agent
RED_TEAM_AGENT = True


def parse_arguments():
    parser = argparse.ArgumentParser(description='Parse Command Line Arguments')
    # Paths
    parser.add_argument('--LOGS_DIR', default='./logs')
    parser.add_argument('--MODELS_DIR', default='./models')
    # Model Parameters
    parser.add_argument('--HIDDEN1', type=int, default=164)
    parser.add_argument('--HIDDEN2', type=int, default=150)
    # Training Parameters
    parser.add_argument('--OPTIMIZER', default='Adam')
    parser.add_argument('--LEARNING_RATE', type=float, default=.01)
    parser.add_argument('--EPSILON', type=float, default=1)
    parser.add_argument('--GAMMA', type=float, default=0.9)
    parser.add_argument('--CHECKPOINT_STEP', type=int, default=5000)
    parser.add_argument('--SUMMARY_STEP', type=int, default=5000)
    parser.add_argument('--OBSERVE', type=int, default=10000)
    parser.add_argument('--TRAIN_FRAMES', type=int, default=50000)
    parser.add_argument('--BATCH_SIZE', type=int, default=100)
    parser.add_argument('--BUFFER_SIZE', type=int, default=50000)
    parser.add_argument('--USE_RED_TEAM', action='store_true', default=False)
    parser.add_argument('--TRAIN_RED_TEAM', action='store_true', default=False)
    # Game Parameters
    parser.add_argument('--CAR_SENSORS', type=int, default=6)
    parser.add_argument('--CAT_SENSORS', type=int, default=12)
    parser.add_argument('--NUM_ACTIONS', type=int, default=3)
    parser.add_argument('--CAR_CRASH_PENALTY', type=int, default=-500)
    parser.add_argument('--CAT_CRASH_PENALTY', type=int, default=-500)
    parser.add_argument('--CAT_SUCCESS_REWARD', type=int, default=1000)
    parser.add_argument('--USE_OBSTACLES', action='store_true', default=False)
    parser.add_argument('--DRAW_SCREEN', action='store_true', default=False)
    parser.add_argument('--SHOW_SENSORS', action='store_true', default=False)
    parser.add_argument('--REWARD_TYPE', default='MinDistance')

    return parser.parse_args()

args = parse_arguments()
for arg in vars(args):
    print(arg, getattr(args, arg))
print('\n')


class Stats():
    def __init__(self):
        self.red_team_crashes = []
        self.obstacle_crashes = []
        self.crashes = []


def get_game_params():
    game_params = {
        # Screen graphics
        'show_sensors': args.SHOW_SENSORS,
        'draw_screen': args.DRAW_SCREEN,
        # Objects to be added in game
        'use_red_team': args.USE_RED_TEAM,
        'trained_red_team': args.TRAIN_RED_TEAM,
        'use_obstacles': args.USE_OBSTACLES,
        # Rewards and penalties
        'car_crash_penalty': args.CAR_CRASH_PENALTY,
        'cat_crash_penalty': args.CAT_CRASH_PENALTY,
        'cat_success_reward': args.CAT_SUCCESS_REWARD,
        'reward_type': args.REWARD_TYPE
    }
    return game_params


# Taken from TensorFlow MNIST Tutorial
# https://goo.gl/gkZs36

# We can't initialize these variables to 0 - the network will get stuck.
def weight_variable(shape):
    """Create a weight variable with appropriate initialization."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """Create a bias variable with appropriate initialization."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def variable_summaries(var, prefix):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope(prefix + 'summaries'):
        mean = tf.reduce_mean(var)
        tf.scalar_summary(prefix + 'mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.scalar_summary(prefix + 'stddev', stddev)
        tf.scalar_summary(prefix + 'max', tf.reduce_max(var))
        tf.scalar_summary(prefix + 'min', tf.reduce_min(var))
        tf.histogram_summary(prefix + 'histogram', var)


def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    """Reusable code for making a simple neural net layer.
    It does a matrix multiply, bias add, and then uses relu to nonlinearize.
    It also sets up name scoping so that the resultant graph is easy to read,
    and adds a number of summary ops.
    """
    # Adding a name scope ensures logical grouping of the layers in the graph.
    with tf.name_scope(layer_name):
        # This Variable will hold the state of the weights for the layer
        with tf.name_scope('weights'):
            weights = weight_variable([input_dim, output_dim])
            variable_summaries(weights, layer_name + 'weights')
        with tf.name_scope('biases'):
            biases = bias_variable([output_dim])
            variable_summaries(biases, layer_name + 'biases')
        with tf.name_scope('Wx_plus_b'):
            preactivate = tf.matmul(input_tensor, weights) + biases
            tf.histogram_summary(layer_name + 'pre_activations', preactivate)
        if act is not None:
            activations = act(preactivate, name='activation')
            tf.histogram_summary(layer_name + 'activations', activations)
            return activations
        return preactivate


def get_net(x, q_action, action, input_size, prefix):
    # Model
    with tf.name_scope(prefix + 'Model'):
        l1 = nn_layer(x, input_size, args.HIDDEN1, prefix + 'L1')
        print(l1)
        l2 = nn_layer(l1, args.HIDDEN1, args.HIDDEN2, prefix + 'L2')
        print(l2)
        q_pred = nn_layer(l2, args.HIDDEN2, args.NUM_ACTIONS, prefix + 'QPred', act=None)
        print(q_pred)

    one_hot_action = tf.one_hot(action, depth=args.NUM_ACTIONS, name=prefix + 'OneHotAction')
    print(one_hot_action)

    q_action_pred = tf.reduce_sum(q_pred * one_hot_action, 1, keep_dims=True,
                                  name=prefix + 'PredictedQVal')
    print(q_action_pred)

    # Loss
    with tf.name_scope(prefix + 'Loss'):
        q_action_diff = tf.sub(q_action, q_action_pred , name=prefix + 'QActionDiff')
        print(q_action_diff)
        loss = tf.reduce_sum(tf.square(q_action_diff, name=prefix + 'SquaredLoss'),
                             name=prefix + 'BatchLoss')
        print(loss)

    # Optimizer
    with tf.name_scope(prefix + 'Optimizer'):
        if args.OPTIMIZER == 'Adam':
            optimizer = tf.train.AdamOptimizer(args.LEARNING_RATE).minimize(loss)
        elif args.OPTIMIZER == 'RMSProp':
            optimizer = tf.train.RMSPropOptimizer(args.LEARNING_RATE).minimize(loss)
        else:
            raise Exception('Unsupported Optimizer')

    # Summaries
    # Will this cause a problem if this function is called multiple times ?
    with tf.name_scope(prefix + 'Summaries'):
        # Create a summary to monitor cost tensor
        tf.scalar_summary(prefix + "loss", loss)
        # Merge all summaries into a single op
        summary = tf.merge_all_summaries()

    net = {
        'model': q_pred,
        'optimizer': optimizer,
        'loss': loss,
        'summary': summary
    }
    return net


# Mostly adopted from learning.py
def get_random_action():
    return np.random.randint(0, args.NUM_ACTIONS)  # random


def get_epsilon_greedy_action(state, model, red_team):
    # Choose an action.
    if random.random() < args.EPSILON:
        return get_random_action()
    else:
        # Get Q values for each action.
        if red_team:
            qval = sess.run(model, feed_dict={cat_x: state.reshape(1, len(state))})
        else:
            qval = sess.run(model, feed_dict={car_x: state.reshape(1, len(state))})
        return np.argmax(qval)


def get_action(t, state, model, red_team):
    if red_team and not args.TRAIN_RED_TEAM:
        return None
    if t < args.OBSERVE:
        action = get_random_action()
    else:
        action = get_epsilon_greedy_action(state, model, red_team)
    return action


def train_batch(t, replay, net, red_team):
    # If we're done observing, start training.
    if t > args.OBSERVE:
        # If we've stored enough in our buffer, pop the oldest.
        if len(replay) > args.BUFFER_SIZE:
            replay.pop(0)
        X_train, y_train, train_actions = get_minibatch(replay, net['model'], red_team)
        # Train the model on this batch.
        if red_team:
            # print('Calling train batch for red team')
            sess.run(net['optimizer'],
                     feed_dict={cat_x: X_train, cat_q: y_train, cat_action: train_actions})
        else:
            # print('Calling train batch for agent')
            sess.run(net['optimizer'],
                     feed_dict={car_x: X_train, car_q: y_train, car_action: train_actions})
        if t % args.SUMMARY_STEP == 0:
            # summary_writer.add_summary(s, t)
            s = sess.run(net['summary'],
                         feed_dict={car_x: X_train, car_q: y_train, car_action: train_actions})
            summary_writer.add_summary(s, t)


def get_minibatch(replay, model, red_team):
    # Randomly sample our experience replay memory
    minibatch = random.sample(replay, args.BATCH_SIZE)
    # Get training values.
    return process_minibatch(minibatch, model, red_team)


def process_minibatch(minibatch, model, red_team):
    X_train, y_train, y_train_actions = [], [], []
    # Loop through our batch and create arrays for X and y
    # so that we can fit our model at every step.
    for record in minibatch:
        # Get stored values.
        old_state, action, reward, new_state = record
        # Get prediction on new state.
        if red_team:
            new_qval = sess.run(model, feed_dict={cat_x: new_state.reshape(1, len(new_state))})
        else:
            new_qval = sess.run(model, feed_dict={car_x: new_state.reshape(1, len(new_state))})
        # Get best move for new state.
        max_qval = np.max(new_qval)
        update = get_update(reward, max_qval, red_team)
        X_train.append(old_state)
        y_train.append(update)
        y_train_actions.append(action)


    X_train = np.array(X_train)
    y_train = np.array(y_train).reshape(len(y_train), 1)
    y_train_actions = np.array(y_train_actions).reshape(len(y_train_actions), 1)

    # print('Red Team: ', red_team)
    # print(X_train.shape, y_train.shape, y_train_actions.shape)
    return X_train, y_train, y_train_actions


def get_update(reward, max_qval, red_team=False):
    is_terminal_state = False
    if red_team:
        if reward == args.CAT_CRASH_PENALTY or reward == args.CAT_SUCCESS_REWARD:
            is_terminal_state = True
    else:
        if reward == args.CAR_CRASH_PENALTY:
            is_terminal_state = True
    # Update depending on whether state is terminal or not.
    if is_terminal_state:
        update = (reward + (args.GAMMA * max_qval))
    else:  # terminal state
        update = reward
    return update


def get_experiment_str():
    if args.TRAIN_RED_TEAM:
        agent_type = 'red_team'
    elif args.USE_RED_TEAM:
        agent_type = 'random'
    else:
        agent_type = 'agent'
    return '{0}_{1}_{2}_{3}_{4}'.format(args.REWARD_TYPE, agent_type, args.OPTIMIZER,
                                        args.LEARNING_RATE, args.BATCH_SIZE)


def get_experiment_dir(base_dir):
    directory = os.path.join(base_dir, get_experiment_str())
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory


def test_model():
    stats = Stats()
    car_distance = 0
    game_state = carmunk.GameState(get_game_params())

    # Get initial state by doing nothing and getting the state.
    car_state, car_reward, cat_state, cat_reward = game_state.frame_step(
        get_random_action(), get_random_action())
    cat_test_action = None

    for t in range(20000):
        if t % 500 == 0:
            print('Finished iteration: {0}. Crashes: {1}'.format(t, len(stats.crashes)))
        car_distance += 1

        # Choose action.
        qval = sess.run(car_net["model"], feed_dict={car_x: car_state.reshape(1, len(car_state))})
        car_test_action = np.argmax(qval)
        # For evaluating random agent comment above 2 lines and uncomment the one below
        # car_test_action = get_random_action()

        if args.USE_RED_TEAM:
            if args.TRAIN_RED_TEAM:
                qval = sess.run(cat_net["model"], feed_dict={cat_x: cat_state.reshape(1, len(cat_state))})
                cat_test_action = np.argmax(qval)
            else:
                cat_test_action = get_random_action()

        car_state, car_reward, cat_state, cat_reward = game_state.frame_step(
            car_test_action, cat_test_action)

        if car_reward == args.CAR_CRASH_PENALTY:
            stats.crashes.append(t)
            if cat_reward is not None and cat_reward == args.CAT_SUCCESS_REWARD:
                stats.red_team_crashes.append(t)
            else:
                stats.obstacle_crashes.append(t)

    print('Crashes: {0}, Obstacle Crashes: {1}, Red Team Crashes: {2}'
          .format(len(stats.crashes), len(stats.obstacle_crashes), len(stats.red_team_crashes)))

tf.reset_default_graph()

global_step = tf.Variable(0)

# Op to write logs to Tensorboard
summary_writer = tf.train.SummaryWriter(get_experiment_dir(args.LOGS_DIR), graph=tf.get_default_graph())

# Input state
car_x = tf.placeholder(tf.float32, shape=[None, args.CAR_SENSORS], name='CarX')
# Reward + GAMMA * QMAX[next_state]
car_q = tf.placeholder(tf.float32, shape=[None, 1], name='CarQ')
# Action taken in input state
car_action = tf.placeholder(tf.int32, shape=[None, 1], name='CarAction')
car_net = get_net(car_x, car_q, car_action, args.CAR_SENSORS, 'CarNet')

# Saver for model check pointing.
saver = tf.train.Saver()

cat_net = None
if args.TRAIN_RED_TEAM:
    # Input state
    cat_x = tf.placeholder(tf.float32, shape=[None, args.CAT_SENSORS], name='CatX')
    # Reward + GAMMA * QMAX[next_state]
    cat_q = tf.placeholder(tf.float32, shape=[None, 1], name='CatQ')
    cat_action = tf.placeholder(tf.int32, shape=[None, 1], name='CatAction')
    cat_net = get_net(cat_x, cat_q, cat_action, args.CAT_SENSORS, 'CatNet')

with tf.Session() as sess:
    init = tf.initialize_all_variables()
    sess.run(init)
    # model_file = os.path.join(args.MODELS_DIR + '/SumDistance_agent_Adam_0.001_100', 'model_75000.ckpt')
    # model_file = os.path.join(args.MODELS_DIR + '/SumDistance_red_team_Adam_0.001_100', 'model_65000.ckpt')
    # model_file = os.path.join(args.MODELS_DIR + '/MinDistance_red_team_Adam_0.001_100', 'model_90000.ckpt')
    # model_file = os.path.join(args.MODELS_DIR + '/SumDistance_random_Adam_0.001_100', 'model_150000.ckpt')
    model_file = os.path.join(args.MODELS_DIR + '/SumDistance_red_team_Adam_0.001_100_small_screen', 'model_105000.ckpt')
    saver.restore(sess, model_file)
    print("Model loaded")
    test_model()

    summary_writer.flush()
    summary_writer.close()
