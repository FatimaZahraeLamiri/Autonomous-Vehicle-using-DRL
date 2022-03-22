import random
from collections import deque
import numpy as np
import cv2
import time
import tensorflow as tf
import tensorflow.python.keras.backend as backend
from tensorflow.keras.models import load_model
from DQNagent import CarEnv, MEMORY_FRACTION
import random

MODEL_PATH = 'models/Mod64x4___-78.00max_-190.20avg_-269.00min__1647910202.model'

#I am using this function to print model details for debugging purposes
def print_total_parameters():
    total_parameters = 0 
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parameters = 1 
        for dim in shape:
            variable_parameters *= dim.value
        print('%s  dim=%i shape=%s params=%i' % ( 
                    variable.name,
                    len(shape),
                    shape,
                    variable_parameters,
                    ))  
        total_parameters += variable_parameters
    print('total_parameters = %i' % (total_parameters))

if __name__ == '__main__':

    # Memory fraction
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=MEMORY_FRACTION)
    backend.set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))

    # Load the model
    model = load_model(MODEL_PATH)
    # print_total_parameters()
    # Create environment
    env = CarEnv()

    # For agent speed measurements - keeps last 60 frametimes
    fps_counter = deque(maxlen=60)

    # Initialize predictions - first prediction takes longer as of initialization that has to be done
    # It's better to do a first prediction then before we start iterating over episode steps
    model.predict(np.ones((1, env.im_height, env.im_width, 3)))

    # Loop over episodes
    while True:

        print('Restarting episode')

        # Reset environment and get initial state
        current_state = env.reset()
        env.collision_hist = []

        done = False

        # Loop over steps
        while True:

            # For FPS counter
            step_start = time.time()

            # Show current frame
            cv2.imshow(f'Agent - preview', current_state)
            cv2.waitKey(1)

            # Predict an action based on current observation space
            qs = model.predict(np.array(current_state).reshape(-1, *current_state.shape)/255)[0]
            # print(len(qs))
            qs = [x * 100 for x in qs]
            winner = np.argwhere(qs == np.amax(qs))
            action = int(random.choice(winner)[0])

            # Step environment (additional flag informs environment to not break an episode by time limit)
            new_state, reward, done, _ = env.step(action)

            # Set current step for next loop iteration
            current_state = new_state

            # If done - agent crashed, break an episode
            if done:
                break

            # Measure step time, append to a deque, then print mean FPS for last 60 frames, q values and taken action
            frame_time = time.time() - step_start
            fps_counter.append(frame_time)
            print(f'Agent: {len(fps_counter)/sum(fps_counter):>4.3f} FPS | Action: [{qs[0]:>5.2f}, {qs[1]:>5.2f}, {qs[2]:>5.2f}, {qs[3]:>5.2f}, {qs[4]:>5.2f}, {qs[5]:>5.2f},{ qs[6]:>5.2f}, {qs[7]:>5.2f}, {qs[8]:>5.2f}, {qs[9]:>5.2f}, {qs[10]:>5.2f}, {qs[11]:>5.2f}, {qs[12]:>5.2f}, {qs[13]:>5.2f}, {qs[14]:>5.2f}] {action}')

        # Destroy an actor at end of episode
        for actor in env.actor_list:
            actor.destroy()