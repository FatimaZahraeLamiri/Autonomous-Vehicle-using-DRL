import glob
import os
import sys
import random
import time
import numpy as np
import cv2
import math
from collections import deque
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.layers import  Dense, GlobalAveragePooling2D, Input, Concatenate, Conv2D, AveragePooling2D, Activation, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import  Sequential,Model
from tensorflow.keras.callbacks import TensorBoard
import tensorflow as tf
import tensorflow.python.keras.backend as backend
from threading import Thread
global sess
global graph
from tqdm import tqdm

try:
    sys.path.append(glob.glob('../PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla


SHOW_PREVIEW = False
IM_WIDTH = 640
IM_HEIGHT = 480
SECONDS_PER_EPISODE = 180
REPLAY_MEMORY_SIZE = 5_000
MIN_REPLAY_MEMORY_SIZE = 1_000
MINIBATCH_SIZE = 16
PREDICTION_BATCH_SIZE = 1
TRAINING_BATCH_SIZE = MINIBATCH_SIZE // 4
UPDATE_TARGET_EVERY = 5
MODEL_NAME = "Mod64x4"
MEMORY_FRACTION = 0.4
MIN_REWARD = -400
EPISODES = 1000
DISCOUNT = 0.9
epsilon = 1
EPSILON_DECAY = 0.995 #0.997  0.9975 99975
MIN_EPSILON = 0.001
AGGREGATE_STATS_EVERY = 10
COUNT_SUCCESS_EVERY = 100

# Tensorboard class
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
    def _write_logs(self, logs, index):
        for name, value in logs.items():
            if name in ['batch', 'size']:
                continue
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value
            summary_value.tag = name
            self.writer.add_summary(summary, index)
        self.writer.flush()



class CarEnv:
    SHOW_CAM = SHOW_PREVIEW
    im_width = IM_WIDTH
    im_height = IM_HEIGHT
    front_camera = None

    def __init__(self):
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(2000.0)
        self.world = self.client.get_world()
        self.world= self.client.load_world('town05')
        self.blueprint_library = self.world.get_blueprint_library()
        self.model_3 = self.blueprint_library.filter("model3")[0]

    def reset(self):
        self.collision_hist = []
        self.actor_list = []
        self.transform = random.choice(self.world.get_map().get_spawn_points())
        self.destination= random.choice(self.world.get_map().get_spawn_points())
        self.transform = random.choice(self.world.get_map().get_spawn_points())
        self.vehicle = self.world.spawn_actor(self.model_3, self.transform)
        self.initialDistance = self.vehicle.get_location().distance(self.destination.location) 
        self.actor_list.append(self.vehicle)
        self.rgb_cam = self.blueprint_library.find('sensor.camera.rgb')
        self.rgb_cam.set_attribute("image_size_x", f"{self.im_width}")
        self.rgb_cam.set_attribute("image_size_y", f"{self.im_height}")
        self.rgb_cam.set_attribute("fov", f"110")
        transform = carla.Transform(carla.Location(x=2.5, z=0.7))
        self.sensor = self.world.spawn_actor(self.rgb_cam, transform, attach_to=self.vehicle)
        self.actor_list.append(self.sensor)
        self.sensor.listen(lambda data: self.process_img(data))

        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        time.sleep(4)

        colsensor = self.blueprint_library.find("sensor.other.collision")
        self.colsensor = self.world.spawn_actor(colsensor, transform, attach_to=self.vehicle)
        self.actor_list.append(self.colsensor)
        self.colsensor.listen(lambda event: self.collision_data(event))
        points_list=self.world.get_map().get_spawn_points()
        
        for point in points_list:
            if self.vehicle.get_location().distance(point.location) <= 100:
                self.destination = point
                break
        

        self.initialDistance = int(self.vehicle.get_location().distance(self.destination.location))

        while self.front_camera is None:
            time.sleep(0.01)

        self.episode_start = time.time()
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))

        return self.front_camera

    def get_location_state(self):

        location_state = []

        veh_location = self.vehicle.get_transform().location
        veh_rotation = self.vehicle.get_transform().rotation
    
        location_state.append(veh_location.x)
        location_state.append(veh_location.y)
        location_state.append(veh_location.z)

     
        location_state.append(self.destination.location.x)
        location_state.append(self.destination.location.y)
        location_state.append(self.destination.location.z)

       
        location_state.append(veh_rotation.pitch)
        location_state.append(veh_rotation.yaw)
        location_state.append(veh_rotation.roll)
        
        result= np.array(location_state)

        return result.reshape(1, 9)

    def collision_data(self, event):
        self.collision_hist.append(event)

    def process_img(self, image):
        i = np.array(image.raw_data)
        i2 = i.reshape((self.im_height, self.im_width, 4))
        i3 = i2[:, :, :3]
        if self.SHOW_CAM:
            cv2.imshow("", i3)
            cv2.waitKey(1)
        self.front_camera = i3

    def step(self, action):
        if action == 0:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=-1))
        elif action == 1:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer= 0))
        elif action == 2:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=1))
        elif action == 3:
             self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=-0.5))
        elif action == 4:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0.5))
        elif action == 5:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.5, steer=1))
        elif action == 6:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.5, steer=0))
        elif action == 7:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.5, steer=-1))
        elif action == 8:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.5, steer=0.5))
        elif action == 9:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.5, steer=-0.5))
        elif action == 10:
            self.vehicle.apply_control(carla.VehicleControl(reverse=True ))
        elif action == 11:
            self.vehicle.apply_control(carla.VehicleControl(reverse=True , steer=-0.5))
        elif action == 12:
            self.vehicle.apply_control(carla.VehicleControl(reverse=True , steer= 0.5))
        elif action == 13:
            self.vehicle.apply_control(carla.VehicleControl(brake=1.0))
        elif action == 14:
            self.vehicle.apply_control(carla.VehicleControl(brake=0.5))
           

       #calculate the velocity of the vehicle in kmh 
        v = self.vehicle.get_velocity()
        kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))
      
        #calculate the distance left to the destination 
        distance_left = int(self.vehicle.get_location().distance(self.destination.location))
         
        #initialize rewards
        reward1 =  -1
        reward2 =  0
 
        #calculate new localisation state
        new_state = self.get_location_state() 
        success = False
        #calculate reawrd values
        if (kmh >= 30 and kmh <= 45):
            reward1 = 1

        if len(self.collision_hist) != 0:
            done = True
            reward1 = -200
            print("Collision", distance_left)

        elif distance_left < 1:
            print("SUCCESS")
            done = True
            success = True
            reward2 = 100

        else:
            done = False
            reward2 = - distance_left/self.initialDistance

        if self.episode_start + SECONDS_PER_EPISODE < time.time():
            print("Time Out")
            done = True

        return self.front_camera, new_state, reward1, reward2, done, success, None



class DQNAgent:
    def __init__(self):
        self.model = self.create_model()
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        self.model2 = self.create_MLP_model()
        self.target_model2= self.create_MLP_model()
        self.target_model2.set_weights(self.model2.get_weights())

        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/{MODEL_NAME}-{int(time.time())}")
        self.target_update_counter = 0

        self.graph = tf.get_default_graph()
        self.terminate = False
        self.last_logged_episode = 0
        self.training_initialized = False

    def create_model(self):
        model = Sequential()
        model.add(Conv2D(64, (3, 3), input_shape=(IM_HEIGHT, IM_WIDTH, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same'))

        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same'))

        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same'))
        model.add(Flatten())
        model.add(Dense(64))
        model.add(Dense(15, activation='linear'))
        model.compile(loss="mse", optimizer=Adam(lr=0.001, decay=0.0), metrics=['accuracy'])
        return model

    def create_MLP_model(self):

        model = Sequential()

        model.add(Dense(20, input_dim=9, kernel_initializer='he_uniform', activation='relu'))
        model.add(Dense(15))

        model.compile(loss='mae', optimizer='adam')

        return model

    def update_replay_memory(self, transition):
        # transition = (current_state, action, reward, new_state, done)
        self.replay_memory.append(transition)

    def train(self):
     
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)
        current_qs_list2 = []
        future_qs_list2 = []
        current_states = np.array([transition[0] for transition in minibatch])/255
        current_states2 = np.array([transition[1] for transition in minibatch])
        with self.graph.as_default():
            backend.set_session(sess)
            current_qs_list = self.model.predict(current_states, PREDICTION_BATCH_SIZE)
            for i  in range(16):
                current_qs_list2.append(self.model2.predict(current_states2[i])) 
        new_current_states = np.array([transition[5] for transition in minibatch])/255
        new_current_states2 =np.array([transition[6] for transition in minibatch])
        with self.graph.as_default():
            backend.set_session(sess)
            future_qs_list = self.target_model.predict(new_current_states, PREDICTION_BATCH_SIZE)
            for i  in  range(16):
                future_qs_list2.append(self.target_model2.predict(new_current_states2[i])) 
      
        X = []
        y = []

        Z = []
        W = []

        for index, (current_state, current_state2, action, reward1, reward2, new_state, new_state2, done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                max_future_q2 = np.max(future_qs_list2[index][0])
                new_q = reward1 + DISCOUNT * max_future_q
                new_q2 = reward2 + DISCOUNT* max_future_q2
            else:
                new_q = reward1
                new_q2 = reward2

            current_qs = current_qs_list[index]
            current_qs[action] = new_q
            

            current_qs2 = current_qs_list2[index][0]
            current_qs2[action] = new_q2

            X.append(current_state)
            y.append(current_qs)
            
            Z.append(current_state2)
            ll = []
            ll.append(current_qs2)
            W.append(ll)

        log_this_step = False
        if self.tensorboard.step > self.last_logged_episode:
            log_this_step = True
            self.last_logged_episode = self.tensorboard.step

        with self.graph.as_default():
            backend.set_session(sess)
            self.model.fit(np.array(X)/255, np.array(y), batch_size=TRAINING_BATCH_SIZE, verbose=0, shuffle=False, callbacks=[self.tensorboard] if log_this_step else None)
            for i in range(16):
                self.model2.fit(np.array(Z[i]), np.array(W[i]), verbose=0, shuffle=False, callbacks=[self.tensorboard] if log_this_step else None)
        if log_this_step:
            self.target_update_counter += 1

        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_model2.set_weights(self.model2.get_weights())
            self.target_update_counter = 0

    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape)/255)[0]
    
    def get_qs2(self, state2):
         return self.model2.predict(state2)[0]

    def train_in_loop(self):
        X = np.random.uniform(size=(1, IM_HEIGHT, IM_WIDTH, 3)).astype(np.float32)
        y = np.random.uniform(size=(1, 15)).astype(np.float32)
        
        Z = np.random.uniform(size=(1, 9)).astype(np.float32)
        W = np.random.uniform(size=(1, 15)).astype(np.float32)

        with self.graph.as_default():
            backend.set_session(sess)
            self.model.fit(X,y, verbose=False, batch_size=1)
            self.model2.fit(Z, W)

        self.training_initialized = True

        while True:
            if self.terminate:
                return
            self.train()
            time.sleep(0.01)



if __name__ == '__main__':
    FPS = 60
    # For stats
    ep_rewards = [-200]
    Reward1 = [-200]
    Reward2 =[-100]
    success_sum = 0
    # For more repetitive results
    random.seed(1)
    np.random.seed(1)
    tf.set_random_seed(1)

    # Memory fraction, used mostly when trai8ning multiple agents
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=MEMORY_FRACTION)
    tf_config=tf.ConfigProto(gpu_options=gpu_options)
    sess=tf.Session(config=tf_config)
    backend.set_session(sess)

    # Create models folder
    if not os.path.isdir('models'):
        os.makedirs('models')

    if not os.path.isdir('models2'):
        os.makedirs('models2')
    # Create agent and environment
    agent = DQNAgent()
    env = CarEnv()


    # Start training thread and wait for training to be initialized
    trainer_thread = Thread(target=agent.train_in_loop, daemon=True)
    trainer_thread.start()
    while not agent.training_initialized:
        time.sleep(0.01)

    # Initialize predictions
    agent.get_qs(np.ones((env.im_height, env.im_width, 3)))
    agent.get_qs2(np.ones((1,9), dtype=int))

    # Iterate over episodes
    for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):
        #try:
            env.collision_hist = []
            # Update tensorboard step every episode
            agent.tensorboard.step = episode
            # Restarting episode - reset episode reward and step number
            episode_reward = 0
            sum_r1 = 0
            sum_r2 = 0
            step = 1
            # Reset environment and get initial state
            current_state = env.reset()
            current_state2  = env.get_location_state()
            # Reset flag and start iterating until episode ends
            done = False
            # episode_start = time.time()
            # Play for given number of seconds only
            while True:
                # This part stays mostly the same, the change is to query a model for Q values
                if np.random.random() > epsilon:

                    Q_table1 = agent.get_qs(current_state)

                    Q_table2 = agent.get_qs2(current_state2)

                    AvgTables =  [(x+y)/2 for x,y in zip(Q_table1, Q_table2)]

                    action = np.argmax(AvgTables)
                else:
                    # Get random action
                    action = np.random.randint(0, 15)
                    time.sleep(1/FPS)

                new_state, new_state2, reward1, reward2, done, success, _  = env.step(action)

                # Transform new continous state to new discrete state and count reward
                episode_reward += reward1 + reward2
                sum_r1 += reward1
                sum_r2 += reward2
                # Every step we update replay memory
                agent.update_replay_memory((current_state, current_state2, action, reward1, reward2, new_state, new_state2, done))
                # agent.train(done, step)
                current_state = new_state
                current_state2 = new_state2
                step += 1
                if success:
                    success_sum += 1
                if done:
                    break

            # End of episode - destroy agents
            for actor in env.actor_list:
                actor.destroy()

            # Append episode reward to a list and log stats (every given number of episodes)
            Reward1.append(sum_r1)
            Reward2.append(sum_r2)
            ep_rewards.append(episode_reward)
            if not episode % AGGREGATE_STATS_EVERY or episode == 1:
                average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
                min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
                max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
                Reward1_Avg =sum(Reward1[-AGGREGATE_STATS_EVERY:])/len(Reward1[-AGGREGATE_STATS_EVERY:])
                Reward2_Avg =sum(Reward2[-AGGREGATE_STATS_EVERY:])/len(Reward2[-AGGREGATE_STATS_EVERY:])
                agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward,Reward1 = Reward1_Avg, Reward2 = Reward2_Avg, epsilon=epsilon)

                # Save model, but only when min reward is greater or equal a set value
                if min_reward >= MIN_REWARD:
                    agent.model.save(f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')
                    agent.model2.save(f'models2/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')
            # Decay epsilon
            if epsilon > MIN_EPSILON:
                epsilon *= EPSILON_DECAY
                epsilon = max(MIN_EPSILON, epsilon)


    # Set termination flag for training thread and wait for it to finish
    agent.terminate = True
    trainer_thread.join()
    agent.model.save(f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')
    agent.model2.save(f'models2/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')
