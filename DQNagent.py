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
import math

try:
    sys.path.append(glob.glob('../PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla

MIN_SPEED= 30
MAX_SPEED= 60
IM_WIDTH = 640
IM_HEIGHT = 480
SECONDS_PER_EPISODE = 60
REPLAY_MEMORY_SIZE = 5_000
MIN_REPLAY_MEMORY_SIZE = 1_000
MINIBATCH_SIZE = 16
PREDICTION_BATCH_SIZE = 1
TRAINING_BATCH_SIZE = MINIBATCH_SIZE // 4
UPDATE_TARGET_EVERY = 5
MODEL_NAME = "Mod64x4"
MEMORY_FRACTION = 0.4
MIN_REWARD = -200
EPISODES = 2000
DISCOUNT = 0.997
epsilon = 1
EPSILON_DECAY = 0.998 #0.997  0.9975 99975
MIN_EPSILON = 0.005
INCREMENT = 0.25   #0.5    0.2    0.1
AGGREGATE_STATS_EVERY = 10
SHOW_PREVIEW = False
NUMBER_OF_ACTIONS= int((2/INCREMENT + 1 ) * (1 + 1/INCREMENT +1 )) 



#funtion that generates action parameters for the action ID 
def generate_Action(action):
  Throttle = 0.0
  Steer    = 0.0
  Reverse  = False 
  Brake = 0.0    
  Steer =  INCREMENT * (action % (2/INCREMENT + 1)) - 1
  
  if (action < ( 2/INCREMENT +1)* ((1/INCREMENT)+1)):
     Throttle = INCREMENT * (math.floor(action /(2/INCREMENT + 1)))
  elif (action < (2/INCREMENT + 1 ) * (1/INCREMENT +2) ):
    Reverse = True 
  else:
    Steer= 0.0
    Brake= INCREMENT*(NUMBER_OF_ACTIONS-action) 
    
  return round(Throttle, 2), round(Steer, 2), Reverse, round(Brake, 2)




# Tensorboard class, taken from open source, and used to generate log files to generate the plots
class ModifiedTensorBoard(TensorBoard):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.FileWriter(self.log_dir)
   
    def set_model(self, model):
        pass
    
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)
    
    def on_batch_end(self, batch, logs=None):
        pass
    def on_train_end(self, _):
        pass
  
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

# environment class
class CarEnv:
    SHOW_CAM = SHOW_PREVIEW
    im_width = IM_WIDTH
    im_height = IM_HEIGHT
    front_camera = None

    def __init__(self):
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(200.0)
        self.world = self.client.get_world()
        #self.world= self.client.load_world('town04')
        self.blueprint_library = self.world.get_blueprint_library()
        self.model_3 = self.blueprint_library.filter("model3")[0]

    #resets the position of the vehicle, the camera, the collision sensor, and returns the camera output
    def reset(self):
        self.collision_hist = []
        self.actor_list = []
        self.transform = random.choice(self.world.get_map().get_spawn_points())
        self.destination= random.choice(self.world.get_map().get_spawn_points())
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

        colsensor = self.blueprint_library.find("sensor.other.collision")
        self.colsensor = self.world.spawn_actor(colsensor, transform, attach_to=self.vehicle)
        self.actor_list.append(self.colsensor)
        self.colsensor.listen(lambda event: self.collision_data(event))
       
        while self.front_camera is None:
            time.sleep(0.01)

        self.episode_start = time.time()
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))

        return self.front_camera
    
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

    #applies the given action, an returns the reward R,  the new State S' and boolean defining whether S' is a final state     
    def step(self, action):
    
        Throttle, Steer, Reverse, Brake =  generate_Action(action)
      
        self.vehicle.apply_control(carla.VehicleControl(throttle=Throttle, steer=Steer, reverse= Reverse, brake= Brake))
       
        v = self.vehicle.get_velocity()
        kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))
        
      
        if len(self.collision_hist) != 0:
                done = True
                reward = -100
        elif kmh < MIN_SPEED or kmh > MAX_SPEED:
                done = False
                reward = -1
        else:
                done = False
                reward = 1

        if self.episode_start + SECONDS_PER_EPISODE < time.time():
                done = True

        return self.front_camera, reward, done, None



#The agent Class
class DQNAgent:

    def __init__(self):

        self.model = self.create_model()
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

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

        model.add(Dense(NUMBER_OF_ACTIONS, activation='linear'))
        model.compile(loss="mse", optimizer=Adam(lr=0.001, decay=0.0), metrics=['accuracy'])
        return model


    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def train(self):
     
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        current_states = np.array([transition[0] for transition in minibatch])/255
      
        with self.graph.as_default():
            backend.set_session(sess)
            current_qs_list = self.model.predict(current_states, PREDICTION_BATCH_SIZE)

        new_current_states = np.array([transition[3] for transition in minibatch])/255
        
        with self.graph.as_default():
            backend.set_session(sess)
            future_qs_list = self.target_model.predict(new_current_states, PREDICTION_BATCH_SIZE)

        X = []
        y = []

        for index, (current_state, action, reward, new_state, done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            X.append(current_state)
            y.append(current_qs)

        log_this_step = False
        if self.tensorboard.step > self.last_logged_episode:
            log_this_step = True
            self.last_logged_episode = self.tensorboard.step

        with self.graph.as_default():
            backend.set_session(sess)
            self.model.fit(np.array(X)/255, np.array(y), batch_size=TRAINING_BATCH_SIZE, verbose=0, shuffle=False, callbacks=[self.tensorboard] if log_this_step else None)

        if log_this_step:
            self.target_update_counter += 1

        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape)/255)[0]

    def train_in_loop(self):
        X = np.random.uniform(size=(1, IM_HEIGHT, IM_WIDTH, 3)).astype(np.float32)
        y = np.random.uniform(size=(1, NUMBER_OF_ACTIONS)).astype(np.float32)
        with self.graph.as_default():
            backend.set_session(sess)
            self.model.fit(X,y, verbose=False, batch_size=1)

        self.training_initialized = True

        while True:
            if self.terminate:
                return
            self.train()
            time.sleep(0.01)



if __name__ == '__main__':
    ep_rewards = [-200]

    random.seed(1)
    np.random.seed(1)
    tf.set_random_seed(1)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=MEMORY_FRACTION)
    tf_config=tf.ConfigProto(gpu_options=gpu_options)
    sess=tf.Session(config=tf_config)
    backend.set_session(sess)

    # Create models folder
    if not os.path.isdir('models'):
        os.makedirs('models')

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

    # Iterate over episodes
    for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):
            env.collision_hist = []
            # Update tensorboard step every episode
            agent.tensorboard.step = episode
            # Restarting episode - reset episode reward and step number
            episode_reward = 0
            step = 1
            # Reset environment and get initial state
            current_state = env.reset()
            # Reset flag and start iterating until episode ends
            done = False
            episode_start = time.time()
            # Play for the given number of seconds only
            while True:
                # This part applies the epsilon greedy policy for the action selection
                if np.random.random() > epsilon:
                    # Get action from Q table
                    Q_table = agent.get_qs(current_state)
                    action = np.argmax(Q_table)
                else:
                    # Get random action
                    action = np.random.randint(0, NUMBER_OF_ACTIONS)
                    time.sleep(1/30)
                
                #apply the action on the env
                new_state, reward, done, _ = env.step(action)

                # sum the reward for the episode
                episode_reward += reward

                # Every step we update replay 
                agent.update_replay_memory((current_state, action, reward, new_state, done))
                current_state = new_state
                step += 1

                if done:
                    break

            # End of episode - destroy agents
            for actor in env.actor_list:
                actor.destroy()

            # Append episode reward to a list and log stats (every given number of episodes)
            ep_rewards.append(episode_reward)
            if not episode % AGGREGATE_STATS_EVERY or episode == 1:
                average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
                min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
                max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
                agent.tensorboard.update_stats(Reward_AVG=average_reward, Reward_min=min_reward, Reward_max=max_reward, epsilon=epsilon)

                # Save model, but only when min reward is greater or equal a set value
                if min_reward >= MIN_REWARD:
                    agent.model.save(f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

            # Decay epsilon
            if epsilon > MIN_EPSILON:
                epsilon *= EPSILON_DECAY
                epsilon = max(MIN_EPSILON, epsilon)


    # Set termination flag for training thread and wait for it to finish
    agent.terminate = True
    trainer_thread.join()
    agent.model.save(f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')