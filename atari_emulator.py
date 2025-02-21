import numpy as np
from ale_py import ALEInterface
import random
from environment import BaseEnvironment, FramePool,ObservationPool
from PIL import Image

# from scipy.misc import imresize
import cv2

IMG_SIZE_X = 84
IMG_SIZE_Y = 84
NR_IMAGES = 4
ACTION_REPEAT = 4
MAX_START_WAIT = 30
FRAMES_IN_POOL = 2


class AtariEmulator(BaseEnvironment):
    def __init__(self, actor_id, args):
        self.ale = ALEInterface()
        self.ale.setInt(b"random_seed", args.random_seed * (actor_id +1))
        # For fuller control on explicit action repeat (>= ALE 0.5.0)
        self.ale.setFloat(b"repeat_action_probability", 0.0)
        # Disable frame_skip and color_averaging
        # See: http://is.gd/tYzVpj
        self.ale.setInt(b"frame_skip", 1)
        self.ale.setBool(b"color_averaging", False)
        full_rom_path = args.rom_path + "/" + args.game + ".bin"
        self.ale.loadROM(str.encode(full_rom_path))
        self.legal_actions = self.ale.getMinimalActionSet()
        self.screen_width, self.screen_height = self.ale.getScreenDims()
        self.lives = self.ale.lives()

        self.random_start = args.random_start
        self.single_life_episodes = args.single_life_episodes
        self.call_on_new_frame = args.visualize

        # Processed historcal frames that will be fed in to the network 
        # (i.e., four 84x84 images)
        self.observation_pool = ObservationPool(np.zeros((IMG_SIZE_X, IMG_SIZE_Y, NR_IMAGES), dtype=np.uint8))
        self.rgb_screen = np.zeros((self.screen_width, self.screen_height, 3), dtype=np.uint8)
        self.gray_screen = np.zeros((self.screen_width, self.screen_height), dtype=np.uint8)

        # frame pool keeps a pool of (2, 210,160) images, __process_frame_pool return a (210, 160) size image
        self.frame_pool = FramePool(np.empty((FRAMES_IN_POOL, self.screen_width,self.screen_height), dtype=np.uint8),
                                    self.__process_frame_pool)

    def get_legal_actions(self):
        return self.legal_actions

    def __get_screen_image(self):
        """
        Get the current frame luminance
        :return: the current frame
        """

        self.ale.getScreenGrayscale(self.gray_screen)

        if self.call_on_new_frame:
            self.ale.getScreenRGB(self.rgb_screen)
            self.on_new_frame(self.rgb_screen)
        

        return np.squeeze(self.gray_screen)

    def get_rgb_screen(self):
        self.ale.getScreenRGB(self.rgb_screen)
        return self.rgb_screen

    def on_new_frame(self, frame):
        pass

    def __new_game(self):
        """ Restart game """
        self.ale.reset_game()
        self.lives = self.ale.lives()
        if self.random_start:
            wait = random.randint(0, MAX_START_WAIT)
            for _ in range(wait):
                self.ale.act(self.legal_actions[0])

    def __process_frame_pool(self, frame_pool):
        """ Preprocess frame pool : return a 84 x 84 grayscale image array by taking max over two frames in the pool """
        
        img = np.amax(frame_pool, axis=0)
        # img = imresize(img, (IMG_SIZE_X, IMG_SIZE_Y), interp='nearest')
        img = cv2.resize(img, (IMG_SIZE_X, IMG_SIZE_Y))

        img = img.astype(np.uint8)
        print()
        return img

    def __action_repeat(self, a, times=ACTION_REPEAT):
        """ Repeat action ACTION_REPEAT and grab last FRAMES_IN_POOL number of screen into frame pool, also collect the reward sum from all step """
        reward = 0
        for i in range(times - FRAMES_IN_POOL):
            reward += self.ale.act(self.legal_actions[a])
        # Only need to add the last FRAMES_IN_POOL frames to the frame pool
        
        for i in range(FRAMES_IN_POOL):
            reward += self.ale.act(self.legal_actions[a])
            frame = self.__get_screen_image()
            self.frame_pool.new_frame(frame)
        return reward

    def get_initial_state(self):
        """ Get the initial state """
        self.__new_game()
        for step in range(NR_IMAGES):
            """
                - when starting, repeat action 0 ACTION_REPEAT number of times and update the frame_pool with last FRAMES_IN_POOL frames
                - further at each of ACTION_REPEAT time, add a new max_pooled observation in the observation pool
            """
            _ = self.__action_repeat(0)     
            self.observation_pool.new_observation(self.frame_pool.get_processed_frame())
        if self.__is_terminal():
            raise Exception('This should never happen.')
        return self.observation_pool.get_pooled_observations()

    def next(self, action):
        """ Get the next state, reward, and game over signal
        - At every time step, you repeat action 'a' ACTION_REPEAT number of times and put the last FRAMES_IN_POOL frames in frame_pool
        - Then you do the max_pool of the two frames in frame_pools and add to the curren time index % 4 index in the observation pool
        - The collective reward from ACTION_REPEAT frames and last NR_IMAGES max_pooled (84,84) images forms the reward and observation.
        
        """
        reward = self.__action_repeat(np.argmax(action))
        self.observation_pool.new_observation(self.frame_pool.get_processed_frame())
        terminal = self.__is_terminal()
        self.lives = self.ale.lives()
        observation = self.observation_pool.get_pooled_observations()
        return observation, reward, terminal, self.lives
            
    def save_frame(self, frame):
        pass

    def __is_terminal(self):
        if self.single_life_episodes:
            return self.__is_over() or (self.lives > self.ale.lives())
        else:
            return self.__is_over()

    def __is_over(self):
        return self.ale.game_over()

    def get_noop(self):
        return [1.0, 0.0]
