import numpy as np
from gym.utils import seeding
from gym import spaces
import gym

class ClctFourRooms(gym.Env):
    def __init__ (self, num_goals=10, num_types=2, object_priors=[1, 1], reward_vec=None, horizon=50, p=0, config=2, layout='open', seed=1234, cont=True):
        """
        config -> configouration of the state space
            0 - returns tabular index of the state
            1 - returns one hot encoded vector of the state
            2 - returns matrix form of the state
        """
        if(layout == '4rooms'):
            layout = """\
wwwwwwwwwwwww
w     w     w
w     w     w
w           w
w     w     w
w     w     w
ww wwww     w
w     www www
w     w     w
w     w     w
w           w
w     w     w
wwwwwwwwwwwww
"""
        elif(layout == '3rooms'):
            layout = """\
wwwwwwwwwwwww
w   w   w   w
w   w       w
w   w   w   w
w   w   w   w
w   w   w   w
w   w   w   w
w   w   w   w
w   w   w   w
w       w   w
w   w   w   w
w   w   w   w
wwwwwwwwwwwww
"""
        elif(layout == '3roomsh'):
            layout = """\
wwwwwwwwwwwww
w           w
w           w
wwwwwwwww www
w           w
w           w
w           w
w           w
ww wwwwwwwwww
w           w
w           w
w           w
wwwwwwwwwwwww
"""
        elif(layout == 'maze'):
            layout = """\
wwwwwwwwwwwww
w           w
w ww wwwwww w
w w       w w
w w wwwww w w
w w w   w w w
w w   w   www
w w w   w w w
w w wwwww w w
w w       w w
w ww wwwwww w
w           w
wwwwwwwwwwwww
"""
        elif(layout == 'open'):
            layout = """\
wwwwwwwwwwwww
w           w
w           w
w           w
w           w
w           w
w           w
w           w
w           w
w           w
w           w
w           w
wwwwwwwwwwwww
"""
        else:
            raise
            
        assert num_types == len(object_priors)
            
        self.p = p
        self.config = config
        self.num_goals = num_goals
        self.num_types = num_types
        self.horizon = horizon
        # object priors is the vector of priors of types of each object (so we have to normalize it first)
        self.object_priors = np.array(object_priors) / np.sum(object_priors)
        # reward_vec is the reward vector for different type of objects
        if reward_vec is None:
            self.reward_vec = np.ones(shape=(self.num_types,))
        else:
            self.reward_vec = reward_vec
            
        # store the number of updates
        self.update = 0
        
        # if we would like to create another object when one object is collected
        self.cont = False
        
        # fix the random seed for reproducibility
        np.random.seed(seed)
        self.rng = np.random.RandomState(seed)
        
        # occupancy of the layout(1 -> walls, 0 -> elsewhere)
        self.occupancy = np.array([list(map(lambda c: 1 if c=='w' else 0, line)) for line in layout.splitlines()])
        
        # action space of the env
        # 0: UP
        # 1: DOWN
        # 2: LEFT
        # 3: RIGHT
        self.a_space = np.array([0, 1, 2, 3])
        self.obs_space = np.zeros(np.sum(self.occupancy == 0))
        
        # observation space (Not used)
        # Setting the observation space based on the config
        # The observation space is for the network to know what shape of input it should expect
        if(config <= 1):
            self.observation_space = spaces.Box(low=np.zeros(np.sum(self.occupancy == 0)), high=np.ones(np.sum(self.occupancy == 0)), dtype=np.uint8)
        elif(config == 2):
            self.observation_space = spaces.Box(low=np.zeros([4, 169]), high=np.ones([4, 169]), dtype=np.uint8)
        
        # action space
        self.action_space = spaces.Discrete(4)
        
        # direction of actions
        self.directions = [np.array((-1,0)), np.array((1,0)), np.array((0,-1)), np.array((0,1))]
        
        # a dictionary that can convert state index (scalar) to state cell location (2-dimension vector)
        self.tostate = {}
        statenum = 0
        for i in range(13):
            for j in range(13):
                # first entry is the y index (vertival)
                # second entry is the x index (horizontal)
                if self.occupancy[i,j] == 0:
                    self.tostate[(i,j)] = statenum
                    statenum += 1
        self.tocell = {v:k for k, v in self.tostate.items()}
        
        # get a list of all available states
        self.area = list(range(self.obs_space.shape[0]))
        
        self.channels_all = len(self.reward_vec) + 2
        
        # now we put goals to the env
        # self.reset()
        
    # generate a random location in the environment
    def random_pos(self):
        # choose an random location based on the area
        return np.random.choice(self.area)
        
    def observation(self):
        if self.config == 1:
            # return one hot encoded map
            temp = np.zeros(len(self.obs_space))
            temp[self.agent_pos] = 1
            return temp
        if self.config == 2:
            area = np.zeros([self.channels_all, 13, 13], dtype=np.float32)
            # first channel is the wall
            area[0, :, :] = self.occupancy
            
            # second channel is the agent
            agent_cell = self.tocell[self.agent_pos]
            area[1, agent_cell[0], agent_cell[1]] = 1
            
            # the following channels are objects
            for pos, obj in self.objects.items():
                x, y = self.tocell[pos]
                area[2:, x, y] = obj
                
            return area
        
    def render(self):
        area = np.array(self.occupancy) * (-1)
        agent_cell = self.tocell[self.agent_pos]
        area[agent_cell[0], agent_cell[1]] = 10
        for pos, obj in self.objects.items():
            x, y = self.tocell[pos]
            area[x, y] = obj[0] * 5 + 1
            
        return area
        
        
    def reset(self):
        # clear the update count
        self.update = 0
        
        # first put objects to the env (each object is at different places)
        self.objects = {}
        for _ in range(self.num_goals):
            while True:
                new_pos = self.random_pos()
                if new_pos not in self.objects:
                    self.objects[new_pos] = np.random.multinomial(1, self.object_priors)
                    break
                    
        # Then we put the agent to the env
        self.agent_pos = self.random_pos()
        # if the agent is put onto an object, re-locate the agent
        while self.agent_pos in self.objects:
            self.agent_pos = self.random_pos()
        
        # now we have already put the agent and the objects into the env properly
        
        return self.observation()
    
    # step function for scavenger env
    def step_scavenger(self, action):
        # increase the update count
        self.update += 1
        
        # determine the next position
        agent_cell = self.tocell[self.agent_pos]
        next_cell = tuple(agent_cell + self.directions[action])
        
        # check if the next position is legal (not in the wall), if it is legal, make the move
        # otherwise, stay still
        if not self.occupancy[next_cell]:
            self.agent_pos = self.tostate[next_cell]
        
        # then we see if any object is collected
        # if no object is consumed, we will get an zero vector
        consumed = self.objects.pop(self.agent_pos, np.zeros(shape=(len(self.reward_vec,))))
           
        # calculate the reward
        reward = np.dot(self.reward_vec, consumed)
        
        # if this game is continuous (never end until the horizon is reached), then we would need to create another object if one object is collected
        if self.cont and reward != 0:
            while True:
                new_pos = self.random_pos()
                if new_pos not in self.objects and new_pos != self.agent_pos:
                    self.objects[new_pos] = np.random.multinomial(1, self.object_priors)
                    break
        
        # make sure that the objects is removed
        assert self.agent_pos not in self.objects
                                    
        # then we check if this episode is over
        done = False
        if (self.update >= self.horizon):
            done = True
        
                                    
        # return the observations and the reward
        return self.observation(), reward, done, {}
    
    
    # step function for simple env (there exists terminal state)
    def step_simple(self, action):
        # increase the update count
        # print(self.update)
        self.update += 1
        
        # determine the next position
        agent_cell = self.tocell[self.agent_pos]
        next_cell = tuple(agent_cell + self.directions[action])
        # print(next_cell)
        
        # check if the next position is legal (not in the wall), if it is legal, make the move
        # otherwise, stay still
        if not self.occupancy[next_cell]:
            self.agent_pos = self.tostate[next_cell]
            
        # # then we see if any object is collected
        # # if no object is consumed, we will get an zero vector
        # consumed = self.objects.pop(self.agent_pos, np.zeros(shape=(len(self.reward_vec,))))
           
        # # calculate the reward
        # reward = np.dot(self.reward_vec, consumed)
        
        done = False
        if self.agent_pos == 62:
            done = True
            reward = 0
        else:
            reward = -1

        if self.update >= self.horizon:
            done = True

        
            
        return self.observation(), reward, done, {}
    
    def step(self, action):
        # return self.step_simple(action)
        return self.step_scavenger(action)
            

    def seed(self, seed=None):
        """
        Setting the seed of the agent for replication
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


class ClctFourRoomsNoTerm(ClctFourRooms):
    def __init__(self, num_goals=10, num_types=2, object_priors=[1, 1], reward_vec=None, horizon=50, p=0, config=2, layout='open', seed=1234, cont=True):
        ClctFourRooms.__init__(self, num_goals=10, num_types=2, object_priors=[1, 1], reward_vec=None, horizon=50, p=0, config=2, layout='open', seed=1234, cont=True)
    
    def step(self, action):
        # increase the update count
        

        # determine the next position
        agent_cell = self.tocell[self.agent_pos]
        next_cell = tuple(agent_cell + self.directions[action])
        
        # check if the next position is legal (not in the wall), if it is legal, make the move
        # otherwise, stay still
        if not self.occupancy[next_cell]:
            self.agent_pos = self.tostate[next_cell]
           
        # calculate the reward
        reward = 0
        
        if(self.rng.uniform() < 0.01): # randomly check if the agent dies
            done = 1
        else:
            done = 0
        
                                    
        # return the observations and the reward
        return self.observation(), reward, done, {}