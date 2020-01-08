import time                         # Needed to set dynamic q-table filenames
import pickle                       # Save and load q table
import numpy as np                  # Array types of stuff
from PIL import Image               #
import cv2                          # OpenCV
import matplotlib.pyplot as plt     # Plotting
from matplotlib import style        # make graph pretty
from time import sleep

style.use("ggplot")

# Recreation of the original environment
SIZE_X = 9
SIZE_Y = 6
BOUNDARIES =   [(1, 1), (2, 1), (3, 1), (5, 1), (6, 1), (7, 1),
                (1, 2), (5, 2), (6, 2), (7, 2),
                (1, 3), (2, 3), (4, 3), (5, 3), (6, 3), (7, 3),
                (1, 4), (2, 4), (4, 4), (5, 4), (6, 4), (7, 4)]

# Declaring Rewards and Penalties
MOVE_PENALTY = 2
BOUNDARY_PENALTY = 4
FOOD_REWARD = 50
STEPS_PER_EPISODE = 25

PLAYER = 1
FOOD = 2
BOUNDARY = 3

START_Q_TABLE = "QTable-TASK3-1578490824.pickle"       # or filename, if you want to continue training

COLOR_DICT =    {1: (255, 175, 0),  # BGR Format
                 2: (0, 255, 0),
                 3: (0, 0, 255)}

# Parameter Tuning
HM_EPISODES = 500    # how many episodes
LEARNING_RATE = 0.2
DISCOUNT = 0.95
EPSILON = 0.1               # setting and decaying works good
EPSILON_DECAY = 0.99
SHOW_EVERY = 1


# Remember: Position of everything is depending on the agent's observation!
class QAgent:
    def __init__(self):
        '''
        Set Agent at predetermined position
        '''
        # Starting Position is set
        self.x = 3
        self.y = 3
        self.heading = 3 # heading north
        self.collision = False # wall collision
        self.L = 0      # Left Memory
        self.R = 0      # Right Memory

    def __str__(self):
        ''' 
        To string method returning the agent's position, ``x`` and  ``y`` coordinates
        '''
        return f"{self.x}, {self.y}"

    def __sub__(self, other):
        '''
        Difference vector (aka Distance) to given ``other`` instance
        '''
        return (self.x-other.x, self.y-other.y)

    def action(self, choice):
        '''
        Perform action based on ``choice``.
        '''
        if abs(self.heading - choice) == 2:
            # "Vorwärts immer - rückwärts nimmer!"     -Erich Honecker
            # Agent trying to go back, not possible, Move Penalty will be induced
            return

        if choice == 0:
            self.move(x=1) # right
        elif choice == 1:
            self.move(y=1) # up
        elif choice == 2:
            self.move(x=-1) # left
        elif choice == 3:
            self.move(y=-1) # down

    def move(self, x=False, y=False):
        '''
        Updates the agent's position, while keeping him in bounds.
        '''
        if x:
            self.x += x
            if (self.x, self.y) in BOUNDARIES:
                self.x -= x # dont move
                self.collision = True
            else:
                if x > 0:
                    self.heading = 0
                else:
                    self.heading = 2
        if y:
            self.y += y
            if (self.x, self.y) in BOUNDARIES:
                self.y -= y # dont move
                self.collision = True
            else:
                if y > 0:
                    self.heading = 1
                else:
                    self.heading = 3
        if not x and not y:
            # do random
            c = np.random.randint(0, 4)
            self.action(c)

        # Stay in Global Outer Boundaries
        if self.x < 0:
            self.x = 0
        elif self.x > SIZE_X-1:
            self.x = SIZE_X-1
        if self.y < 0:
            self.y = 0
        elif self.y > SIZE_Y-1:
            self.y = SIZE_Y-1

    def update_memory(self, task, ep, success=False):
        '''
        Updates the Memory Cells. Depending on task, set corresponding cell to 2.
        '''
        if self.L > 0:
            self.L -= 1
        if self.R > 0:
            self.R -= 1
        if success:
            if task == 1:
                self.L = 2
            elif task == 2:
                self.R = 2
            elif task == 3:
                if ep % 2 == 0:
                    self.L = 2
                else:
                    self.R = 2



class QFood:
    def __init__(self, task, ep):
        '''
        Set Food position depending on task.
        '''
        if task == 1: # Reward on the Left
            self.x = 0
            self.y = 2
        elif task == 2: # Reward on the Right
            self.x = 8
            self.y = 2
        else:
            if ep % 2 == 0:
                self.x = 0
                self.y = 2
            else:
                self.x = 8
                self.y = 2

    def __str__(self):
        return f"{self.x}, {self.y}"

    def __sub__(self, other):
        '''
        Difference vector (aka Distance) to given ``other`` instance
        '''
        return (self.x-other.x, self.y-other.y)


# Set the Task
TASK = 3

# Set the Q table
if START_Q_TABLE is None:
    q_table = {}
    for x1 in range(-SIZE_X+1, SIZE_X):              
        for y1 in range(-SIZE_Y+1, SIZE_Y):          
            for x2 in range(-SIZE_X+1, SIZE_X):      
                for y2 in range(-SIZE_Y+1, SIZE_Y):   
                    q_table[((x1, y1), (x2, y2))] = [np.random.uniform(-10, 0) for i in range(4)]
else:
    with open(START_Q_TABLE, "rb") as f:
        q_table = pickle.load(f)


# RUN
episode_rewards = []
tmp_L = 0
tmp_R = 0
for episode in range(HM_EPISODES):
    player = QAgent()
    player.L = tmp_L
    player.R = tmp_R
    food = QFood(TASK, episode)

    if episode % SHOW_EVERY == 0:
        print(f"on # {episode}, epsilon: {EPSILON}, Memory {(player.L, player.R)}")
        print(f"{HM_EPISODES} ep mean {np.mean(episode_rewards[-SHOW_EVERY:])}")
        show = True
    else:
        show = False

    episode_reward = 0
    for i in range(STEPS_PER_EPISODE):
        obs = (player-food, (player.L, player.R))
        if np.random.random() > EPSILON:
            action = np.argmax(q_table[obs])
        else:
            action = np.random.randint(0, 4)
        player.action(action)


        # Handle the Rewarding
        if player.x == food.x and player.y == food.y:
            reward = FOOD_REWARD
        elif player.collision:
            player.collision = False
            reward = -BOUNDARY_PENALTY
        else:
            reward = -MOVE_PENALTY

        new_obs = (player-food, (player.L, player.R))
        max_future_q = np.max(q_table[new_obs])
        current_q = q_table[obs][action]

        if reward == FOOD_REWARD:
            new_q = FOOD_REWARD
        else:
            new_q = (1-LEARNING_RATE)*current_q + LEARNING_RATE*(reward + DISCOUNT * max_future_q)

        q_table[obs][action] = new_q

        if show:
            # Visualize everything
            env = np.zeros((SIZE_Y, SIZE_X, 3), dtype=np.uint8)
            env[food.y][food.x] = COLOR_DICT[FOOD]
            env[player.y][player.x] = COLOR_DICT[PLAYER]
            for b in BOUNDARIES:
                env[b[1]][b[0]] = COLOR_DICT[BOUNDARY]

            img = Image.fromarray(env, "RGB")
            img = img.resize((500, 300))
            cv2.imshow("", np.array(img))
            sleep(0.05)
            # break
            if reward == FOOD_REWARD:
                if cv2.waitKey(500) & 0xFF == ord("q"):
                    break
            else: 
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        episode_reward += reward
        if reward == FOOD_REWARD:
            # Update Memory with reward
            player.update_memory(task=TASK, ep=episode, success=True)
            print(f"Success on Episode {episode}")
            break
        if i == STEPS_PER_EPISODE-1:
            # Memory will just be shifted
            player.update_memory(task=TASK, ep=episode)
        
    tmp_L = player.L
    tmp_R = player.R
    episode_rewards.append(episode_reward)
    EPSILON *= EPSILON_DECAY

moving_avg = np.convolve(episode_rewards, np.ones((SHOW_EVERY, )) / SHOW_EVERY, mode="valid")

plt.plot([i for i in range(len(moving_avg))], moving_avg)
plt.ylabel(f"Reward during {SHOW_EVERY} Step Interval")
plt.xlabel("Number of Episodes")
plt.show()

with open(f"QTable-TASK{TASK}-{int(time.time())}.pickle", "wb") as f:
    pickle.dump(q_table, f)
