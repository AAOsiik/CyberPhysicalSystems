import time             # set dynamic q-table filenames
import pickle           # Save and load q table
import numpy as np      # Array types of stuff
from PIL import Image   #
import cv2
import matplotlib.pyplot as plt  # Plotting
from matplotlib import style    # make graph pretty
from time import sleep

style.use("ggplot")

SIZE_X = 9
SIZE_Y = 6
BOUNDARIES = [  (1, 1), (2, 1), (3, 1), (5, 1), (6, 1), (7, 1),
                (1, 2), (5, 2), (6, 2), (7, 2),
                (1, 3), (2, 3), (4, 3), (5, 3), (6, 3), (7, 3),
                (1, 4), (2, 4), (4, 4), (5, 4), (6, 4), (7, 4)]
HM_EPISODES = 25000 # how many episodes
MOVE_PENALTY = 10
# ENEMY_PENALTY = 300
FOOD_REWARD = 25
epsilon = 0.1               # setting and decaying works good
EPS_DECAY = 0.9998
SHOW_EVERY = 200

start_q_table = None # or filename, if you want to continue training

LEARNING_RATE = 0.1
DISCOUNT = 0.95

PLAYER_N = 1
FOOD_N = 2
BOUNDARY = 3
# ENEMY_N = 3

d = {   1: (255 , 175 , 0),  # bgr
        2: (0 , 255 , 0),
        3: (0, 0 , 255)}

# position of everything is depending on observation

class Blob:
    def __init__(self):
        self.x = 3
        self.y = 2
        
    def __str__(self):
        return f"{self.x}, {self.y}"

    def __sub__(self, other):
        return (self.x-other.x, self.y-other.y)

    def action(self, choice):
        if choice == 0:
            self.move(x=1) # right
        elif choice == 1:
            self.move(y=1) # up
        elif choice == 2:
            self.move(x=-1) # left
        elif choice == 3:
            self.move(y=-1) # down

    def move(self, x=False, y=False):
        if x:
            self.x += x
            if (self.x, self.y) in BOUNDARIES:
                self.x -= x # dont move
        
        if y:
            self.y += y
            if (self.x, self.y) in BOUNDARIES:
                self.y -= y # dont move

        if not(x) and not(y):
            # do random random
            c = np.random.randint(0, 4)
            self.action(c)

        if self.x < 0:
            self.x = 0
        elif self.x > SIZE_X-1:
            self.x = SIZE_X-1

        if self.y < 0:
            self.y = 0
        elif self.y > SIZE_Y-1:
            self.y = SIZE_Y-1


class Food:
    def __init__(self):
        self.x = 0
        self.y = 2
        
    def __str__(self):
        return f"{self.x}, {self.y}"

    def __sub__(self, other):
        return (self.x-other.x, self.y-other.y)


if start_q_table is None:
    q_table = {}
    for x1 in range(-SIZE_X+1, SIZE_X):              # delta to the food
        for y1 in range(-SIZE_Y+1, SIZE_Y):          # (x1,y1)(x2,y2)
            for x2 in range(-SIZE_X+1, SIZE_X):      
                for y2 in range(-SIZE_Y+1, SIZE_Y):   
                    q_table[((x1,y1), (x2,y2))] = [np.random.uniform(-5, 0) for i in range(4)]
else:
    with open(start_q_table, "rb") as f:
        q_table = pickle.load(f)


episode_rewards = []
for episode in range(HM_EPISODES):
    player = Blob()
    food = Food()
    enemy = Blob()

    if episode % SHOW_EVERY == 0:
        print(f"on # {episode}, epsilon: {epsilon}")
        print(f"{HM_EPISODES} ep mean {np.mean(episode_rewards[-SHOW_EVERY:])}")
        show = True
    else:
        show = False

    episode_reward = 0
    for i in range(20):
        obs = (player-food, player-enemy)
        if np.random.random() > epsilon:
            action = np.argmax(q_table[obs])
        else:
            action = np.random.randint(0, 4)

        player.action(action)

        #### maybe later
        # enemy.move()
        # food.move()
        ################

        # if player.x == enemy.x and player.y == enemy.y:
        #     reward = -ENEMY_PENALTY
        if player.x == food.x and player.y == food.y:
            reward = FOOD_REWARD
        else: 
            reward = -MOVE_PENALTY

        new_obs = (player-food, player-enemy)
        max_future_q = np.max(q_table[new_obs])
        current_q = q_table[obs][action]

        if reward == FOOD_REWARD:
            new_q = FOOD_REWARD
        # elif reward == -ENEMY_PENALTY:
        #     new_q = -ENEMY_PENALTY
        else:
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
        
        q_table[obs][action] = new_q

        if show:
            env = np.zeros((SIZE_Y, SIZE_X, 3), dtype=np.uint8)
            env[food.y][food.x] = d[FOOD_N]
            env[player.y][player.x] = d[PLAYER_N]
            for b in BOUNDARIES:
                env[b[1]][b[0]] = d[BOUNDARY]

            img = Image.fromarray(env, "RGB")
            img = img.resize((500, 300))
            cv2.imshow("", np.array(img))
            sleep(0.05)
            if reward == FOOD_REWARD:
                if cv2.waitKey(500) & 0xFF == ord("q"):
                    break
            else: 
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        episode_reward += reward
        if reward == FOOD_REWARD:
            break

    episode_rewards.append(episode_reward)
    epsilon *= EPS_DECAY

moving_avg = np.convolve(episode_rewards, np.ones((SHOW_EVERY, )) / SHOW_EVERY, mode="valid")

plt.plot([i for i in range(len(moving_avg))], moving_avg)
plt.ylabel(f"reward {SHOW_EVERY}ma")
plt.xlabel("episode #")
plt.show()

with open(f"qtableTEST-{int(time.time())}.pickle", "wb") as f:
    pickle.dump(q_table, f)
