## What is the idea of Q Learning?

- To have Q values for every action you could possibly take for every state
- Model free learning: applicable in every environment!

## Programming
- After each step we get a new state (position and velocity)

initialize q table with random values
explore/do random stuff, slowly update q values as time goes on
- Table will blow out memory, so we want to convert continous values to discrete values
- reward will be -1, until reaching flag (reward 0
- need values for every action that is possible)

## In the real World, we have to figure out the parameters!
- simple tracking system: Reward
- Basic Q-Learning not meant for complex environment