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

# Creating a Custom Environment
- each observation needs 4 valiues if we have 4 actions

# Deep Q Learning (DQN)
- multilayer deep nn image->(conv layer, conv layer, FC layer, FC layer) many output, regression model
- Here: input, 2 L, Output
- barely increase size of q table: quadratic space increase!
- DOWNSIDE: take a lot longer to train! minutes for q tables, hours for deep q learning!
- BENEFIT: waaaaay less memory required, so deep Q is pretty cool

### How to?
- every step, we need to update q value
- then resample environment, caluclate new q, fit operation

### Deep Q
- train on batch of specific size (32 or 64), train on 10.000