# Discrete DeepQ

## Parameters
| Variable                    | Default | Description |
|-----------------------------|---------|-------------|
| `model`                     | N/A     | model that takes the state from your simulator and outputs scores for each of the discrete actions. |
| `optimizer`                 | N/A     | which optimizer to use |
| `exploration_period`        | 10000   | how many actions need to be taken before random action probability reaches its final value. |
| `random_action_probability` | 0.05    | final value of random action probability. |
| `discount_rate`             | 0.95    | how much we care about future rewards. must be less than 1. |
| `store_every_nth`           | 1       | to further decorrelate samples do not store all transitions, but rather every nth transition. For example if store_every_nth is 5, then only 20% of all the transitions is stored. |
| `replay_buffer_size`        | 10000   | Size of the replay buffer |
| `train_every_nth`           | 1       | normally training_step is invoked every time action is executed. Depending on the setup that might be too often. When this variable is set set to n, then only every n-th transition will the training procedure be executed. |
| `minibatch_size`            | 30      | number of state,action,reward,newstate tuples considered during experience reply |
| `target_network_update_rate | 0.001   | how much to update target network after each iteration. Let's call target_network_update_rate alpha, target network T, and network N. Every time N gets updated we execute: T = (1-alpha)*T + alpha*N |
