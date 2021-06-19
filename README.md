# openai-gym-taxi
solution for 1 and 2 passengers with jupyter notebook

[HTML Q_LEARNING 1 PASSENGER](https://fahimbench.github.io/openai-gym-taxi/html/Q_Learning-1passenger.html)

[HTML Q_LEARNING 2 PASSENGERS](https://fahimbench.github.io/openai-gym-taxi/html/Q_Learning-2passengers.html)

Require with jupyter notebook scripts:
 - Pandas
 - Plotly
 - gym

Gym is a toolkit for developing and comparing reinforcement learning algorithms. It supports teaching agents everything from walking to playing games like Pong or Pinball. Here we use it to learn how it works with the Taxi game.

We need to teach to our agent(Taxi) to pickup a person on a point and drop it off on another point. Ant with more difficult, 2 persons on 2 different points.

We used Q-learning for our agent. Q-learning is a reinforcement learning algorithm that seeks to find the best possible next action given its current state, in order to maximise the reward it receives.

The agent have a determinate actions to use :
For 1 passenger - 6 actions :
 0 - go to south
 1 - go to north
 2 - go to east
 3 - go to west
 4 - pickup passenger
 5 - dropoff passenger

For 2 passengers - 8 actions :  
 0 - go to south
 1 - go to north
 2 - go to east
 3 - go to west
 4 - pickup passenger 1
 5 - dropoff passenger 1
 6 - pickup passenger 2
 7 - dropoff passenger 2

The agent have an observable state :
For 1 passenger - max is 4,4,4,3 (encoded by 500)
 4 - Row
 4 - Column
 4 - the place where to pickup the passenger (4 means it's a passenger in the taxi)
 3 - the place where to dropoff the passenger

For 1 passenger - max is 4,4,4,3,4,3 (encoded by 10000)
 4 - Row
 4 - Column
 4 - the place where to pickup the passenger 1 (4 means in the taxi)
 3 - the place where to dropoff the passenger 1
 4 - the place where to pickup the passenger 2 (4 means in the taxi)
 3 - the place where to dropoff the passenger 2
