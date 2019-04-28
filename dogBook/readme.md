# 1. MDP & Bellman Equation

## MDP 
> 순차적 행동 결정 문제를 수학적으로 정의한 것

- agent
- action
- state
- policy
- reward
- discount factor
- return

## Value Function
> agent가 어떤 policy가 더 좋은 policy인지 판단하는 기준

- state-value function
- action-value function

## Bellman Equation
> 현재 상태의 가치함수와 다음 상태 가치함수의 관계식

- bellman expect equation
- bellman optimality equation 


<br>

# 2. Dynamic Programming 

## Dynamic Programming
> 변화하는 대상을 계획하는 것. <br>
> 작은 문제가 큰 문제 안에 중첩돼 있는 경우 작은 문제의 답을 다른 작은 문제에서 이용하여 효율적으로 계산하는 방법 <br> 
> 다이나믹프로그래밍에서는 에이전트가 모든 상태에 대해 벨만 방정식으로 계산한다. 

> *하지만, 계산 복잡도가 크고, 차원의 저주에 빠진며, 환경에 대한 완벽한 정보가 필요하다.*

- policy iteration : 구한 value function을 통해 최대의 reward을 얻는 action을 선택
	- policy evaluation
	- policy improvement
- value iteration : 최적의 policy를 가정하고 순차적 action을 결정  

# 3. Reinforcement Learning
## 3.1. Prediction
> value function을 구하는 과정
### Monte-Carlo
> 무작위로 무엇인가를 해본다. 
### Temporal-Difference 
> episode마다가 아니라 time-step마다 업데이트를 한다.
## 3.2. Control
> prediction과 함께 policy를 발전시키는 과정  
## SARSA
> on-policy
## Q-learning
> off-policy

# 4. Deep Reinforcement Learning

# 5. Example