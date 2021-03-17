import numpy as np
import matplotlib.pyplot as plt
import math

def getSamplar():
    mu=np.random.normal(0,10)
    sd=abs(np.random.normal(5,2))
    getSample=lambda: np.random.normal(mu,sd)
    return getSample

def e_greedy(Q, e):
    p = np.random.uniform(0,1) 
    if p < e:
        action = np.random.choice(list(Q))
    else:
        #get actions with max reward, break ties randomly
        maxActions = [a for a in Q.keys() if Q[a] == max(Q.values())]
        action = np.random.choice(maxActions)
    
    return action
    
def upperConfidenceBound(Q, N, c):
    #get actions where n = 0
    n_zero = [key for key in N.keys() if N[key] == 0]
    #find t
    t = sum(N.values())+1
    
    if len(n_zero) == 0:
        #estimate expected reward, break ties randomly
        upper = {a:Q[a]+c*math.sqrt(math.log(t)/N[a]) for a in Q.keys()} 
        maxActions = [a for a in upper.keys() if upper[a] == max(upper.values())]
        action = np.random.choice(maxActions)
    else:
        action = np.random.choice(n_zero)
    
    return action

def updateQN(action, reward, Q, N):
    NNew = N.copy(); QNew = Q.copy()
    NNew[action] = NNew[action] + 1
    QNew[action] = QNew[action] + (1/NNew[action])*(reward - QNew[action])
 
    return QNew, NNew

def decideMultipleSteps(Q, N, policy, bandit, maxSteps):
    actionReward = []
    for steps in range(maxSteps):
        action = policy(Q,N) #take action
        reward = bandit(action) #get reward
        actionReward.append((action,reward))
        Q,N = updateQN(action,reward,Q,N)
 
    return {'Q':Q, 'N':N, 'actionReward':actionReward}

def plotMeanReward(actionReward,label):
    maxSteps=len(actionReward)
    reward=[reward for (action,reward) in actionReward]
    meanReward=[sum(reward[:(i+1)])/(i+1) for i in range(maxSteps)]
    plt.plot(range(maxSteps), meanReward, linewidth=0.9, label=label)
    plt.xlabel('Steps')
    plt.ylabel('Average Reward')

def main():
    np.random.seed(2020)
    K=10
    maxSteps=1000
    Q={k:0 for k in range(K)}
    N={k:0 for k in range(K)}
    testBed={k:getSamplar() for k in range(K)}
    bandit=lambda action: testBed[action]()
    
    policies={}
    policies["e-greedy-0.5"]=lambda Q, N: e_greedy(Q, 0.5)
    policies["e-greedy-0.1"]=lambda Q, N: e_greedy(Q, 0.1)
    policies["UCB-2"]=lambda Q, N: upperConfidenceBound(Q, N, 2)
    policies["UCB-20"]=lambda Q, N: upperConfidenceBound(Q, N, 20)
    
    allResults = {name: decideMultipleSteps(Q, N, policy, bandit, maxSteps) for (name, policy) in policies.items()}
    
    for name, result in allResults.items():
         plotMeanReward(allResults[name]['actionReward'], label=name)
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',ncol=2, mode="expand", borderaxespad=0.)
    plt.show()
    


if __name__=='__main__':
    main()
