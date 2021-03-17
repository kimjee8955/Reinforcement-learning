import numpy as np
import seaborn as sb 
import matplotlib.pyplot as plt

def expect(xDistribution, function):
    expectation=sum([function(x)*px for x, px in xDistribution.items()])
    return expectation

def e_greedyProbability(Q, e):
    actionMaxQ=[action for action in Q.keys() if Q[action]==max(Q.values())]
    def probabilityOfAction(action):
        if action in actionMaxQ:
            return (1-e)/len(actionMaxQ)+e/len(Q)
        else:
            return e/len(Q)
    actionDistribution={action: probabilityOfAction(action) for action in Q.keys()}
    return actionDistribution

def Bellman(s, policy, V, transitionTable, getSPrimeRDistribution, gamma):
    Q = {}
    for action in transitionTable[s].keys():
        #calculate Q 
        sPrimeDist = getSPrimeRDistribution(s,action)
        cumReward = lambda sPrime_Reward:sPrime_Reward[1] + gamma*V[sPrime_Reward[0]]
        Q[action] = expect(sPrimeDist,cumReward)
    
    #calculate policy using Q
    pi = policy(Q)
    
    #calculate V(s) using policy and Q
    V_s = expect(pi, lambda a: Q[a])
  
    return V_s

def viewDictionaryStructure(d, levels, indent=0):
    for key, value in d.items():
        print('\t' * indent + str(levels[indent]) + ": "+ str(key))
        if isinstance(value, dict):
            viewDictionaryStructure(value, levels, indent+1)
        else:
            print('\t' * (indent+1) + str(levels[indent+1])+ ": " + str(value))


def drawHeatMap(V, policyName):
    fig, ax=plt.subplots(figsize=(12,7))
    title=f"Value Map: {policyName}"
    plt.title(title, fontsize=18)
    ttl=ax.title
    ttl.set_position([0.5, 1.05])
    x ,y, v=([x for (x, y), v in V.items()], [y for (x, y), v in V.items()], [v for (x, y), v in V.items()])
    maxX=max(x)+1
    maxY=max(y)+1
    label=[str(key)+":"+str(round(value,3)) for key,value in V.items()]
    label, v=(np.array(label).reshape(maxX,maxY).transpose(), np.array(v).reshape(maxX,maxY).transpose())
    heatMap=sb.heatmap(v, annot=label, fmt="", cmap='RdYlGn', linewidths=0.30, center=0)
    plt.show()
    return heatMap

def main():
    
    transitionTable = {(0, 0): {(1, 0): {(1, 0): 1},(0, 1): {(0, 1): 1},(-1, 0): {(0, 0): 1},(0, -1): {(0, 0): 1}},(0, 1): {(1, 0): {(1, 1): 1},(0, 1): {(0, 2): 1},(-1, 0): {(0, 1): 1},(0, -1): {(0, 0): 1}},(0, 2): {(1, 0): {(1, 2): 1},(0, 1): {(0, 3): 1},(-1, 0): {(0, 2): 1},(0, -1): {(0, 1): 1}},(0, 3): {(1, 0): {(1, 3): 1},(0, 1): {(0, 4): 1},(-1, 0): {(0, 3): 1},(0, -1): {(0, 2): 1}},(0, 4): {(1, 0): {(1, 4): 1},(0, 1): {(0, 4): 1},(-1, 0): {(0, 4): 1},(0, -1): {(0, 3): 1}},(1, 0): {(1, 0): {(2, 0): 1},(0, 1): {(1, 1): 1},(-1, 0): {(0, 0): 1},(0, -1): {(1, 0): 1}},(1, 1): {(1, 0): {(2, 1): 1},(0, 1): {(1, 2): 1},(-1, 0): {(0, 1): 1},(0, -1): {(1, 0): 1}},(1, 2): {(1, 0): {(2, 2): 1},(0, 1): {(1, 3): 1},(-1, 0): {(0, 2): 1},(0, -1): {(1, 1): 1}},(1, 3): {(1, 0): {(2, 3): 1},(0, 1): {(1, 4): 1},(-1, 0): {(0, 3): 1},(0, -1): {(1, 2): 1}},(1, 4): {(1, 0): {(2, 4): 1},(0, 1): {(1, 4): 1},(-1, 0): {(0, 4): 1},(0, -1): {(1, 3): 1}},(2, 0): {(1, 0): {(2, 0): 1},(0, 1): {(2, 1): 1},(-1, 0): {(1, 0): 1},(0, -1): {(2, 0): 1}},(2, 1): {(1, 0): {(2, 1): 1},(0, 1): {(2, 2): 1},(-1, 0): {(1, 1): 1},(0, -1): {(2, 0): 1}},(2, 2): {(1, 0): {(2, 2): 1},(0, 1): {(2, 3): 1},(-1, 0): {(1, 2): 1},(0, -1): {(2, 1): 1}},(2, 3): {(1, 0): {(2, 3): 1},(0, 1): {(2, 4): 1},(-1, 0): {(1, 3): 1},(0, -1): {(2, 2): 1}},(2, 4): {(1, 0): {(2, 4): 1},(0, 1): {(2, 4): 1},(-1, 0): {(1, 4): 1},(0, -1): {(2, 3): 1}}}
    rewardTable = {(0, 0): {(1, 0): {(1, 0): -1},(0, 1): {(0, 1): -1},(-1, 0): {(0, 0): -1},(0, -1): {(0, 0): -1}},(0, 1): {(1, 0): {(1, 1): -1},(0, 1): {(0, 2): -1},(-1, 0): {(0, 1): -1},(0, -1): {(0, 0): -1}},(0, 2): {(1, 0): {(1, 2): -1000},(0, 1): {(0, 3): -1},(-1, 0): {(0, 2): -1},(0, -1): {(0, 1): -1}},(0, 3): {(1, 0): {(1, 3): -1},(0, 1): {(0, 4): -1},(-1, 0): {(0, 3): -1},(0, -1): {(0, 2): -1}},(0, 4): {(1, 0): {(1, 4): -1},(0, 1): {(0, 4): -1},(-1, 0): {(0, 4): -1},(0, -1): {(0, 3): -1}},(1, 0): {(1, 0): {(2, 0): -1},(0, 1): {(1, 1): -1},(-1, 0): {(0, 0): -1},(0, -1): {(1, 0): -1}},(1, 1): {(1, 0): {(2, 1): -1},(0, 1): {(1, 2): -1000},(-1, 0): {(0, 1): -1},(0, -1): {(1, 0): -1}},(1, 2): {(1, 0): {(2, 2): -1},(0, 1): {(1, 3): -1},(-1, 0): {(0, 2): -1},(0, -1): {(1, 1): -1}},(1, 3): {(1, 0): {(2, 3): -1},(0, 1): {(1, 4): -1},(-1, 0): {(0, 3): -1},(0, -1): {(1, 2): -1000}},(1, 4): {(1, 0): {(2, 4): 100},(0, 1): {(1, 4): -1},(-1, 0): {(0, 4): -1},(0, -1): {(1, 3): -1}},(2, 0): {(1, 0): {(2, 0): -1},(0, 1): {(2, 1): -1},(-1, 0): {(1, 0): -1},(0, -1): {(2, 0): -1}},(2, 1): {(1, 0): {(2, 1): -1},(0, 1): {(2, 2): -1},(-1, 0): {(1, 1): -1},(0, -1): {(2, 0): -1}},(2, 2): {(1, 0): {(2, 2): -1},(0, 1): {(2, 3): -1},(-1, 0): {(1, 2): -1000},(0, -1): {(2, 1): -1}},(2, 3): {(1, 0): {(2, 3): -1},(0, 1): {(2, 4): 100},(-1, 0): {(1, 3): -1},(0, -1): {(2, 2): -1}},(2, 4): {(1, 0): {(2, 4): 100},(0, 1): {(2, 4): 100},(-1, 0): {(1, 4): -1},(0, -1): {(2, 3): -1}}}
    convergenceTolerance = 10e-7
    gamma = .5
   
    '''
    levelsReward  = ["state", "action", "next state", "reward"]
    levelsTransition  = ["state", "action", "next state", "probability"]

    viewDictionaryStructure(transitionTable, levelsTransition)
    viewDictionaryStructure(rewardTable, levelsReward)
    '''
    
    policies={}
    policies["Completely random"]=lambda Q: e_greedyProbability(Q, 1)
    policies["e-greedy-0.5"]=lambda Q: e_greedyProbability(Q, 0.5)
    policies["e-greedy-0.1"]=lambda Q: e_greedyProbability(Q, 0.1)
    
    def getSPrimeRDistribution(s, action):
        reward=lambda sPrime: rewardTable[s][action][sPrime]
        p=lambda sPrime: transitionTable[s][action][sPrime]
        sPrimeRDistribution={(sPrime, reward(sPrime)): p(sPrime) for sPrime in transitionTable[s][action].keys()}
        return sPrimeRDistribution
    
    
    def updateMultipleSteps(policy, transitionTable, rewardTable, convergenceTolerance, gamma):
        S=list(transitionTable.keys())
        V={s:0 for s in S}
        deltas={s:np.Inf for s in S}
        
        getValueOfAState=lambda s: Bellman(s, policy, V, transitionTable, getSPrimeRDistribution, gamma)
        
        while max(deltas.values()) > convergenceTolerance:
            deltas={s:0 for s in S}
            v=V.copy()
            V={s: getValueOfAState(s) for s in S}
            deltas={s: abs(V[s]-v[s]) for s in S}
        
        return V
                
    policyEvaluation=lambda policy: updateMultipleSteps(policy, transitionTable, rewardTable, convergenceTolerance, gamma)
        
    allResults = {name: policyEvaluation(policy) for (name, policy) in policies.items()}
    
    for name, result in allResults.items():
        drawHeatMap(result, name)
        

if __name__=='__main__':
    main()

    

    
    

