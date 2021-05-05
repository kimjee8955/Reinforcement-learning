


def expect(xDistribution, function):
    fxProduct=[px*function(x) for x, px in xDistribution.items()]
    expectation=sum(fxProduct)
    return expectation


def forward(xT_1Distribution, eT, transitionTable, sensorTable):
    temp = {xT:expect(xT_1Distribution,lambda xt_1:transitionTable[xt_1][xT])*sensorTable[xT][eT] for xT in xT_1Distribution.keys()}
    if sum(temp.values()) == 0:
        alpha = 0
    else:
        alpha = 1/sum(temp.values())
    xT = {nextState:alpha*prob for nextState,prob in temp.items()}

    return xT

def main():
    
    pX0={0:0.3, 1:0.7}
    e=1
    transitionTable={0:{0:0.6, 1:0.4}, 1:{0:0.3, 1:0.7}}
    sensorTable={0:{0:0.6, 1:0.3, 2:0.1}, 1:{0:0, 1:0.5, 2:0.5}}
    
    xTDistribution=forward(pX0, e, transitionTable, sensorTable)
    print(xTDistribution)

if __name__=="__main__":
    main()