#! /usr/bin/env python3

import BinaryTrellis
import random

def deletionChannelSimulation(codeword, p, seed, trimmedZerosAtEdges=False):
    N = len(codeword)
    receivedWord = []

    random.seed(seed)


    for i in range(N):
        r = random.random()

        if r < p:
            pass
        else:
            receivedWord.append(codeword[i])


    if trimmedZerosAtEdges == True:
        trimmedReceivedWord = []

        firstOneIndex = -1

        for i in range(len(receivedWord)):
            if receivedWord[i] == 1:
                firstOneIndex = i
                break
        
        if firstOneIndex == -1:
            return trimmedReceivedWord # which is empty

        lastOneIndex = -1
        for i in range(len(receivedWord)-1,-1,-1):
            if receivedWord[i] == 1:
                lastOneIndex = i
                break

        assert(lastOneIndex != -1)

        for i in range(firstOneIndex, lastOneIndex+1):
            trimmedReceivedWord.append(receivedWord[i])

        return trimmedReceivedWord

    else:
        return receivedWord

def temp():
    bt = BinaryTrellis.BinaryTrellis(2)
    
    # print(bt)
    
    fromVertex_stateId = 3
    fromVertex_verticalPosInLayer = 17
    fromVertex_layer = 0
    
    toVertex_stateId = 4
    toVertex_verticalPosInLayer = 14
    toVertex_layer = 1
    
    edgeLabel = 0
    edgeProb = 0.5
    
    bt.addToEdgeProb(fromVertex_stateId, fromVertex_verticalPosInLayer, fromVertex_layer, toVertex_stateId, toVertex_verticalPosInLayer, toVertex_layer, edgeLabel, edgeProb)
    
    print(bt)

codeword = [0,0,1,0,1,0,1,1]

xi = 0.1
n = 3
n0 = 1

withGuardBand = BinaryTrellis.addDeletionGuardBands(codeword, n, n0, xi)

print(codeword)
print(withGuardBand)
print(BinaryTrellis.removeDeletionGuardBands(withGuardBand, n, n0))

# deletionProb = 0.01
# seed = 0
# trimmedZerosAtEdges=True
#
# receivedWord = deletionChannelSimulation(codeword, deletionProb, seed, trimmedZerosAtEdges)
# trellis = BinaryTrellis.buildTrellis_uniformInput_deletion(receivedWord, len(codeword), deletionProb, trimmedZerosAtEdges)
#
# print(codeword)
# print(receivedWord)
# print(trellis)
