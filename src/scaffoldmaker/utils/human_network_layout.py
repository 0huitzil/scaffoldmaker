#%%
# Number of elements per segment. Used to calculate the number of nodes per segment. 
humanElementCounts = {
    'headElementsCount': 3, 
    'neckElementsCount': 2, 
    'shoulderElementsCount': 2, 
    'brachiumElementsCount': 3, 
    'antebrachiumElementsCount': 3, 
    'handElementsCount': 1, 
    'thoraxElementsCount': 3, 
    'abdomenElementsCount': 4, 
    'hipElementsCount': 2, 
    'upperLegElementsCount': 4,
    'lowerLegElementsCount': 3,
    'footElementsCount': 1
}


def createSegment(nodeCount:int, networkLayout:str, nodeIdentifier:int, endSegment=False, version=0):
    """
    Construct a segment of the human network node

    :param nodeCount: Number of nodes to add.
    :param networkLayout: String containing the current network layout.
    :param nodeIdentifier: Integer denoting the current node.
    :param endSegment: If true, adds a comma at the end of the segment.
    :param version: If > 0, adds version number on the last node of the segment.
    :return networklayout: String containing the updated layout.
    :return nodeIdentifier: The updated nodeIdentifier after adding the segment.
    """
    for i in range(nodeCount):
        networkLayout = networkLayout + str(nodeIdentifier)
        if i == nodeCount - 1:
            if endSegment:
                if version == 0:
                    segmentConnector = ','
                else:
                    segmentConnector = '.' + str(version) + ','
                networkLayout = networkLayout + segmentConnector
            else:
                networkLayout = networkLayout + '-'
                nodeIdentifier += 1
        else:
            networkLayout = networkLayout + '-'
            nodeIdentifier += 1 
    return networkLayout, nodeIdentifier

def constructNetworkLayoutStructure(humanElementCounts:dict):
    """
    Construct the network layout of the human wholebody scaffold. 
    The network layout consists of the following segments: 
    head, neck, thorax, abdomen, right/left arm, right/left leg. 
    Arms are subdivided into brachium, antebrachium and hand. 
    Legs are subdivided into upper leg, lower leg and foot. 

    :param humanElementCounts: Dictionary containing the number of elements 
        corresponding to each segment. 
    :return humanNetworkLayout: String containing the network layout
    """
    nodeIdentifier = 1
    # Head
    headNetworkLayout = str(nodeIdentifier) + '-'
    nodeIdentifier += 1 
    headNetworkLayout, nodeIdentifier = createSegment(
        humanElementCounts['headElementsCount'], 
        headNetworkLayout, nodeIdentifier, endSegment=True)
    # Neck
    neckNetworkLayout = str(nodeIdentifier) + '-'
    nodeIdentifier += 1 
    neckNetworkLayout, nodeIdentifier = createSegment(
        humanElementCounts['neckElementsCount'], 
        neckNetworkLayout, nodeIdentifier, endSegment=True, version=1)
    neckJointNode = nodeIdentifier
    # Thorax 
    thoraxNetworkLayout = str(nodeIdentifier) + '.1-'
    nodeIdentifier += 1 
    thoraxNetworkLayout, nodeIdentifier = createSegment(
        humanElementCounts['thoraxElementsCount'], 
        thoraxNetworkLayout, nodeIdentifier, endSegment=True)
    # Abdomen 
    abdomenNetworkLayout = str(nodeIdentifier) + '-'
    nodeIdentifier += 1 
    abdomenNetworkLayout, nodeIdentifier = createSegment(
        humanElementCounts['abdomenElementsCount'], 
        abdomenNetworkLayout, nodeIdentifier, endSegment=True, version=1)
    pelvisNodeJoint = nodeIdentifier
    # Arms
    arms = []
    for i in range(2):
        version = 2 if (i == 0) else 3 #Left is 2, right is 3 
        armNetworkLayout = str(neckJointNode) + '.' + str(version) + '-'
        nodeIdentifier += 1  
        # Shoulder 
        armNetworkLayout, nodeIdentifier = createSegment(
            humanElementCounts['shoulderElementsCount'], 
            armNetworkLayout, nodeIdentifier, endSegment=True)
        # Brachium 
        armNetworkLayout = armNetworkLayout + str(nodeIdentifier) + '-'
        nodeIdentifier += 1 
        armNetworkLayout, nodeIdentifier = createSegment(
            humanElementCounts['brachiumElementsCount'], 
            armNetworkLayout, nodeIdentifier, endSegment=True)
        # Antebrachium 
        armNetworkLayout = armNetworkLayout + str(nodeIdentifier) + '-'
        nodeIdentifier += 1 
        armNetworkLayout, nodeIdentifier = createSegment(
            humanElementCounts['antebrachiumElementsCount'], 
            armNetworkLayout, nodeIdentifier, endSegment=True)
        # Hand
        armNetworkLayout = armNetworkLayout + str(nodeIdentifier) + '-'
        nodeIdentifier += 1 
        armNetworkLayout, nodeIdentifier = createSegment(
            humanElementCounts['handElementsCount'], 
            armNetworkLayout, nodeIdentifier, endSegment=True)
        arms.append(armNetworkLayout)
    #Legs 
    legs = []
    for i in range(2):
        version = 2 if (i == 0) else 3 #Left is 2, right is 3 
        legNetworkLayout = str(pelvisNodeJoint) + '.' + str(version) + '-'
        nodeIdentifier += 1
        # Hip
        legNetworkLayout, nodeIdentifier = createSegment(
            humanElementCounts['hipElementsCount'], 
            legNetworkLayout, nodeIdentifier, endSegment=True)
        # Upper leg
        legNetworkLayout = legNetworkLayout + str(nodeIdentifier) + '-'
        nodeIdentifier += 1 
        legNetworkLayout, nodeIdentifier = createSegment(
            humanElementCounts['upperLegElementsCount'], 
            legNetworkLayout, nodeIdentifier, endSegment=True)
        # Lower leg 
        legNetworkLayout = legNetworkLayout + str(nodeIdentifier) + '-'
        nodeIdentifier += 1 
        legNetworkLayout, nodeIdentifier = createSegment(
            humanElementCounts['lowerLegElementsCount'], 
            legNetworkLayout, nodeIdentifier, endSegment=True)
        # Foot 
        legNetworkLayout = legNetworkLayout + str(nodeIdentifier) + '-'
        nodeIdentifier += 1 
        legNetworkLayout, nodeIdentifier = createSegment(
            humanElementCounts['footElementsCount'], 
            legNetworkLayout, nodeIdentifier, endSegment=True)
        legs.append(legNetworkLayout)
    humanNetworkLayout = headNetworkLayout + neckNetworkLayout + arms[0] + arms[1]  + thoraxNetworkLayout + abdomenNetworkLayout  + legs[0] + legs[1]
    humanNetworkLayout = humanNetworkLayout[:-1] #Remove a comma at the end
    return humanNetworkLayout