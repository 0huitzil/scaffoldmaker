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

def createLayoutSegment(nodeCount:int, nodeIdentifier:int, initialJointNode = 0, versionStart=0, versionEnd=0):
    """
    Construct a segment of the human network node

    :param nodeCount: Number of nodes to add.
    
    :param networkLayout: String containing the current network layout.
    :type nodeCount: int
    :param nodeIdentifier: Integer denoting the current node.
    :type nodeCount: int
    :param versionStart: If > 0, adds version number on the first node of the segment.
    :param versionEnd: If > 0, adds version number on the last node of the segment.
    :return networklayout: String containing the layout segment.
    :return nodeIdentifier: The updated nodeIdentifier after adding the segment.
    """
    if initialJointNode == 0:
        networkLayout = str(nodeIdentifier)
    else:
        networkLayout = str(initialJointNode)
    if versionStart == 0:
        segmentConnector = '-'
    else:
        segmentConnector = '.' + str(versionStart) + '-'
    networkLayout = networkLayout + segmentConnector
    nodeIdentifier += 1
    for i in range(nodeCount):
        networkLayout = networkLayout + str(nodeIdentifier)
        if i < nodeCount - 1:
            segmentConnector = '-'
            nodeIdentifier += 1
        else:
            if versionEnd == 0:
                segmentConnector = ','
            else:
                segmentConnector = '.' + str(versionEnd) + ','
        networkLayout = networkLayout + segmentConnector
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
    # Head
    nodeIdentifier = 1
    headNetworkLayout, nodeIdentifier = createLayoutSegment(humanElementCounts['headElementsCount'], nodeIdentifier)
    # Neck
    neckNetworkLayout, nodeIdentifier = createLayoutSegment(
        humanElementCounts['neckElementsCount'], nodeIdentifier, versionEnd=1)
    neckJointNode = nodeIdentifier
    # Thorax 
    thoraxNetworkLayout, nodeIdentifier = createLayoutSegment(
        humanElementCounts['thoraxElementsCount'], nodeIdentifier, versionStart=1)
    # Abdomen 
    abdomenNetworkLayout, nodeIdentifier = createLayoutSegment(
        humanElementCounts['abdomenElementsCount'], nodeIdentifier, versionEnd=1)
    pelvisJointNode = nodeIdentifier
    # Arms
    arms = []
    for i in range(2):
        version = 2 if (i == 0) else 3 #Left is 2, right is 3 
        # Shoulder 
        shoulderNetworkLayout, nodeIdentifier = createLayoutSegment(
            humanElementCounts['shoulderElementsCount'], nodeIdentifier, 
            initialJointNode=neckJointNode, versionStart=version)
        # Brachium 
        brachiumNetworkLayout, nodeIdentifier = createLayoutSegment(
            humanElementCounts['brachiumElementsCount'], nodeIdentifier)
        # Antebrachium 
        antebrachiumNetworkLayout, nodeIdentifier = createLayoutSegment(
            humanElementCounts['antebrachiumElementsCount'], nodeIdentifier)
        # Hand
        handNetworkLayout, nodeIdentifier = createLayoutSegment(
            humanElementCounts['handElementsCount'], nodeIdentifier, versionEnd=1)
        handJointNode = nodeIdentifier
        # Join arm
        armNetworkLayout = shoulderNetworkLayout + brachiumNetworkLayout + antebrachiumNetworkLayout + handNetworkLayout
        arms.append(armNetworkLayout)
    #Legs 
    legs = []
    for i in range(2):
        version = 2 if (i == 0) else 3 #Left is 2, right is 3 
        # Hip
        hipNetworkLayout, nodeIdentifier = createLayoutSegment(
            humanElementCounts['hipElementsCount'], nodeIdentifier, 
            initialJointNode=pelvisJointNode, versionStart=version)
        # Upper leg
        upperLegNetworkLayout, nodeIdentifier = createLayoutSegment(
            humanElementCounts['upperLegElementsCount'], nodeIdentifier)
        # Lower leg 
        lowerLegNetworkLayout, nodeIdentifier = createLayoutSegment(
            humanElementCounts['lowerLegElementsCount'], nodeIdentifier)
        # Foot 
        footNetworkLayout, nodeIdentifier = createLayoutSegment(
            humanElementCounts['footElementsCount'], nodeIdentifier)
        # Join leg
        legNetworkLayout = hipNetworkLayout + upperLegNetworkLayout + lowerLegNetworkLayout + footNetworkLayout
        legs.append(legNetworkLayout)
    # Joint network
    humanNetworkLayout = headNetworkLayout + neckNetworkLayout + arms[0] + arms[1]  + thoraxNetworkLayout + abdomenNetworkLayout  + legs[0] + legs[1]
    #Remove an extra comma at the end
    humanNetworkLayout = humanNetworkLayout[:-1] 
    return humanNetworkLayout

# constructNetworkLayoutStructure(humanElementCounts).replace(',', ',\n').splitlines()
