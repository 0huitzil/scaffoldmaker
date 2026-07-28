#%%
# Number of elements per segment. Used to calculate the number of nodes per segment.
human_network_element_counts = {
    'headElementsCount': 3,
    'neckElementsCount': 2,
    'shoulderElementsCount': 2,
    'brachiumElementsCount': 3,
    'antebrachiumElementsCount': 2,
    'handElementsCount': 1,
    'thoraxElementsCount': 3,
    'abdomenElementsCount': 4,
    'hipElementsCount': 2,
    'upperLegElementsCount': 2,
    'lowerLegElementsCount': 2,
    'footElementsCount': 2
}

def create_segment_layout(nodeCount:int, nodeIdentifier:int,
                        initialJointNode:int = 0, versionStart:int = 0, versionEnd:int = 0):
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

def generate_network_layout_structure(humanElementCounts:dict):
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
    headNetworkLayout, nodeIdentifier = create_segment_layout(
        humanElementCounts['headElementsCount'], nodeIdentifier)
    # Neck
    neckNetworkLayout, nodeIdentifier = create_segment_layout(
        humanElementCounts['neckElementsCount'], nodeIdentifier, versionEnd=1)
    neckJointNode = nodeIdentifier
    # Thorax
    thoraxNetworkLayout, nodeIdentifier = create_segment_layout(
        humanElementCounts['thoraxElementsCount'], nodeIdentifier, versionStart=1)
    # Abdomen
    abdomenNetworkLayout, nodeIdentifier = create_segment_layout(
        humanElementCounts['abdomenElementsCount'], nodeIdentifier, versionEnd=1)
    pelvisJointNode = nodeIdentifier
    # Arms
    arms = []
    for i in range(2):
        version = 2 if (i == 0) else 3 #Left is 2, right is 3
        # Shoulder
        # shoulderNetworkLayout, nodeIdentifier = create_segment_layout(
        #     humanElementCounts['shoulderElementsCount'], nodeIdentifier,
        #     initialJointNode=neckJointNode, versionStart=version)
        # Shoulder and brachium
        brachiumNetworkLayout, nodeIdentifier = create_segment_layout(
            humanElementCounts['brachiumElementsCount'] \
            + humanElementCounts['shoulderElementsCount'], nodeIdentifier,
            initialJointNode=neckJointNode, versionStart=version)
        # Antebrachium
        antebrachiumNetworkLayout, nodeIdentifier = create_segment_layout(
            humanElementCounts['antebrachiumElementsCount'], nodeIdentifier)
        # Hand
        handNetworkLayout, nodeIdentifier = create_segment_layout(
            humanElementCounts['handElementsCount'], nodeIdentifier, versionEnd=1)
        # Join arm
        armNetworkLayout =  brachiumNetworkLayout \
              + antebrachiumNetworkLayout + handNetworkLayout
        arms.append(armNetworkLayout)
    #Legs
    legs = []
    for i in range(2):
        version = 2 if (i == 0) else 3 #Left is 2, right is 3
        # Hip
        # hipNetworkLayout, nodeIdentifier = create_segment_layout(
        #     humanElementCounts['hipElementsCount'], nodeIdentifier,
        #     initialJointNode=pelvisJointNode, versionStart=version)
        # Upper leg and hip
        upperLegNetworkLayout, nodeIdentifier = create_segment_layout(
            humanElementCounts['upperLegElementsCount'] \
            + humanElementCounts['hipElementsCount'],
            nodeIdentifier, initialJointNode=pelvisJointNode, versionStart=version)
        # Lower leg
        lowerLegNetworkLayout, nodeIdentifier = create_segment_layout(
            humanElementCounts['lowerLegElementsCount'], nodeIdentifier)
        # Foot 
        footNetworkLayout, nodeIdentifier = create_segment_layout(
            humanElementCounts['footElementsCount'], nodeIdentifier)
        # Join leg
        legNetworkLayout = upperLegNetworkLayout \
            + lowerLegNetworkLayout + footNetworkLayout
        legs.append(legNetworkLayout)
    # Joint network
    humanNetworkLayout = '(' + headNetworkLayout + neckNetworkLayout + arms[0] + arms[1] \
          + thoraxNetworkLayout + abdomenNetworkLayout  + legs[0] + legs[1]
    #Remove an extra comma at the end
    humanNetworkLayout = humanNetworkLayout[:-1]
    return humanNetworkLayout

# structure = generate_network_layout_structure(human_network_element_counts)
# structure = structure.replace(',', ',\n').splitlines()
# print(structure)
