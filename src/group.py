import numpy as np

'''
Generate groups for a pandas DataFrame using keys for splitting
'''
def generateGroups(frame):
    groups = np.empty(frame.index.size, dtype=int)

    groupKeys = {}
    i = 0
    for ind, key in enumerate(frame.index):
        groupChar = key[0]
        if groupChar not in groupKeys:
            groupKeys[groupChar] = i
            i += 1
        groups[ind] = groupKeys[groupChar]
    return groups

def generateGroups2(frame):
    groups = np.empty(frame.index.size, dtype=int)

    groupKeys = {}
    i = 0
    for ind, key in enumerate(frame.index):
        groupChar = key[0]
        if groupChar not in groupKeys:
            groupKeys[groupChar] = i
            i += 1
        groups[ind] = groupKeys[groupChar]
    groupCount = np.max(groups)+1

    clusters = []
    for i in xrange(groupCount):
        clusters.append(np.where(groups == i)[0])
    return clusters
