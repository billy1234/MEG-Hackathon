import numpy as np
import mne

X_POLAR_RANGE = (0,2)
Y_POLAR_RANGE = (-1.6,1.6)
#Important these may not generalize over different machines well, fiddling may be reqired (-2,2) should be the max

def normScale(x,scale):
    return np.array([
        round(((x[0] - X_POLAR_RANGE[0]) / (X_POLAR_RANGE[1] - X_POLAR_RANGE[0]))*scale),
        round(((x[1] - Y_POLAR_RANGE[0]) / (Y_POLAR_RANGE[1] - Y_POLAR_RANGE[0]))*scale),
    ])

def partitionSpace(arr : np.ndarray,partitions : int =5):        
    return np.array([normScale(e,partitions) for e in arr])

def cart2polar(x):
    #theta then pi r is ommited. im taking x[0],1,2 to be x y z
    #taken from https://en.wikipedia.org/wiki/Spherical_coordinate_system
    radius = np.linalg.norm(x)
    return np.array([
        np.arccos(x[2]/radius), #arccos(z/r)
        np.arctan(x[1]/x[0]) #arctan(y/x)
    ]) 

def spacialPartitionSensors(record: mne.io.Raw,partitions: int):
    '''
    returns the cartesian and polar coordiantes of the sensors relative to the paitents head
    additionaly groups the sensors into partition x partition categories, this needs to only be done
    once per machine and can be matched up with all records from said machine
    '''
    positions = np.array([e['loc'][:3] for e in record.info['chs']])
    polarPositions = np.array([cart2polar(e) for e in positions])
    groupings = partitionSpace(polarPositions,partitions=partitions)

    return positions, polarPositions, groupings,record.info['chs']