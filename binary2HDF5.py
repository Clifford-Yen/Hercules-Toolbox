import numpy as np
import pandas as pd
import gc
import os
import time
import argparse
import progressbar

class Timer(object):
    def __init__(self, name=None):
        self.name = name
    def __enter__(self):
        self.tstart = time.time()
    def __exit__(self, type, value, traceback):
        if self.name:
            print('[%s]' % self.name,)
        print('Elapsed: %.2f secs' % (time.time() - self.tstart))

def getDataFromInputFile(inputFilePath: str) -> dict:
    ''' getDataFromInputFile returns a dict with all Hercules'
    parameters as the dict keys and values as the dict values. '''
    with open(inputFilePath) as f:
        lines = f.readlines()
        lines = [line.rstrip().split('=') for line in lines if (line[0] != '#' and line.rstrip() != '')]
    inputData = {}
    heldKey = None
    for line in lines:
        if len(line) == 2 and line[1] != '':
            inputData[line[0].rstrip()] = line[1].lstrip()
            heldKey = None
        elif len(line) == 2 and line[1] == '':
            heldKey = line[0].rstrip()
            inputData[line[0].rstrip()] = []
        elif len(line) == 1:
            inputData[heldKey].append(line[0].split())
        else:
            raise ValueError('Unexpected input file format.')
    return inputData

def backupAndRemoveFile(filePath):
    if os.path.exists(filePath):
        backupFilePath = filePath + '.bak'
        if os.path.exists(backupFilePath):
            os.remove(backupFilePath)
        os.rename(filePath, backupFilePath)

def getDisplacementHistory(pathToBinaryFile, planeData, deltaT, columns=['timeStep', 'time', 'x', 'y', 'z', 'u', 'v', 'w']):
    # NOTE: planeData is the plane geometry that comes from the parameters.in file
    depth = planeData[2]
    alongStrike = planeData[4] # Number of Nodes in X direction
    stepAlongStrike = planeData[3] # Distance between nodes in X direction (dstrk)
    downDip = planeData[6] # Number of Nodes in Y direction
    stepDownDip = planeData[5] # Distance between nodes in Y direction
    numGridPoints = alongStrike*downDip
    numTotalComponents = numGridPoints*3 # 3 for displacements in x, y, and z directions
    with open(pathToBinaryFile, 'rb') as f:
        fullArray = np.fromfile(f, dtype=np.float64)
        # /// Handling the case when the analysis is not completed
        if fullArray.size % numTotalComponents != 0:
            # /// Drop the last time step if it is not completed
            fullArray = fullArray[:-(fullArray.size % numTotalComponents)]
    numTimeSteps = int(fullArray.size/numTotalComponents)
    df = pd.DataFrame(np.hstack([np.repeat(np.arange(numTimeSteps), numGridPoints).reshape(-1, 1),
        np.repeat(np.arange(numTimeSteps)*deltaT, numGridPoints).reshape(-1, 1),
        np.tile(np.repeat(np.arange(0, stepAlongStrike*alongStrike, stepAlongStrike), downDip), numTimeSteps).reshape(-1, 1),
        np.tile(np.tile(np.arange(0, stepDownDip*downDip, stepDownDip), alongStrike), numTimeSteps).reshape(-1, 1),
        np.repeat(depth, numTimeSteps*numGridPoints).reshape(-1, 1),
        fullArray.reshape(-1, 3)]), columns=columns)
    # dataType = {'timeStep': 'int32', 'time': 'float32', 'x': 'float32', 'y': 'float32', 'z': 'float32'}
    # NOTE: category data type is used to reduce memory usage and improve performance. 
    # However, it can only compare equality and not greater than or less than.
    # dataType = {'timeStep': 'category', 'time': 'category'}
    # NOTE: The following line creates a copy of the dataframe, which results in a memory increase. 
    # df = df.astype(dataType)
    # NOTE: The following line only creates a copy of the column, which only results in a slight memory increase.
    df['timeStep'] = df['timeStep'].astype(int)
    return df

def binaryPlane2HDF5(inputFilePath, dbFolder='database'):
    binaryFileName='planedisplacements' # NOTE: binaryFileName should not include .0 or .1, etc.
    inputData = getDataFromInputFile(inputFilePath)
    # NOTE: Convert strings to floats first, then convert the floats to integers by using the built-in is_integer() function.
    # Note that the conversion to integers is critical as this program will use nx and ny as the inputs of range() function 
    # for determining the coordinates of grid points.
    # And since range() function only takes integers as input, failing to convert nx and ny correctly will raise the error.
    planesData = [[float(x) for x in plane] for plane in inputData['output_planes']]
    planesData = [[int(x) if x.is_integer() else x for x in plane] for plane in planesData]
    # Remove the planes that are not considered in the analysis
    planesData = planesData[:int(inputData['number_output_planes'])]
    domainSurfaceCorners = [[float(x) for x in corner] for corner in inputData['domain_surface_corners']]
    deltaT = int(inputData['output_planes_print_rate']) * float(inputData['simulation_delta_time_sec'])
    # Create dbFolder if it does not exist.
    if not os.path.exists(dbFolder):
        os.makedirs(dbFolder)
    dbPath = os.path.join(dbFolder, binaryFileName+'.hdf5')
    backupAndRemoveFile(dbPath)
    df = pd.DataFrame(planesData, columns=['x', 'y', 'z', 'dx', 'nx', 'dy', 'ny', 'strk', 'dp'])
    df.to_hdf(dbPath, key='planesData', mode='w')
    df = pd.DataFrame(domainSurfaceCorners, columns=['x', 'y'])
    df.to_hdf(dbPath, key='domainSurfaceCorners', mode='a')
    print('Processing planes...')
    for i, planeData in enumerate(progressbar.progressbar(planesData)):
        pathToBinaryFile = os.path.join(inputData['output_planes_directory'], binaryFileName+'.%i'%i)
        if not os.path.exists(pathToBinaryFile):
            continue
        df = getDisplacementHistory(pathToBinaryFile, planeData, deltaT)
        df.to_hdf(dbPath, key='plane%i'%i, mode='a', format='table', data_columns=True, index=False, complevel=9, complib="blosc:lz4")
        # To keep the memory usage low (without using swap memory), delete df and collect the memory back.
        del df
        gc.collect()
    return

def getMeshCoordinates(coordinateFiles, meshFolder):
    print('Getting mesh coordinates...')
    # ===== Without geid =====
    # finalArray = np.hstack([np.fromfile(os.path.join(meshFolder, coordinateFile), dtype=np.float64)
    #     for coordinateFile in progressbar.progressbar(coordinateFiles)])
    # finalArray = finalArray.reshape(-1, 3)
    # numElements = int(finalArray.shape[0]/8)
    # # NOTE: eleIDs are different from geid used in Hercules. Just to make it easier to compare with meshdata.
    # eleID = np.arange(numElements)
    # finalArray = np.hstack((eleID.repeat(8).reshape(-1, 1), finalArray))
    # ===== With geid =====
    finalArray = np.hstack([np.fromfile(os.path.join(meshFolder, coordinateFile), 
        dtype=[('geid', np.int64), ('x', np.float64), ('y', np.float64), ('z', np.float64)]) 
        for coordinateFile in progressbar.progressbar(coordinateFiles)])
    return finalArray

def getMeshData(meshDataFiles, meshFolder):
    print('Getting mesh data...')
    # ===== Without geid =====
    # finalArray = np.hstack([np.fromfile(os.path.join(meshFolder, meshDataFile), dtype=np.float32)
    #     for meshDataFile in progressbar.progressbar(meshDataFiles)])
    # finalArray = finalArray.reshape(-1, 3)
    # # NOTE: eleIDs are different from geid used in Hercules. Just to make it easier to compare with meshcoordinates.
    # eleID = np.arange(finalArray.shape[0])
    # finalArray = np.hstack((eleID.reshape(-1, 1), finalArray))
    # ===== With geid =====
    finalArray = np.hstack([np.fromfile(os.path.join(meshFolder, meshDataFile), 
        dtype=[('geid', np.int64), ('Vs', np.float32), ('Vp', np.float32), ('rho', np.float32)]) 
        for meshDataFile in progressbar.progressbar(meshDataFiles)])
    return finalArray

def binaryMesh2HDF5(inputFilePath, dbFolder='database'):
    coordinateFileName = 'mesh_coordinates'
    meshDataFileName = 'mesh_data'
    inputData = getDataFromInputFile(inputFilePath)
    meshFolder = inputData['mesh_coordinates_directory_for_matlab']
    coordinateFiles = [fileName for fileName in os.listdir(meshFolder) if fileName.startswith(coordinateFileName)]
    coordinateFiles.sort()
    meshDataFiles = [fileName for fileName in os.listdir(meshFolder) if fileName.startswith(meshDataFileName)]
    meshDataFiles.sort()
    # /// Create dbFolder if it does not exist.
    if not os.path.exists(dbFolder):
        os.makedirs(dbFolder)
    dbPath = os.path.join(dbFolder, 'mesh.hdf5')
    backupAndRemoveFile(dbPath)
    meshCoordinates = getMeshCoordinates(coordinateFiles, meshFolder)
    # ===== Without geid =====
    # df = pd.DataFrame(meshCoordinates, columns=['geid', 'x', 'y', 'z'])
    # df['geid'] = df['geid'].astype(int)
    # ===== With geid =====
    df = pd.DataFrame(meshCoordinates)
    # ===== =====
    df.to_hdf(dbPath, key='meshCoordinates', mode='w', format='table', data_columns=True, index=False, complevel=9, complib="blosc:lz4")
    # To keep the memory usage low (without using swap memory), delete df and collect the memory back.
    del df
    gc.collect()
    meshData = getMeshData(meshDataFiles, meshFolder)
    # ===== Without geid =====
    # df = pd.DataFrame(meshData, columns=['geid', 'Vs', 'Vp', 'rho'])
    # df['geid'] = df['geid'].astype(int)
    # ===== With geid =====
    df = pd.DataFrame(meshData)
    # ===== =====
    df.to_hdf(dbPath, key='meshData', mode='a', format='table', data_columns=True, index=False, complevel=9, complib="blosc:lz4")
    return

if __name__ == '__main__':
    # DEBUGGING: Change the current working directory to the directory of this file for debugging purpose.
    # os.chdir(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(prog='binary2HDF5',
        description="Convert Hercules' binary files to HDF5 format. The generated \
            HDF5 files will be stored in the 'database' folder and can be used \
            for making velocity/displacement propagation animations or plotting \
            velocity profiles.")
    inputFileHelp = 'Path to the input file. Default: inputfiles/parameters.in'
    parser.add_argument('inputFilePath', nargs='?', help=inputFileHelp, default='inputfiles/parameters.in')
    binaryTypeHelp = "Type of the binary file. Default is 'plane', and another option is 'mesh'."
    parser.add_argument('--binaryType', '-t', help=binaryTypeHelp, default='plane')
    inputFilePath = parser.parse_args().inputFilePath
    binaryType = parser.parse_args().binaryType
    with Timer():
        if binaryType == 'plane':
            binaryPlane2HDF5(inputFilePath)
        elif binaryType == 'mesh':
            binaryMesh2HDF5(inputFilePath)
        else:
            raise ValueError("Invalid binary type. Only 'plane' and 'mesh' are supported.")