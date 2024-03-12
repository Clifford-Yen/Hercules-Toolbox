import pandas as pd
import numpy as np
import gc
from tqdm import tqdm
import matplotlib.pyplot as plt
# plt.rcParams.update({'text.usetex': True, 'font.family': 'serif', 'font.serif': ['Computer Modern'], 
#     'font.size': 13, 'font.weight': 'regular'})
plt.rcParams.update({'font.size': 13, 'font.weight': 'regular'})

def get_MPI_data(comm=None, size=None, rank=None):
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        size = comm.Get_size() # total number of processors
        rank = comm.Get_rank() # the no. of current processor
        # print('multi-threading, processor %i out of %i' % (rank, size))
        if size > 1:
            MPI_enabled = True
        elif size == 1:
            MPI_enabled = False
        else: # size <= 0
            raise ValueError("Total number of processors should not be less than 1.")
    except ImportError:
        MPI_enabled = False
    return MPI_enabled, comm, size, rank

def getMesh(filePath) -> dict[str, pd.DataFrame]:
    df = {}
    with pd.HDFStore(filePath, mode='r') as store: 
        for key in store.keys():
            df[key.strip('/')] = store.get(key)
    return df

def getSpacing(pointList: np.ndarray|list[float]) -> float:
    pointList.sort()
    return np.diff(pointList).max()

def getElementCentroids(df: pd.DataFrame) -> pd.DataFrame:
    pointsPerElement = len(df.loc[df['geid'] == df['geid'].iloc[0]])
    numElements = len(df)//pointsPerElement
    eleCentroids = pd.DataFrame(np.zeros((numElements, 4)), columns=['geid', 'x', 'y', 'z'])
    for i in range(numElements):
        subset = df.iloc[i*pointsPerElement:(i+1)*pointsPerElement-1]
        eleCentroids.loc[i, 'geid'] = subset['geid'].iloc[0]
        eleCentroids.loc[i, 'x'] = subset['x'].mean()
        eleCentroids.loc[i, 'y'] = subset['y'].mean()
        eleCentroids.loc[i, 'z'] = subset['z'].mean()
    return eleCentroids

def getElementsOnLine(df: pd.DataFrame, xmin: float, xmax: float, ymin: float, ymax: float, zmin: float|None = None, zmax: float|None = None) -> tuple[pd.DataFrame, float]:
    # # Compute the mesh grid size
    # h = df.loc[1, 'x'] - df.loc[0, 'x']
    # subset = df.loc[(df['x'] > x-h/2) & (df['x'] < x+h/2) & (df['y'] >= ymin-h/2) & (df['y'] <= ymax+h/2)]
    # geids = subset['geid'].unique()
    # elementsCloseToLine = df.loc[df['geid'].isin(geids)]
    # elementCentroids = getElementCentroids(elementsCloseToLine)
    
    # xs = elementCentroids['x'].unique()
    # if len(xs) > 1:
    #     # If there are more than one x value, choose the closest one to x
    #     targetX = xs[np.argmin(np.abs(xs-x))]
    #     elementCentroids = elementCentroids.loc[elementCentroids['x'] == targetX]
    #     # NOTE: Hercules might not generate the mesh in a way that the centroids are exactly the same along the z-axis.
    #     elementCentroids = elementCentroids.loc[(elementCentroids['x'] > targetX-h/2) & (elementCentroids['x'] < targetX+h/2)]
    # NOTE: Hercules might not generate the mesh in a way that the centroids are exactly the same along the z-axis.
    # elementCentroids = elementCentroids.drop_duplicates(subset=['y', 'z'])
    
    # elementsOnLine = df.loc[df['geid'].isin(elementCentroids['geid'])]
    # return elementsOnLine, elementCentroids

    geids = []
    zList = df['z'].unique()
    zList.sort()
    if zmin is None:
        zmin = zList.min()
    if zmax is None:
        zmax = zList.max()
    # NOTE: For unknown reason, the spacing to find the subset of elements is not 
    # large enough to find the elements containing points in userMeshPlane in the
    # shallow part of the model.
    zShallow = zList.max()/16.0
    maxSpacing = np.diff(zList).max()
    zAbove = zList[0]
    print('Extracting elements on the line at different depths...')
    for z in tqdm(zList[1:]):
        # NOTE: Set spacing as follows should be enough for our purpose.
        h = z - zAbove
        if zAbove <= zShallow:
            h = 2*h
        if z < zmin:
            zAbove = z
            continue
        if z > zmax:
            z = zmax
        subset = df.loc[(df['z'] <= z) & (df['z'] >= zAbove)] # get partial nodes on elements
        # NOTE: although we have to execute the following lines to get the complete node sets 
        # of elements, it will takes a significant amount of time. And since what we need is 
        # to get geids, the partial nodes on elements are enough for our purpose.
        # subset = df.loc[subset.index.unique()] # get complete node sets of elements
        # NOTE: Getting spacing this way might be wrong if the elements we get are not continuous.
        # hx = getSpacing(subset['x'].unique())
        # hy = getSpacing(subset['y'].unique())
        # if hx > maxSpacing:
        #     maxSpacing = hx
        # if hy > maxSpacing:
        #     maxSpacing = hy
        subset = subset.loc[(subset['x'] >= xmin-h) & (subset['x'] <= xmax+h) & (subset['y'] >= ymin-h) & (subset['y'] <= ymax+h)]
        geids += subset.index.unique().tolist()
        zAbove = z
        if z == zmax:
            break
        # NOTE: For unknown reason, the progress bar does not show up in the console 
        # when MPI is enabled and only the rank 0 processor prints the progress bar.
        # To fix this, print a new line and clear the line does the trick.
        # \x1b is the ANSI escape character, 1A moves the cursor up one line, 2K 
        # clears the entire line. The end='\r' is the ASCII Carriage Return (CR) 
        # character which moves the cursor to the beginning of the line without 
        # moving to the next line. For more info about ANSI escape codes, see:
        # https://en.wikipedia.org/wiki/ANSI_escape_code
        print('')
        print('\x1b[1A\x1b[2K', end='\r')
    geids = list(set(geids)) # Keep only the unique elements
    return df.loc[geids], maxSpacing
    # elementCentroids = getElementCentroids(elementsOnLine)
    # return elementsOnLine, elementCentroids

def plotProperties(df: pd.DataFrame, direction: str = 'y', outputFileName: str = 'properties.pdf', 
        minorTicksIncluded: bool = False) -> None:
    propsTable = {'Vs': '$V_s\ (m/s)$', 'Vp': '$V_p\ (m/s)$', 'rho': '$\\rho\ (kg/m^3)$'}
    fig, ax = plt.subplots(1, 3, figsize=(20, 5))
    fig.set_tight_layout(True)
    # ===== Plot a 2D scatter plot of Vs, Vp, and rho =====
    xMin = df[direction].min()
    xMax = df[direction].max()
    yMin = df['z'].min()
    if yMin > 0:
        yMin = 0
    yMax = df['z'].max()
    for i, prop in enumerate(propsTable.keys()):
        scatter = ax[i].scatter(df[direction], df['z'], c=df[prop])
        if minorTicksIncluded:
            ax[i].set_xticks([xMin, xMax], [int(xMin), int(xMax)], minor=True)
            ax[i].tick_params(axis='x', which='minor', length=20, color='tab:gray', labelcolor='tab:gray')
            ax[i].xaxis.labelpad = -10
        ax[i].set_yticks(np.append(ax[i].get_yticks(), [yMin, yMax]))
        ax[i].set_xlim(xMin, xMax)
        ax[i].set_ylim(yMin, yMax)
        ax[i].set_xlabel('$y\ (m)$')
        ax[i].set_ylabel('$z\ (m)$')
        ax[i].set_title(propsTable[prop])
        ax[i].invert_yaxis()
        fig.colorbar(scatter, ax=ax[i])
    # ===== Plot a 2D contour plot of Vs, Vp, and rho =====
    # for i, prop in enumerate(propsTable.keys()):
    #     x = df['y'].drop_duplicates()
    #     y = df['z'].drop_duplicates()
    #     levels = np.linspace(df[prop].min(), df[prop].max(), 100)
    #     contour = ax[i].contourf(x, y, df.pivot(index='z', columns='y', values=prop), levels)
    #     ax[i].set_xlabel('$y$')
    #     ax[i].set_ylabel('$z$')
    #     ax[i].set_title(propsTable[prop])
    #     ax[i].invert_yaxis()
    #     fig.colorbar(contour, ax=ax[i])
    # ===== save the figure =====
    fig.savefig(outputFileName)
    return

def plot3DVelocityModel(fileName: str, xmin: float, xmax: float, ymin: float, 
        ymax: float, zmin: float|None = None, zmax: float|None = None, 
        outputFileName='3DVelocityModelProperties.pdf', 
        minorTicksIncluded: bool = False) -> None:
    df = pd.read_csv(fileName)
    df = df.loc[(df['x'] >= xmin) & (df['x'] <= xmax) & (df['y'] >= ymin) & (df['y'] <= ymax)]
    if zmin is not None:
        df = df.loc[df['z'] >= zmin]
    if zmax is not None:
        df = df.loc[df['z'] <= zmax]
    df.loc[:, 'rho'] = df['rho']*1000
    df.columns = ['x', 'y', 'z', 'Vs', 'Vp', 'rho']
    if xmin == xmax:
        direction = 'y'
    elif ymin == ymax:
        direction = 'x'
    else:
        ValueError('The user-defined mesh plane should be either in the x-z or y-z plane.')
    plotProperties(df, direction=direction, outputFileName=outputFileName, minorTicksIncluded=minorTicksIncluded)
    return

def plotVelocityProfileWithUserMeshPlane(meshDatabaseFilePath: str, xmin: float, xmax: float, 
        ymin: float, ymax: float, zmax: float, method: str = 'fast'):
    MPI_enabled, comm, size, rank = get_MPI_data()
    if not MPI_enabled or rank == 0:
        dfMeshCoordinates = pd.read_hdf(meshDatabaseFilePath, key='meshCoordinates')
        # NOTE: Pandas allows duplicate indices. To make the searching process faster 
        # in the later steps, the indices are set to the 'geid' column.
        dfMeshCoordinates.set_index('geid', inplace=True)
        elementsOnLine, h = getElementsOnLine(dfMeshCoordinates, xmin, xmax, ymin, ymax, zmax=zmax)
        # To keep the memory usage low (without using swap memory), delete df and collect the memory back.
        del dfMeshCoordinates
        gc.collect()
        dfMeshData = pd.read_hdf(meshDatabaseFilePath, key='meshData')
        dfMeshData.set_index('geid', inplace=True)
        # ===== Define a mesh plane =====
        # Generate a grid of points
        if xmin == xmax:
            direction = 'y'
            y, z = np.mgrid[ymin:ymax+spacing:spacing, 0:zmax+zSpacing:zSpacing]
            x = np.full_like(y, xmin)
        elif ymin == ymax:
            direction = 'x'
            x, z = np.mgrid[xmin:xmax+spacing:spacing, 0:zmax+zSpacing:zSpacing]
            y = np.full_like(x, ymin)
        else:
            ValueError('The user-defined mesh plane should be either in the x-z or y-z plane.')
        # Stack the arrays along the last axis to get an array of coordinates
        userMeshPlane = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    else:
        elementsOnLine = None
        h = None
        dfMeshData = None
        userMeshPlane = None
    if MPI_enabled:
        comm.barrier()
        elementsOnLine = comm.bcast(elementsOnLine, root=0)
        h = comm.bcast(h, root=0)
        dfMeshData = comm.bcast(dfMeshData, root=0)
        userMeshPlane = comm.bcast(userMeshPlane, root=0)
        totalMeshPoints = len(userMeshPlane)
        if rank > totalMeshPoints:
            exit()
        userMeshPlane = userMeshPlane[rank::size]
    userMeshData = pd.DataFrame(np.zeros((len(userMeshPlane), 6)), columns=['x', 'y', 'z', 'Vs', 'Vp', 'rho'])
    userMeshData[['x', 'y', 'z']] = userMeshPlane
    if not MPI_enabled or rank == 0:
        print('Extracting velocity profiles to the user-defined mesh plane...')
    if MPI_enabled: # makes the proper printing order
        comm.barrier()
    for i, point in enumerate(tqdm(userMeshPlane, position=rank, desc=f'Processor {rank}')):
        # # First check whether the point is at any of the corners of the elements
        # subset = elementsOnLine.loc[(elementsOnLine['x'] == point[0]) & (elementsOnLine['y'] == point[1]) & (elementsOnLine['z'] == point[2])]
        # if len(subset) > 0:
        #     for prop in ['Vs', 'Vp', 'rho']:
        #         userMeshData.loc[i, prop] = dfMeshData.loc[subset.index][prop].mean()
        # else:
        # Retrieve the elements that are close to the point
        subset = elementsOnLine.loc[(elementsOnLine['x'] >= point[0]-h/2) & (elementsOnLine['x'] <= point[0]+h/2) & 
            (elementsOnLine['y'] >= point[1]-h/2) & (elementsOnLine['y'] <= point[1]+h/2) & 
            (elementsOnLine['z'] >= point[2]-h/2) & (elementsOnLine['z'] <= point[2]+h/2)]
        geids = subset.index.unique()
        hLocal = h
        for geid in geids:
            element = elementsOnLine.loc[geid]
            # isInElement = True
            # ===== Safer way to check whether the point is inside the element =====
            # for coord in ['x', 'y', 'z']:
            #     numBiggerThan = (element[coord] > userMeshData.loc[i, coord]).sum()
            #     numSmallerThan = (element[coord] < userMeshData.loc[i, coord]).sum()
            #     # NOTE: If both numBiggerThan and numSmallerThan are 4, the point is inside the element
            #     # If one of them is 4 and the other is 0, the point is on the boundary of the element
            #     if numBiggerThan == 8 or numSmallerThan == 8: # outside the element
            #         isInElement = False
            #         break
            # if isInElement:
            # ===== Faster way. Assuming the cube is aligned with the axes =====
            # if (element['x'].min() <= point[0] <= element['x'].max() and
            #     element['y'].min() <= point[1] <= element['y'].max() and
            #     element['z'].min() <= point[2] <= element['z'].max()):
            # ===== Even faster way. Assuming the first node is the minimum and the last node is the maximum =====
            if (element['x'].iloc[0] <= point[0] <= element['x'].iloc[-1] and
                element['y'].iloc[0] <= point[1] <= element['y'].iloc[-1] and
                element['z'].iloc[0] <= point[2] <= element['z'].iloc[-1]):
                # /// For a faster result, use the first element that contains the point
                if method == 'fast':
                    for prop in ['Vs', 'Vp', 'rho']:
                        userMeshData.loc[i, prop] = dfMeshData.loc[geid][prop]
                    break
                # /// For a more accurate result, use the smallest element to represent the point
                elif method == 'accurate':
                    hz = getSpacing(element['z'].unique())
                    if hz <= hLocal:
                        hLocal = hz
                        for prop in ['Vs', 'Vp', 'rho']:
                            userMeshData.loc[i, prop] = dfMeshData.loc[geid][prop]
        if userMeshData.loc[i, 'Vs'] == 0:
            raise ValueError(f'No element is found for the point ({point[0]}, {point[1]}, {point[2]}).')
    if MPI_enabled:
        if rank > 0 and totalMeshPoints > 0:
            comm.send(userMeshData, dest=0, tag=41) # tag is arbitrary
        else:
            for i in range(1, size):
                userMeshData = pd.concat([userMeshData, comm.recv(source=i, tag=41)])
            userMeshData.sort_values(by=['x', 'y', 'z'], inplace=True)
    if not MPI_enabled or rank == 0:
        userMeshData.to_csv('userMeshPlane.csv', index=False)
        plotProperties(userMeshData, direction=direction, minorTicksIncluded=True)
    return

if __name__ == '__main__':
    # /// Change the current working directory to the directory of this file for debugging purpose.
    import os
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    filePath = 'database/mesh.hdf5'
    # keys = ['meshCoordinates', 'meshData']
    # ===== Example 1: along the y-axis at x = 51500 =====
    xmin = 51500
    xmax = 51500
    ymin = 0
    ymax = 62500
    # ===== Example 2: along the x-axis at y = 55000 =====
    # xmin = 35000
    # xmax = 60000
    # ymin = 55000
    # ymax = 55000
    # ===== =====
    zmax = 21000
    spacing = 500 # spacing between the points in the user-defined mesh plane in both x and y directions
    zSpacing = 200

    # NOTE: Since the domain used in Hercules is larger than the velocity model 
    # (to include all the source points), the distance between them should be 
    # taken into account when querying the mesh data.
    xDiff = 16000
    yDiff = 5000

    MPI_enabled, comm, size, rank = get_MPI_data()
    if not MPI_enabled or rank == 0:
        plot3DVelocityModel('./inputfiles/materialfiles/herculesVelocityModel.csv', xmin, xmax, ymin, ymax, zmin=0, minorTicksIncluded=True)
    plotVelocityProfileWithUserMeshPlane(filePath, xmin+xDiff, xmax+xDiff, ymin+yDiff, ymax+yDiff, zmax, method='fast')

    # ===== Plot the velocity profile at centroids of the elements =====
    # NOTE: This is not used anymore.
    # df = getMesh(filePath)
    # elementsOnLine, elementCentroids = getElementsOnLine(df['meshCoordinates'], x+xDiff, ymin+yDiff, ymax+yDiff, zmax)
    # # Keep only the elements that are on the line
    # df['meshData'] = df['meshData'].loc[df['meshData']['geid'].isin(elementCentroids['geid'])]
    # # combine df['meshData'] and elementCentroids
    # df['meshData'] = df['meshData'].merge(elementCentroids, on='geid')
    # df['meshData'].to_csv('elementsOnLine.csv', index=False)
    # plotProperties(df['meshData'])
    # ===== =====
