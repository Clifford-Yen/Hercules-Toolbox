import pandas as pd
import numpy as np
import gc
from tqdm import tqdm
import argparse
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
        print("mpi4py is not installed. Running the code in a single processor mode.")
        MPI_enabled = False
    return MPI_enabled, comm, size, rank

def getSpacing(pointList: np.ndarray|list[float], returnType='max') -> float:
    pointList.sort()
    if returnType == 'max':
        return np.diff(pointList).max()
    elif returnType == 'min':
        return np.diff(pointList).min()
    else:
        raise ValueError('The returnType should be either "max" or "min".')

def getElementsOnPlane(df: pd.DataFrame, xmin: float, xmax: float, ymin: float, ymax: float, 
    zmin: float|None = None, zmax: float|None = None) -> tuple[pd.DataFrame, float]:
    geids = []
    zList = df['z'].unique()
    zList.sort()
    # NOTE: For unknown reason, the spacing to find the subset of elements is not 
    # large enough to find the elements containing points in userMeshPlane in the
    # shallow part of the model. To mitigate this problem, the spacing is set to 
    # be 2 times larger in the shallow part of the model.
    zShallow = zList.max()/16.0
    if zmin is None:
        zmin = zList.min()
    if zmax is None:
        zmax = zList.max()
    else:
        res = np.where(zList > zmax)[0]
        if res.size > 0:
            zList = zList[:res[0]+1]
    maxSpacing = 0
    zAbove = zList[0]
    for z in tqdm(zList[1:]):
        # NOTE: Set spacing as follows should be enough for our purpose.
        h = z - zAbove
        if zAbove <= zShallow:
            h = 2*h
        if maxSpacing < h:
            maxSpacing = h
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
        # NOTE: Even if we only halve the spacing for the direction of the crossing line, and 
        # ideally, the maximum distance between the points and the line should be h/2, we still
        # can't find the elements containing the points in some parts of the model (and it's not
        # in the shallow part of the model). As a result, the spacing is still set to be h.
        # hx = h/2 if xmin == xmax else h
        # hy = h/2 if ymin == ymax else h
        subset = subset.loc[(subset['x'] >= xmin-h) & (subset['x'] <= xmax+h) & (subset['y'] >= ymin-h) & (subset['y'] <= ymax+h)]
        geids += subset.index.unique().tolist()
        zAbove = z
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
    elementsOnPlane = df.loc[geids]
    maxSpacing = max(maxSpacing, getSpacing(elementsOnPlane['z'].unique()))
    return elementsOnPlane, maxSpacing

def plotProperties(df: pd.DataFrame, planeNormal: str = 'x', outputFileName: str = 'properties.pdf', 
        minorTicksIncluded: bool = False) -> None:
    propsTable = {'Vs': '$V_s\ (m/s)$', 'Vp': '$V_p\ (m/s)$', 'rho': '$\\rho\ (kg/m^3)$'}
    fig, ax = plt.subplots(1, 3, figsize=(20, 5))
    fig.set_tight_layout(True)
    if planeNormal == 'x':
        dirs = ['y', 'z']
    elif planeNormal == 'y':
        dirs = ['x', 'z']
    elif planeNormal == 'z':
        dirs = ['y', 'x']
    # ===== Calculate a marker size to fill the entire plot =====
    # NOTE: Have no idea how to calculate the marker size that exactly fills the entire plot.
    markerSize = max([ax[0].bbox.size[i]/(len(df[dirs[i]].unique())-1) for i in range(2)])/2
    # print(f'markerSize: {markerSize}')
    # ===== Plot a 2D scatter plot of Vs, Vp, and rho =====
    xMin = df[dirs[0]].min()
    xMax = df[dirs[0]].max()
    yMin = df[dirs[1]].min()
    yMax = df[dirs[1]].max()
    # Remove air elements (Vs=1e10, Vp=-1.0, rho=0) as defined in Hercules
    df = df.loc[(df['Vs'] != 1e10) & (df['Vp'] != -1.0) & (df['rho'] != 0)]
    for i, prop in enumerate(propsTable.keys()):
        scatter = ax[i].scatter(df[dirs[0]], df[dirs[1]], c=df[prop], marker='s', linewidths=0, s=markerSize**2)
        if minorTicksIncluded:
            ax[i].set_xticks([xMin, xMax], [int(xMin), int(xMax)], minor=True)
            ax[i].tick_params(axis='x', which='minor', length=20, color='tab:gray', labelcolor='tab:gray')
            ax[i].xaxis.labelpad = -10
        ax[i].set_yticks(np.append(ax[i].get_yticks(), [yMin, yMax]))
        ax[i].set_xlim(xMin, xMax)
        ax[i].set_ylim(yMin, yMax)
        ax[i].set_xlabel('$%s\ (m)$'%dirs[0])
        ax[i].set_ylabel('$%s\ (m)$'%dirs[1])
        ax[i].set_title(propsTable[prop])
        if planeNormal != 'z':
            ax[i].invert_yaxis()
        fig.colorbar(scatter, ax=ax[i])
    fig.savefig(outputFileName)
    return

def createUserMeshPlane(xmin: float, xmax: float, ymin: float, ymax: float, zmin: float, zmax: float, 
        spacing: float, zSpacing: float) -> tuple[np.ndarray, str]:
    # NOTE: The +eps is added to the maximum values to include the last point 
    # in the mesh plane. Not sure whether there's a better way to do this.
    eps = np.finfo(np.float32).resolution
    if xmin == xmax:
        planeNormal = 'x'
        y, z = np.mgrid[ymin:ymax+eps:spacing, zmin:zmax+eps:zSpacing]
        x = np.full_like(y, xmin)
    elif ymin == ymax:
        planeNormal = 'y'
        x, z = np.mgrid[xmin:xmax+eps:spacing, zmin:zmax+eps:zSpacing]
        y = np.full_like(x, ymin)
    elif zmin == zmax:
        planeNormal = 'z'
        x, y = np.mgrid[xmin:xmax+eps:spacing, ymin:ymax+eps:spacing]
        z = np.full_like(x, zmin)
    else:
        ValueError('The user-defined mesh plane should be either in the x-z or y-z plane.')
    # Stack the arrays along the last axis to get an array of coordinates
    userMeshPlane = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    return userMeshPlane, planeNormal

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
    df.loc[:, 'rho'] = df['rho']*1000.0
    df.columns = ['x', 'y', 'z', 'Vs', 'Vp', 'rho']
    if xmin == xmax:
        planeNormal = 'x'
    elif ymin == ymax:
        planeNormal = 'y'
    elif zmin == zmax:
        planeNormal = 'z'
    else:
        ValueError('The user-defined mesh plane should be either in the x-z, y-z, or x-y plane.')
    plotProperties(df, planeNormal=planeNormal, outputFileName=outputFileName, minorTicksIncluded=minorTicksIncluded)
    return

def plotVelocityProfileWithUserMeshPlane(meshDatabaseFilePath: str, xmin: float, 
        xmax: float, ymin: float, ymax: float, zmin: float = 0.0, zmax: float|None = None, 
        spacing: float|None = None, zSpacing: float|None = None, method: str = 'fast', 
        minorTicksIncluded: bool = False) -> None:
    MPI_enabled, comm, size, rank = get_MPI_data()
    if not MPI_enabled or rank == 0:
        # NOTE: Reading the mesh coordinates from the HDF5 file could be a memory-consuming
        # task. To keep the memory usage low, it is done in the main process (rank 0) only.
        dfMeshCoordinates = pd.read_hdf(meshDatabaseFilePath, key='meshCoordinates')
        # NOTE: Pandas allows duplicate indices. To make the searching process faster 
        # in the later steps, the indices are set to the 'geid' column.
        dfMeshCoordinates.set_index('geid', inplace=True)
        print('Extracting elements on the plane...')
        elementsOnPlane, maxSpacing = getElementsOnPlane(dfMeshCoordinates, xmin, xmax, ymin, ymax, zmin=zmin, zmax=zmax)
        # To keep the memory usage low (without using swap memory), delete df and collect the memory back.
        del dfMeshCoordinates
        gc.collect()
        dfMeshData = pd.read_hdf(meshDatabaseFilePath, key='meshData')
        dfMeshData.set_index('geid', inplace=True)
        # ===== Define a mesh plane =====
        if spacing is None:
            xSpacing = getSpacing(elementsOnPlane['x'].unique(), returnType='min')
            ySpacing = getSpacing(elementsOnPlane['y'].unique(), returnType='min')
            spacing = min(xSpacing, ySpacing)
        if zSpacing is None:
            zSpacing = getSpacing(elementsOnPlane['z'].unique(), returnType='min')
        userMeshPlane, planeNormal = createUserMeshPlane(xmin, xmax, ymin, ymax, zmin, zmax, spacing, zSpacing)
        print('Extracting velocity profiles to the user-defined mesh plane...')
    else:
        # NOTE: For ranks other than 0, the variables are initialized to None to avoid
        # the "UnboundLocalError" error. The variables are assigned to the proper values
        # after the MPI communication.
        elementsOnPlane = None
        maxSpacing = None
        dfMeshData = None
        userMeshPlane = None
    if MPI_enabled: # Distribute the data to all the processors
        comm.barrier() # All the processors wait until the main process reads the mesh data
        elementsOnPlane = comm.bcast(elementsOnPlane, root=0)
        maxSpacing = comm.bcast(maxSpacing, root=0)
        dfMeshData = comm.bcast(dfMeshData, root=0)
        userMeshPlane = comm.bcast(userMeshPlane, root=0)
        totalMeshPoints = len(userMeshPlane)
        if rank > totalMeshPoints: # This is very unlikely to happen, but just in case
            exit()
        userMeshPlane = userMeshPlane[rank::size]
    userMeshData = pd.DataFrame(np.zeros((len(userMeshPlane), 6)), columns=['x', 'y', 'z', 'Vs', 'Vp', 'rho'])
    userMeshData[['x', 'y', 'z']] = userMeshPlane
    for i, point in enumerate(tqdm(userMeshPlane, position=rank, desc=f'Processor {rank}')):
        # NOTE: Searching with maxSpacing/2 is enough to find the elements containing 
        # the point for some cases, which is faster than searching with maxSpacing.
        h = maxSpacing/2
        while userMeshData.loc[i, 'Vs'] == 0 and h <= maxSpacing:
            subset = elementsOnPlane.loc[(elementsOnPlane['x'] >= point[0]-h) & (elementsOnPlane['x'] <= point[0]+h) & 
                (elementsOnPlane['y'] >= point[1]-h) & (elementsOnPlane['y'] <= point[1]+h) & 
                (elementsOnPlane['z'] >= point[2]-h) & (elementsOnPlane['z'] <= point[2]+h)]
            geids = subset.index.unique()
            hLocal = np.inf
            for geid in geids:
                element = elementsOnPlane.loc[geid]
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
                            userMeshData.loc[i, prop] = dfMeshData.loc[geid, prop]
                        break
                    # /// For a more accurate result, use the smallest element to represent the point
                    elif method == 'accurate':
                        hz = getSpacing(element['z'].unique())
                        if hz <= hLocal:
                            hLocal = hz
                            for prop in ['Vs', 'Vp', 'rho']:
                                userMeshData.loc[i, prop] = dfMeshData.loc[geid][prop]
            h = 2*h
        if userMeshData.loc[i, 'Vs'] == 0:
            # NOTE: If the point is still can't be found even if h=maxSpacing, it's problematic.
            # In fact, either Vs, Vp, or rho is 0 is problematic. But checking one of them is enough.
            raise ValueError(f'No element is found for the point ({point[0]}, {point[1]}, {point[2]}).')
    if MPI_enabled: # Collect the results from all the processors
        if rank > 0 and totalMeshPoints > 0:
            comm.send(userMeshData, dest=0, tag=41) # tag is arbitrary
        else:
            for i in range(1, size):
                userMeshData = pd.concat([userMeshData, comm.recv(source=i, tag=41)])
            userMeshData.sort_values(by=['x', 'y', 'z'], inplace=True)
    if not MPI_enabled or rank == 0:
        userMeshData.to_csv('userMeshPlane.csv', index=False)
        plotProperties(userMeshData, planeNormal=planeNormal, minorTicksIncluded=minorTicksIncluded)
    return

if __name__ == '__main__':
    # DEBUGGING: Change the current working directory to the directory of this file for debugging purpose.
    # import os
    # os.chdir(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(prog='plotVelocityProfile', 
        description='Plot the velocity profile with the user-defined mesh plane.')
    dimensionsHelp = 'The dimensions of the user-defined mesh plane. The order is \
        xmin, xmax, ymin, ymax, zmin, and zmax. If zmin and zmax are not provided, \
        the zmin is set to 0 and the zmax is set to the maximum z value in the mesh \
        database. If 5 values are provided, the fifth value is considered as zmax \
        and the zmin is set to 0. \n \
        Note that the user-defined mesh plane should be either in the x-z, y-z, or \
        x-y plane. That is, either xmin==xmax, ymin==ymax, or zmin==zmax has to be \
        true. \n \
        Also, the dimensions between the 3D velocity model and the mesh database \
        generated from Hercules might be different. Users have to take this into \
        account when querying the mesh data.'
    spacingHelp = 'The spacings between points on the user-defined mesh plane in \
        horizontal (x and y) and vertical (z) directions. Note that it only takes \
        two values as the first value stands for spacings between points in both x \
        and y directions. If not provided, the spacings are set to the minimum \
        spacings between the points on the plane in horizontal and vertical directions.' 
    targetHelp = 'The target to plot the velocity profile. The target could be either \
        "3DVelocityModel" or "MeshDatabase". If not provided, the target is set to \
        "MeshDatabase". \n \
        Note that the mesh database should be a HDF5 file with two groups: \
        "meshCoordinates" and "meshData". The "meshCoordinates" should have the \
        columns "geid", "x", "y", and "z". The "meshData" group should have the \
        columns "geid", "Vs", "Vp", and "rho". \n \
        Plotting the velocity profile from the mesh database supports MPI. \
        However, please also note that reading a large HDF5 file could be a very \
        memory-consuming task. Running the task with MPI might or might not be \
        beneficial depending on the size of the mesh database and the available \
        memory on the system.'
    accurateHelp = 'Whether to use the accurate method to find the elements containing \
        the points in the user-defined mesh plane.'
    parser.add_argument('dimensions', type=float, nargs='+', help=dimensionsHelp)
    parser.add_argument('--target', '-t', type=str, help=targetHelp, 
        default='MeshDatabase')
    parser.add_argument('--spacing', '-s', type=float, nargs=2, help=spacingHelp, default=[None, None])
    parser.add_argument('--filePath', '-f', type=str, help='The path to the mesh database file.', 
        default='database/mesh.hdf5')
    parser.add_argument('--velocityModelPath', '-v', type=str, help='The path to the velocity model file.', 
        default='./inputfiles/materialfiles/herculesVelocityModel.csv')
    parser.add_argument('--includeMinorTicks', '-i', action='store_true', help='Include minor ticks in the plot.')
    parser.add_argument('--accurate', '-a', action='store_true', help=accurateHelp)
    args = parser.parse_args()
    # ===== Handling the dimensions =====
    if len(args.dimensions) == 4:
        xmin, xmax, ymin, ymax = args.dimensions
        zmin = 0
        zmax = None
    elif len(args.dimensions) == 5:
        xmin, xmax, ymin, ymax, zmax = args.dimensions
        zmin = 0
    elif len(args.dimensions) == 6:
        xmin, xmax, ymin, ymax, zmin, zmax = args.dimensions
    else:
        ValueError('The number of dimensions should be either 4, 5, or 6.')
    # ===== Run the target function =====
    if args.target == '3DVelocityModel':
        plot3DVelocityModel(args.velocityModelPath, xmin, xmax, ymin, ymax, 
            zmin=zmin, zmax=zmax, minorTicksIncluded=args.includeMinorTicks)
    elif args.target == 'MeshDatabase':
        spacing = args.spacing[0]
        zSpacing = args.spacing[1]
        method = 'fast' if not args.accurate else 'accurate'
        plotVelocityProfileWithUserMeshPlane(args.filePath, xmin, xmax, ymin, ymax, 
            zmin=zmin, zmax=zmax, spacing=spacing, zSpacing=zSpacing, 
            method=method, minorTicksIncluded=args.includeMinorTicks)

    """
    In the following examples, the domain used in Hercules is larger than the 
    velocity model (to include all the source points), the distance between them 
    should be taken into account when querying the mesh data.
        xDiff = 16000
        yDiff = 5000
    Running plotVelocityProfileWithUserMeshPlane function supports MPI. To run it 
    with MPI, simply add `mpirun -n numProcessors` before the command. For example:
        mpirun -n 4 python3 plotVelocityProfile.py 67500 67500 5000 67500 0 21000 -s 500 200 -i
    """
    # ===== Example 1: along the y-axis at x = 51500 =====
    """
    Plot the 3D velocity model:
        python3 plotVelocityProfile.py 51500 51500 0 62500 0 21000 -t 3DVelocityModel -s 500 200 -i
    Plot the velocity profile with the user-defined mesh plane:
        python3 plotVelocityProfile.py 67500 67500 5000 67500 0 21000 -s 500 200 -i
    """
    # ===== Example 2: along the x-axis at y = 55000 =====
    """
    Plot the 3D velocity model:
        python3 plotVelocityProfile.py 35000 60000 55000 55000 0 21000 -t 3DVelocityModel -s 500 200
    Plot the velocity profile with the user-defined mesh plane:
        python3 plotVelocityProfile.py 51000 76000 60000 60000 0 21000 -s 500 200 -i
    """
    # ===== Example 3: x-y plane at z = 0 =====
    """
    Plot the 3D velocity model:
        python3 plotVelocityProfile.py 0 74000 0 62500 0 0 -t 3DVelocityModel -s 500 200
    Plot the velocity profile with the user-defined mesh plane:
        python3 plotVelocityProfile.py 16000 90000 5000 67500 0 0 -s 500 200 -i
    """
