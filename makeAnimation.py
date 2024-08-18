import pandas as pd
from matplotlib.animation import FuncAnimation
import numpy as np
import progressbar
import argparse
from plotMap import plotMap
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family': 'serif', 
    'font.serif': ['Courier New'], 'font.weight': 'regular'})

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

def plotResponseMagnitude(fileName, key, response='velocity', maxVel=0.5, fps=24, 
        includeMap=False, mapFile='map.png', threeDMagnitude=False, 
        parameterFile='inputfiles/parameters.in', **kwargs):
    # Read the HDF5 file and load the data into a DataFrame
    df = pd.read_hdf(fileName, key=key)
    if response == 'displacement':
        title = 'Displacement Magnitude (m)'
    elif response == 'velocity':
        title = 'Velocity Magnitude (m/s)'
        timePoints = df['time'].unique()
        print('Computing velocity...')
        # ===== The fastest way to compute the gradient with pivot table =====
        # Create a pivot table to reshape the DataFrame
        # NOTE: the ``pivot`` method reshapes the DataFrame into a 3D structure 
        # where the index is the timeStep, and the columns are a MultiIndex of x,
        # y, and z. By doing so, we can compute the gradient for each component 
        # at all nodes simultaneously. This is much faster than looping through
        # each node and computing the gradient one by one.
        # Also note that, although ``pivot_table`` can also be used here, it will 
        # aggregate the values if there are multiple values for the same index/column
        # instead of raising an error. This is not what we want. There shouldn't 
        # be multiple values for the same index/column in this case.
        # NOTE 2: since we don't really need the 'time' column, it's dropped here.
        df_pivot = df.pivot(index='timeStep', columns=['x', 'y', 'z'], values=['u', 'v', 'w'])
        # Compute the gradient for each component
        for component in progressbar.progressbar(['u', 'v', 'w']):
            df_pivot.loc[:, component] = np.gradient(df_pivot.loc[:, component], timePoints, axis=0)
        # Flatten the pivot table back to the original DataFrame structure
        # NOTE: set future_stack=True to avoid the warning message
        df = df_pivot.stack(['x', 'y', 'z'], future_stack=True).reset_index()
        # Reclaim memory by getting rid of the pivot table
        del df_pivot
        # ===== An even faster way (by slicing the DataFrame) =====
        # df_uniqueNodes = df.drop_duplicates(subset=['x', 'y', 'z'])
        # numUniqueNodes = len(df_uniqueNodes)
        # for i in progressbar.progressbar(range(numUniqueNodes)):
        #     df_subset = df.iloc[i::numUniqueNodes]
        #     for column in ['u', 'v', 'w']:
        #         df.loc[df_subset.index, column] = np.gradient(df.loc[df_subset.index, column], timePoints)
        # ===== Faster, but only applicable to Hercules' current plane output format =====
        # for i in progressbar.progressbar(range(len(df_uniqueNodes))):
        #     row = df_uniqueNodes.iloc[i]
        #     df_subset = df[(df['x'] == row['x']) & (df['y'] == row['y']) & (df['z'] == row['z'])]
        #     for column in ['u', 'v', 'w']:
        #         df.loc[df_subset.index, column] = np.gradient(df.loc[df_subset.index, column], timePoints)
        # ===== More general way, but slower =====
        # processedIndices = []
        # bar = progressbar.ProgressBar(max_value=len(df))
        # for i, row in df.iterrows():
        #     if i in processedIndices:
        #         continue
        #     if len(processedIndices) == len(df):
        #         break
        #     df_subset = df[(df['x'] == row['x']) & (df['y'] == row['y']) & (df['z'] == row['z'])]
        #     processedIndices.extend(df_subset.index.to_list())
        #     for column in ['u', 'v', 'w']:
        #         df.loc[df_subset.index, column] = np.gradient(df.loc[df_subset.index, column], timePoints)
        #     bar.update(len(processedIndices))
    if threeDMagnitude:
        # /// Magnitude of 3 directional responses
        df['response'] = (df['u']**2 + df['v']**2 + df['w']**2)**0.5
    else:
        # /// Magnitude of 2 (horizontal) directional responses
        df['response'] = (df['u']**2 + df['v']**2)**0.5
    # Create a figure and axis for the animation
    fig, ax = plt.subplots()
    x = df['y'].drop_duplicates()
    y = df['x'].drop_duplicates()
    cm = plt.cm.ScalarMappable(cmap='jet')
    if response == 'velocity':
        levels = np.linspace(0.01, maxVel, 101, endpoint=True)
        cm.set_clim(levels[0], levels[-1])
        cb = plt.colorbar(cm, ax=ax, extend='max')
    else:
        levels = np.linspace(0.01, df['response'].max().round(2), 100, endpoint=True)
        cm.set_clim(levels[0], levels[-1])
        cb = plt.colorbar(cm, ax=ax)
    # Update the colorbar ticks
    ticks = cb.get_ticks()
    ticks = ticks[(ticks > levels[0]) & (ticks < levels[-1])]
    ticks = [levels[0], *ticks, levels[-1]]
    cb.set_ticks(ticks)
    # Include the map if required
    if includeMap:
        plotMap(mapFile, parameterFile, overwriteExisting=False, includeTopography=True, useColorMap=True)
        map = plt.imread(mapFile)
        # Set the fig size of the plot based on the map
        colorbarWidth = 1.2 # TODO: This is an estimate. May find a better way to calculate this
        fig.set_size_inches(5+colorbarWidth, map.shape[0]/map.shape[1]*5)
        # Set the axis limits. This is not necessary if the map is not included
        # since ax.contourf will automatically set the axis limits
        ax.set_xlim(df['y'].min(), df['y'].max())
        ax.set_ylim(df['x'].min(), df['x'].max())
        extent = ax.get_xlim() + ax.get_ylim()
    # ===== Define the update function for the animation =====
    def updateFrame(frame):
        ax.clear() # Clear the previous frame
        if includeMap: # Plot the map
            ax.imshow(map, extent=extent, aspect='auto')
        # Get the data for the current frame
        frame_data = df[df['timeStep'] == frame]
        # Turn frame_data['displacement'] into 2D array
        responseMagnitude = frame_data.pivot(index='x', columns='y', values='response')
        # Create a 2D contour plot of the displacement magnitude
        contour = ax.contourf(x, y, responseMagnitude, levels=levels, cmap='jet', extend='max')
        if includeMap:
            contour.set_alpha(0.5)
            contour.set_antialiased(True)
        plt.title(title+'\nTime: %6.2f s'%(timePoints[frame]))
        return contour,
    # ===== Define the initialization function for the animation =====
    def plotFirstFrame():
        return updateFrame(frame=0)
    # Create the animation
    print('Creating animation...')
    animation = FuncAnimation(fig, func=updateFrame, frames=progressbar.progressbar(df['timeStep'].unique()), 
        init_func=plotFirstFrame, blit=True, cache_frame_data=False)
    animation.save(response+'_'+key+'.mp4', writer='ffmpeg', fps=fps)

if __name__ == '__main__':
    # DEBUGGING: Change the working directory to the directory of this file for debugging
    # import os
    # os.chdir(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(prog='makeAnimation', 
        description='Create an animation of the velocity/displacement propagation in the domain.')
    helpString = "The response to be plotted. It can be 'displacement' or 'velocity'. Default is 'velocity'."
    parser.add_argument('response', nargs='?', help=helpString, default='velocity')
    parser.add_argument('--fps', '-f', type=int, help='Frames per second. Default is 24.', default=24)
    inputFileHelp = 'Path to the input file. Default: inputfiles/parameters.in'
    parser.add_argument('--parameterFile', '-i', type=str, help=inputFileHelp, default='inputfiles/parameters.in')
    parser.add_argument('--includeMap', '-m', action='store_true', help='Plot on the provided map. Default is False.')
    parser.add_argument('--maxVel', '-v', type=float, help='Maximum velocity for the colorbar. Default is 0.5 (m/s).', default=0.5)
    parser.add_argument('--numPlanes', '-n', type=int, help='Number of planes to plot. Default is 1 (only plane0 will be plotted).', default=1)
    parser.add_argument('--threeDMagnitude', '-t', action='store_true', help='Plot the magnitude of 3 directional responses. Default is False (only horizontal responses).')
    args = parser.parse_args()
    if args.fps is not None and args.fps < 1:
        raise ValueError('fps should be greater than 0.')
    
    fileName = 'database/planedisplacements.hdf5'
    planesData = pd.read_hdf(fileName, key='/planesData')
    for index in planesData.index:
        if index >= args.numPlanes:
            break
        key = 'plane'+str(index)
        plotResponseMagnitude(fileName, key, **vars(args))