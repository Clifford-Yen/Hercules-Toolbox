import pandas as pd
from matplotlib.animation import FuncAnimation
import numpy as np
import progressbar
import argparse
import matplotlib.pyplot as plt
# plt.rcParams.update({'text.usetex': True, 'font.family': 'serif', 'font.serif': ['Computer Modern'], 
#     'font.weight': 'regular'})

def plotResponseMagnitude(fileName, key, response='velocity', fps=None):
    # Read the HDF5 file and load the data into a DataFrame
    df = pd.read_hdf(fileName, key=key)
    if response == 'displacement':
        title = 'Displacement Magnitude ($m$)'
    elif response == 'velocity':
        title = 'Velocity Magnitude ($m/s$)'
        timePoints = df['time'].unique()
        df_uniqueNodes = df.drop_duplicates(subset=['x', 'y', 'z'])
        print('Computing velocity...')
        # ===== An even faster way (by slicing the DataFrame) =====
        numUniqueNodes = len(df_uniqueNodes)
        for i in progressbar.progressbar(range(numUniqueNodes)):
            df_subset = df.iloc[i::numUniqueNodes]
            for column in ['u', 'v', 'w']:
                df.loc[df_subset.index, column] = np.gradient(df.loc[df_subset.index, column], timePoints)
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
    df['response'] = (df['u']**2 + df['v']**2 + df['w']**2)**0.5
    # Create a figure and axis for the animation
    fig, ax = plt.subplots()
    # Set the axis limits
    ax.set_xlim(df['y'].min(), df['y'].max())
    ax.set_ylim(df['x'].min(), df['x'].max())
    x = df['y'].drop_duplicates()
    y = df['x'].drop_duplicates()
    levels = np.linspace(0, df['response'].max(), 100)
    frame_data = df[df['timeStep'] == 0]
    contour = ax.contourf(x, y, frame_data.pivot(index='x', columns='y', values='response'), levels=levels)
    plt.xlabel('$y$')
    plt.ylabel('$x$')
    plt.colorbar(contour)
    plt.title(title)
    # plt.grid(True)
    # plt.show()
    # ===== Define the update function for the animation =====
    def update(frame):
        # Get the data for the current frame
        frame_data = df[df['timeStep'] == frame]
        # Turn frame_data['displacement'] into 2D array
        displacementMagnitude = frame_data.pivot(index='x', columns='y', values='response')
        # Create a 2D contour plot of the displacement magnitude
        contour = ax.contourf(x, y, displacementMagnitude, levels=levels)
        # plt.title(title+', Time: {:.2f} s'.format(df.loc[df['timeStep'] == frame, 'time'].iloc[0]))
        fig.axes[1].set(title='Time: {:.2f} s'.format(df.loc[df['timeStep'] == frame, 'time'].iloc[0]))
        return contour,
    # Create the animation
    print('Creating animation...')
    animation = FuncAnimation(fig, update, frames=progressbar.progressbar(df['timeStep'].unique()), blit=True, cache_frame_data=False)
    if fps is None:
        animation.save(response+'.mp4', writer='ffmpeg')
    else:  
        animation.save(response+'.mp4', writer='ffmpeg', fps=fps)

if __name__ == '__main__':
    # /// Change the working directory to the directory of this file for debugging
    import os
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser()
    helpString = "The quantity to be plotted. It can be 'displacement' or 'velocity'. Default is 'velocity'."
    parser.add_argument('quanity', nargs='?', help=helpString, default='velocity')
    parser.add_argument('--fps', '-f', type=int, help='Frames per second. Default is 10.', default=10)
    quantity = parser.parse_args().quanity
    fps = parser.parse_args().fps
    if fps is not None and fps < 1:
        raise ValueError('fps should be greater than 0.')
    
    fileName = 'database/planedisplacements.hdf5'
    key = 'plane0'
    plotResponseMagnitude(fileName, key, response=quantity, fps=fps)