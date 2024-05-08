import pandas as pd
import numpy as np
import scipy
import os
import io
import argparse
import matplotlib.pyplot as plt
plt.rcParams.update({'text.usetex': True, 'font.family': 'serif', 'font.serif': ['Computer Modern'], 
    'font.size': 13, 'font.weight': 'regular', 'lines.linewidth': 0.5})

# def getStationList(folderPath: str) -> list[str]:
#     return [f for f in os.listdir(folderPath) if os.path.isfile(os.path.join(folderPath, f))]

def getStationList(folderPath: str, neglect_subfolder=True) -> list[str]:
    stationList = []
    for dirpath, dirNames, fileNames in os.walk(folderPath):
        for fileName in fileNames:
            prefix, suffix = os.path.splitext(fileName)
            if prefix == 'station' or suffix == '.csv':
                stationList.append(fileName)
        if neglect_subfolder:
            dirNames[:] = [] # clear subdirectories. This act would neglect subdirectories after searching at current directory
    stationList.sort()
    return stationList

def getFileWithoutUnnecessaryHeading(filePath: str) -> io.StringIO:
    with open(filePath, 'r') as f:
        lines = f.readlines()
        lines[0] = lines[0].lstrip('#')
    return io.StringIO(''.join(lines))

def getFilteredData(df: pd.Series, cutoffFrequency: float, filterOrder: int, 
        filterName='Butterworth', passes='zero-phase') -> np.ndarray:
    dt = df.index[1] - df.index[0]
    if filterName == 'Butterworth':
        sos = scipy.signal.butter(filterOrder, cutoffFrequency, output='sos', fs=1/dt)
    elif filterName == 'Bessel':
        sos = scipy.signal.bessel(filterOrder, cutoffFrequency, output='sos', fs=1/dt)
    else:
        raise ValueError(f'Invalid filterName: {filterName}')
    if passes == 'zero-phase':
        return scipy.signal.sosfiltfilt(sos, df)
    elif passes == 'causal':
        return scipy.signal.sosfilt(sos, df)
    else:
        raise ValueError(f'Invalid passes: {passes}')

def getTimeHistoryFromStation(stationFolder: str = './outputfiles/stations') -> list[pd.DataFrame]:
    stationList = getStationList(stationFolder)
    stations = {}
    for stationFileName in stationList:
        stationNum = int(stationFileName.split('.')[1])
        stationFile = getFileWithoutUnnecessaryHeading(os.path.join(stationFolder, stationFileName))
        # NOTE: The 'delim_whitespace' keyword in pd.read_csv is deprecated and will be removed in a future version. Use ``sep='\s+'`` instead
        df = pd.read_csv(stationFile, sep='\s+', index_col='Time(s)')
        stations[stationNum] = df
    return stations

def plotAndSaveResults(locations: dict[str, dict[str, int]], stations: dict[int, pd.DataFrame], 
        outputFolder: str='./outputfiles/timeHistories', 
        cutoffFrequency: float|None = None, filterOrder: float=3) -> None:
    columnSeries = {'Displacement ($m$)': {'$u$': 'X|(m)', '$v$': 'Y-(m)', '$w$': 'Z.(m)'},
        'Velocity ($m/s$)': {'$\dot{u}$': 'X|(m/s)', '$\dot{v}$': 'Y-(m/s)', '$\dot{w}$': 'Z.(m/s)'},
        'Acceleration ($m/s^2$)': {'$\ddot{u}$': 'X|(m/s2)', '$\ddot{v}$': 'Y-(m/s2)', '$\ddot{w}$': 'Z.(m/s2)'}}
    # Create the outputFolder if it does not exist
    if not os.path.exists(outputFolder):
        os.makedirs(outputFolder)
    for locationName, stationNum in locations.items():
        fig, axes = plt.subplots(3, 3, sharex='col', sharey='row')
        fig.set_size_inches([8.5, 8.5])
        times = stations[stationNum].index
        for i, (quantity, columns) in enumerate(columnSeries.items()):
            for j, (title, column) in enumerate(columns.items()):
                if cutoffFrequency is not None:
                    axes[i, j].plot(times, getFilteredData(stations[stationNum].loc[times, column], 
                        cutoffFrequency, filterOrder))
                else:
                    axes[i, j].plot(times, stations[stationNum].loc[times, column])
                axes[i, j].set(title=title)
                axes[-1, j].set(xlabel='Time ($s$)')
            axes[i, 0].set(ylabel=quantity)
        fig.suptitle('Station '+locationName, y=0.95)
        fig.savefig(os.path.join(outputFolder, locationName+'.pdf'))

if __name__ == '__main__':
    # DEBUGGING: Change the current working directory to the directory of this file for debugging purpose.
    # os.chdir(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(prog='plotStations', 
        description='Plot the responses at the stations stored in the outputfiles/stations folder.')
    stationsHelp = 'The names to each file in the stations folder in the ascending \
        order of the station number. The station number is the number in the \
        station file name after the dot. Type the station names separated with spaces. \
        If not provided, the station numbers will be used as the station names.'
    parser.add_argument('stations', type=str, nargs='*', help=stationsHelp)
    parser.add_argument('--cutOffFrequency', '-f', type=float, default=None, 
        help='The cut-off frequency for the low-pass filter. Default is None.')
    args = parser.parse_args()
    stationNames = parser.parse_args().stations
    stations = getTimeHistoryFromStation()
    if stationNames is not None:
        locations = dict(zip(stationNames, stations.keys()))
        if len(stationNames) < len(stations):
            for key in list(stations.keys())[len(stationNames):]:
                locations[key] = key
    else:
        locations = {str(i): i for i in stations.keys()}
    plotAndSaveResults(locations, stations, cutoffFrequency=args.cutOffFrequency)