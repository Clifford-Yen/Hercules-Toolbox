import matplotlib.pyplot as plt
import pandas as pd
import os
import io

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
    return stationList

def getFileWithoutUnnecessaryHeading(filePath: str) -> io.StringIO:
    with open(filePath, 'r') as f:
        lines = f.readlines()
        lines[0] = lines[0].lstrip('#')
    return io.StringIO(''.join(lines))

def getTimeHistoryFromStation(stationFolder: str) -> list[pd.DataFrame]:
    stationList = getStationList(stationFolder)
    stations = {}
    for stationFileName in stationList:
        stationNum = int(stationFileName.split('.')[1])
        stationFile = getFileWithoutUnnecessaryHeading(os.path.join(stationFolder, stationFileName))
        # NOTE: The 'delim_whitespace' keyword in pd.read_csv is deprecated and will be removed in a future version. Use ``sep='\s+'`` instead
        df = pd.read_csv(stationFile, sep='\s+', index_col='Time(s)')
        stations[stationNum] = df
    return stations

def plotAndSaveResults(locations: dict[str, dict[str, int]], stations: dict[int, pd.DataFrame], outputFolder='') -> None:
    columnSeries = {'Displacement (m)': {'$u$': 'X|(m)', '$v$': 'Y-(m)', '$w$': 'Z.(m)'},
        'Velocity (m/s)': {'$\dot{u}$': 'X|(m/s)', '$\dot{v}$': 'Y-(m/s)', '$\dot{w}$': 'Z.(m/s)'},
        'Acceleration (m/s$^2$)': {'$\ddot{u}$': 'X|(m/s2)', '$\ddot{v}$': 'Y-(m/s2)', '$\ddot{w}$': 'Z.(m/s2)'}}
    plt.rcParams.update({'font.size': 10, 'font.weight': 'regular'})
    for locationName, stationNum in locations.items():
        fig, axes = plt.subplots(3, 3, sharex='col', sharey='row')
        fig.set_size_inches([8.5, 8.5])
        times = stations[stationNum].index
        for i, (quantity, columns) in enumerate(columnSeries.items()):
            for j, (title, column) in enumerate(columns.items()):
                axes[i, j].plot(times, stations[stationNum].loc[times, column], '-', linewidth=0.5)
                # axes[i, j].legend(frameon=False)
                axes[i, j].set(title=title)
                axes[-1, j].set(xlabel='Time (s)')
            axes[i, 0].set(ylabel=quantity)
        fig.savefig(os.path.join(outputFolder, locationName+'.pdf'))

if __name__ == '__main__':
    # os.chdir(os.path.dirname(os.path.abspath(__file__)))
    stationFolder = './outputfiles/stations'
    stations = getTimeHistoryFromStation(stationFolder)
    locations = {'3513': 0, '3514': 1, '3518': 2, '3520': 3, '3522': 4, '3526': 5}
    plotAndSaveResults(locations, stations, outputFolder='./outputfiles/timeHistories')