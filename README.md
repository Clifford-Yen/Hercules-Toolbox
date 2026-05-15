# Hercules Toolbox

The Hercules Toolbox is a suite of Python-based post-processing tools developed for the Hercules finite element earthquake simulator. It is designed to efficiently convert simulation outputs, create visualizations, and facilitate result interpretation.

## Authorship
This toolbox was authored and developed by Clifford (Chu-Han) Yen. It was created to address the lack of pre- and post-processing utilities in the original Hercules codebase, modernizing the workflow for physics-based earthquake simulations.

## Tools Included

The toolbox provides several independent Python scripts. Each script is highly configurable via command-line arguments.

### 1. `binary2HDF5.py`
Converts Hercules binary output files (such as displacement histories at points on specified planes) into efficient HDF5 database files using Blosc LZ4 compression. It supports parallel I/O for accelerated conversion and query operations.

### 2. `makeAnimation.py`
Creates animations of displacement or velocity propagation across a plane.
* **Optional Arguments:**
    * `response`: The response type to animate (displacement or velocity).
    * `--fps`: The frame rate of the animation (default is 24).
    * `--parameterFile`: Path to the Hercules parameter input file for automatic domain extraction.
    * `--includeMap`: Include a GMT map as the background.
    * `--maxVel`: The maximum velocity value for the color bar.
    * `--numPlanes`: The number of planes to animate.
    * `--threeDMagnitude`: Plot the response magnitude considering three directional components.

### 3. `plotStations.py`
Visualizes time histories for displacement, velocity, and acceleration at designated virtual stations.
* **Optional Arguments:**
    * `stations`: A list of user-defined, descriptive station names.
    * `--cutOffFrequency`: The cut-off frequency (in Hz) for applying a zero-phase Butterworth low-pass filter.

### 4. `plotVelocityProfile.py`
Visualizes velocity profiles in arbitrary $x$-$z$, $y$-$z$, or $x$-$y$ planes to inspect and verify the velocity model before and after the simulation.
* **Optional Arguments:**
    * `dimensions`: The user-defined mesh plane specified as $x_{min}$, $x_{max}$, $y_{min}$, $y_{max}$, $z_{min}$, and $z_{max}$.
    * `--target`: Defines whether the visualized model is from the HDF5 mesh database or the input velocity model.
    * `--spacing`: The horizontal and vertical spacings between points on the user-defined mesh plane.

## Dependencies
These scripts rely on modern Python data and visualization libraries, including:
* `pandas` (for data handling and database interaction)
* `h5py` (for HDF5 file operations)
* `matplotlib` (for plotting and visualizations)
* `ffmpeg` (for generating animations)