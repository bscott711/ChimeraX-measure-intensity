# ChimeraX-measure-intensity

## Steps to measure intensity of surface

    1. Generate the KDtree from nonzero points in 3D image
    2. Query the KDTree with a given radius (r=15)
    3. Calculate intensities from the queried radius
    4. Normalize the intensity to the mean

The initial profiling (of similar meSPIM code) in Matlab took ~15 minutes to complete with the majority of time spent at step 5 (querying KDTree).
Once the refactoring was complete, this reduced this to ~3 seconds.

## Usage

### Clone the repo and install in ChimeraX

    cd $Repo_dir
    devel build . ; devel install . ; devel clean . ;

### Download the demo data to X:\Demo_data: [Demo Data](https://github.com/bscott711/ChimeraX-measure-intensity/blob/main/demo_data/)

### Run the following commands in ChimeraX

    open X:\\Demo_data\\*ch0_*.tif format images
    open X:\\Demo_data\\*ch1_*.tif format images
    volume #1.1 level 42 style surface
    volume #2.1 style image maximumIntensityProjection true level 0,0 level 300,1 level 1260,1
    measure intensity #1.1-10 toMaps #2.1-10 range 0,5
    measure distance #1.1-9 toSurfaces #1.2-10 range 0,15
    surface recolor #1 metric intensity
    surface recolor #1 metric distance palette spectral

## Representative Intensity at Time=0

|                  Initial Isosurface                  |                   Secondary Intensity                   |                     Intensity                      |
| :--------------------------------------------------: | :-----------------------------------------------------: | :------------------------------------------------: |
| ![Surface Image](/readme_images/Initial_Surface.png) | ![Volume Image](/readme_images/Secondary_Intensity.png) | ![Intensity](/readme_images/Surface_Intensity.png) |

## Representative Distance between Time=0 and Time=1

|                        Time 0                        |                       Time 1                       |                     Distance                     |
| :--------------------------------------------------: | :------------------------------------------------: | :----------------------------------------------: |
| ![Surface Image](/readme_images/Initial_Surface.png) | ![Surface Image](/readme_images/Surface_Time2.png) | ![Distance](/readme_images/Surface_Distance.png) |
