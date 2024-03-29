# ChimeraX-measure-intensity

## Steps to measure intensity of surface

    1. Generate the KDtree from nonzero points in 3D image
    2. Query the KDTree with a given radius (r=15)
    3. Calculate intensities from the queried radius
    4. Normalize the intensity to the mean

The initial profiling (of similar meSPIM code) in Matlab took ~15 minutes to complete with the majority of time spent querying KDTree.
Once the refactoring was complete, this reduced this to ~3 seconds.

## Usage

### Clone the repo and install in ChimeraX

    gh repo clone bscott711/ChimeraX-measure-intensity
    cd $Repo_dir
    devel build . ; devel install . ; devel clean . ;

    If Needed Remove the Old.
    toolshed uninstall measure-intensity

### Download the demo data to X:\Demo_data: [Demo Data](https://github.com/bscott711/ChimeraX-measure-intensity/blob/main/demo_data/)

## Run the following commands in ChimeraX

    cd X:\\demo_data\\
    open X:\\demo_data\\*ch0_*.tif format images
    open X:\\demo_data\\*ch1_*.tif format images
    open X:\\demo_data\\*ch2_*.tif format images
    volume #1.1 level 46 style surface
    volume #2.1 style image maximumIntensityProjection true level 0,0 level 192,1 level 1260,1
    volume #3.1 level 31 style surface
    measure intensity #3.1-5 toMap #2.1-5 range 0,5
    measure distance #1.1-4 toSurface #1.2-5 range 0,40
    surface recolor #1 metric distance palette spectral range 0,30

## Representative Intensity

|             Initial Isosurfaces              |                 Secondary Intensity                  |                Surface Intensity                |
| :------------------------------------------: | :--------------------------------------------------: | :---------------------------------------------: |
| ![Surface Image](/readme_images/initial.png) | ![Volume Image](/readme_images/intensity_volume.png) | ![Intensity](/readme_images/intensity_only.png) |

## Representative Distance between Time=x and Time=x+1

|               Initial Surfaces               |               Distance in next frame               |               Distance and Intensity               |
| :------------------------------------------: | :------------------------------------------------: | :------------------------------------------------: |
| ![Surface Image](/readme_images/initial.png) | ![Surface Image](/readme_images/distance_only.png) | ![Distance](/readme_images/intensity_distance.png) |

## Create interactive surface

    1. Save surfaces in GLB format
    2. DRACO compress with optimized settings
    3. Upload files for display in Babylon.JS

## ChimeraX commands to run

    hide #!1-3 models;
    perframe "show #!1,3.$1 models; wait 1; save Surfaces_$1.glb floatColors true; wait 1; hide #!1,3.$1 models; wait 1;" ranges 1,5

## Ensure gltf-pipeline is installed and on the PATH. Run in the cmd prompt

    cd X:\\demo_data\\

    This will save the file in the subdirectory draco with the same name.
    for /F %x in ('dir /b *.glb') do (gltf-pipeline -d true -i %x -o ./draco/%x)

    Use the following for a maximally compressed gltf.
    for /F %x in ('dir /b *.glb') do (gltf-pipeline -i %x -o ./draco/%x -d true --draco.quantizePositionBits 10 --draco.quantizeNormalBits 5)

### If MParallel is installed you can gain a 5x speed increase: [MParallel](https://github.com/lordmulder/MParallel)

    cd X:\\demo_data\\

    Run the Draco compressed batch file in the new directory.
    $Repo_dir\draco_compressed.bat

## Compression results

| Surface ID | Raw GLB (KB) | Draco Compressed (KB) | Compression Ratio X |
| :--------: | :----------: | :------------------: | :-----------------: |
| Surface 1  |    28,922    |         1563         |        18.5         |
| Surface 47 |    42,248    |         2305         |        18.3         |

## Update to using a more advanced compression algorithm that includes mesh simplification

There is no perceptual difference between the meshes, but the file size tells us there is a difference, and the loading time does as well.

    Install Gltf-Pack
    npm install -g gltf-pack

    cd X:\\demo_data\\

    Use the following for a simplified and maximally compressed gltf.
    if not exist ".\draco" mkdir ".\draco"
    for /F %x in ('dir /b *.glb') do (gltfpack -i %x -o .\draco\%x -cc -tc -kn -si 0.5 -vp 10 -vt 10 -vn 5)

### The same can be done using MParallel

    cd X:\\demo_data\\

    Run the Draco compressed batch file in the new directory.
    $Repo_dir\draco_pack.bat

## Compression results for gltf-pack

| Surface ID  | Raw GLB (KB) | Draco Compressed (KB) | Compression Ratio X |
| :---------: | :----------: | :------------------: | :-----------------: |
| Surface 10  |    31,145    |         1538         |        20.25        |
| Surface 130 |    44,943    |         2491         |        18.04        |

## If the files need renamed, you can do that using Powershell

Move into the folder and run the following:
    Get-ChildItem *.glb | Rename-Item -NewName { $_.Name -replace 'Surfaces_','Eat_' }

The first string is the original file name and the second string will be what it is replaced with
It is recommended to have an underscore before the number so that is all that changes as it is indexed
