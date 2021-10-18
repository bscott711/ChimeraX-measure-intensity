# ChimeraX-measure_intensity

Steps to measure intensity of surface

    1. Generate the KDtree from nonzero points in 3D image
    2. Query the KDTree with a given radius (r=15)
    3. Calculate intensities from the queried radius
    4. Normalize the intensity to the mean

The initial profiling (of similar meSPIM code) in Matlab took ~15 minutes to complete with the majority of time spent at step 5 (querying KDTree).
Once the refactoring was complete, this reduced this to ~3 seconds.

# Usage

1. Clone the repo and install in chimeraX
```
cd $Repo_dir
devel build . ; devel install . ; devel clean . ;
```
2. Download the demo data to X:\Demo_data:
https://github.com/bscott711/ChimeraX-measure-intensity/blob/main/demo_data/

3. Run the following commands
```
open X:\\Demo_data\\*ch0_*.tif format images
open X:\\Demo_data\\*ch1_*.tif format images
volume #1.1 level 42 style surface
volume #2.1 style image maximumIntensityProjection true level 0,0 level 300,1 level 1260,1
measure intensity #1.1-10 toMaps #2.1-10 range 0,5
measure distance #1.1-9 toSurfaces #1.2-10 range 0,15
```