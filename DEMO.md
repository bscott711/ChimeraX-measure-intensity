# Create interactive surface

    1. Color isosurface based on secondary information
           - Local intensity
           - Local motion
    2. Save surfaces in GLB format
    3. DRACO compress with optimized settings
    4. Upload files for display in Babylon.JS

## If necessary, clone the repo and install in ChimeraX

    gh repo clone bscott711/ChimeraX-measure-intensity
    cd $Repo_dir
    devel build . ; devel install . ; devel clean . ;

## Run the following commands in ChimeraX

    cd X:\\Demo_data\\
    open X:\\Demo_data\\*ch0_*.tif format images
    open X:\\Demo_data\\*ch1_*.tif format images
    open X:\\Demo_data\\*ch2_*.tif format images
    volume #1.1 level 45.9 style surface
    volume #2.1 style image maximumIntensityProjection true level 0,0 level 192,1 level 1260,1
    volume #3.1 level 31.4 style surface
    measure intensity #3.1-50 toMaps #2.1-50 range 0,5
    measure distance #1.1-49 toSurfaces #1.2-50 range 0,40
    surface recolor #1 metric distance palette spectral range 0,30
    hide #!1-3 models;
    perframe "show #!1,3.$1 models; wait 1; save Surfaces_$1.glb floatColors true; wait 1; hide #!1,3.$1 models; wait 1;" ranges 1,50

## Ensure gltf-pipeline is installed and on the PATH

    cd X:\\Demo_data\\

    This will save the file in the subdirectory draco with the same name.
    for /F %x in ('dir /b *.glb') do (gltf-pipeline -d true -i %x -o ./draco/%x)

    Use the following for a maximally compressed gltf.
    for /F %x in ('dir /b *.glb') do (gltf-pipeline -i %x -o ./draco/%x -d true --draco.quantizePositionBits 10 --draco.quantizeNormalBits 5)

## Compression results

| Surface ID | Raw GLB (KB) | Draco Comressed (KB) | Compression Ratio X |
| :--------: | :----------: | :------------------: | :-----------------: |
| Surface 1  |    28,922    |         1563         |        18.5         |
| Surface 47 |    42,248    |         2305         |        18.3         |
