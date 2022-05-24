# gltf compression optimization

## Using draco compression and gltf-pipeline / gltf-pack

### Use powershell and gltf-pipeline to use draco compression

    foreach ($file in get-ChildItem *.glb) {
    $suffix = "_d.glb"
    $saveName = $file.basename+$suffix
    gltf-pipeline -i $file.name -o $saveName -d
    }

### Save the file in the subdirectory pack with the same name

    for /F %x in ('dir /b *.glb') do (gltfpack -cc -kn -i %x -o ./pack/%x)

### Save the file in the same directory with the draco_ appended to as prefix to name

    for /F %x in ('dir /b *.glb') do (gltf-pipeline -i %x -o draco_%x -d)

### This will save the file in the subdirectory draco with the same name

    for /F %x in ('dir /b *.glb') do (gltf-pipeline -i %x -o ./draco/%x -d)

Raw glb file from chimeraX: 36127 KB
%% Maximizing the compression while maintaining the quality.

|PosBit|Size(KB)|Quality|Percent of total|
| :-----:|:-----:|:-----:|:----:|
|9|1203|Distorted |85%|
|10|1302|Highest|90%|
|11|1439|Highest|Default setting|

|NormBit|Size(KB)|Quality|Percent of total|
| :-----:|:-----:|:-----:|:----:|
|4| 888|Distorted|68%|
|5| 985|Highest|76%|
|6|1090|Highest|84%|
|7|1196|Highest|92%|
|8|1302|Highest|New Default Setting (used PosQ 10 from above)|


### compressionLevel has no impact on size beyond 7 which is the default

### quantizeColorBits has no impact on the size or quality. Use the default of 8

### quantizeTexcoordBits has no impact on the size or quality. Use the default of 10

### quantizeGenericBits has no impact on the size or quality. Use the default of 8

## So the overall compressionFlags are as follows

    for /F %x in ('dir /b *.glb') do (gltf-pipeline -i %x -o ./draco/%x -d true --draco.quantizePositionBits 10 --draco.quantizeNormalBits 5)

## Maximally compressed file size with highest quality: 985 KB

## Percent of raw: 2.7% ==> For all 100 timepoints it is 2.82% (3.74 GB reduced to 108 MB)
