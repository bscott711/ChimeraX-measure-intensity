@REM for /F %x in ('dir /b *.glb'); do
@REM (gltf-pipeline -i %x -o ./draco/%x -d true --draco.quantizePositionBits 10 --draco.quantizeNormalBits 5)

@REM This took 500 seconds to complete the 150 files ~3.3 seconds per file

@REM The following requires MParallel.exe https://github.com/lordmulder/MParallel

@REM The following took 95 seconds to run through the 150 files. ~0.63 seconds per file: A factor of 5 speedup

@REM chcp 65001 sets the stdout to utf-8. Required to pipe properly into stdin
chcp 65001
@REM These need to be modified as needed where MParallel and gltf-pipeline are.
set "MPARALLEL64=C:\Apps\mparallel\MParallel.exe"
set "GLTF_PATH=C:\Users\bscott\AppData\Roaming\npm\gltf-pipeline.cmd"

set "PATTERN=\"%GLTF_PATH%\" -i \"{{0}}\" -o \".\draco\{{0}}\" --draco.compressMeshes true --draco.quantizePositionBits 10 --draco.quantizeNormalBits 5"

dir /B "*.glb" | "%MPARALLEL64%" --stdin --no-split-lines --pattern="%PATTERN%"