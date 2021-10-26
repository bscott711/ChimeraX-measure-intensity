chcp 65001
set "MPARALLEL64=C:\Apps\mparallel\MParallel.exe"
set "GLTF_PATH=C:\Users\bscott\AppData\Roaming\npm\gltf-pipeline.cmd"
set "PATTERN=\"%GLTF_PATH%\" -i \"{{0}}\" -o \".\draco\{{0}}\" -d true --draco.quantizePositionBits 10 --draco.quantizeNormalBits 5"
dir /B "*.glb" | "%MPARALLEL64%" --stdin --no-split-lines --pattern="%PATTERN%"