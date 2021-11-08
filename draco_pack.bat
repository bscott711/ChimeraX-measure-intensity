if not exist ".\draco" mkdir ".\draco"
chcp 65001
set "MPARALLEL64=C:\Apps\mparallel\MParallel.exe"
set "PATTERN=gltfpack.cmd -i {{0}} -o .\draco\{{0}} -cc -tc -kn -si 0.5 -vp 10 -vt 10 -vn 5"
dir /B "*.glb" | "%MPARALLEL64%" --stdin --no-split-lines --pattern="%PATTERN%"