# ChimeraX-measure_intensity

Steps to measure intensity of surface

    1. Load in a single 3D timepoint
    2. Create a surface using marching cubes (verts, faces)
    3. Find the barycenter of each triangle
    4. Generate the KDtree from nonzero points in 3D image
    5. Query the KDTree with a given radius (r=9)
    6. Calculate intensities from the queried radius
    7. Normalize the intensity to the mean

The initial profiling (of similar meSPIM code) in Matlab took ~15 minutes to complete with the majority of time spent at step 5 (querying KDTree).
Once the refactoring was complete, this reduced this to 6 seconds, and the query was ~3 seconds.

This repository is simplified since the surface is already generated in ChimeraX, and it will just need to query the surface and additional channel.

Using the barycenter doesn't seem to be entirely necessary since we are coloring the vertices.
