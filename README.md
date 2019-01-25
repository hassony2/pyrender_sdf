# Minimal Signed Distance Function Object pure python renderer

This repo was mostly done as an **exercise** to understand Signed-Distance-Function-based rendering.

It provides basic utilities to render silhouettes and normals of objects represented by their Signed Distance Function.

`demo.py` shows an example on a sphere.

No parallelization is operated, and (this is no surprise! :) ) the computation is **slow!** (~5seconds for the minimal sphere-rendering example)

Very simple ray-marching is used (for more details see [ray-marching-signed-distance-functions](http://jamie-wong.com/2016/07/15/ray-marching-signed-distance-functions/))
