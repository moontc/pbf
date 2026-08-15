# pbf cuda

![output](gif/output.gif)

This project reproduces *Position Based Fluids* on CUDA.

Rendering is done with OpenGL.

You need an NVIDIA GPU to run this project.

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target pbf --config Release
```

## Controls

| Input | Action |
|---|---|
| **Left mouse drag** | Orbit the camera |
| **Mouse wheel** | Zoom in / out |
| **R** | Reset the simulation to the initial fluid block |
| **Esc** | Quit |