# Voronoi Tessellation

A Python implementation of Voronoi tessellation, visualising the simultaneous spatial growth of randomly initialised centroids across a 2D grid. 

Centroids expand simultaneously across the grid through successive square boundaries. Newly reached, unvisited pixels are assigned to their nearest centroid using Euclidean distance. The process continues until the entire grid is partitioned, producing a coloured Voronoi tessellation.

## Demo

![Voronoi Tessellation](sprawl.mp4)

## Installation

```bash
pip install numpy matplotlib
```

FFmpeg is required for MP4 export.

## Usage

```bash
python voronoi.py
```

Optional arguments:

| Argument | Default | Description |
|---|---:|---|
| `--centroids` | 10 | Number of centroids |
| `--size` | 60 | Grid dimensions |
| `--batch` | 20 | Pixels per animation frame |
| `--interval` | 30 | Frame interval (ms) |
| `--fps` | 30 | Video FPS |
| `--output` | voronoi.mp4 | Output filename |
| `--seed` | None | Random seed |

