
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap


def evaluate_distance(points, centroids):
    
    diff = points[:, None, :] - centroids[None, :, :]
    return (diff ** 2).sum(axis=2)


def generate_grid(n_centroids=6, size=256):

    # Ensure grid is large enough
    if size <= 10:
        raise ValueError("Grid size must be greater than 10.")

    # Create list of possible centroids. Keeping away from edge
    available = [(x, y)
                 for x in range(5, size - 5)
                 for y in range(5, size - 5)]

    # Ensure number of clusters is correct
    if n_centroids < 1 or n_centroids > len(available):
        raise ValueError("Invalid number of centroids for grid size.")

    # Sample centroids and generate empty grid
    candidate_centroids = random.sample(available, n_centroids)
    grid = np.zeros((size, size), dtype=int)

    return grid, candidate_centroids


def generate_cmap(centroids):
    # Each centroid receives a unique integer label
    return {i: i + 1 for i in range(len(centroids))}


def generate_palette(n_centroids):
    # Use categorical colours for easily distinguishable regions
    if n_centroids <= 20:
        colours = list(plt.get_cmap("tab20").colors[:n_centroids])
    else:
        colours = list(
            plt.get_cmap("turbo")(
                np.linspace(0, 1, n_centroids, endpoint=False)
            )
        )

    # Label 0 is the unvisited background.
    return ListedColormap(["#101018"] + colours)


def extract_surroundings(centroid, width, size):
    c_x, c_y = centroid

    # Sweeping range
    offsets = np.arange(start=-width, stop=width + 1, step=1)

    top = np.stack(
        (c_x + offsets, np.full_like(offsets, c_y - width)), axis=1
    )
    bottom = np.stack(
        (c_x + offsets, np.full_like(offsets, c_y + width)), axis=1
    )

    # Ignore extremes and they are accounted for in top and bottom
    left = np.stack(
        (np.full_like(offsets[1:-1], c_x - width), c_y + offsets[1:-1]),
        axis=1
    )
    right = np.stack(
        (np.full_like(offsets[1:-1], c_x + width), c_y + offsets[1:-1]),
        axis=1
    )

    # Concatenate together and check they are in bounds
    pts = np.concatenate((top, bottom, left, right), axis=0)
    mask = (pts >= 0).all(axis=1) & (pts < size).all(axis=1)

    return pts[mask]


def update_grid(grid, points, colour):
    points = np.asarray(points)
    grid[points[:, 1], points[:, 0]] = colour
    return grid


def simulate(grid, centroids, cmap, size, batch=20):
    centroids_arr = np.array(centroids)

    # Boolean map to store visited points. Set centroids to visited
    visited = np.zeros((size, size), dtype=bool)
    visited[centroids_arr[:, 1], centroids_arr[:, 0]] = True

    n_visited = int(visited.sum())
    yield grid

    width = 1

    # Run till all points have been visited
    while n_visited < size * size:
        # Store relevant points and colours to grid update per iteration
        pts_list, col_list = [], []

        for i, centroid in enumerate(centroids):
            edges = extract_surroundings(centroid, width, size)

            if len(edges) == 0:
                continue

            # Keep only non visited edges
            edges = edges[~visited[edges[:, 1], edges[:, 0]]]

            if len(edges) == 0:
                continue

            # Smallest of the distance to other centroids
            owner = evaluate_distance(edges, centroids_arr).argmin(axis=1)
            mine = edges[owner == i]

            # Assign new edges to visisted
            visited[mine[:, 1], mine[:, 0]] = True

            pts_list.append(mine)
            col_list.append(np.full(len(mine), cmap[i]))

        if pts_list:
            pts = np.concatenate(pts_list)
            cols = np.concatenate(col_list)

            # Creates the sprawl effect
            order = np.random.permutation(len(pts))
            pts, cols = pts[order], cols[order]

            n_visited += len(pts)

            # Update in bathches for sprawl
            for k in range(0, len(pts), batch):
                update_grid(grid, pts[k:k + batch], cols[k:k + batch])
                yield grid

        width += 1


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Centroid-based spatial growth simulation."
    )

    parser.add_argument(
        "--centroids", type=int, default=10,
        help="Number of centroids to initialise"
    )
    parser.add_argument(
        "--size", type=int, default=60,
        help="Square grid size"
    )
    parser.add_argument(
        "--batch", type=int, default=20,
        help="Number of pixels updated per animation frame"
    )
    parser.add_argument(
        "--interval", type=int, default=30,
        help="Delay between animation frames in milliseconds"
    )
    parser.add_argument(
        "--fps", type=int, default=30,
        help="Output video frames per second"
    )
    parser.add_argument(
        "--output", type=str, default="voronoi.gif",
        help="Output video filename"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    if args.batch < 1 or args.fps < 1 or args.interval < 1:
        parser.error("Batch, FPS and interval must be positive.")

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    n_centroids = args.centroids
    size = args.size

    grid, centroids = generate_grid(
        n_centroids=n_centroids,
        size=size
    )

    cmap = generate_cmap(centroids)
    palette = generate_palette(n_centroids)

    # Initialise each centroid with same colour
    for i, centroid in enumerate(centroids):
        update_grid(grid, [centroid], 255)

    fig, ax = plt.subplots(figsize=(6, 6))

    im = ax.imshow(
        grid,
        cmap=palette,
        vmin=0,
        vmax=n_centroids,
        interpolation="nearest"
    )

    ax.axis("off")

    def update(g):
        im.set_data(g)
        return (im,)

    ani = FuncAnimation(
        fig,
        update,
        frames=simulate(
            grid, centroids, cmap, size,
            batch=args.batch
        ),
        interval=args.interval,
        blit=True,
        repeat=False,
        cache_frame_data=False
    )

    ani.save(args.output, fps=args.fps)
    plt.close(fig)

    print(f"Animation saved to {args.output}")
