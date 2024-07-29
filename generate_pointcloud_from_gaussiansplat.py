from plyfile import PlyData, PlyElement
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from multiprocessing import Pool, cpu_count
import multiprocessing
from collections import deque
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import signal


def init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)

def count_voxels_chunk(vertices_chunk, voxel_size):
    """Count the number of points in each voxel for a chunk of vertices."""
    voxel_counts = {}
    for vertex in vertices_chunk:
        voxel_coords = (int(vertex['x'] / voxel_size), int(vertex['y'] / voxel_size), int(vertex['z'] / voxel_size))
        if voxel_coords in voxel_counts:
            voxel_counts[voxel_coords] += 1
        else:
            voxel_counts[voxel_coords] = 1
    
    return voxel_counts

def parallel_voxel_counting(vertices, voxel_size=1.0):
    """Counts the number of points in each voxel in a parallelized manner."""
    num_processes = cpu_count()
    chunk_size = len(vertices) // num_processes
    chunks = [vertices[i:i + chunk_size] for i in range(0, len(vertices), chunk_size)]

    num_cores = max(1, multiprocessing.cpu_count() - 1)
    with Pool(processes=num_cores, initializer=init_worker) as pool:
        results = pool.starmap(count_voxels_chunk, [(chunk, voxel_size) for chunk in chunks])

    # Aggregate results from all processes
    total_voxel_counts = {}
    for result in results:
        for k, v in result.items():
            if k in total_voxel_counts:
                total_voxel_counts[k] += v
            else:
                total_voxel_counts[k] = v

    return total_voxel_counts

def get_neighbors(voxel_coords):
    """Get the face-touching neighbors of the given voxel coordinates."""
    x, y, z = voxel_coords
    neighbors = [
        (x-1, y, z), (x+1, y, z),
        (x, y-1, z), (x, y+1, z),
        (x, y, z-1), (x, y, z+1)
    ]
    return neighbors

def knn_worker(args):

    """Utility function for parallel KNN computation."""
    coords, tree, k = args
    coords = coords.reshape(1, -1)  # Reshape to a 2D array
    distances, _ = tree.kneighbors(coords)
    avg_distance = np.mean(distances[:, 1:])

    return avg_distance

def construct_list_of_attributes(features_dc, features_rest, scaling, rotation, rgb):
    l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
    # All channels except the 3 DC
    for i in range(features_dc.shape[1]):
        l.append('f_dc_{}'.format(i))
    for i in range(features_rest.shape[1]):
        l.append('f_rest_{}'.format(i))
    l.append('opacity')
    for i in range(scaling.shape[1]):
        l.append('scale_{}'.format(i))
    for i in range(rotation.shape[1]):
        l.append('rot_{}'.format(i))
    # for i in range(rgb.shape[1]):
    #     l.append('rgb_{}'.format(i))
    # l.append('red')
    # l.append('green')
    # l.append('blue')
    return l

def main():
    args = argparse.ArgumentParser()
    #args.add_argument("--device", type=str, default="cuda", help="Device")
    args.add_argument("--ply_path", type=str, default="", help="Gaussian splatting ply file path")
    args.add_argument("--output_path", type=str, default="", help="Output pointcloud ply file path")
    args.add_argument("--voxel_filtering_threshold", type=float, default=0.32, help="Filtering thresold for dense voxelisation")
    args.add_argument("--voxel_size", type=float, default=1.0, help="Size of the voxels to be used in the filtering")
    args.add_argument("--scale_filtering_size", type=float, default=70.0, help="Size of the Gaussians to be removed in the filtering")
    args.add_argument("--knn_threshold_factor", type=float, default=10.5, help="Threshold for filtering in knn filtering")
    args.add_argument("--knn_k", type=int, default=25, help="Number of neighbours to use in knn")
    args.add_argument("--knn_chunk_size", type=int, default=50000, help="Chunk size for knn")
    args.add_argument("--knn_repeats", type=int, default=3, help="Number of times to run the knn filtering")

    args = args.parse_args()

#    device = "cuda" if torch.cuda.is_available() else "cpu"

    ## Load ply file
    plydata = PlyData.read(args.ply_path)

    ## Voxelised filtering
    vertices = plydata.elements[0]
    # Convert threshold_percentage into a ratio
    threshold_ratio = args.voxel_filtering_threshold / 100.0

    # Parallelized voxel counting
    voxel_counts = parallel_voxel_counting(vertices, args.voxel_size)

    threshold = int(len(vertices) * threshold_ratio)
    dense_voxels = {k: v for k, v in voxel_counts.items() if v >= threshold}

    visited = set()
    max_cluster = set()
    for voxel in dense_voxels:
        if voxel not in visited:
            current_cluster = set()
            queue = deque([voxel])
            while queue:
                current_voxel = queue.popleft()
                visited.add(current_voxel)
                current_cluster.add(current_voxel)
                for neighbor in get_neighbors(current_voxel):
                    if neighbor in dense_voxels and neighbor not in visited:
                        queue.append(neighbor)
                        visited.add(neighbor)
            if len(current_cluster) > len(max_cluster):
                max_cluster = current_cluster

    # Filter vertices to only include those in dense voxels
    # filtered_vertices = [vertex for vertex in vertices if (int(vertex['x'] / voxel_size), int(vertex['y'] / voxel_size), int(vertex['z'] / voxel_size)) in max_cluster]
    filtered_vertices = np.asarray([list(vertex) for vertex in vertices if (int(vertex['x'] / args.voxel_size), int(vertex['y'] / args.voxel_size), int(vertex['z'] / args.voxel_size)) in max_cluster])

    ## Filtering based on size
    scale_filtered = np.stack((np.asarray(filtered_vertices[:,55]),
                             np.asarray(filtered_vertices[:,56]),
                             np.asarray(filtered_vertices[:,57])),  
                   axis=1)

    s_st_filtered = np.einsum('Bi,iB->B', scale_filtered, scale_filtered.T)

    filtered_vertices = filtered_vertices[s_st_filtered>args.scale_filtering_size]

    ## Filtering based on number of neighbours
    for _ in range(args.knn_repeats):
        # Extract vertex data from the current object's data
        vertices = filtered_vertices
        num_vertices = len(vertices)

        # Adjust k based on the number of vertices
        k = max(3, min(args.knn_k, num_vertices // 100))  # Example: ensure k is between 3 and 1% of the total vertices

        # Number of chunks
        num_chunks = (num_vertices + args.knn_chunk_size - 1) // args.knn_chunk_size  # Ceiling division
        masks = []

        # Create a pool of workers
        num_cores = max(1, cpu_count() - 1)  # Leave one core free
        with Pool(processes=num_cores, initializer=init_worker) as pool:
            for i in range(num_chunks):
                start_idx = i * args.knn_chunk_size
                end_idx = min(start_idx + args.knn_chunk_size, num_vertices)  # Avoid going out of bounds
                # chunk_coords = np.vstack((vertices['x'][start_idx:end_idx], vertices['y'][start_idx:end_idx], vertices['z'][start_idx:end_idx])).T
                chunk_coords = np.vstack((vertices[start_idx:end_idx, 0], vertices[start_idx:end_idx, 1], vertices[start_idx:end_idx, 2])).T

                # Compute K-Nearest Neighbors for the chunk
                nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(chunk_coords)
                avg_distances = pool.map(knn_worker, [(coord, nbrs, k) for coord in chunk_coords])

                # Calculate the threshold for removal based on the mean and standard deviation of the average distances
                threshold = np.mean(avg_distances) + args.knn_threshold_factor * np.std(avg_distances)

                # Create a mask for points to retain for this chunk
                mask = np.array(avg_distances) < threshold
                masks.append(mask)

        # Combine masks from all chunks
        combined_mask = np.concatenate(masks)

        # Apply the mask to the vertices and store the result in self.data
        filtered_vertices = vertices[combined_mask]

    ## Extract filtered data
    xyz_filtered = np.stack((np.asarray(filtered_vertices[:,0]),
                             np.asarray(filtered_vertices[:,1]),
                             np.asarray(filtered_vertices[:,2])),
                   axis=1)

    alpha_filtered = np.asarray(filtered_vertices[:,54:55]) * 0.28209479177387814 + 0.5

    rgb_filtered = np.stack((np.asarray(filtered_vertices[:,6]),
                             np.asarray(filtered_vertices[:,7]),
                             np.asarray(filtered_vertices[:,8])),  
                   axis=1) * 0.28209479177387814 + 0.5

    rgba_filtered = np.concatenate([rgb_filtered, alpha_filtered], axis=-1)

    rgba_filtered[rgba_filtered<0] = 0
    rgba_filtered[rgba_filtered>1] = 1
    rgb_filtered[rgb_filtered<0] = 0
    rgb_filtered[rgb_filtered>1] = 1
    alpha_filtered[alpha_filtered<0] = 0
    alpha_filtered[alpha_filtered>1] = 1

    scale_filtered = np.stack((np.asarray(filtered_vertices[:,55]),
                             np.asarray(filtered_vertices[:,56]),
                             np.asarray(filtered_vertices[:,57])),  
                   axis=1)

    s_st_filtered = np.einsum('Bi,iB->B', scale_filtered, scale_filtered.T)

    xyz_filtered_c = xyz_filtered - np.mean(xyz_filtered, axis=0)

#    M = xyz_filtered - np.mean(xyz_filtered, axis=0)
#    C = M.T @ M
#    eigenvalues, eigenvectors = np.linalg.eig(C)

#    xyz_filtered_r = xyz_filtered_c @ eigenvectors

    ## save ply
    path_out = 'output_models/red_wine/point_cloud/iteration_30000/point_cloud_gs_to_pc.ply'

    xyz = xyz_filtered_c
    normals = np.zeros_like(xyz)
    f_dc = np.asarray(filtered_vertices[:,6:9])
    f_rest = np.asarray(filtered_vertices[:,9:54])
    opacities = np.asarray(filtered_vertices[:,54:55])
    scale = np.asarray(filtered_vertices[:,55:58])
    rotation = np.asarray(filtered_vertices[:,58:62])

    rgb_filtered = np.stack((np.asarray(filtered_vertices[:,6]),
                             np.asarray(filtered_vertices[:,7]),
                             np.asarray(filtered_vertices[:,8])),  
                   axis=1) * 0.28209479177387814 + 0.5
    rgb_filtered[rgb_filtered<0] = 0
    rgb_filtered[rgb_filtered>1] = 1
    rgb_filtered = np.array(rgb_filtered*255, dtype=int)

    dtype_full = [(attribute, 'f4') for attribute in construct_list_of_attributes(f_dc, f_rest, scale, rotation, rgb_filtered)]
    dtype_full += [(attribute, 'B') for attribute in ['red', 'green', 'blue']]

    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation, rgb_filtered), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(args.output_path)

    print(f'Done!')

if __name__ == "__main__":
    main()
