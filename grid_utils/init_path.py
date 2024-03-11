import dijkstra3d
import numpy as np

# Perform coarse path planning
def astar3D(field, source, target, feasible):

    #Performs A* 
    inds = dijkstra3d.binary_dijkstra(field, source, target, connectivity=6, background_color=1)
    inds = inds.astype(np.int32)

    # print(f'Returning path of length {len(inds)}')
    traj = feasible[inds[:, 0], inds[:, 1], inds[:, 2]]

    return traj, inds

class Astar():
    # Uses A* as path initialization
    def __init__(self, grid_occupied, grid_points) -> None:
        self.grid_occupied = grid_occupied      # Binary Field (Nx x Ny x Nz)
        self.grid_points = grid_points          # Points corresponding to the field (Nx x Ny x Nz)

        self.cell_sizes = self.grid_points[1, 1, 1] - self.grid_points[0, 0, 0]

    def create_path(self, x0, xf, num_sec=10):
        source = self.get_indices(x0)   # Find nearest grid point and find its index
        target = self.get_indices(xf)

        # source_occupied = self.grid_occupied[source[0], source[1], source[2]]
        target_occupied = self.grid_occupied[target[0], target[1], target[2]]

        if target_occupied:
            print(f'Target {xf, target} is in occupied voxel. Projecting to nearest free point.')
            free_points = self.grid_points[~self.grid_occupied]

            dist = np.linalg.norm(free_points - xf[None, :], axis=-1)
            closest_pt_ind = np.argmin(dist)

            xf = free_points[closest_pt_ind]
            target = self.get_indices(xf)

            print(xf, target)

        # if source_occupied:
        #     raise ValueError('Source is in occupied voxel. Please choose another starting point.')
        
        path3d, indices = astar3D(self.grid_occupied, source, target, self.grid_points)

        try:
            assert len(path3d) > 2
        except:
            raise AssertionError('Could not find a feasible initialize path. Please change the initial/final positions to not be in collision.')

        return path3d

    def get_indices(self, point):
        min_bound = self.grid_points[0, 0, 0]

        transformed_pt = point - min_bound

        indices = np.floor(transformed_pt / self.cell_sizes)

        return_indices = indices.copy()
        # If querying points outside of the bounds, project to the nearest side
        for i, ind in enumerate(indices):
            if ind < 0.:
                return_indices[i] = 0

                print('Point is outside of minimum bounds. Projecting to nearest side. This may cause unintended behavior.')

            elif ind > self.grid_occupied.shape[i]:
                return_indices[i] = self.grid_occupied.shape[i]

                print('Point is outside of maximum bounds. Projecting to nearest side. This may cause unintended behavior.')

        return_indices = return_indices.astype(np.uint32)

        return return_indices