#import time
import argparse
import os
import sys
import numpy as np

from numba import njit

sys.path.append("/home/bcov/sc/random/npose")
#sys.path.append("/home/bcov/sc/random/stable_npose/npose")
import voxel_array
import npose_util as nu

from scipy.spatial import ConvexHull
from scipy.spatial import Delaunay

import atomic_depth



parser = argparse.ArgumentParser(description='arguments')
parser.add_argument('--rolling_ball', type=bool, default=False, 
    help='bool; Use the rolling_ball instead of convex hull for tigher surface.')
parser.add_argument('--rolling_ball_radius', type=bool, default=20.0, 
    help='float; radius for rolling ball method')
parser.add_argument('--debug_dumping', type=bool, default=False, 
    help='bool; Dump some debugging pdbs')
parser.add_argument('--pocket_features', type=bool, default=True, 
    help='bool; Write pocket features to scorefile? They are calculated regardless.')
parser.add_argument('--pocket_residues', type=bool, default=False, 
    help='bool; Do you want do write out pocket residues? Might be a major slow down...')
parser.add_argument('--oligomer_mode', type=bool, default=False, 
    help='bool; For cyclic oligomer asymmetric pockets.')
parser.add_argument('--resl', type=float, default=0.5, 
    help='float; Resolution of 0.5 is suggested; Use 1.0 for more speed.')
parser.add_argument('--pdbs',nargs='+', default=None,
    help='list; list of pdbs ie *pdb')
parser.add_argument('--pdbs_list_file',default=None,
    help='str; list file containing pdb paths')
parser.add_argument('--score_name', type=str, default="pocket_volume.sc", 
    help='str; Name for pocket features scorefile.')

args = parser.parse_args()


def outer_product(a, b):
    return np.einsum('ij,ik->ijk',a,b)


@njit(cache=True, fastmath=True)
def indices_box(lb_idx, ub_idx ):
    sizes = ub_idx - lb_idx + 1
    size = sizes[0]*sizes[1]*sizes[2]

    indices = np.zeros((size, 3), np.int_)

    count = 0
    for k in range(lb_idx[2], ub_idx[2]+1):
        for j in range(lb_idx[1], ub_idx[1]+1):
            for i in range(lb_idx[0], ub_idx[0]+1):
                indices[count, 0] = i
                indices[count, 1] = j
                indices[count, 2] = k
                count = count + 1

    return indices



@njit(cache=True, fastmath=True)
def points_above_plane(pts, normal, pt_plane, padding=0):
    if ( padding != 0 ):
        normal = normal / np.linalg.norm(normal)
        pt_plane = pt_plane - normal*padding

    # to_pts = pts - pt_plane
    # dots = np.dot(to_pts, normal)
    # return dots > 0

    result = np.zeros(len(pts), np.bool_)

    for i in range(len(pts)):
        a = (pts[i, 0] - pt_plane[0]) * normal[0]
        b = (pts[i, 1] - pt_plane[1]) * normal[1]
        c = (pts[i, 2] - pt_plane[2]) * normal[2]

        result[i] = (a + b + c) > 0

    return result


@njit(cache=True, fastmath=True)
def triangular_prism_inner(fill_indices, fill_centers, to_other_face, triangle, upper_triangle, padding):

    # when looking from inside the shape, this triangle runs clockwise
    cw_triangle = triangle.copy()
    # make upper counter clockwise so it's not super confusing
    ccw_upper = upper_triangle.copy()

    if ( np.dot( to_other_face, nu.cross(triangle[0]-triangle[1],
                                         triangle[2]-triangle[1])) < 0 ):
        cw_triangle[1] = triangle[2]
        cw_triangle[2] = triangle[1]
        ccw_upper[1] = upper_triangle[2]
        ccw_upper[2] = upper_triangle[1]


    normal = nu.cross(cw_triangle[0]-cw_triangle[1],
                                       cw_triangle[2]-cw_triangle[1])

    # original triangle
    mask = points_above_plane(fill_centers,
                              nu.cross(cw_triangle[0]-cw_triangle[1],
                                       cw_triangle[2]-cw_triangle[1]),
                              cw_triangle[0],
                              padding
                              )
    fill_indices = fill_indices[mask]
    fill_centers = fill_centers[mask]


    # upper triangle

    mask = points_above_plane(fill_centers,
                              -nu.cross(ccw_upper[0]-ccw_upper[1],
                                        ccw_upper[2]-ccw_upper[1]),
                              ccw_upper[0],
                              padding
                              )
    fill_indices = fill_indices[mask]
    fill_centers = fill_centers[mask]

    # walls
    for iwall in range(3):
        # we are going to pretend triangle is the floor and we're inside
        #  the box
        bottom_right = cw_triangle[iwall]
        top_right = ccw_upper[iwall]
        bottom_left = cw_triangle[(iwall-1)%3]

        mask = points_above_plane(fill_centers,
                                  nu.cross(top_right - bottom_right,
                                           bottom_left - bottom_right),
                                  bottom_right,
                                  padding
                                  )
        fill_indices = fill_indices[mask]
        fill_centers = fill_centers[mask]

    return fill_indices


def triangular_prism_indices(vx, triangle, to_other_face, padding=0):
    upper_triangle = triangle + to_other_face

    # just going to assume it's cubic
    resl = vx.cs[0]

    lb = np.min( np.r_[upper_triangle, triangle], axis=0) - padding - resl
    ub = np.max( np.r_[upper_triangle, triangle], axis=0) + padding + resl

    lb_idx, ub_idx = vx.floats_to_indices(np.array([lb, ub]))

    fill_indices = indices_box(lb_idx, ub_idx)

    fill_centers = vx.indices_to_centers(fill_indices)

    return triangular_prism_inner(fill_indices, fill_centers, to_other_face, triangle, upper_triangle, padding)


@njit(cache=True, fastmath=True)
def fill_pockets_inner(color, arr, lb, ub, cs, shape):
  for x in range(2, shape[0] -2):
      for y in range(2, shape[1] -2):
          for z in range(2, shape[2] -2):
              if ( arr[x, y, z] == 0 ):
                  arr[x, y, z] = color
                  voxel_array.numba_flood_fill_3d_from_here(color, 0, np.array([x, y, z], np.int_), arr, lb, ub, cs, shape)
                  color += 1
  return color


def fill_pockets(vox_arr, color=4):
   color = fill_pockets_inner(color, vox_arr.arr, vox_arr.lb, vox_arr.ub, vox_arr.cs, vox_arr.arr.shape)
   return color



def make_surf_grid(verts, vert_normals, surfgrid):

       atom_size = 2.0

       surfgrid.add_to_clashgrid(verts-vert_normals*-1.5, atom_size, -1)
       surfgrid.add_to_clashgrid(verts-vert_normals*-0.5, atom_size, -1)
       surfgrid.add_to_clashgrid(verts-vert_normals*0.5, atom_size, -1)
       surfgrid.add_to_clashgrid(verts-vert_normals*1.5, atom_size, -1)



first = True

pdbs = []
if args.pdbs:
    pdbs = args.pdbs
elif args.pdbs_list_file:
    with open(args.pdbs_list_file, "r") as f:
        pdbs = f.readlines()
        pdbs = [x.strip() for x in pdbs] 

if len(pdbs) == 0:
    print("no input pdbs")
    sys.exit()

#start = time.time()
for pdb_name in pdbs:
    #print(f"working on {pdb_name}")

    if args.oligomer_mode:
        npose_in, symm = nu.npose_from_file_fast(pdb_name, chains=True)
        symm = len(''.join(set(symm)))
        asu_size = int(nu.nsize(npose_in)/symm)
        npose = npose_in[:nu.R*asu_size*2]
    else:
        npose = nu.npose_from_file_fast(pdb_name)


    atom_size = 2.2
    shell_thickness = 0.01
    padding = 7.0
    resl = args.resl


    clash_grid = nu.ca_clashgrid_from_npose(npose, atom_size, resl, padding=3) #padding=3

    vx = voxel_array.VoxelArray(clash_grid.lb[:3], clash_grid.ub[:3], clash_grid.cs[:3], dtype=np.bool)
 

    if args.rolling_ball:
        

        cas = nu.extract_atoms(npose, [nu.CA])[:,:3]
        
        probe_radius = args.rolling_ball_radius

        radii = np.repeat(2, len(cas))
        surf = atomic_depth.AtomicDepth(cas[:,:3].reshape(-1), radii, probe_radius, 1.0, True, 1)

        verts = surf.get_surface_vertex_bases().reshape(-1, 3)

        if args.debug_dumping:
            nu.dump_pts(verts, "vertices.pdb")

        vert_normals = surf.get_surface_vertex_normals().reshape(-1, 3)
        face_centers = surf.get_surface_face_centers().reshape(-1, 3)
        face_normals = surf.get_surface_face_normals().reshape(-1, 3)

        if args.debug_dumping:
            nu.dump_pts(verts-vert_normals*1.45, "vert_vert_normals.pdb")
        
        try: #had a weird error in voxel grid with spaghetti protein
            make_surf_grid(verts, vert_normals, vx)
        except:
            score_dict = {
                    "pocket_volume": float('nan'),
                    "dimension_1": float('nan'),
                    "dimension_2": float('nan'),
                    "dimension_3": float('nan'),
                    "I1": float('nan'),
                    "I2": float('nan'),
                    "I3": float('nan'),
                    "I1/I3": float('nan'),
                    "I2/I3": float('nan'),
                    }
            if args.pocket_features:
                nu.add_to_score_file(pdb_name, args.score_name, first, score_dict)
                first = False
        

    else:
        hull = ConvexHull(npose[:,:3])

        vertices = hull.points[hull.vertices]
        
        if args.debug_dumping:
            nu.dump_pts(vertices, "vertices.pdb")

        cluster_resl = 5
        centers, _ = nu.cluster_points(vertices, cluster_resl, find_centers=True)

        hull = ConvexHull(vertices[centers])
        delaunay_hull = Delaunay(vertices[centers])

        for simplex_idxs in hull.simplices:
            
            simplex_vertices = hull.points[simplex_idxs]

            normal = np.cross( simplex_vertices[0] - simplex_vertices[1],
                               simplex_vertices[0] - simplex_vertices[2]
                               )
            normal /= np.linalg.norm(normal)
            center = np.mean(simplex_vertices, axis=0)

            test_pt = center + normal*0.01

            test_inside = delaunay_hull.find_simplex(test_pt) >= 0

            if ( test_inside ):
                normal *= -1

            fill = triangular_prism_indices(vx, simplex_vertices, normal*shell_thickness, padding)

            vx.arr[ tuple(fill.T) ] = True

        if args.debug_dumping:
            nu.dump_pts(simplex_vertices, "triangle.pdb")
            vx.dump_mask_true("face.pdb", vx.arr, fraction=0.01)


    number_grid = voxel_array.VoxelArray(vx.lb, vx.ub, vx.cs, np.int)

    number_grid.arr.flat[vx.arr.reshape(-1)] = 1
    number_grid.arr.flat[clash_grid.arr.reshape(-1)] = 2


    # technically the border is supposed to be 0
    # so go to 1, 1, 1
    number_grid.arr[1, 1, 1] = 3

    voxel_array.numba_flood_fill_3d_from_here(3, 0, np.array([1, 1, 1], np.int_), number_grid.arr, number_grid.lb, number_grid.ub, number_grid.cs, number_grid.arr.shape)
    color = fill_pockets(number_grid)

    num_colors = color
    """
    bounding box = 0
    convex hull = 1
    clash grid = 2
    outside hull = 3
    pockfill = 4+
    """

    if args.debug_dumping:
        vx.dump_mask_true("polygon.pdb", number_grid.arr == 1, fraction=1)
        vx.dump_mask_true("clash_grid.pdb", number_grid.arr == 2, fraction=0.1)
        vx.dump_mask_true("outside_hull.pdb", number_grid.arr == 3, fraction=0.01)

    size_by_color = [0, 0, 0, 0]

    for color in range(4, num_colors+1):
        volume = np.sum(number_grid.arr == color) * number_grid.cs[0] * number_grid.cs[1] * number_grid.cs[2]
        size_by_color.append(volume)

    largest_color = np.argmax(size_by_color)
    pocket_volume = size_by_color[largest_color]

    if args.debug_dumping:
        colors_to_dump = [largest_color]
        #uncomment if you reeeaaalllly want to dump all 
        #colors_to_dump = range(0, num_colors+1)
        for color in colors_to_dump:
            number_grid.dump_mask_true("pocket_{}.pdb".format(color), number_grid.arr == color, fraction=0.1)


    #get 3 main dimensions of the pocket
    all_mass_idx = np.array(list(np.where(number_grid.arr == largest_color))).T
    all_mass = number_grid.indices_to_centers(all_mass_idx)

    mass_com = nu.center_of_mass(all_mass)

    # https://en.wikipedia.org/wiki/Moment_of_inertia#Inertia_tensor
    r = all_mass - mass_com
    r_norm2 = np.sum( np.square(r), axis=-1 )

    r_outer_prod = outer_product( r, r )

    dx_dy_dz = np.prod( number_grid.cs )

    inertia_tensor = np.sum( r_norm2[:,None,None] * np.identity(3) - r_outer_prod, axis=0 ) * dx_dy_dz

    # https://ccrma.stanford.edu/~jos/pasp/Principal_Axes_Rotation.html
    eigen_values, eigen_vectors = np.linalg.eig(inertia_tensor)
    eigen_vectors = eigen_vectors.T


    argsort = np.argsort(-eigen_values)
    dimensions = []
    for i, eig_i in enumerate(argsort):

        #print("Eigenvalue %i: %8.3f  -- sqrt() %8.3f"%(i, eigen_values[eig_i], np.sqrt(eigen_values[eig_i])))

        vector = eigen_vectors[eig_i]

        projections = np.dot(r, vector)
        size_in_this_dim = np.max(projections) - np.min(projections)
        dimensions.append(size_in_this_dim)
        #print("Eigenvalue %i length = %8.3f"%(i, dst))

        if args.debug_dumping:
            nu.dump_lines([mass_com - vector * (size_in_this_dim/2)], [vector], size_in_this_dim, "eig%i.pdb"%i)

    dimensions.sort()
    pmoi = []
    for i in range(len(dimensions)):
        #for each principal moment of inertia, add the other 2 axes squared
        #technically this should also be multiplied by the mass of the object
        #but the mass cancels out in the ratios I1/I3 and I2/I3, so i didn't include it
        #i.e. Ixx =(sum of the mass) * (y^2 + x^2)
        #For info on the primary moment of inertia graph, see (Sauer and Schwarz 2003)
        pmoi.append((dimensions[i]*dimensions[i]) + (dimensions[(i+1)%3] * dimensions[(i+1)%3]))
    pmoi.sort()    

    score_dict = {
    "pocket_volume": pocket_volume,
    "dimension_1": dimensions[0],
    "dimension_2": dimensions[1],
    "dimension_3": dimensions[2],
    "I1": pmoi[0],
    "I2": pmoi[1],
    "I3": pmoi[2],
    "I1/I3": pmoi[0]/pmoi[2],
    "I2/I3": pmoi[1]/pmoi[2], 
    }

    if args.pocket_features:
        nu.add_to_score_file(pdb_name, args.score_name, first, score_dict)

        first = False


    if args.pocket_residues:

        """
        try to get the residues lining the pocket now
        """

        # clash grid for pocket
        pocket_clash_grid = nu.clashgrid_from_points(all_mass, 1.1, 0.5, padding=100)
        
        #protein Cbs
        cbs = nu.extract_atoms(npose, [nu.CB])[:,:3]

        pocket_res = []
        for i, cb in enumerate(cbs):
            clash = np.any( pocket_clash_grid.arr[ tuple( pocket_clash_grid.indices_within_x_of( 2.5, cb ).T) ] )
            if clash:
                pocket_res.append(i+1)
        '''
        if args.oligomer_mode:
            pocket_res_1 = []
            for res in pocket_res:
                if res <= asu_size:
                    pocket_res_1.append(res)
                else:
                    if res - asu_size not in pocket_res_1:
                        pocket_res_1.append(res - asu_size)
            pocket_res = pocket_res_1
        '''
        pocket_res_str = [str(x) for x in pocket_res]
        #for pymol selection...
        #print(f"select resi {'+'.join(pocket_res_str)}")

        base = os.path.basename(pdb_name)
        outname = os.path.splitext(base)[0] + ".pos"

        with open(outname, "w") as f:
            for resi in pocket_res_str:
                f.write(resi)
                f.write("\n")


#print(f"run time = {time.time() - start}")
#
