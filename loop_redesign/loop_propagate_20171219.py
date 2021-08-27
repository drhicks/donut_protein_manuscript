from __future__ import print_function

import argparse
import numpy as np
import pyrosetta
import sys
import pyrosetta.toolbox.numpy_utils as np_utils
from os import path, listdir, makedirs


def my_own_2D_numpy_to_rosetta(np_arr):
    ros_container = pyrosetta.rosetta.numeric.xyzMatrix_double_t(0.)
    ros_container.xx = np_arr[0, 0]
    ros_container.xy = np_arr[0, 1]
    ros_container.xz = np_arr[0, 2]

    ros_container.yx = np_arr[1, 0]
    ros_container.yy = np_arr[1, 1]
    ros_container.yz = np_arr[1, 2]

    ros_container.zx = np_arr[2, 0]
    ros_container.zy = np_arr[2, 1]
    ros_container.zz = np_arr[2, 2]
    return ros_container


def my_own_rotate_pose(p, R):
    '''Apply a rotation matrix to all of the coordinates in a Pose.

    Args:
        p (Pose): The Pose instance to manipulate
        R (np.mat): A rotation matrix to apply to the Pose coordinates

    Returns:
        None. The input Pose is manipulated.

    '''
    # t must be an xyzMatrix_Real
    for i in range(1, p.size() + 1):
        for j in range(1, p.residue(i).natoms() + 1):
            v = p.residue(i).atom(j).xyz()

            x = R.xx * v.x + R.xy * v.y + R.xz * v.z
            y = R.yx * v.x + R.yy * v.y + R.yz * v.z
            z = R.zx * v.x + R.zy * v.y + R.zz * v.z

            p.residue(i).atom(j).xyz(pyrosetta.rosetta.numeric.xyzVector_double_t(x, y, z))

def rmsd_2_np_arrays(crds1,
                     crds2):
    #"""Returns RMSD between 2 sets of [nx3] numpy array"""
    #D assert(crds1.shape[1] == 3)
    #D assert(crds1.shape == crds2.shape)

    ##Corrected to account for removal of the COM
    COM1 = np.sum(crds1,axis=0) / crds1.shape[0]
    COM2 = np.sum(crds2,axis=0) / crds2.shape[0]
    crds1-=COM1
    crds2-=COM2
    n_vec = np.shape(crds1)[0]
    correlation_matrix = np.dot(np.transpose(crds1), crds2)
    v, s, w_tr = np.linalg.svd(correlation_matrix)
    is_reflection = (np.linalg.det(v) * np.linalg.det(w_tr)) < 0.0
    if is_reflection:
            s[-1] = - s[-1]
            v[:,-1] = -v[:,-1]
    E0 = sum(sum(crds1 * crds1)) + sum(sum(crds2 * crds2))
    rmsd_sq = (E0 - 2.0*sum(s)) / float(n_vec)
    rmsd_sq = max([rmsd_sq, 0.0])

    return np.sqrt(rmsd_sq)

def rmsd_2_np_arrays_no_aln(crds1,
                     crds2):
    crds1 = crds1.flatten()
    crds2 = crds2.flatten()
    assert(crds1.shape == crds2.shape)
    rmsd_sq = sum([ (crds1[x]-crds2[x])**2 for x in range(len(crds1)) ])
    return np.sqrt(rmsd_sq)

def rmsd_by_ndxs_atoms(pose1,
                     init_res1,
                     end_res1,
                     pose2,
                     init_res2,
                     end_res2,
                    target_atoms=["CA","C","O","N"]):

    numRes=(end_res1-init_res1+1)
    coorA=np.zeros(((len(target_atoms)*numRes),3), float)
    coorB=np.zeros(((len(target_atoms)*numRes),3), float)

    counter=0
    for ires in range (init_res1, (end_res1+1)):
        for jatom in target_atoms:
            for dim in range(0,3):
                coorA[counter,dim]=(pose1.residue(ires).xyz(jatom)[dim])
            counter+=1

    counter=0
    for ires in range (init_res2, (end_res2+1)):
        for jatom in target_atoms:
            for dim in range(0,3):
                coorB[counter,dim]=(pose2.residue(ires).xyz(jatom)[dim])
            counter+=1

    #Calculate the RMSD
    rmsdVal = rmsd_2_np_arrays_no_aln(coorB, coorA)

    return rmsdVal

def get_anchor_coordinates_from_pose(p, reslist):
    """Pack atomic coordinates in a single residue into a numpy matrix.

    Args:
        p (Pose): Pose from coordinates will be extracted
        start (int): the pose-numbered start residue
        stop (int): the pose-numbered stop residue

    Returns:
        Set of coordinates (np.mat) for the atoms in the Pose

    """
    _bb_atoms = ['N', 'CA', 'C', 'O']
    coords = list()
    for resNo in reslist:
        res = p.residue(resNo)

        # only iterate over relevant atoms
        for i in _bb_atoms:
            coords.append([res.xyz(i).x, res.xyz(i).y, res.xyz(i).z])

    return np.mat(coords)


def align_loop_pose_to_anchor_coords(p, coords, Nterm_only=False):
    """Compute and apply the transformation to superpose the residue onto the
    stub.

    Args:
        p (Pose): The Pose instance to manipulate
        coords (np.mat): The coordinates of the atoms in the stub

    Returns:
        The Pose

    """
    if Nterm_only:
        moveable_coords = get_anchor_coordinates_from_pose(p, [1])
    else:
        moveable_coords = get_anchor_coordinates_from_pose(p, [1, p.size()])

    R, t = np_utils.rigid_transform_3D(moveable_coords, coords)
    #np_utils.rotate_pose(p, np_utils.numpy_to_rosetta(R)) # JHL, sth wrong w/ dig's pyrosetta: xx() not callable, but xx directly accessible
    my_own_rotate_pose(p, my_own_2D_numpy_to_rosetta(R))  # JHL, so I had to rewrite np->rosetta and rotation function to change xx() to xx
    np_utils.translate_pose(p, np_utils.numpy_to_rosetta(t.T))


    return p


def insert_loop_into_repeat_scaffold_Cterm(p, loop_pose, anchor): 
    c = get_anchor_coordinates_from_pose(p, [anchor]) 
    looooop = align_loop_pose_to_anchor_coords(loop_pose, c, Nterm_only=True)
    p.delete_residue_range_slow(anchor, p.size()-1) 
    smashed = pyrosetta.rosetta.protocols.grafting.insert_pose_into_pose(p, looooop, anchor-1)
    return smashed


def insert_loop_into_repeat_scaffold(p, loop_pose, anchor_range): 
    c = get_anchor_coordinates_from_pose(p, anchor_range) 
    looooop = align_loop_pose_to_anchor_coords(loop_pose, c)
    p.delete_residue_range_slow(anchor_range[0], anchor_range[1]) 
    smashed = pyrosetta.rosetta.protocols.grafting.insert_pose_into_pose(p, looooop, anchor_range[0]-1)
    return smashed


def find_repeat(seq, probe_len=10):
    probe = seq[:probe_len]
    repeatunitseq, repeat_num = '', 0
    for i in range(probe_len, len(seq)-probe_len):
        if seq[i:i+probe_len] == probe:
            repeatunitseq = seq[:i]
            assert( (len(seq)%len(repeatunitseq)) == 0 )
            repeat_num = len(seq) / len(repeatunitseq)
            break
    return repeatunitseq, int(repeat_num)

def find_loop_by_seq(pseq, refseq, ala_check=True, reftaglen=20):
    '''
        find the first polyAla loop
    '''
    start, end = -1, -1
    reftag = ''
    anchor_n, anchor_c = -1, -1
    for i in range(len(pseq)):
        if pseq[i] != refseq[i]:
            if start == -1:
                if ala_check and pseq[i] == 'A' or (not ala_check):
                    start = i+1
                    anchor_n = i+1
                    reftag = refseq[i:i+reftaglen]
            #elif (ala_check and pseq[i] != 'A') or pseq[i:i+reftaglen] == reftag:
            elif  (ala_check and pseq[i] != 'A') or pseq[i:i+5] in reftag:
                end = i
                anchor_c = refseq.index(pseq[i:i+5])
                break
    return range(start, end+1), range(anchor_n, anchor_c+1) 

def find_loop_by_dssp(p, min_helix_length=6):
    '''
        look for the two loops connecting Helix1-Helix2 and Helix2-Helix3
    '''
    dssp_obj = pyrosetta.rosetta.core.scoring.dssp.Dssp(p)
    dssp_obj.dssp_reduced()
    dssp = dssp_obj.get_dssp_secstruct()
    helices = []
    prev = ''
    start = -1
    for i in range(len(dssp)):
        if dssp[i] == 'H' and dssp[i] != prev:
            start = i+1
        elif dssp[i] != 'H' and prev == 'H' and i - start >= min_helix_length:
            helices.append([start,i])
            start = -1
        prev = dssp[i]
    loop1 = [helices[0][1]+1, helices[1][0]-1]
    loop2 = [helices[1][1]+1, helices[2][0]-1]
    assert(loop1[1] - loop1[0] > 0)
    assert(loop2[1] - loop2[0] > 0)
    return loop1, loop2

def find_loop_by_alignment(ref, p, fragsize=3, rmsd_cutoff=0.01, flank=1):
    match_list = []
    p_to_ref, ref_to_p = {}, {}
    for curr in range(1, ref.size()+1-fragsize):
        ref_frag = range(curr, curr+fragsize)
        for p_curr in range(curr, p.size()+1-fragsize):
            p_frag = range(p_curr, p_curr+fragsize)
            rmsd = rmsd_by_ndxs_atoms(ref, ref_frag[0], ref_frag[-1], p, p_frag[0], p_frag[-1], ["CA"])
            #print(ref_frag, p_frag, rmsd)
            if rmsd < rmsd_cutoff:
                for i in range(len(ref_frag)):
                    #match_list.append(p_frag[i])
                    p_to_ref[p_frag[i]] = ref_frag[i]
                    ref_to_p[ref_frag[i]] = p_frag[i]
                break
    #print('p_to_ref', p_to_ref)
    #print('ref_to_p', ref_to_p)

    #look for loop residues
    #loopres = [x for x in range(1,int(p.size())) if x not in match_list]
    loopres = [x for x in range(1,int(p.size())) if x not in p_to_ref]
    #print(loopres)
    loops = []
    loops_ref = []
    start = loopres[0]
    for i in range(1,len(loopres)):
        if loopres[i] != loopres[i-1]+1:
            loops.append([start-flank,loopres[i-1]+flank])
            loops_ref.append([p_to_ref[loops[-1][0]],p_to_ref[loops[-1][1]]])
            start = loopres[i]
        elif i == len(loopres)-1:
            #loops.append([start,loopres[i]])
            loops.append([start-flank,loopres[i]+flank])
            loops_ref.append([p_to_ref[loops[-1][0]],p_to_ref[loops[-1][1]]])
            break
    
    return loops[0], loops[1], loops_ref[0], loops_ref[1]


def main(argv):
    parser = argparse.ArgumentParser(description='Program')
    parser.add_argument('-i', '--in_file', action='store', type=str,required=True,
                        help='input single pdb or pdblist of poses with one loop inserted')
    parser.add_argument('-r', '--ref_file', action='store', type=str,required=True,
                        help='refence scaffold pdb')
    parser.add_argument('-n', '--repeat_num', action='store', type=int,default=4,
                        help='number of repeat in the input')
    parser.add_argument('-f', '--flank', action='store', type=int,default=1,
                        help='number of flanking residues at each side to be included during grafting')

    args = parser.parse_args()
    in_fname = args.in_file.strip()
    ref_fname = args.ref_file.strip()
    flank = args.flank

    pyrosetta.init(extra_options=' '.join(['-mute all']))

    ref = pyrosetta.pose_from_file(ref_fname)
    refseq = ref.sequence()
    repeat_num = args.repeat_num
    repeatlen = int(len(refseq) / repeat_num)

    if in_fname[-4:] == '.pdb':
        inputlist = [in_fname.strip()]
    else:
        fin = open(in_fname, 'r')
        inputlist = fin.read().split('\n')[:-1]
    for line in inputlist:    
        inputpdb = line.strip()
        #print(inputpdb)
        p = pyrosetta.pose_from_file(inputpdb)
        #loop1, loop2 = find_loop_by_dssp(p)
        loop1, loop2, loop1_ref, loop2_ref = find_loop_by_alignment(ref, p)
        #print(loop1, loop2, loop1_ref, loop2_ref)
        loop1_pose, loop2_pose = p.clone(), p.clone()
        loop1_pose.delete_residue_range_slow(loop1[1]+1,p.size())
        loop1_pose.delete_residue_range_slow(1,loop1[0]-1)
        loop2_pose.delete_residue_range_slow(loop2[1]+1,p.size())
        loop2_pose.delete_residue_range_slow(1,loop2[0]-1)
        #loop1_pose.dump_pdb('loop1.pdb')
        #loop2_pose.dump_pdb('loop2.pdb')

        new_pose = ref.clone()
        last_repeat_pos = (repeat_num-1)*repeatlen
        #print([loop2_ref[0]+last_repeat_pos,loop2_ref[1]+last_repeat_pos], [loop1_ref[0]+last_repeat_pos,loop1_ref[1]+last_repeat_pos])
        new_pose = insert_loop_into_repeat_scaffold_Cterm(new_pose, loop2_pose, loop2_ref[0]+last_repeat_pos)
        #new_pose.dump_pdb('last2.pdb')
        new_pose = insert_loop_into_repeat_scaffold(new_pose, loop1_pose, [loop1_ref[0]+last_repeat_pos,loop1_ref[1]+last_repeat_pos])
        #new_pose.dump_pdb('last1.pdb')
        for k in range(repeat_num-2,-1,-1):
            curr_repeat_pos = k*repeatlen
            #print([loop2_ref[0]+curr_repeat_pos,loop2_ref[1]+curr_repeat_pos], [loop1_ref[0]+curr_repeat_pos,loop1_ref[1]+curr_repeat_pos])
            new_pose = insert_loop_into_repeat_scaffold(new_pose, loop2_pose, [loop2_ref[0]+curr_repeat_pos,loop2_ref[1]+curr_repeat_pos])       
            new_pose = insert_loop_into_repeat_scaffold(new_pose, loop1_pose, [loop1_ref[0]+curr_repeat_pos,loop1_ref[1]+curr_repeat_pos])       
            #new_pose.dump_pdb('prop_{}.pdb'.format(k))
        new_pose.delete_residue_range_slow(new_pose.size()-flank-1, new_pose.size()) # delete the flanking residues in the last loop
        outpre = '{}_prop'.format(inputpdb.split('.pdb')[0])
        new_pose.dump_pdb(outpre+'.pdb')

if __name__ == '__main__':
    main(sys.argv)

