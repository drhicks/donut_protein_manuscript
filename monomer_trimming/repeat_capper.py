#!/software/miniconda3/envs/pyrosetta3/bin/python

from __future__ import division


import os
import sys
import glob
import numpy as np

import pyrosetta
pyrosetta.init("-beta_cart")

def extract_mer(in_pose, mer_start=1, mer_end=9):
    fragment_pose = pyrosetta.rosetta.core.pose.Pose()
    fragment_pose.append_residue_by_jump(in_pose.residue(mer_start), 1)
    for i in  range(mer_start + 1, mer_end+1):
        fragment_pose.append_residue_by_bond(in_pose.residue(i))
    return fragment_pose

def get_ss_lengths(pose):
    repeat_unit_dssp = pyrosetta.rosetta.core.scoring.dssp.Dssp(pose)
    repeat_unit_ss = repeat_unit_dssp.get_dssp_secstruct()
    from itertools import groupby
    groups = groupby(repeat_unit_ss)
    ss_lengths = [(label, sum(1 for _ in group)) for label, group in groups]
    if ss_lengths[-2][1] < 8:
        tmp_ss_lengths = ss_lengths[0:-3]
        tmp_ss_lengths.append(['L', ss_lengths[-1][1]+ss_lengths[-2][1]+ss_lengths[-3][1]])
        ss_lengths = tmp_ss_lengths
    return ss_lengths

def calculate_ddg_per_residue(base_pose, term_pose, scorefxn=None):
    num_residues = term_pose.size()
    tmp_pose = base_pose.clone()
    tmp_pose.append_pose_by_jump(term_pose, tmp_pose.size())
    if scorefxn == None:
        scorefxn = pyrosetta.get_fa_scorefxn()
    tmp_pose_score = scorefxn(tmp_pose)
    translate = pyrosetta.rosetta.protocols.rigid.RigidBodyTransMover(tmp_pose,1)
    translate.step_size(100)
    translate.trans_axis(pyrosetta.rosetta.numeric.xyzVector_double_t(1,0,0))
    translate.apply(tmp_pose)
    ### should repack and min chain B, but haven't added yet
    
    ### should repack and min chain B, but haven't added yet
    tmp_pose_after_score = scorefxn(tmp_pose)
    print(tmp_pose_after_score)
    ddg = tmp_pose_score - tmp_pose_after_score
    ddg_per_residue = ddg / num_residues
    #ddg_per_residue = ddg
    return ddg_per_residue

def get_repeat_length(pose):
    if pose.size() % 4 != 0:
        print('not a repeat of 4')
    else:
        repeat_length = int(pose.size() / 4)
        return(repeat_length)

def create_extended_pose(pose, both=True, N_only=False, C_only=False, trim_repeat=False):
    if N_only == True and C_only == True:
        print("cannot ONLY add to BOTH N and C")
        return
    if N_only == True or C_only == True:
        both=False
        
    #get building blocks
    repeat_length = get_repeat_length(pose)
    ss_lengths = get_ss_lengths(pose)
    H1_L1_H2_L2_H1_L2 = extract_mer(pose, repeat_length+1, repeat_length*2+ss_lengths[4][1]+ss_lengths[5][1])
    H2_L2_H1_L2_H2_L2 = extract_mer(pose, repeat_length+1-ss_lengths[6][1]-ss_lengths[7][1], repeat_length*2)
    internal_repeat = extract_mer(pose, repeat_length+1, repeat_length*3)
    repeat_start = extract_mer(pose, 1, repeat_length)
    repeat_end = extract_mer(pose, repeat_length*3+1, pose.size())
    #superimpose the building blocks
    vec1 = pyrosetta.rosetta.utility.vector1_unsigned_long()
    for i in range(1, repeat_start.size()+1):
        vec1.append(i)
    #super impose on start
    #pyrosetta.rosetta.protocols.grafting.superimpose_overhangs_heavy(repeat_start, H2_L2_H1_L2_H2_L2, False, 2, 20, 2, 2)
    pyrosetta.rosetta.protocols.toolbox.pose_manipulation.superimpose_pose_on_subset_CA(H2_L2_H1_L2_H2_L2, repeat_start, vec1, -(repeat_start.size()-H2_L2_H1_L2_H2_L2.size()))
    #super impose on end
    #pyrosetta.rosetta.protocols.grafting.superimpose_overhangs_heavy(repeat_end, H1_L1_H2_L2_H1_L2, False, 2, 20, 2, 2)
    pyrosetta.rosetta.protocols.toolbox.pose_manipulation.superimpose_pose_on_subset_CA(H1_L1_H2_L2_H1_L2, repeat_end, vec1, 0)
    
    
    if both == True: 
        #create the extended pose by piecing together the building blocks
        pose_extended = pyrosetta.rosetta.core.pose.Pose()
        for j in range(1, H2_L2_H1_L2_H2_L2.size()+1):
            pose_extended.append_residue_by_bond(H2_L2_H1_L2_H2_L2.residue(j))
        for j in range(1, internal_repeat.size()+1):
            pose_extended.append_residue_by_bond(internal_repeat.residue(j))
        for j in range(1, H1_L1_H2_L2_H1_L2.size()+1):
            pose_extended.append_residue_by_bond(H1_L1_H2_L2_H1_L2.residue(j))
        return pose_extended
    elif N_only == True and trim_repeat == False:
        my_new_N_pose = pyrosetta.rosetta.core.pose.Pose()
        for j in range(1, H2_L2_H1_L2_H2_L2.size()+1):
            my_new_N_pose.append_residue_by_bond(H2_L2_H1_L2_H2_L2.residue(j))
        for j in range(repeat_length+1, pose.size()+1):
            my_new_N_pose.append_residue_by_bond(pose.residue(j))
        return my_new_N_pose
    elif C_only == True and trim_repeat == False:
        my_new_C_pose = pyrosetta.rosetta.core.pose.Pose()
        for j in range(1, pose.size()-repeat_length+1):
            my_new_C_pose.append_residue_by_bond(pose.residue(j))
        for j in range(1, H1_L1_H2_L2_H1_L2.size()+1):
            my_new_C_pose.append_residue_by_bond(H1_L1_H2_L2_H1_L2.residue(j))
        return my_new_C_pose
    elif N_only == True and trim_repeat == True:
        #do something
        my_new_N_pose = pyrosetta.rosetta.core.pose.Pose()
        for j in range(1, H2_L2_H1_L2_H2_L2.size()+1):
            my_new_N_pose.append_residue_by_bond(H2_L2_H1_L2_H2_L2.residue(j))
        for j in range(repeat_length+1, pose.size()-repeat_length+1):
            my_new_N_pose.append_residue_by_bond(pose.residue(j))
        return my_new_N_pose
    elif C_only == True and trim_repeat == True:
        #do something
        my_new_C_pose = pyrosetta.rosetta.core.pose.Pose()
        for j in range(repeat_length+1, pose.size()-repeat_length+1):
            my_new_C_pose.append_residue_by_bond(pose.residue(j))
        for j in range(1, H1_L1_H2_L2_H1_L2.size()+1):
            my_new_C_pose.append_residue_by_bond(H1_L1_H2_L2_H1_L2.residue(j))
        return my_new_C_pose    

def get_pose_variants(pose):
    pose_ss_lengths = get_ss_lengths(pose)
    
    pose_without_N_term_helix = extract_mer(pose, 
                                            pose_ss_lengths[0][1]+pose_ss_lengths[1][1]+pose_ss_lengths[2][1]+1, 
                                            pose.size())
    
    pose_without_C_term_helix = extract_mer(pose, 
                                            1, 
                                            pose.size()-(pose_ss_lengths[-1][1]+pose_ss_lengths[-2][1]+pose_ss_lengths[-3][1]))
    
    pose_N_term = extract_mer(pose,
                             1,
                             pose_ss_lengths[0][1]+pose_ss_lengths[1][1])
    
    pose_C_term = extract_mer(pose,
                             pose.size()-(pose_ss_lengths[-1][1]+pose_ss_lengths[-2][1])+1,
                             pose.size())
    
    return pose_without_N_term_helix, pose_without_C_term_helix, pose_N_term, pose_C_term

def average_vector(in_vector):
    # input format should be [[x,y,z],[x,y,z],[x,y,z],[x,y,z]]
    # output should be [x,y,z]
    in_vector = np.matrix(in_vector)
    out_vector = np.array(in_vector.mean(0)).tolist()
    return out_vector

def distance_between_vectors(a,b):
    a = np.array(a)
    b = np.array(b)
    dist = np.linalg.norm(a-b)
    return dist

def trim_extra_residues(term_pose, base_pose, termini):
    if termini == 'N':
        #do something
        # trim N terminal side of terminal helix to be close to C terminal side of first helix
        ss_lengths_base = get_ss_lengths(base_pose)
        #print(ss_lengths_base)
        i_xyz = []
        sum_v = (
                term_pose.residue(1).xyz('CA')+
                term_pose.residue(2).xyz('CA')+
                term_pose.residue(3).xyz('CA')+
                term_pose.residue(4).xyz('CA')
                )
        i_xyz.append(sum_v.x/4)
        i_xyz.append(sum_v.y/4)
        i_xyz.append(sum_v.z/4)
        
        term_distance = [0,999]
        base_xyz = []
        for i in range(1, base_pose.size()+1):
            tmp_base_xyz = []
            tmp_base_xyz.append(base_pose.residue(i).xyz('CA').x)
            tmp_base_xyz.append(base_pose.residue(i).xyz('CA').y)
            tmp_base_xyz.append(base_pose.residue(i).xyz('CA').z)
            i_distance = distance_between_vectors(i_xyz, tmp_base_xyz)
            if i_distance < term_distance[1]:
                term_distance = [i, i_distance]
                base_xyz = tmp_base_xyz
        #print(base_xyz)
        
        index_distance = [0,999]
        for i in range(1, term_pose.size()-3):
            i_xyz = []
            sum_v = (
                term_pose.residue(i).xyz('CA')+
                term_pose.residue(i+1).xyz('CA')+
                term_pose.residue(i+2).xyz('CA')+
                term_pose.residue(i+3).xyz('CA')
                    )
            i_xyz.append(sum_v.x/4)
            i_xyz.append(sum_v.y/4)
            i_xyz.append(sum_v.z/4)
            
            i_plus_one_xyz = []
            sum_v = (
                term_pose.residue(i+1).xyz('CA')+
                term_pose.residue(i+2).xyz('CA')+
                term_pose.residue(i+3).xyz('CA')+
                term_pose.residue(i+4).xyz('CA')
                    )
            i_plus_one_xyz.append(sum_v.x/4)
            i_plus_one_xyz.append(sum_v.y/4)
            i_plus_one_xyz.append(sum_v.z/4)
            
            i_distance = distance_between_vectors(i_xyz, base_xyz)
            i_plus_one_distance = distance_between_vectors(i_plus_one_xyz, base_xyz)
            
            if i_distance < index_distance[1]:
                index_distance = [i, i_distance]
            
            #print(distance_between_vectors(i_xyz, base_xyz))
            #print(distance_between_vectors(i_plus_one_xyz, base_xyz))
            
            #if i_distance > i_plus_one_distance:
            #    continue
            #else:
            #    #do something
            #    new_term_pose = extract_mer(term_pose, i, term_pose.size())
            #    return [new_term_pose, i]
            #    break
        
        new_term_pose = extract_mer(term_pose, index_distance[0], term_pose.size())
        return [new_term_pose, index_distance[0]]
    
    elif termini == 'C':
        #do something else
        # trim C terminal side of terminal helix to be close to N terminal side of last helix
        ss_lengths_base = get_ss_lengths(base_pose)
        #print(ss_lengths_base)
        i_xyz = []
        sum_v = (
                term_pose.residue(term_pose.size()).xyz('CA')+
                term_pose.residue(term_pose.size()-1).xyz('CA')+
                term_pose.residue(term_pose.size()-2).xyz('CA')+
                term_pose.residue(term_pose.size()-3).xyz('CA')
                )
        i_xyz.append(sum_v.x/4)
        i_xyz.append(sum_v.y/4)
        i_xyz.append(sum_v.z/4)
        
        term_distance = [0,999]
        base_xyz = []
        for i in range(1, base_pose.size()+1):
            tmp_base_xyz = []
            tmp_base_xyz.append(base_pose.residue(i).xyz('CA').x)
            tmp_base_xyz.append(base_pose.residue(i).xyz('CA').y)
            tmp_base_xyz.append(base_pose.residue(i).xyz('CA').z)
            i_distance = distance_between_vectors(i_xyz, tmp_base_xyz)
            if i_distance < term_distance[1]:
                term_distance = [i, i_distance]
                base_xyz = tmp_base_xyz
        #print(base_xyz)
        
        index_distance = [0,999]
        for i in range(term_pose.size(), 4, -1):
            #print(i)
            i_xyz = []
            sum_v = (
                term_pose.residue(i).xyz('CA')+
                term_pose.residue(i-1).xyz('CA')+
                term_pose.residue(i-2).xyz('CA')+
                term_pose.residue(i-3).xyz('CA')
                    )
            i_xyz.append(sum_v.x/4)
            i_xyz.append(sum_v.y/4)
            i_xyz.append(sum_v.z/4)
            
            i_plus_one_xyz = []
            sum_v = (
                term_pose.residue(i-1).xyz('CA')+
                term_pose.residue(i-2).xyz('CA')+
                term_pose.residue(i-3).xyz('CA')+
                term_pose.residue(i-4).xyz('CA')
                    )
            i_plus_one_xyz.append(sum_v.x/4)
            i_plus_one_xyz.append(sum_v.y/4)
            i_plus_one_xyz.append(sum_v.z/4)
            
            i_distance = distance_between_vectors(i_xyz, base_xyz)
            i_plus_one_distance = distance_between_vectors(i_plus_one_xyz, base_xyz)
            
            if i_distance < index_distance[1]:
                index_distance = [i, i_distance]
            
            #print(distance_between_vectors(i_xyz, base_xyz))
            #print(distance_between_vectors(i_plus_one_xyz, base_xyz))
            
            #if i_distance > i_plus_one_distance:
            #    continue
            #else:
            #    #do something
            #    j = term_pose.size() - i # how many residues to remove
            #    new_term_pose = extract_mer(term_pose, 1, i)
            #    return [new_term_pose, j]
            #    break
            
        new_term_pose = extract_mer(term_pose, 1, index_distance[0])
        return [new_term_pose, term_pose.size() - index_distance[0]]


pdb_folder = sys.argv[1]
print('searching for',pdb_folder)
pdb_files = glob.glob('{}/*pdb'.format(pdb_folder))
print('found',pdb_files)
for pdb in pdb_files:
    base_name = os.path.basename(pdb)[0:-4]
    pose = pyrosetta.pose_from_file(pdb)

    scorefxn = pyrosetta.get_fa_scorefxn()
    extended_pose = create_extended_pose(pose)

    #get all the pose variant pieces
    pose_without_N_term_helix, pose_without_C_term_helix, pose_N_term, pose_C_term = get_pose_variants(pose)
    extended_pose_without_N_term_helix, extended_pose_without_C_term_helix, extended_pose_N_term, extended_pose_C_term = get_pose_variants(extended_pose)

    ## trim extra residues
    pose_N_term = trim_extra_residues(pose_N_term, pose_without_N_term_helix, 'N')
    pose_C_term = trim_extra_residues(pose_C_term, pose_without_C_term_helix, 'C')
    extended_pose_N_term = trim_extra_residues(extended_pose_N_term, extended_pose_without_N_term_helix, 'N')
    extended_pose_C_term = trim_extra_residues(extended_pose_C_term, extended_pose_without_C_term_helix, 'C')
    ## trim extra residues
    

    pose_N_term_ddg = calculate_ddg_per_residue(pose_without_N_term_helix, pose_N_term[0])
    pose_C_term_ddg = calculate_ddg_per_residue(pose_without_C_term_helix, pose_C_term[0])
    extended_pose_N_term_ddg = calculate_ddg_per_residue(extended_pose_without_N_term_helix, extended_pose_N_term[0])
    extended_pose_C_term_ddg = calculate_ddg_per_residue(extended_pose_without_C_term_helix, extended_pose_C_term[0])

    print(pose_N_term_ddg, pose_C_term_ddg, extended_pose_N_term_ddg, extended_pose_C_term_ddg)
    print(get_ss_lengths(extended_pose))

    if extended_pose_N_term_ddg < pose_N_term_ddg and extended_pose_C_term_ddg < pose_C_term_ddg:
        #output two poses
        #pose1 has N_term extended and C_term double trim 
        #pose1 has N_term double trim and C_term extended
        p1 = create_extended_pose(pose, both=False, N_only=True, C_only=False, trim_repeat=True)
        p1 = extract_mer(p1, extended_pose_N_term[1], p1.size()-pose_C_term[1])
        p2 = create_extended_pose(pose, both=False, N_only=False, C_only=True, trim_repeat=True)
        p2 = extract_mer(p2, pose_N_term[1], p2.size() - extended_pose_C_term[1])
        p1.dump_pdb('{}_capped_a.pdb'.format(base_name))
        p2.dump_pdb('{}_capped_b.pdb'.format(base_name))
    elif extended_pose_N_term_ddg < pose_N_term_ddg and extended_pose_C_term_ddg > pose_C_term_ddg:
        #output pose with extended N_term but trimmed C_term
        p1 = create_extended_pose(pose, both=False, N_only=True, C_only=False, trim_repeat=False)
        p1 = extract_mer(p1, extended_pose_N_term[1], p1.size()-pose_C_term[1])
        p1.dump_pdb('{}_capped.pdb'.format(base_name))
    elif extended_pose_N_term_ddg > pose_N_term_ddg and extended_pose_C_term_ddg < pose_C_term_ddg:
        #output pose with trimmed N_term but extended C_term
        p1 = create_extended_pose(pose, both=False, N_only=False, C_only=True, trim_repeat=False)
        p1 = extract_mer(p1, pose_N_term[1], p1.size() - extended_pose_C_term[1])
        p1.dump_pdb('{}_capped.pdb'.format(base_name))
    elif extended_pose_N_term_ddg > pose_N_term_ddg and extended_pose_C_term_ddg > pose_C_term_ddg:
        #output original pose
        p1 = pose
        p1 = extract_mer(p1, pose_N_term[1], p1.size()-pose_C_term[1])
        p1.dump_pdb('{}_capped.pdb'.format(base_name))

"""
p1 = create_extended_pose(pose, both=False, N_only=True, C_only=False, trim_repeat=True)
p1 = extract_mer(p1, extended_pose_N_term[1], p1.size()-pose_C_term[1])
p2 = create_extended_pose(pose, both=False, N_only=False, C_only=True, trim_repeat=True)
p2 = extract_mer(p2, pose_N_term[1], p2.size() - extended_pose_C_term[1])
p1.dump_pdb('test1.pdb')
p2.dump_pdb('test2.pdb')
p1 = create_extended_pose(pose, both=False, N_only=True, C_only=False, trim_repeat=False)
p1 = extract_mer(p1, extended_pose_N_term[1], p1.size()-pose_C_term[1])
p1.dump_pdb('test3.pdb')
p1 = create_extended_pose(pose, both=False, N_only=False, C_only=True, trim_repeat=False)
p1 = extract_mer(p1, pose_N_term[1], p1.size() - extended_pose_C_term[1])
p1.dump_pdb('test4.pdb')
p1 = pose
p1 = extract_mer(p1, pose_N_term[1], p1.size()-pose_C_term[1])
p1.dump_pdb('test5.pdb')
pose.dump_pdb('pose.pdb')
"""

#what the hell would this be useful for
"""
getting crystal structures of repeat proteins that have floppy termini
functionalizing the termini for binding
designing homodimers where termini form interface
generally removing floppy tails when functionalizing a different area
improved forward folding prediction
"""
#alternative use
""" 
Could be modified to output multiple variants that differ only in the cap
This would change outcomes of downstream design ie homodimers, one sided binders
Simple method to tune size of derroid pores
"""

#things to do
"""
add pack and min to ddg calculation
"""

"""
def current_cap_scores(pose):
    scorefxn = pyrosetta.get_fa_scorefxn()
    extended_pose = create_extended_pose(pose)
    
    #get all the pose variant pieces
    pose_without_N_term_helix, pose_without_C_term_helix, pose_N_term, pose_C_term = get_pose_variants(pose)
       
    
    #calculate the ddg
    pose_N_term_ddg = calculate_ddg_per_residue(pose_without_N_term_helix, pose_N_term)
    pose_C_term_ddg = calculate_ddg_per_residue(pose_without_C_term_helix, pose_C_term)
    
    #do some calculations
    if pose_N_term_ddg < pose_C_term_ddg:
        best_cap_score = pose_N_term_ddg
        worst_cap_score = pose_C_term_ddg
    else:
        best_cap_score = pose_C_term_ddg
        worst_cap_score = pose_N_term_ddg
    average_cap_score = (float(pose_C_term_ddg) + float(pose_N_term_ddg))/2
    sum_cap_scores = float(pose_C_term_ddg) + float(pose_N_term_ddg)
    
    return best_cap_score, worst_cap_score, average_cap_score, sum_cap_scores


pdb_folder = sys.argv[1]
print('searching for',pdb_folder)
pdb_files = glob.glob('{}/*pdb'.format(pdb_folder))
print('found',pdb_files)
for pdb in pdb_files:
    base_name = os.path.basename(pdb)
    with open('{}.capscore'.format(base_name), 'w') as fout:
       pose = pyrosetta.pose_from_file(pdb)
       best_cap_score, worst_cap_score, average_cap_score, sum_cap_scores = current_cap_scores(pose)
       out_list = [str(base_name), str(best_cap_score), str(worst_cap_score), str(average_cap_score), str(sum_cap_scores)]
       fout.write('description best_cap_score worst_cap_score average_cap_score sum_cap_scores')
       fout.write('\n') 
       fout.write(" ".join(out_list))
       fout.write('\n')

"""
