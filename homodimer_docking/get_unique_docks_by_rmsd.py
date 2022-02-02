#!/usr/bin/env python
from argparse import ArgumentParser
import os
import os.path
import glob
from multiprocessing import Pool
import shutil
parser  = ArgumentParser()
import sys
from shutil import copy
import subprocess
from pyrosetta import *
from pyrosetta.rosetta import *
init("-packing:prevent_repacking")

_proc_status = '/proc/%d/status' % os.getpid()

_scale = {'kB': 1024.0, 'mB': 1024.0*1024.0,
          'KB': 1024.0, 'MB': 1024.0*1024.0}


def _VmB(VmKey):
    '''Private.
    '''
    global _proc_status, _scale
     # get pseudo file  /proc/<pid>/status
    try:
        t = open(_proc_status)
        v = t.read()
        t.close()
    except:
        return 0.0  # non-Linux?
     # get VmKey line e.g. 'VmRSS:  9999  kB\n ...'
    i = v.index(VmKey)
    v = v[i:].split(None, 3)  # whitespace
    if len(v) < 3:
        return 0.0  # invalid format?
     # convert Vm value to bytes
    #print('VmB =',float(v[1]) * _scale[v[2]])
    return float(v[1]) * _scale[v[2]]


def memory(since=0.0):
    '''Return memory usage in bytes.
    '''
    print('memory usage in GB =', (_VmB('VmSize:') - since) / (1024.0*1024.0*1024.0))
    return _VmB('VmSize:') - since


def resident(since=0.0):
    '''Return resident memory usage in bytes.
    '''
    print('Return resident memory usage in GB =', (_VmB('VmRSS:') - since) / (1024.0*1024.0*1024.0))
    return _VmB('VmRSS:') - since


def stacksize(since=0.0):
    '''Return stack size in bytes.
    '''
    print('stack size in MB =', (_VmB('VmStk:') - since) / (1024.0*1024.0))
    return _VmB('VmStk:') - since

# get just the pdb name, not directory junk
def get_pdb_names(pdbs):
    pdbs_names = []
    for i in pdbs:
        split_i = i.split('/')
        pdbs_names.append(split_i[-1])
    return(pdbs_names)

#load all the poses
def load_all_poses(pdbs):
    pose_v = []
    for i in range(len(pdbs)):
        pose_v.append(pose_from_file(pdbs[i]))
    return(pose_v)

#delete dummy residues = chain A and chain C
def del_dummy_residues(pose_v):
    for pose in pose_v:
        chains = utility.vector1_unsigned_long()
        chains.append(3)
        chains.append(1)
        del_chain = protocols.simple_moves.DeleteChainsMover(chains)
        del_chain.apply(pose)

def group_by_rg(pdbs, pose_v):
    #group by radius of gyration
    RadG = ScoreFunction()
    RadG.set_weight(core.scoring.rg , 1)
    rg_v = [1]*len(pdbs)
    for i in range(len(pdbs)):
        tmp_rg = RadG(pose_v[i])
        rg_v[i] = "%.5f" % tmp_rg
    
    rg_dic = {}
    rmsd_bool_dict = {}
    for i in range(len(rg_v)):
        tmp_v = []
        rmsd_bool_dict[i] = 1
        if rg_v[i] in rg_dic:
            rg_dic.setdefault(rg_v[i], []).append(i)
        else:
            tmp_v = [i]
            rg_dic[rg_v[i]] = tmp_v
    return([rg_dic,rmsd_bool_dict])

def find_unique_docks(rg_dic,pose_v,rmsd_bool_dict):
    # find the unique docks
    # first just compare neighbors within the same rg containier
    # this should remove most/all identicle docks but avoid O(n^2)
    for key in rg_dic:
        for i in range(len(rg_dic[key]) -1):
            if core.scoring.CA_rmsd(pose_v[rg_dic[key][i]], pose_v[rg_dic[key][i+1]]) < 0.05:
                rmsd_bool_dict[rg_dic[key][i+1]] = 0
    
    # for only the remaining unique docks
    # perform all by all comparison to ensure no duplicates 
    # should run in O(n^2) which is slow for large n
    # but should be ok if the previous step worked
    unique_keys = []
    for key in rmsd_bool_dict:
        if rmsd_bool_dict[key] == 1:
            unique_keys.append(key)
    for i in range(len(unique_keys)):
        for j in range(i+1, len(unique_keys)):
            if core.scoring.CA_rmsd(pose_v[unique_keys[i]], pose_v[unique_keys[j]]) < 0.05:
               rmsd_bool_dict[unique_keys[j]] = 0
    unique_keys = []
    for key in rmsd_bool_dict:
        if rmsd_bool_dict[key] == 1:
            unique_keys.append(key)
    return(unique_keys)

#dump the unique docks
def dump_unique_docks(unique_keys,pose_v,pdbs_names,ref_pose):
    for i in unique_keys:
            ref_pose_copy_1 = ref_pose.clone()
            ref_pose_copy_2 = ref_pose.clone()
            tmp_pose = pose_v[i]
            split_pose = tmp_pose.split_by_chain()
            core.scoring.calpha_superimpose_pose(ref_pose_copy_1, split_pose[1])
            core.scoring.calpha_superimpose_pose(ref_pose_copy_2, split_pose[2])
            new_pose = ref_pose_copy_1
            new_pose.append_pose_by_jump(ref_pose_copy_2, 1)
            new_pose.dump_pdb('unique_docks/%s' % pdbs_names[i])

#find unique docks in each docking directory
def run_program(my_dir):
    ref_pdb = glob.glob(my_dir+"/26*pdb")
    pdbs = glob.glob(my_dir+"/output/26H/C2/26*pdb")
    pdbs_names = get_pdb_names(pdbs)
    ref_pose = pose_from_file(ref_pdb[0])
    print(ref_pose) 
    print('loading',len(pdbs),'poses into memory')
    pose_v = load_all_poses(pdbs)
    print('deleting the dummy residues')
    #del_dummy_residues(pose_v)
    print('grouping by radius of gyration')
    rg_dic,rmsd_bool_dict = group_by_rg(pdbs,pose_v)
    print('finding unique docks')
    unique_keys = find_unique_docks(rg_dic,pose_v,rmsd_bool_dict)
    print('saving',len(unique_keys),'pdbs')
    dump_unique_docks(unique_keys,pose_v,pdbs_names,ref_pose)


directories = glob.glob("26*rd2")
myPool    = Pool( processes = 1)
myResults = myPool.map(run_program,directories)

