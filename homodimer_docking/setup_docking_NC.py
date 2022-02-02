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
from rosetta import *

def get_seq(pose):
    seq = pose.sequence()
    return(seq)

def get_dssp(pose):
    dssp = core.scoring.dssp.Dssp(pose)
    dssp.insert_ss_into_pose(pose)
    sec = pose.secstruct()
    return(sec)

def get_ss_lengths(pose):
    repeat_unit_dssp = pyrosetta.rosetta.core.scoring.dssp.Dssp(pose)
    repeat_unit_ss = repeat_unit_dssp.get_dssp_secstruct()
    from itertools import groupby
    groups = groupby(repeat_unit_ss)
    ss_lengths = [(label, sum(1 for _ in group)) for label, group in groups]
    if ss_lengths[0][0] == 'L':
        tmp_ss_lengths = ss_lengths[2:]
        tmp_ss_lengths.insert(0,['H' , ss_lengths[0][1]+ss_lengths[1][1]])
        ss_lengths = tmp_ss_lengths
    if ss_lengths[-1][0] == 'L':
        tmp_ss_lengths = ss_lengths[0:-2]
        tmp_ss_lengths.append(['H' , ss_lengths[-1][1]+ss_lengths[-2][1]])
        ss_lengths = tmp_ss_lengths
    return ss_lengths

def del_NandC_loops(pdb_fn):
    pose = pose_from_pdb(pdb_fn)
    seq = get_seq(pose)
    dssp = get_dssp(pose)
    pdb =  os.path.basename(pdb_fn).split(".")[0]

    in_loop = 1
    N_term_loop = []
    for i in range(len(seq)):    
        if dssp[i] == 'L' and in_loop == 0:
            break
        elif dssp[i] == 'L' and in_loop == 1:
            N_term_loop.append(i+1)
        elif dssp[i] == 'H' and in_loop == 0:
            break
        elif dssp[i] == 'H' and in_loop == 1:
            in_loop = 0
            break
        else:
            print('problem')

    in_loop = 1
    C_term_loop = []
    for i in reversed(range(len(seq))):    
        if dssp[i] == 'L' and in_loop == 0:
            break
        elif dssp[i] == 'L' and in_loop == 1:
            C_term_loop.append(i+1)
        elif dssp[i] == 'H' and in_loop == 0:
            break
        elif dssp[i] == 'H' and in_loop == 1:
            break
            in_loop = 0
        else:
            print('problem')
    
    C_term_loop = list(reversed(C_term_loop))    
    print('trimming loop residues ',N_term_loop,C_term_loop,'for ',pdb) 
    pose.delete_residue_range_slow(int(C_term_loop[0]),int(C_term_loop[-1]))
    pose.delete_residue_range_slow(int(N_term_loop[0]),int(N_term_loop[-1]))
    pose.dump_file(pdb+'/'+pdb+'.pdb')
   
    ss_lengths = get_ss_lengths(pose)
    print(ss_lengths)

    with open(pdb+'/pos1.file', 'w') as fout:
        middle_residue = int(ss_lengths[0][1]/2)
        positions = []
        for i in range(middle_residue-2,middle_residue+3):
            positions.append(i)
        fout.write(' '.join(map(str,positions)))
        fout.write('\n')

    with open(pdb+'/pos2.file', 'w') as fout:
        middle_residue = int(pose.size()-(ss_lengths[-1][1]/2))
        positions = []
        for i in range(middle_residue-2,middle_residue+3):
            positions.append(i)
        fout.write(' '.join(map(str,positions)))
        fout.write('\n')

def get_pdb_length(pdb_fn):
    tmp_pose = pose_from_file(pdb_fn)
    length = tmp_pose.size()
    return(length)

def setup_dir(pdb_fn):
    pdb =  os.path.basename(pdb_fn).split(".")[0]
    if  os.path.isdir(pdb):
        shutil.rmtree(pdb)
    os.makedirs(pdb)
    del_NandC_loops(pdb_fn)
    #os.system("cp {} {}".format(pdb_fn,pdb))
    os.system("cp input/* {}".format(pdb))
    fl = open(pdb+".run", "w")
    fl.write('#!/bin/bash\n')
    fl.write('#SBATCH -p medium\n')
    fl.write('#SBATCH -n 1\n')
    fl.write('#SBATCH --mem=10G\n')
    fl.write('cd %s\n' % pdb)
    fl.write('j=$(cat pos1.file) ; sed -i "s/XXXX/$j/" dock.flags\n')
    fl.write('j=$(cat pos2.file) ; sed -i "s/YYYY/$j/" dock.flags\n')
    fl.write('bash digs_command_c2.sh\n')
    print("{} done".format(pdb))

init()
pdbs = glob.glob("/home/drhicks1/capped_pdbs/PDBS/forward_fold/*pdb")
myPool    = Pool( processes = 15)
myResults = myPool.map(setup_dir,pdbs)
