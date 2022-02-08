import os
import re
import glob
import pyrosetta as pyrosetta
import rosetta as rosetta
import numpy as np
import math
from pyrosetta import toolbox
pyrosetta.init()

def get_sec_structure(in_pose):
    pose_sec_struct = pyrosetta.rosetta.core.scoring.dssp.Dssp(in_pose).get_dssp_secstruct()
    print(pose_sec_struct)
    return(pose_sec_struct)

def get_loop_residues(in_pose, pose_sec_struct):
    loop_residues = []
    tmp_loop_stretch = []
    residue_index = 0
    for i in pose_sec_struct:
        residue_index += 1
        if i == 'L':
            tmp_loop_stretch.append(residue_index)
        else:
            if len(tmp_loop_stretch) > 0:
                loop_residues.append(tmp_loop_stretch)
                tmp_loop_stretch = []
            else:
                tmp_loop_stretch = []
    return(loop_residues)

def poses_from_files(files):
    for file_ in files:
        yield file_, pyrosetta.pose_from_file(file_)

with open('replace_loops.cmds', 'w') as fout:
	pdb_files = glob.glob('/home/drhicks1/500_curved_repeats/*pdb')
	for i, (pdb_name, pose) in enumerate(poses_from_files(pdb_files)):
		loopresidues = get_loop_residues(pose, get_sec_structure(pose))
		fout.write('/software/rosetta/latest/bin/rosetta_scripts.hdf5.linuxgccrelease -parser:protocol fixallloops.xml -indexed_structure_store:fragment_store /home/drhicks1/DB/hdf5/ss_grouped_vall_helix_shortLoop.h5 -s {} -parser:script_vars repeat_length={} length={}\n'.format(pdb_name, int(pose.size()/4), pose.size()))
