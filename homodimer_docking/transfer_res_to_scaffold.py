import glob
import sys
import os
import random
import numpy as np
from multiprocessing import Pool
#sys.path.append('/software/pyrosetta3/latest/setup/')
import pyrosetta
pyrosetta.init("-prevent_repacking")

def do_the_transfer(to_pdb, from_pdb):
    #if os.path.isfile(os.path.basename(from_pdb)):
    #    return 1
    try:
        print("hello")
    except:
        print("no")
    if 1:
        from_pose1 = pyrosetta.pose_from_file(from_pdb).split_by_chain()[1]
        from_pose2 = pyrosetta.pose_from_file(from_pdb).split_by_chain()[1]
        to_pose1 = pyrosetta.pose_from_file(to_pdb).split_by_chain()[1]
        to_pose2 = pyrosetta.pose_from_file(to_pdb).split_by_chain()[2]
        assert(from_pose1.size() == to_pose1.size())
        assert(from_pose2.size() == to_pose2.size())

        seqposs = pyrosetta.rosetta.utility.vector1_unsigned_long()
        for i in range(1, from_pose1.size() + 1):
            seqposs.append(i)


        #pyrosetta.rosetta.protocols.toolbox.pose_manipulation.superimpose_pose_on_subset_CA( to_pose1, from_pose1, seqposs, 0)

        for i in range(1, from_pose1.size() + 1):
            res = from_pose1.residue(i)
            to_pose1.replace_residue(i, res, True)
            
        #pyrosetta.rosetta.protocols.toolbox.pose_manipulation.superimpose_pose_on_subset_CA( to_pose2 ,from_pose2, seqposs, 0)

        for i in range(1, from_pose2.size() + 1):
            res = from_pose2.residue(i)
            to_pose2.replace_residue(i, res, True)
        
        name = os.path.basename(to_pdb)
        to_pose1.append_pose_by_jump(to_pose2, to_pose1.size())
        to_pose1.dump_pdb(name)

    #except:
    #    print('something went wrong')

def find_docks(monomer):
    print(monomer)
    monomer_basename = os.path.basename(monomer)[0:-4]
    sicdocks = glob.glob('/home/drhicks1/run_sicdoc/unique_docks_[A-Z][A-Z]/{}*pdb'.format(monomer_basename))
    print(sicdocks)
    pdb_pairs = []
    for dock in sicdocks:
        pdb_pairs.append([dock,monomer])
    myPool = Pool( processes = 1)
    myResults = myPool.starmap(do_the_transfer,pdb_pairs)

monomer = sys.argv[1]
find_docks(monomer)
