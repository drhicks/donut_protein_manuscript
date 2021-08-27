import os
import re
import glob
import subprocess
import sys

all_files = glob.glob('/home/drhicks1/36K_curved_repeats_enumerate_ideal_loops/passing_0_4rmsd/FROM*/*/*pdb')
print('number of files found =', len(all_files))

#sys.exit()
#print('globbed all loop files')
#for pdb in all_files:
#ref_name = os.path.basename(pdb)
#sbatch_file = str(i)+"_"+str(j)+".sh"
#os.makedirs(str(i)+"_"+str(j))
#with open(sbatch_file, 'w') as fout:
#    fout.write('#!/bin/bash \n#SBATCH -p medium \n#SBATCH -n 1 \n#SBATCH --mem=4g\n')
#    fout.write('cd {} \n'.format(str(i)+"_"+str(j)))
#    fout.write('/home/brunette/src/Rosetta_master/main/source/bin/rosetta_scripts.hdf5.linuxgccrelease -l ../{}_{}.list -parser:protocol ../wost9mer.xml -indexed_structure_store:fragment_store /home/brunette/DBs/hdf5/ss_grouped_vall_helix_shortLoop.h5'.format(i,j)) 
#bashCommand = 'sbatch {}'.format(sbatch_file)
#process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
#output, error = process.communicate()

finished_files = []
with open('done.list', 'r') as fin:
    for i in fin.readlines():
        finished_files.append(i.strip())
tmp_dict = {}
for i in all_files:
    tmp_dict[i] = i

for i in finished_files:
        #print(i)
    if i in tmp_dict:
        print('removing',i)
        tmp_dict.pop(i)

out_list_all = []
for key in tmp_dict:
    out_list_all.append(tmp_dict[key])

for i in range(12,31):
    for j in range(2,5):
        r = re.compile('.*X_h{}_l{}.*pdb'.format(i,j))
        newlist = list(filter(r.match, out_list_all))
        if not newlist:
            continue
        print(len(newlist))
        for group in range(1,int(len(newlist)/8000)+2):
            tmp_list = newlist[((group-1)*8000):((group-1)*8000)+8000]
            #print('len of list =',len(tmp_list))
            with open('{}_{}_{}.list'.format(i,j,group), 'w') as listout:
                listout.write('\n'.join(tmp_list))
                listout.write('\n')
            sbatch_file = '{}_{}_{}.sh'.format(i,j,group)
            if not os.path.exists(str(i)+"_"+str(j)):
                os.makedirs(str(i)+"_"+str(j))
            with open(sbatch_file, 'w') as fout:
                fout.write('#!/bin/bash \n#SBATCH -p short \n#SBATCH -n 1 \n#SBATCH --mem=4g\n')
                fout.write('cd {} ; '.format(str(i)+"_"+str(j)))
                fout.write('/home/drhicks1/Rosetta_master/Rosetta/main/source/bin/rosetta_scripts.default.linuxgccrelease -l ../{}_{}_{}.list -parser:protocol /home/drhicks1/36K_curved_repeats_enumerate_ideal_loops/passing_0_4rmsd/test_geometry.xml -no_nstruct_label \n'.format(i,j,group)) 
        #bashCommand = 'sbatch {}'.format(sbatch_file)
        #process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        #output, error = process.communicate()

