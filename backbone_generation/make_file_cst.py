#!/usr/bin/env python
from argparse import ArgumentParser
import os
import glob
from multiprocessing import Pool
parser  = ArgumentParser()
import sys
from shutil import copy

# Max omega set to 0.1
# Deleted: max radius
# Deleted: l2
# Restored: lattice_space and cst_tolerance from create_design()
# Restored: atom_pair_constraints from flags and weights
# Modified in python script: samples helices that are of equal length and those within 3 residues in length distant from each other
# Modified :  weights file all rpce is 1.

def create_design(name,directory_name,ss_elements,lattice_space,cst_tolerance):
    overlap_fragment_sampling = 3
    path_name = name
    if directory_name != "":
        path_name = "{}/{}".format(directory_name,name)
    if not os.path.exists(path_name):
        os.makedirs(path_name)
    copy_necessary_files(path_name)
    create_blueprint(path_name,ss_elements)
    create_csts(path_name,ss_elements,lattice_space,cst_tolerance)
    add_flags(path_name,name)
    fl = open(path_name+"/design.xml","w")
    print("""<ROSETTASCRIPTS>
<TASKOPERATIONS>
</TASKOPERATIONS>
<SCOREFXNS>
    <ScoreFunction name="sfn_centroid" weights="abinitio_remodel_cen.wts"/>
    <ScoreFunction name="sfn_motif" weights="empty">
        <Reweight scoretype="cen_pair_motifs" weight="1"/>
    </ScoreFunction>
    <ScoreFunction name="sfn_motif_degree" weights="empty">
        <Reweight scoretype="cen_pair_motif_degree" weight="1"/>
    </ScoreFunction>
    <ScoreFunction name="VDW" weights="empty">
        <Reweight scoretype="vdw" weight="1"/>
    </ScoreFunction>
</SCOREFXNS>
<FILTERS>
    <worst9mer name="worst9mer_h" threshold="0.15" only_helices="true"/>
    <ScoreType name="VDW" scorefxn="VDW" threshold="100" confidence="1" />
    <ScoreType name="motif_score" scorefxn="sfn_motif" threshold="-1" confidence="1" />
    <ScoreType name="motif_degree_score" scorefxn="sfn_motif_degree" threshold="-0.3" confidence="1" />
    <SSDegree name="ss_degree_avg" report_avg="true" ignore_terminal_ss="2"/>
    <SSDegree name="ss_degree_worst" report_avg="false" ignore_terminal_ss="2"/>
    <RepeatParameter name="radius" param_type="radius" numb_repeats="4" min="0" max="99999999" confidence="1"/> #replace with correct number of repeats
    <RepeatParameter name="rise" param_type="rise" numb_repeats="4" min="0" max="99999999" confidence="1"/>
    <RepeatParameter name="omega" param_type="omega" numb_repeats="4" min="0" max="99999999" confidence="1"/>
</FILTERS>
<MOVERS>
<RemodelMover name="remodel_mover" blueprint="design.blueprint"/>
</MOVERS>
<PROTOCOLS>
    <Add mover_name="remodel_mover"/>
    <Add filter_name="VDW"/>
    <Add filter_name="worst9mer_h"/>
    <Add filter_name="motif_score"/>
    <Add filter_name="motif_degree_score"/>
    <Add filter_name="ss_degree_worst"/>
    <Add filter_name="radius"/>
    <Add filter_name="rise"/>
    <Add filter_name="omega"/>
</PROTOCOLS>
<OUTPUT scorefxn="sfn_centroid"/>
</ROSETTASCRIPTS>""", file=fl)
    fl.close()

def copy_necessary_files(name):
    os.system("cp rd1_files/flags_cst " + name + "/flags")
    os.system("cp rd1_files/abinitio_remodel_cen_stage0a.wts "+ name +"/abinitio_remodel_cen_stage0a.wts")
    os.system("cp rd1_files/abinitio_remodel_cen_stage0b.wts "+ name +"/abinitio_remodel_cen_stage0b.wts")
    os.system("cp rd1_files/abinitio_remodel_cen_stage1.wts "+ name +"/abinitio_remodel_cen_stage1.wts")
    os.system("cp rd1_files/abinitio_remodel_cen_stage2.wts "+ name +"/abinitio_remodel_cen_stage2.wts")
    os.system("cp rd1_files/abinitio_remodel_cen.wts "+ name +"/abinitio_remodel_cen.wts")
    os.system("cp rd1_files/cmd "+ name + "/cmd")
    os.system("cp rd1_files/start.pdb " + name + "/start.pdb")


def add_flags(path_name,name):
    fl = open(path_name+"/flags","a")
    fl.write("-score:motif_residues ")
    design_length = get_design_length(name)
    #print("design_length:{}",design_length)
    for ii in range(design_length,design_length*2):
        fl.write("{},".format(ii))
    fl.write("{}".format(design_length*2))
    fl.close

def get_design_length(name):
    name_wo_filename = os.path.splitext(name)[0]
    component_list = name_wo_filename.split("_")
    length = 0
    for ii in range(1,(len(component_list)-1)):
        tmpItem = component_list[ii].replace('h','')
        tmpItem2 = tmpItem.replace('l','')
        if(tmpItem2[0]!="t"):
            length +=(int(tmpItem2))
    return(length)

def create_csts(name,ss_elements,lattice_space,cst_tolerance):
	ss_lengths = [ int(element[1:]) for element in ss_elements ]
	# for i, element in enumerate(ss_elements):
	#    for x in range(ss_lengths[i]):
	# 	  ss_list.append(ss_elements[0][0])
	rep_len = sum(ss_lengths)
	cst_template = 'AtomPair CA %s CA %s HARMONIC %s %s'
	c = str('%.2f' % lattice_space).rjust(4)
	d = str('%.2f' % cst_tolerance).rjust(2)
	with open(name+'/lattice_csts.cst', 'w') as fout:
		for rep_unit in [0]:
			for i in range(1,ss_lengths[0]+1):
				# if ss_list[i-1] == 'h':
				a = str(int(i+rep_unit*rep_len)).rjust(2)
				b = str(int(rep_len+i+rep_unit*rep_len)).rjust(3)
				line = cst_template%(a,b,c,d)
				fout.write(line+'\n')
			for i in range(ss_lengths[0]+ss_lengths[1]+1, ss_lengths[0]+ss_lengths[1]+ss_lengths[2]+1):
				# if ss_list[i-1] == 'h':
				a = str(int(i+rep_unit*rep_len)).rjust(2)
				b = str(int(rep_len+i+rep_unit*rep_len)).rjust(3)
				line = cst_template%(a,b,c,d)
				fout.write(line+'\n')

def create_blueprint(name,ss_elements):
    fl = open(name+"/design.blueprint","w")
    first = True
    for ssElement in ss_elements:
        ssLength = int(ssElement[1:])
        ssType = ssElement[0]
        if(ssType == "h"):
            tmpType = "HA"
        if(ssType == "l"):
            tmpType = "LD"
        if(ssType == "e"):
            tmpType = "ED"
        if(first):
            print("{} {} {}".format(1,"A",tmpType), file=fl)
            first = False
            for ii in range(1,ssLength):
                print("{} {} {}".format(0,"x",tmpType), file=fl)
        else:
            for ii in range(0,ssLength):
                print("{} {} {}".format(0,"x",tmpType), file=fl)
    fl.close()

ct = 0
max_files_per_directory = 2000
directory_name_int = 0 
num_dirs = 1
lattice_space = 10.9 
cst_tolerance = 0.5
#h1s = ['h16','h17','h18','h19','h20','h21','h22','h23','h24','h25','h26','h27','h28','h29','h30']
#l1s = ['l2','l3','l4']
l1s = ['l3']
#h2s = ['h16','h17','h18','h19','h20','h21','h22','h23','h24','h25','h26','h27','h28','h29','h30']
#l2s = ['l2','l3','l4']
l2s = ['l3']

#h1s = list(range(10, 30))
h1s = list(range(20, 21))
#### modified how many h1s, l1s, h2s, l2s to try for testing simplicity

for d in range(1,2):
    for h1 in h1s:
        h1_size = h1
        for l1 in l1s:
            #h2s = list(range(10,30))
            #h2s = list(range(h1-2,h1+3))
            h2s = list(range(20,21))
            for h2 in h2s:
                h2_size = h2
                h_diff = abs(h1_size - h2_size)
                if h_diff <= 9999:
                    for l2 in l2s:
                        for idir in range(num_dirs):
                            name = "X_{}_{}_{}_{}_c{}".format('h'+str(h1),l1,'h'+str(h2),l2,d)
                            ss_elements = ['h'+str(h1),l1,'h'+str(h2),l2]
                            directory_name = "run_{}_{}".format(idir,directory_name_int)
                            print(("working on {}".format(name)))
                            if(ct > max_files_per_directory * num_dirs):
                                directory_name_int+=1
                                ct = 0
                            ct+=1
                            create_design(name,directory_name,ss_elements,lattice_space,cst_tolerance)
                else:
                    continue
