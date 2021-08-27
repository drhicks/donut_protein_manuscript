#!/usr/bin/env python
from argparse import ArgumentParser
import os
import glob
from multiprocessing import Pool
parser  = ArgumentParser()
import sys
from shutil import copy


def create_design(name,directory_name,ss_elements):
    overlap_fragment_sampling = 3
    path_name = name
    if directory_name != "":
        path_name = "{}/{}".format(directory_name,name)
    if not os.path.exists(path_name):
        os.makedirs(path_name)
    copy_necessary_files(path_name)
    create_blueprint(path_name,ss_elements)
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
    <ScoreType name="VDW" scorefxn="VDW" threshold="100" confidence="1" />
    <worst9mer name="worst9mer" threshold="0.15" only_helices="true" confidence="1" />
    <ScoreType name="motif_score" scorefxn="sfn_motif" threshold="-1.0" confidence="1" />
    <ScoreType name="motif_degree_score" scorefxn="sfn_motif_degree" threshold="-0.3" confidence="1" />
    <SSDegree name="ss_degree_avg" report_avg="true" ignore_terminal_ss="2"/>
    <SSDegree name="ss_degree_worst" report_avg="false" threshold="3" ignore_terminal_ss="2"/>
    <RepeatParameter name="radius" param_type="radius" numb_repeats="4" min="0" max="99999999" confidence="1"/> #replace with correct number of repeats
    <RepeatParameter name="rise" param_type="rise" numb_repeats="4" min="0" max="99999999" confidence="1"/>
    <RepeatParameter name="twist" param_type="omega" numb_repeats="4" min="0" max="99999999" confidence="1"/>
</FILTERS>
<MOVERS>
<RemodelMover name="remodel_mover" blueprint="design.blueprint"/>
</MOVERS>
<PROTOCOLS>
    <Add mover_name="remodel_mover"/>
    <Add filter_name="VDW"/>
    <Add filter_name="worst9mer"/>
    <Add filter_name="motif_score"/>
    <Add filter_name="motif_degree_score"/>
    <Add filter_name="ss_degree_worst"/>
    <Add filter_name="radius"/>
    <Add filter_name="rise"/>
    <Add filter_name="twist"/>
</PROTOCOLS>
<OUTPUT scorefxn="sfn_centroid"/>
</ROSETTASCRIPTS>""", file=fl)
    fl.close()

def copy_necessary_files(name):
    os.system("cp rd1_files/flags " + name + "/flags")
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
    for ii in range(1,(len(component_list))):
        tmpItem = component_list[ii].replace('h','')
        tmpItem2 = tmpItem.replace('l','')
        if(tmpItem2[0]!="t"):
            length +=(int(tmpItem2))
    return(length)

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
#h1s = ['h12','h18','h24']
h1s = ['h12','h13','h14','h15','h16','h17','h18','h19','h20','h21','h22','h23','h24','h25','h26','h27','h28','h29','h30']
l1s = ['l2','l3','l4']
#h2s = ['h12','h18','h24']
h2s = ['h12','h13','h14','h15','h16','h17','h18','h19','h20','h21','h22','h23','h24','h25','h26','h27','h28','h29','h30']
l2s = ['l2','l3','l4']

### only making a single one for testing...  remove this to try all combos
#h1s = ['h24']
#l1s = ['l3']
#h2s = ['h20']
#l2s = ['l3']

for h1 in h1s:
    for l1 in l1s:
        for h2 in h2s:
            for l2 in l2s:
                for idir in range(num_dirs):
                    name = "X_{}_{}_{}_{}".format(h1,l1,h2,l2,)
                    ss_elements = [h1,l1,h2,l2]
                    directory_name = "run_{}_{}".format(idir,directory_name_int)
                    print(("working on {}".format(name)))
                    if(ct > max_files_per_directory * num_dirs):
                        directory_name_int+=1
                        ct = 0
                    ct+=1
                    create_design(name,directory_name,ss_elements)
