import glob
import os
import re
import sys
import numpy as np
#sys.path.append('/software/pyrosetta3/latest/setup/')
import nglview
import pyrosetta
import pyrosetta.bindings.utility
pyrosetta.init(options='-symmetry_definition stoopid -old_sym_min true -out:mute protocols.idealize -out:mute core.pack', extra_options='')


def poses_from_files(files):
    for file_ in files:
        yield os.path.basename(file_)[:-4], pyrosetta.pose_from_file(file_)


# In[ ]:





# In[4]:


# fills in a SymmetryInfo object with the necessary info
def setup_repeat_symminfo(repeatlen, symminfo, nrepeat, base_repeat):
    #print('doing setup_repeat_symminfo')
    base_offset = (base_repeat-1)*repeatlen
    i=1
    while i <= nrepeat: 
        if i == base_repeat: 
            i=i+1
            continue
        offset = (i-1)*repeatlen
        j=1
        while j <= repeatlen:
            symminfo.add_bb_clone(base_offset+j, offset+j )
            symminfo.add_chi_clone( base_offset+j, offset+j )
            j=j+1
        i=i+1


    symminfo.num_virtuals( 1 ) # the one at the end...
    symminfo.set_use_symmetry( True )
    symminfo.set_flat_score_multiply( nrepeat*repeatlen+1, 1 )
    symminfo.torsion_changes_move_other_monomers( True ) # note -- this signals that we are folding between repeats

    ### what is the reason to do this???
    ### If there is a good reason, why not do for repeats after base_repeat???
    ### 
    # repeats prior to base_repeat have score_multiply ==> 0
    """
    i=1
    while i < base_repeat:
        offset = (i-1)*repeatlen
        j=1
        while j <= repeatlen:
            symminfo.set_score_multiply( offset+j, 0 )
            j=j+1
        i=i+1
    """
    symminfo.update_score_multiply_factor()
    #print('finished setup_repeat_symminfo')


# In[ ]:





# In[5]:


# sets up a repeat pose, starting from a non-symmetric pdb with nres=repeatlen*nrepeat
def setup_repeat_pose(pose, numb_repeats_=4, base_repeat=3):
    #print('doing setup_repeat_pose')
    if pyrosetta.rosetta.core.pose.symmetry.is_symmetric(pose):
        return False # not to begin with...
    repeatlen = int(pose.size()/numb_repeats_)
    nrepeat = numb_repeats_
    
    if not nrepeat * repeatlen == pose.size():
        return False

    if not base_repeat > 1:
        return False
    # why? well, the base repeat should get the right context info from nbring repeats
    # but note that with base_repeat>1 we probably can't use linmem_ig and there are other places in the code that
    # assume that monomer 1 is the independent monomer. These should gradually be fixed. Let me (PB) know if you run into
    # trouble.

    nres_protein = nrepeat * repeatlen
    pyrosetta.rosetta.core.pose.remove_upper_terminus_type_from_pose_residue( pose, pose.size() )
    vrtrsd = pyrosetta.rosetta.core.conformation.Residue(pyrosetta.rosetta.core.conformation.ResidueFactory.create_residue
           (pyrosetta.rosetta.core.pose.get_restype_for_pose(pose, "VRTBB" )))
    pose.append_residue_by_bond( vrtrsd, True ) # since is polymer...
    #pose.append_residue_by_jump( vrtrsd, True )
    pose.conformation().insert_chain_ending( nres_protein )
    f = pyrosetta.rosetta.core.kinematics.FoldTree( pose.size() )
    f.reorder( pose.size() )
    pose.fold_tree( f )
    symminfo = pyrosetta.rosetta.core.conformation.symmetry.SymmetryInfo()
    setup_repeat_symminfo( repeatlen, symminfo, nrepeat, base_repeat )

    # now make symmetric
    pyrosetta.rosetta.core.pose.symmetry.make_symmetric_pose(pose, symminfo)
    
    if not pyrosetta.rosetta.core.pose.symmetry.is_symmetric(pose):
        return False

    ### what is the purpose of this???
    ###
    ##TJ adding these to Phil's function
    base_offset = (base_repeat-1)*repeatlen
    """
    for i in range(0,nrepeat):
        j=1
        while j <= repeatlen:
            pos = j + base_offset
            pose.set_phi  ( j+i*repeatlen, pose.phi  (pos) )
            pose.set_psi  ( j+i*repeatlen, pose.psi  (pos) )
            pose.set_omega( j+i*repeatlen, pose.omega(pos) )
            j=j+1
       
    for i in range(0,nrepeat): 
        j=1
        while j <= repeatlen:
            pos = j + base_offset
            oldrsd = pyrosetta.rosetta.core.conformation.Residue( pose.residue(pos).clone() )
            pose.replace_residue( j+i*repeatlen, oldrsd, False )
            j=j+1
    """
    #print('finished setup_repeat_pose')


# In[6]:


#apply_cst = pyrosetta.rosetta.protocols.constraint_generator.AddConstraints()
#all_res = pyrosetta.rosetta.core.select.residue_selector.SecondaryStructureSelector('HLE')
#atom_pair_cst = pyrosetta.rosetta.protocols.constraint_generator.AtomPairConstraintGenerator()
#atom_pair_cst.set_weight(1.0)
#atom_pair_cst.set_max_distance(10.0)
#atom_pair_cst.set_min_seq_sep(0)
#atom_pair_cst.set_sd(1.0)
#atom_pair_cst.set_ca_only(False)
#atom_pair_cst.set_residue_selector(all_res.apply())
##atom_pair_cst.apply(pose)
#cord_cst = pyrosetta.rosetta.protocols.constraint_generator.CoordinateConstraintGenerator()
#dihedral_cst = pyrosetta.rosetta.protocols.constraint_generator.DihedralConstraintGenerator()
#distnace_cst = pyrosetta.rosetta.protocols.constraint_generator.DistanceConstraintGenerator()
#
#apply_cst.add_generator(atom_pair_cst)
#apply_cst.add_generator(cord_cst)
#apply_cst.add_generator(dihedral_cst)
#apply_cst.add_generator(distnace_cst)
##apply_cst.apply(pose)


# In[ ]:





# In[ ]:





# In[7]:


def relax_pose(pose, cartesian_=False):
    #print('doing relax_pose')
    relax_iterations_ = 1
    pdb_pose = pose.clone()
    s = pyrosetta.get_score_function()
    sf = pyrosetta.rosetta.core.scoring.symmetry.symmetrize_scorefunction(s)        
    
    sf.set_weight(pyrosetta.rosetta.core.scoring.atom_pair_constraint, 1.0)
    sf.set_weight(pyrosetta.rosetta.core.scoring.angle_constraint, 1.0)
    sf.set_weight(pyrosetta.rosetta.core.scoring.dihedral_constraint, 1.0)
    sf.set_weight(pyrosetta.rosetta.core.scoring.coordinate_constraint, 0.1)
    fastrelax = pyrosetta.rosetta.protocols.relax.FastRelax( sf , relax_iterations_ )
    #fastrelax.constrain_relax_to_start_coords(True)
    fastrelax.ramp_down_constraints(False)
    fastrelax.min_type('lbfgs_armijo_nonmonotone')
    if cartesian_:
        sf.set_weight(pyrosetta.rosetta.core.scoring.ScoreType.pro_close, 0.0) # has to be zero
        sf.set_weight(pyrosetta.rosetta.core.scoring.ScoreType.cart_bonded, 0.5) # what are good values
        sf.set_weight(pyrosetta.rosetta.core.scoring.ScoreType.cart_bonded_angle, 0.5) # what are good values
        sf.set_weight(pyrosetta.rosetta.core.scoring.ScoreType.cart_bonded_length, 0.5) # what are good values
        sf.set_weight(pyrosetta.rosetta.core.scoring.ScoreType.cart_bonded_torsion, 0.5) # what are good values
        fastrelax.cartesian(True)
        fastrelax.minimize_bond_angles(True)
        fastrelax.minimize_bond_lengths(True)
    movemap = setup_movemap(pose)
    
    #apply_cst = pyrosetta.rosetta.protocols.constraint_generator.AddConstraints()
    #all_res = pyrosetta.rosetta.core.select.residue_selector.SecondaryStructureSelector('HLE')
    #all_res.apply(pose)
    #atom_pair_cst = pyrosetta.rosetta.protocols.constraint_generator.AtomPairConstraintGenerator()
    #atom_pair_cst.set_weight(0.01)
    #atom_pair_cst.set_max_distance(100.0)
    #atom_pair_cst.set_min_seq_sep(0)
    #atom_pair_cst.set_sd(0.001)
    #atom_pair_cst.set_ca_only(False)
    #atom_pair_cst.set_residue_selector(all_res)
    #atom_pair_cst.apply(pose)
    #apply_cst.add_generator(atom_pair_cst)
    #apply_cst.apply(pose)
    
    pyrosetta.rosetta.core.pose.symmetry.make_symmetric_movemap(pose,movemap)
    #movemap.clear() #for testing purposes only, prevents relax from doing anything
    fastrelax.set_movemap( movemap )
    pyrosetta.rosetta.core.scoring.constraints.add_coordinate_constraints(pose, 0.1, False)
    fastrelax.apply( pose )
    rmsd = pyrosetta.rosetta.core.scoring.CA_rmsd(pose, pdb_pose)
    print("rmsd change from relax: ",rmsd)
    #print('finished relax_pose')


# In[ ]:





# In[8]:


def minimize_pose(pose, cartesian_=False):
    #print('doing minimize_pose')
    movemap = setup_movemap(pose)
    pyrosetta.rosetta.core.pose.symmetry.make_symmetric_movemap(pose,movemap)
    s = pyrosetta.get_score_function()
    sf = pyrosetta.rosetta.core.scoring.symmetry.symmetrize_scorefunction(s)    
    sf.set_weight(pyrosetta.rosetta.core.scoring.atom_pair_constraint, 1.0)
    #sf.set_weight(pyrosetta.rosetta.core.scoring.angle_constraint, 1.0)
    #sf.set_weight(pyrosetta.rosetta.core.scoring.dihedral_constraint, 1.0)
    #sf.set_weight(pyrosetta.rosetta.core.scoring.coordinate_constraint, 1.0)
    #pyrosetta.rosetta.core.scoring.constraints.add_coordinate_constraints(pose, 0.1, False)
    
    use_nblist = True
    deriv_check = True
    deriv_check_verbose = False
    min_mover = pyrosetta.rosetta.protocols.simple_moves.symmetry.SymMinMover(movemap,sf,'lbfgs_armijo_nonmonotone',0.1,use_nblist,
                                                                          deriv_check,deriv_check_verbose)
    min_mover.max_iter(1)
    #min_mover.min_type('lbfgs_armijo_nonmonotone')
    
    apply_cst = pyrosetta.rosetta.protocols.constraint_generator.AddConstraints()
    all_res = pyrosetta.rosetta.core.select.residue_selector.SecondaryStructureSelector('HLE')
    #all_res = pyrosetta.rosetta.core.select.residue_selector.ResidueSpanSelector(int((pose.size()-1)/4+1),int(2*(pose.size()-1)/4+1))
    #all_res = pyrosetta.rosetta.core.select.residue_selector.ResidueSpanSelector(1,int((pose.size()-1)/4))
    all_res.apply(pose)
    print(all_res.apply(pose))
    atom_pair_cst = pyrosetta.rosetta.protocols.constraint_generator.AtomPairConstraintGenerator()
    atom_pair_cst.set_weight(1.0)
    atom_pair_cst.set_max_distance(10.0)
    atom_pair_cst.set_min_seq_sep(3)
    atom_pair_cst.set_sd(1.0)
    atom_pair_cst.set_ca_only(False)
    atom_pair_cst.set_residue_selector(all_res)
    atom_pair_cst.apply(pose)
    apply_cst.add_generator(atom_pair_cst)
    apply_cst.apply(pose)
    
    if cartesian_:
        min_mover.cartesian(True)
        sf.set_weight(pyrosetta.rosetta.core.scoring.ScoreType.cart_bonded, 0.5)
        sf.set_weight(pyrosetta.rosetta.core.scoring.ScoreType.pro_close, 0.0)
        sf.set_weight(pyrosetta.rosetta.core.scoring.ScoreType.cart_bonded_angle, 0.5)
        sf.set_weight(pyrosetta.rosetta.core.scoring.ScoreType.cart_bonded_length, 0.5)
        sf.set_weight(pyrosetta.rosetta.core.scoring.ScoreType.cart_bonded_torsion, 0.5)

    min_mover.apply( pose )
    #print('finished minimize_pose')


# In[ ]:





# In[9]:


def setup_movemap(pose):
    #print('doing setup_movemap')
    mm = pyrosetta.rosetta.core.kinematics.MoveMap()
    mm.set_chi( True )
    mm.set_bb( True )
    mm.set_jump( True )
    mm.set_bb ( pose.size(), False ) # # for the virtual residue?
    mm.set_chi( pose.size(), False ) # for the virtual residue?
    #print('finished setup_movemap')
    return mm


# In[ ]:





# In[10]:


def seal_jumps(pose):   
    #print('doing seal_jumps')
    i=1
    while i <= pose.size():
        if pose.residue_type(i).name() == "VRTBB":
            pose.conformation().delete_residue_slow(i)
        i=i+1
    ii=1
    while ii <= pose.size()-1:
        if ( pose.residue( ii ).has_variant_type( pyrosetta.rosetta.core.chemical.CUTPOINT_LOWER ) 
            and pose.residue( ii+1 ).has_variant_type( pyrosetta.rosetta.core.chemical.CUTPOINT_UPPER ) ):
            pyrosetta.rosetta.core.pose.remove_variant_type_from_pose_residue( pose, pyrosetta.rosetta.core.chemical.CUTPOINT_LOWER, ii )
            pyrosetta.rosetta.core.pose.remove_variant_type_from_pose_residue( pose, pyrosetta.rosetta.core.chemical.CUTPOINT_UPPER, ii+1 )
        ii=ii+1
    ft = pose.fold_tree()
    ft.clear()
    ft.add_edge(1,pose.size(),-1)
    pose.fold_tree(ft)
    #print('finished seal_jumps')


# In[ ]:





# In[11]:


def RepeatProteinRelax_apply(pose, modify_symmetry_and_exit_=False, remove_symm_=False, minimize_=False, cartesian_=False):
    #print('doing RepeatProteinRelax_apply')
    if modify_symmetry_and_exit_ and remove_symm_:
        pyrosetta.rosetta.core.pose.symmetry.make_asymmetric_pose(pose)
        seal_jumps(pose)
        return True
    setup_repeat_pose(pose)
    setup_movemap(pose);
    if modify_symmetry_and_exit_ and not remove_symm_:
        return True
    if minimize_:
        minimize_pose(pose, cartesian_)
    else:
        relax_pose(pose, cartesian_)
    pyrosetta.rosetta.core.pose.symmetry.make_asymmetric_pose(pose)
    seal_jumps(pose)
    #print('finished RepeatProteinRelax_apply')


# In[ ]:





# In[12]:


def extract_mer(in_pose, mer_start=1, mer_end=9):
    fragment_pose = pyrosetta.rosetta.core.pose.Pose()
    fragment_pose.append_residue_by_jump(in_pose.residue(mer_start), 1)
    for i in  range(mer_start + 1, mer_end+1):
        fragment_pose.append_residue_by_bond(in_pose.residue(i))
    return fragment_pose


# In[ ]:





# In[13]:


def find_loops_by_rmsd(ref_pose,loop_pose, rmsd_tolerance=0.01):
    rmsd_dict = {}
    for i in range(1, loop_pose.size() -9):
        loop_9_mer = extract_mer(loop_pose, i, i+8)
        for j in range(1, ref_pose.size() -9):
            ref_9_mer = extract_mer(ref_pose, j, j+8)
            rmsd = pyrosetta.rosetta.core.scoring.CA_rmsd(loop_9_mer, ref_9_mer)
            if rmsd < rmsd_tolerance:
                rmsd = 0.0
            if i+3 not in rmsd_dict:
                rmsd_dict[i+3] = rmsd
            else:
                if rmsd < rmsd_dict[i+3]:
                    rmsd_dict[i+3] = rmsd
    for key in rmsd_dict:
        print(key,rmsd_dict[key])
    return rmsd_dict


# In[14]:


def return_loops(rmsd_dict, offset=0):
    loops = {}
    loop = []
    j = 1
    for key in range(1+offset, len(rmsd_dict)+1+offset):
        if rmsd_dict[key] == 0.0 and len(loop) > 0:
            #print('adding to dict')
            loops[j] = [loop[0], loop[-1]]
            loop = []
            j=j+1    
        elif rmsd_dict[key] == 0.0 and len(loop) == 0:
            #print('continuing')
            continue
        elif rmsd_dict[key] > 0.0:
            loop.append(key)
            #print('appending',loop)
        else:
            print('wtf; how?')
    return loops


# In[ ]:





# In[15]:


def find_ref_loop_start_and_end(loop, ref_pose, nrepeat=4):
    start_of_loop = [0, 999]
    loop_start = extract_mer(loop,1,2)
    for i in range(1, int(ref_pose.size()/nrepeat)+10):
        ref_2_mer = extract_mer(ref_pose, i, i+1)
        rmsd = pyrosetta.rosetta.core.scoring.bb_rmsd_including_O(loop_start, ref_2_mer)
        if rmsd < start_of_loop[1]:
            start_of_loop = [i, rmsd]
    
    end_of_loop = [0, 999]
    loop_end = extract_mer(loop,loop.size()-1,loop.size())
    for i in range(1, int(ref_pose.size()/nrepeat)+10):
        ref_2_mer = extract_mer(ref_pose, i, i+1)
        rmsd = pyrosetta.rosetta.core.scoring.bb_rmsd_including_O(loop_end, ref_2_mer)
        if rmsd < end_of_loop[1]:
            end_of_loop = [i+1, rmsd]
    return [start_of_loop, end_of_loop]


# In[ ]:





# In[16]:


def superimpose_pose_on_pose(mod_pose, ref_pose, ref_start, ref_end):
    #tmp_ref_section = extract_mer(ref_pose, ref_start, ref_end)
    #pyrosetta.rosetta.core.scoring.calpha_superimpose_pose(mod_pose, tmp_ref_section)
    #print(ref_pose, mod_pose, False, ref_start+1, ref_end-1, 2, 2)
    pyrosetta.rosetta.protocols.grafting.superimpose_overhangs_heavy(ref_pose, mod_pose, False, ref_start+1, ref_end-1, 2, 2)


# In[ ]:





# In[17]:


def copy_phi_psi_omega(mod_pose, ref_pose, numb_repeats_=4, base_repeat=2):
##TJ adding these to Phil's function
    repeatlen = int(ref_pose.size()/numb_repeats_)
    nrepeat = numb_repeats_
    base_offset = (base_repeat-1)*repeatlen
    for i in range(0,nrepeat):
        j=1
        while j <= repeatlen:
            pos = j + base_offset
            mod_pose.set_phi  ( j+i*repeatlen, ref_pose.phi  (pos) )
            mod_pose.set_psi  ( j+i*repeatlen, ref_pose.psi  (pos) )
            mod_pose.set_omega( j+i*repeatlen, ref_pose.omega(pos) )
            j=j+1


# In[ ]:





# In[18]:


def idealize_to_tolerance(pose, tolerance=0.000001):
    idealize = pyrosetta.rosetta.protocols.idealize.IdealizeMover()
    idealize.atom_pair_constraint_weight(0.01)
    for i in range(1, 10):
        before_pose_copy = pose.clone()
        idealize.apply(pose)
        rmsd = pyrosetta.rosetta.core.scoring.bb_rmsd_including_O(pose, before_pose_copy)
        print('idealizing round', i, rmsd)
        if rmsd < tolerance:
            break


# In[19]:


def superimpose_by_tm_align(loop_pose, ref_pose):
    tmalign = pyrosetta.rosetta.protocols.hybridization.TMalign()
    tmalign.apply(loop_pose,ref_pose)


# In[20]:


def return_loop_start_and_end(loop_pose, ref_pose, nrepeats=4):
    distance = pyrosetta.rosetta.numeric.xyzVector_double_t.distance
    min_ref_distance_dict = {}
    min_loop_distance_dict = {}
    for i in range(1,int(ref_pose.size()+1)):
        for j in range(1,int(loop_pose.size()+1)):
            d = distance(ref_pose.residue(i).xyz('CA'), loop_pose.residue(j).xyz('CA'))
            if i in min_ref_distance_dict:
                if d < min_ref_distance_dict[i]:
                    min_ref_distance_dict[i] = d
            else:
                min_ref_distance_dict[i] = d
            if j in min_loop_distance_dict:
                if d < min_loop_distance_dict[j]:
                    min_loop_distance_dict[j] = d
            else:
                min_loop_distance_dict[j] = d
    return return_loops(min_ref_distance_dict), return_loops(min_loop_distance_dict)


# In[ ]:





# In[57]:


def propogate_loops_idealize_and_symmetrize(loop_pose, ref_pose, num_repeats=4):

    # get repeat length
    if (ref_pose.size()/num_repeats).is_integer():
        repeat_len = int(ref_pose.size()/num_repeats)
    
    # get the start and end of loop1 and loop2
    ref_loops, loops = return_loop_start_and_end(loop_pose, ref_pose)
    #print('new loops',loops)
     
    #loops = return_loops(find_loops_by_rmsd(ref_pose,loop_pose), 3)
    #print('old loops', loops)
    if len(loops) !=2:
        print('could not find two loops; something is wrong; ending')
        return
    #print(loop_pose, loops[1][0], loops[1][1])
    loop1 = extract_mer(loop_pose, loops[1][0], loops[1][1]) 
    #print(loop_pose, loops[2][0], loops[2][1])
    loop2 = extract_mer(loop_pose, loops[2][0], loops[2][1]) 
    #print(loop1)
    #print(loop2)
    
    # get start and end of reference around loop
    if len(ref_loops) !=2:
        print('could not find two loops; something is wrong; ending')
        return
    loop1_s_e = [[ref_loops[1][0]-2], [ref_loops[1][-1]+2]] 
    loop2_s_e = [[ref_loops[2][0]-2], [ref_loops[2][-1]+2]]
    #print('new ref loops', loop1_s_e, loop2_s_e)
    #loop1_s_e = find_ref_loop_start_and_end(loop1,ref_pose)
    #loop2_s_e = find_ref_loop_start_and_end(loop2,ref_pose)
    #print('old ref loops', loop1_s_e, loop2_s_e)
    
    # get four copies of loop1
    loop1_pose_v = [extract_mer(loop_pose, loops[1][0]-2, loops[1][1]+2), 
                    extract_mer(loop_pose, loops[1][0]-2, loops[1][1]+2), 
                    extract_mer(loop_pose, loops[1][0]-2, loops[1][1]+2), 
                    extract_mer(loop_pose, loops[1][0]-2, loops[1][1]+2)]
    #get four copies of loop2 # trim extra off the last loop
    loop2_pose_v = [extract_mer(loop_pose, loops[2][0]-2, loops[2][1]+2), 
                    extract_mer(loop_pose, loops[2][0]-2, loops[2][1]+2), 
                    extract_mer(loop_pose, loops[2][0]-2, loops[2][1]+2), 
                    extract_mer(loop_pose, loops[2][0]-2, loops[2][1]+2-(loop2_s_e[1][0]-repeat_len))]
    
    # get superposition for each loop
    # If this is not done well, will cause horrible problems
    ref_loop1_start = loop1_s_e[0][0]
    ref_loop1_end = loop1_s_e[1][0]
    tmp_index=1
    for tmp_pose in loop1_pose_v:
        #print('superimposing loop1',tmp_index)
        tmp_index=tmp_index+1
        if ref_loop1_end > ref_pose.size():
            ref_loop1_end = ref_pose.size() # this might cause pose length differences... maybe???
        #print(tmp_pose, ref_pose, ref_loop1_start, ref_loop1_end)
        superimpose_pose_on_pose(tmp_pose, ref_pose, ref_loop1_start, ref_loop1_end)
        ref_loop1_start += repeat_len
        ref_loop1_end += repeat_len

    ref_loop2_start = loop2_s_e[0][0]
    ref_loop2_end = loop2_s_e[1][0]
    tmp_index=1
    for tmp_pose in loop2_pose_v:
        #print('superimposing loop2',tmp_index)
        tmp_index+=1
        if ref_loop2_end > ref_pose.size():
            ref_loop2_end = ref_pose.size() # this might cause pose length differences... maybe???
        superimpose_pose_on_pose(tmp_pose, ref_pose, ref_loop2_start, ref_loop2_end)
        ref_loop2_start += repeat_len
        ref_loop2_end += repeat_len
    
        
    ## now we build a new pose by piecing together the reference pose and new superimposed loops
    ## be careful with index, additions, and subtrations
    my_new_pose = pyrosetta.rosetta.core.pose.Pose()
    my_new_pose.append_residue_by_jump(ref_pose.residue(1), 1)
    for i in range(0, num_repeats):
        if i == 0: ### handle the first repeat
            for j in range(2, loop1_s_e[0][0]+repeat_len*i):
                my_new_pose.append_residue_by_bond(ref_pose.residue(j)) #start of repeat
            for j in range(1, loop1_pose_v[i].size()+1):
                my_new_pose.append_residue_by_bond(loop1_pose_v[i].residue(j)) # loop1
            for j in range(loop1_s_e[1][0]+repeat_len*i+1, loop2_s_e[0][0]+repeat_len*i):
                my_new_pose.append_residue_by_bond(ref_pose.residue(j)) # second part of repeat
            for j in range(1, loop2_pose_v[i].size()+1):
                my_new_pose.append_residue_by_bond(loop2_pose_v[i].residue(j)) # loop2
        else: ### handle internal repeats and last repeat
            for j in range((loop2_s_e[1][0]-repeat_len+1)+repeat_len*i, loop1_s_e[0][0]+repeat_len*i):
                my_new_pose.append_residue_by_bond(ref_pose.residue(j))
            for j in range(1, loop1_pose_v[i].size()+1):
                my_new_pose.append_residue_by_bond(loop1_pose_v[i].residue(j))
            for j in range(loop1_s_e[1][0]+repeat_len*i+1, loop2_s_e[0][0]+repeat_len*i):
                my_new_pose.append_residue_by_bond(ref_pose.residue(j))
            for j in range(1, loop2_pose_v[i].size()+1):
                my_new_pose.append_residue_by_bond(loop2_pose_v[i].residue(j))
    
    #print(my_new_pose)
    #my_new_pose.dump_pdb('/home/drhicks1/ADD_LOOPS/test/26H_AGBA_18H_AGB_27_start_0013_rd2_1_rd2_0001_0002_0001_0001_new.pdb')
    
    # need to idealize bond lengths and angles
    # so we can later apply perfect symmetry
    # Very very very bad things happen 
    # when the pose is not PERFECTLY IDEAL and you add symmetry
    my_new_ideal_pose = my_new_pose.clone()
    idealize_to_tolerance(my_new_ideal_pose)
    #my_new_ideal_pose.dump_pdb('/home/drhicks1/ADD_LOOPS/test/26H_AGBA_18H_AGB_27_start_0013_rd2_1_rd2_0001_0002_0001_0001_new_ideal.pdb')
    # pose has to be so perfectly ideal
    # now we just copy phi/psi/omega to a new ideal backbone
    # the pose has to be very close to perfect before this, or bad things can happen
    my_new_ideal_pose2 = pyrosetta.pose_from_sequence(my_new_ideal_pose.sequence())
    copy_phi_psi_omega(my_new_ideal_pose2, my_new_ideal_pose)
    #my_new_ideal_pose2.dump_pdb('/home/drhicks1/ADD_LOOPS/test/26H_AGBA_18H_AGB_27_start_0013_rd2_1_rd2_0001_0002_0001_0001_new_ideal2.pdb')
    
    # now we apply true symmetry and do a constrained relax
    # spent so much time idealizing, maybe don't do cartesian relax
    # but it should be ok
    
    ### sym relax is not working correclty?
    ### causes large rmsd change
    ### constraints do not seem to do anything
    ### there is no reason to need this anyways, the pose is already prefectly symmetric
    ### relax/minimization doesnt make sense until the pose has sequnece anyways 
    
    #sym_pose = my_new_ideal_pose2.clone()
    #RepeatProteinRelax_apply(sym_pose)
    # relax will print an rmsd change for relax
    # this is the rmsd change due to relax and adding symmetry
    # symmetry shouldn't casue much change or there is likley a prior problem
    #print(pyrosetta.rosetta.core.scoring.CA_rmsd(my_new_ideal_pose2, sym_pose))
    #sym_pose.dump_pdb('/home/drhicks1/ADD_LOOPS/test/26H_AGBA_18H_AGB_27_start_0013_rd2_1_rd2_0001_0002_0001_0001_new_sym.pdb')
    
    return my_new_pose, my_new_ideal_pose, my_new_ideal_pose2 #, sym_pose
    

def run_jobs(ref_files):
    all_loops = glob.glob('/home/drhicks1/36K_curved_repeats_enumerate_ideal_loops/loop2/X*.pdb')
    print('globbed all loop files')
    path_dict = {}
    for loopfile in all_loops:
        reference = os.path.basename(loopfile)[0:-4]
        remove_from_end =  re.compile("\_[0-9][0-9][0-9][0-9]$")
        while remove_from_end.search(reference):
            reference = reference[0:-5]
        reference = reference+'.pdb'
        #print(loopfile, reference)
        if reference in path_dict:
            path_dict[reference].append(loopfile)
        else:
            path_dict[reference] = [loopfile]
    #print(path_dict)
    previously_finished = {}
    fout = open('finished_files.loop', 'a+')
    fout.seek(0)
    content = fout.readlines()
    for line in content:
        #print('here is a finsihed one', line)
        line = line.strip()
        previously_finished[line] = 1
    for reference in ref_files:
        ref_name = os.path.basename(reference)
        if ref_name in path_dict:
            i=1
            for loop_file in path_dict[ref_name]:
                if loop_file in previously_finished:
                    print('skipping',loop_file,reference)
                    i=i+1
                    continue
                print('working on',loop_file,reference)
                ref_pose = pyrosetta.pose_from_pdb(reference)
                loop_pose = pyrosetta.pose_from_pdb(loop_file)
                my_new_pose, my_new_ideal_pose, my_new_ideal_pose2 = propogate_loops_idealize_and_symmetrize(loop_pose, ref_pose)
       
                base_ref = os.path.basename(reference)
                #ref_pose.dump_pdb('/home/drhicks1/TEST_LOOP_OUTPUT1/{}'.format(base_ref))
                my_new_ideal_pose2.dump_pdb('/home/drhicks1/36K_curved_repeats_enumerate_ideal_loops/propogate_loops/{}_loop_{}.pdb'.format(base_ref[0:-4], i))
                fout.write(loop_file+'\n')
                i=i+1

    fout.close()
    path_dict2 = {}
    all_loops = glob.glob('/home/drhicks1/36K_curved_repeats_enumerate_ideal_loops/loop2/X*.pdb')
    for loopfile in all_loops:
        reference = os.path.basename(loopfile)[0:-4]
        remove_from_end =  re.compile("\_[0-9][0-9][0-9][0-9]$")
        while remove_from_end.search(reference):
            reference = reference[0:-5]
        reference = reference+'.pdb'
        if reference in path_dict2:
            path_dict[reference].append(loopfile)
        else:
            path_dict2[reference] = [loopfile]
        #loop_files2 = glob.glob('/home/drhicks1/ADD_LOOPS/test/{}_[0-9][0-9][0-9][0-9]_[0-9][0-9][0-9][0-9]_[0-9][0-9][0-9][0-9]_[0-9][0-9][0-9][0-9].pdb'.format(base_ref[0:-4]))
    return path_dict, path_dict2

ref_files = glob.glob('/home/drhicks1/36K_curved_repeats/*/X*.pdb')
i,j = run_jobs(ref_files)
while i != j:
    i,j = run_jobs(ref_files)

