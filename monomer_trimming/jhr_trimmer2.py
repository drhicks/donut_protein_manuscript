#!/usr/bin/env python

import sys
import os


sys.path.append("/home/bcov/sc/random/npose")

import hashlib

from numba import njit

from importlib import reload
import npose_util as nu
reload(nu)

import argparse

import re
import time
import itertools

import atomic_depth

import numpy as np

import helical_worms
reload(helical_worms)
import motif_stuff2

if ( len(sys.argv) > 2 ):
    files = sys.argv[1:]
else:
    files = []
    with open(sys.argv[1]) as f:
        for line in f:
            line = line.strip()
            if ( len(line) == 0 ):
                continue
            files.append(line)


parser = argparse.ArgumentParser()
parser.add_argument("pdbs", type=str, nargs="*")
parser.add_argument("-in:file:silent", type=str, default="")
parser.add_argument("-force_silent", action="store_true")
parser.add_argument("-debug", action="store_true")
parser.add_argument("-allow_nonstandard", action="store_true")
parser.add_argument("-pcscn_cut", type=float, default=0.01) #0.19
parser.add_argument("-avgfive_cut", type=float, default=0.01) #0.70
parser.add_argument("-avgnine_cut", type=float, default=0.01) #0.80
parser.add_argument("-avgnine_two_cut", type=float, default=0.01) #0.80

parser.add_argument("-min_scaff_length", type=float, default=120)
parser.add_argument("-max_scaff_length", type=float, default=300)
parser.add_argument("-helix_worst_gap_cut", type=float, default=100000)


args = parser.parse_args(sys.argv[1:])

pdbs = args.pdbs
silent = args.__getattribute__("in:file:silent")



if ( silent != "" ):
    print("Loading silent")
    nposes, pdbs = nu.nposes_from_silent( silent )

def get_ipdb(i):
    if ( silent != "" ):
        return nposes[i], pdbs[i]

    npose = nu.npose_from_file_fast(pdbs[i])
    return npose, nu.get_tag(pdbs[i])



def get_extraneous(npose):
    # nu.dump_npdb(npose, "test.pdb")
    is_helix = nu.npose_dssp_helix(npose)
    ss_elems = nu.npose_helix_elements(is_helix)

    # print(is_helix)

    assert(is_helix[0] and is_helix[-1])

    is_segment = np.zeros(nu.nsize(npose), np.int)
    is_segment.fill(-1)

    num_helices = 0
    for i, elem in enumerate(ss_elems):
        helix, start, end = elem
        if ( not helix ):
            continue
        is_segment[start:end+1] = i
        num_helices += 1

    care_mask = np.ones(nu.nsize(npose), np.bool)

    ca_cb = nu.extract_atoms(npose, [nu.CA, nu.CB]).reshape(-1, 2, 4)[...,:3]

    return is_helix, is_segment, ss_elems, care_mask, ca_cb

def trim_this_pose(npose, froms, tos, min_length, max_length):

    min_helix_length = 12

    # prepare info
    is_helix, is_segment, ss_elems, care_mask, ca_cb = get_extraneous(npose)

    save_froms = froms
    save_tos = tos


    first_bundle_mask = (is_segment == 0) | (is_segment == 2) | (is_segment == 4)

    k = len(ss_elems)-1
    last_bundle_mask = (is_segment == k-0) | (is_segment == k-2) | (is_segment == k-4)

    # get info about first and last helices
    _, _, last_n_res = ss_elems[0]
    _, first_c_res, _ = ss_elems[-1]

    # trim residues will be removed
    max_n_trim_res = last_n_res - min_helix_length
    max_c_trim_res = first_c_res + min_helix_length

    # cant trim one of the helices
    if ( max_n_trim_res < -1 or max_c_trim_res > nu.nsize(npose) ):
        return None, None, None

    # all possible trimmings
    ntrims = range(-1, max_n_trim_res+1)
    ctrims = range(max_c_trim_res, nu.nsize(npose))

    ntrim_ctrim = list(itertools.product(ntrims, ctrims))
    scores = np.zeros(len(ntrim_ctrim))
    scores.fill(-1)

    for i, (ntrim, ctrim) in enumerate(ntrim_ctrim):

        size = ctrim - ntrim - 1
        if ( size < min_length or size > max_length ):
            continue


        froms = save_froms - ntrim - 1
        tos = save_tos - ntrim - 1

        oob = (tos < 0) | (tos >= size) | (froms < 0) | (froms >= size)

        tos = tos[~oob]
        froms = froms[~oob]

        # a = trim_npose(npose, ntrim+1, ctrim-1)
        # nu.dump_npdb(a, "test.pdb")


        score_map = {}
        fail = score_seg2(is_segment[ntrim+1:ctrim], is_helix[ntrim+1:ctrim], ca_cb[ntrim+1:ctrim],
            froms, tos, first_bundle_mask[ntrim+1:ctrim], score_map, "N")
        if (not fail ):
            fail = score_seg2(is_segment[ntrim+1:ctrim], is_helix[ntrim+1:ctrim], ca_cb[ntrim+1:ctrim],
                        froms, tos, last_bundle_mask[ntrim+1:ctrim], score_map, "C")

        if ( fail ):
            continue
        # _, _, _, _, neighs = helical_worms.get_avg_sc_neighbors( ca_cb[ntrim+1:ctrim], care_mask[ntrim+1:ctrim] )
        # is_core_boundary = neighs > 2

        # _, _, avg5_n, avg5_2_n, _ = helical_worms.worst_second_worst_motif_interaction(is_segment[ntrim+1:ctrim], is_helix[ntrim+1:ctrim],
        #                                         froms, tos, size, is_core_boundary&first_bundle_mask[ntrim+1:ctrim], care_mask[ntrim+1:ctrim], 5)
        # _, _, avg9_n, avg9_2_n, _ = helical_worms.worst_second_worst_motif_interaction(is_segment[ntrim+1:ctrim], is_helix[ntrim+1:ctrim],
        #                                         froms, tos, size, is_core_boundary&first_bundle_mask[ntrim+1:ctrim], care_mask[ntrim+1:ctrim], 9)

        # _, _, avg5_c, avg5_2_c, _ = helical_worms.worst_second_worst_motif_interaction(is_segment[ntrim+1:ctrim], is_helix[ntrim+1:ctrim],
        #                                         froms, tos, size, is_core_boundary&last_bundle_mask[ntrim+1:ctrim], care_mask[ntrim+1:ctrim], 5)
        # _, _, avg9_c, avg9_2_c, _ = helical_worms.worst_second_worst_motif_interaction(is_segment[ntrim+1:ctrim], is_helix[ntrim+1:ctrim],
        #                                         froms, tos, size, is_core_boundary&last_bundle_mask[ntrim+1:ctrim], care_mask[ntrim+1:ctrim], 9)

        avg5_n = score_map["N_avg5"]
        avg9_n = score_map["N_avg9"]
        avg9_2_n = score_map["N_avg9_2"]
        pcscn_n = score_map["N_pcscn"]

        avg5_c = score_map["C_avg5"]
        avg9_c = score_map["C_avg9"]
        avg9_2_c = score_map["C_avg9_2"]
        pcscn_c = score_map["C_pcscn"]

        # this is the important line
        score = (avg5_n + avg9_n + 2*avg9_2_n)**0.5 + (avg5_c + avg9_c + 2*avg9_2_c)**0.5

        scores[i] = score


        # print("%5i %5i %7.2f %7.2f %7.2f"%(ntrim+1, ctrim+1, score, pcscn_n, pcscn_c))


    if ( scores.max() == -1 ):
        return None, None, None

    argmax = np.argmax(scores)

    ntrim, ctrim = ntrim_ctrim[argmax]

    npose = trim_npose(npose, ntrim+1, ctrim-1)

    return npose, ntrim, ctrim

def score_seg2(is_segment, is_helix, ca_cb, froms, tos, mask, score_map, prefix):

    start = np.min(np.where(mask)[0])
    end = np.max(np.where(mask)[0])

    # npose = trim_npose(npose, start, end)
    # nu.dump_npdb(npose, prefix + ".pdb")



    is_segment = is_segment[start:end+1]
    is_helix = is_helix[start:end+1]
    ca_cb = ca_cb[start:end+1]
    mask = mask[start:end+1]

    froms = froms - start
    tos = tos - start
    oob = (tos < 0) | (tos >= len(ca_cb)) | (froms < 0) | (froms >= len(ca_cb))
    froms = froms[~oob]
    tos = tos[~oob]

    _, _, _, _, neighs = helical_worms.get_avg_sc_neighbors( ca_cb, mask )
    is_core_boundary = neighs > 2

    percent_core = (neighs[mask] > 5.2).mean()

    if ( percent_core < args.pcscn_cut ):
        return True

    if ( is_core_boundary.sum() == 0 ):
        print("None")

    _, _, avg9, avg9_2, _ = helical_worms.worst_second_worst_motif_interaction(is_segment, is_helix,
                                            froms, tos, len(ca_cb), is_core_boundary, mask, 9)

    if ( avg9 < args.avgnine_cut ):
        return True
    if ( avg9_2 < args.avgnine_two_cut ):
        return True

    _, _, avg5, avg5_2, _ = helical_worms.worst_second_worst_motif_interaction(is_segment, is_helix,
                                            froms, tos, len(ca_cb), is_core_boundary, mask, 5)

    if ( avg5 < args.avgfive_cut ):
        return True

    score_map[prefix + "_avg5"] = avg5
    score_map[prefix + "_avg9"] = avg9
    score_map[prefix + "_avg9_2"] = avg9_2
    score_map[prefix + "_pcscn"] = percent_core

    return False


def score_seg(npose, is_segment, is_helix, froms, tos, is_core_boundary, mask, score_map, prefix):


    _, _, avg5, avg5_2, _ = helical_worms.worst_second_worst_motif_interaction(is_segment, is_helix,
                                            froms, tos, nu.nsize(npose), is_core_boundary&mask, mask, 5)
    _, _, avg9, avg9_2, _ = helical_worms.worst_second_worst_motif_interaction(is_segment, is_helix,
                                            froms, tos, nu.nsize(npose), is_core_boundary&mask, mask, 9)

    score_map[prefix + "_avg5"] = avg5
    score_map[prefix + "_avg9"] = avg9
    score_map[prefix + "_avg9_2"] = avg9_2

def get_helix_stats(cas, params1, params2):

    _, start1, end1 = params1
    _, start2, end2 = params2

    cas1 = cas[start1:end1+1]
    cas2 = cas[start2:end2+1]

    # output shape should be (cas1.shape[-2], cas2.shape[-2])
    pair_dists = np.linalg.norm(cas1[:,None,:] - cas2, axis=-1)

    closest = np.min(pair_dists)

    # terrible greedy method here

    needed = np.min([len(cas1), len(cas2), 4])

    dists = []

    close_pairs = np.dstack(np.unravel_index(np.argsort(pair_dists.ravel()), pair_dists.shape))[0]
    used1 = set()
    used2 = set()

    for i1, i2 in close_pairs:
        if ( i1 in used1 ):
            continue
        if ( i2 in used2 ):
            continue
        used1.add(i1)
        used2.add(i2)

        dists.append(pair_dists[i1, i2])

    return closest, np.mean(dists)




def get_five_cross(npose, score_map):

    is_helix, is_segment, ss_elems, care_mask, ca_cb = get_extraneous(npose)

    num_fives = score_map['helices'] - 4

    cas = nu.extract_atoms(npose, [nu.CA])

    whole_closests = []
    whole_avgs = []

    for ifive in range(num_fives):
        these_helices = ss_elems[ifive*2:(ifive+5)*2:2]

        closests = []
        avgs = []

        closest, avg = get_helix_stats(cas, these_helices[0], these_helices[3])
        closests.append(closest)
        avgs.append(avg)
        closest, avg = get_helix_stats(cas, these_helices[0], these_helices[4])
        closests.append(closest)
        avgs.append(avg)
        closest, avg = get_helix_stats(cas, these_helices[1], these_helices[3])
        closests.append(closest)
        avgs.append(avg)
        closest, avg = get_helix_stats(cas, these_helices[1], these_helices[4])
        closests.append(closest)
        avgs.append(avg)

        closest = np.min(closests)
        avg = np.min(avgs)

        whole_closests.append(closest)
        whole_avgs.append(avg)

    score_map['helix_worst_gap'] = np.max(whole_closests)
    score_map['helix_worst_gap_avg'] = np.max(whole_avgs)




def score_npose(npose, froms, tos, score_map):

    is_helix, is_segment, ss_elems, care_mask, ca_cb = get_extraneous(npose)

    score_map['helices'] = len(ss_elems)//2 + 1

    # here we make sure that the first and last helical bundles look ok

    first_bundle_mask = (is_segment == 0) | (is_segment == 2) | (is_segment == 4)

    k = len(ss_elems)-1

    last_bundle_mask = (is_segment == k-0) | (is_segment == k-2) | (is_segment == k-4)

    assert(np.all(is_helix[first_bundle_mask]))
    assert(np.all(is_helix[last_bundle_mask]))


    _, _, _, _, neighs = helical_worms.get_avg_sc_neighbors( ca_cb, care_mask )
    is_core_boundary = neighs > 2

    score_seg2(is_segment, is_helix, ca_cb, froms, tos, first_bundle_mask, score_map, "N")
    score_seg2(is_segment, is_helix, ca_cb, froms, tos, last_bundle_mask, score_map, "C")

    any_fail = False
    if ( score_map["N_pcscn"] < args.pcscn_cut or score_map["C_pcscn"] < args.pcscn_cut ):
        # print("Fail pcscn: %6.2f %6.2f"%(score_map["N_pcscn"], score_map["C_pcscn"]))
        any_fail = True

    if ( score_map["N_avg5"] < args.avgfive_cut or score_map["C_avg5"] < args.avgfive_cut ):
        # print("Fail avg5: %6.2f %6.2f"%(score_map["N_avg5"], score_map["C_avg5"]))
        any_fail = True

    if ( score_map["N_avg9"] < args.avgnine_cut or score_map["C_avg9"] < args.avgnine_cut ):
        # print("Fail avg9: %6.2f %6.2f"%(score_map["N_avg9"], score_map["C_avg9"]))
        any_fail = True

    if ( score_map["N_avg9_2"] < args.avgnine_two_cut or score_map["C_avg9_2"] < args.avgnine_two_cut ):
        # print("Fail avg9_2: %6.2f %6.2f"%(score_map["N_avg9_2"], score_map["C_avg9_2"]))
        any_fail = True

    # print("")

    if ( any_fail ):
        return False

    # score_seg(npose, is_segment, is_helix, froms, tos, is_core_boundary&first_bundle_mask, first_bundle_mask, score_map, "N_")
    # score_seg(npose, is_segment, is_helix, froms, tos, is_core_boundary&last_bundle_mask, last_bundle_mask, score_map, "C_")

    get_five_cross(npose, score_map)

    if ( score_map['helix_worst_gap'] > args.helix_worst_gap_cut ):
        return False

    return True


# make this a helper to avoid mistakes
def trim_npose(npose, save_start, save_end):
    return npose[save_start*nu.R:(save_end+1)*nu.R]


def trim_jhr(npose, tag, score_map, string_map):

    is_helix = nu.npose_dssp_helix( npose )
    ss_elements = nu.npose_helix_elements( is_helix )

    if ( not args.allow_nonstandard ):
        if ( len(ss_elements) != 16 ):
            print("Bad secondary structure")
            return None

    if ( not ss_elements[0][0] ):
        print("Doesnt start with helix")
        return None

    if ( not ss_elements[-1][0] ):

        _, last_loop_start, last_loop_end = ss_elements[-1]

        assert(last_loop_end == nu.nsize(npose)-1)

        npose = trim_npose(npose, 0, last_loop_start-1)
        ss_elements = ss_elements[:-1]
    else:
        assert( args.allow_nonstandard)

    if ( len(ss_elements) % 2 != 1 ):
        print("Weird secondary structure, how is this possible?")
    num_input_helices = (len(ss_elements)+1)//2
    
    # max_trim = num_input_helices - 5
    max_trim = 2

    care_mask = np.ones(nu.nsize(npose), np.bool)

    ca_cb = nu.extract_atoms( npose, [nu.CA, nu.CB]).reshape(-1, 2, 4)[...,:3]

    _, _, _, _, neighs = helical_worms.get_avg_sc_neighbors( ca_cb, care_mask )


    # remember, core and boundary positions will change later
    is_core_boundary = neighs > 2

    _, froms, tos, _ = motif_stuff2.motif_score_npose( npose, care_mask, is_core_boundary )

    min_length = args.min_scaff_length
    max_length = args.max_scaff_length

    # ncuts = [0, 1, 2]
    # ccuts = [0, 1, 2]
    ncuts = [0, 1]
    ccuts = [0, 1]


    to_ret = []

    save_npose = npose
    save_froms = froms
    save_tos = tos

    icut = 0
    for ncut, ccut in itertools.product(ncuts, ccuts):
        if ( ncut + ccut > max_trim ):
            continue
        icut += 1
        npose = save_npose.copy()
        froms = save_froms.copy()
        tos = save_tos.copy()

        is_false, first_remove, end = ss_elements[-ccut*2]
        c_removed = 0
        if ( ccut > 0 ):
            assert( not is_false )
            c_removed = nu.nsize(npose) - first_remove
            npose = trim_npose(npose, 0, first_remove-1)

        removed = 0
        if ( ncut > 0 ):
            is_false, start, last_remove = ss_elements[-1+ncut*2]
            assert( not is_false )
            npose = trim_npose(npose, last_remove+1, nu.nsize(npose)-1)
            removed = last_remove+1

        froms -= removed
        tos -= removed

        oob = (tos < 0) | (tos >= nu.nsize(npose)) | (froms < 0) | (froms >= nu.nsize(npose))

        froms = froms[~oob]
        tos = tos[~oob]


        ca_cb = nu.extract_atoms(npose, [nu.CA, nu.CB]).reshape(-1, 2, 4)[...,:3]
        # nu.dump_npdb(npose, "%i.pdb"%(icut))

        # print("icut: %i "%icut + "_n%i_c%i"%(ncut, ccut))
        
        old_size = nu.nsize(npose)
        npose, ntrim, ctrim = trim_this_pose(npose, froms, tos, min_length, max_length)

        if ( npose is None ):
            continue

        # this is what happened
        # npose = trim_npose(npose, ntrim+1, ctrim-1)
        removed += ntrim + 1
        c_removed += old_size - ctrim

        # args.debug = ncut==1 and ccut==2

        froms = froms - ntrim - 1
        tos = tos - ntrim - 1
        oob = (tos < 0) | (tos >= nu.nsize(npose)) | (froms < 0) | (froms >= nu.nsize(npose))
        tos = tos[~oob]
        froms = froms[~oob]

        score_map = {}
        passes = score_npose(npose, froms, tos, score_map)

        score_map['trimmed_N'] = removed
        score_map['trimmed_C'] = c_removed

        if ( passes ):
            to_ret.append([npose, tag + "_n%i_c%i"%(ncut, ccut), score_map, {}])



    return to_ret












if ( silent != "" or args.force_silent ):
    open_silent = open("out.silent", "w")

work_care_mask = np.zeros((400), np.bool)

first_score = True
first_silent = True

start = 0
if ( os.path.exists("ckpt")):
    with open("ckpt") as f:
        try:
            start = int(f.read())
            print("Starting at checkpoint %i"%start)
        except:
            pass

for ipdb in range(start, len(pdbs)):
    with open("ckpt", "w") as f:
        f.write(str(ipdb))
    t0 = time.time()
        # try:
    for k in [1]:

        npose, tag = get_ipdb(ipdb)

        score_map = {}
        string_map = {}

        out_stuff = trim_jhr(npose, tag, score_map, string_map)

        if ( not out_stuff is None):

            to_iterate = [(out_stuff, tag, score_map, string_map)]

            if ( isinstance(out_stuff, list) ):
                to_iterate = out_stuff

            for npose_out, tag_out, score_map_out, string_map_out in to_iterate:

                open_score = open("score.sc", "a")
                nu.add_to_score_file_open(tag_out, open_score, first_score, score_map_out, string_map_out )
                open_score.close()
                first_score = False


                if ( silent == "" and not args.force_silent):
                    nu.dump_npdb(npose_out, tag_out + ".pdb")
                else:
                    nu.add_to_silent_file_open( npose_out, tag_out, open_silent, first_silent, score_map_out, string_map_out)
                    first_silent = False


        seconds = int(time.time() - t0)

        print("protocols.jd2.JobDistributor: " + tag + " reported success in %i seconds"%seconds)


    # except Exception as e:
    #     print("Error!!!")
    #     print(e)



if ( silent != "" or args.force_silent ):
    open_silent.close()