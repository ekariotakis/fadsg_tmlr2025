# %%
import os
import copy
import numpy as np
import networkx as nx
# from dsd import *
import dsd
# from my_dsp import *
from super_greedy_pp import *
from init_graph import *
import utils
from datetime import datetime
import matplotlib.pyplot as plt 
import matplotlib
# %matplotlib inline 

from tqdm import tqdm

import random
import inspect

import csv

import argparse

def main():

    parser = argparse.ArgumentParser(description='Bisection Tester')
    parser.add_argument('--formulation', type=int, default=1, metavar='R', help='Desired Formulation (1 for FADSG-I - 2 for FADSG-II)')
    parser.add_argument('--dataset-name', type=str, default='karate', metavar='S', help='Desired Dataset Name')
    parser.add_argument('--alpha', type=float, default=0.5, metavar='R', help='Target Fairness Level')
    parser.add_argument('--protected-class', type=float, default=1.0, metavar='R', help='Protected Class Label')
    parser.add_argument('--epsilon', type=float, default=1e-8, metavar='R', help='Tolerance')

    args = parser.parse_args()

    formulation = copy.deepcopy(args.formulation)
    dataset_name = copy.deepcopy(args.dataset_name)
    alpha = copy.deepcopy(args.alpha)
    protected_class = copy.deepcopy(args.protected_class)
    epsilon = copy.deepcopy(args.epsilon)

    # %%
    G, protected_nodes, lam_max = init_graph(dataset_name, formulation, protected_class)

    n = G.number_of_nodes()
    print('Number of Nodes:', n)
    m = G.number_of_edges()
    print('Number of Edges:', m)

    # %%
    
    # T=5
    # super_greedy_pp_R = super_greedy_pp(G, T, protected_nodes, lam=0, weight=None, formulation=formulation)
    # super_greedy_pp_R_nodes = list(super_greedy_pp_R[0].nodes())
    # super_greedy_pp_R_nodes.sort()
    # density_0 =  utils.compute_density(super_greedy_pp_R[0])

    # print('flowless method')
    # start = datetime.now()
    # flowless_R = dsd.flowless(G, 5)
    # density_0 = flowless_R[1]

    exact_R = dsd.exact_densest(G)
    exact_R[0].sort()
    density_0 = exact_R[1]

    lam_vec = np.linspace(0, lam_max, num=20)

    # T=3
    T=5
    Psi_vec = []
    r_vec = []

    # %%
    # epsilon = 1e-3
    # epsilon = 1e-5
    lam_max_ = lam_max
    lam_min_ = 0

    lam_mid_vec = []

    density_vec = []
    num_of_nodes = []
    num_of_protected_vec = []
    protected_portion_in_sub_vec = []
    protected_portion_in_prot_vec = []
    PoF_vec = []
    LB_UB_vec = []
    fairness_vec = []
    r_vec = []

    max_iters = np.ceil(np.log2(lam_max/epsilon))
    for i in tqdm(range(int(max_iters))):
        lam_mid = (lam_max_+lam_min_)/2
        lam_mid_vec.append(lam_mid)
        protected_nodes_all = copy.deepcopy(protected_nodes)

        super_greedy_pp_R = super_greedy_pp(G, T, protected_nodes, lam=lam_mid, weight=None, formulation=formulation)
        super_greedy_pp_R_nodes = list(super_greedy_pp_R[0].nodes())
        super_greedy_pp_R_nodes.sort()

        induced_protected_nodes = super_greedy_pp_R[2]
        
        num_of_nodes.append(len(super_greedy_pp_R_nodes))
        
        density =  utils.compute_density(super_greedy_pp_R[0])
        density_vec.append(density)  

        PoF = 1 - density/density_0
        PoF_vec.append(PoF)

        LB_UB = super_greedy_pp_R[3]
        LB_UB_vec.append(LB_UB)
        
        fairness = utils.find_num_common(super_greedy_pp_R_nodes, protected_nodes)
        fairness_vec.append(fairness)

        num_of_protected = len(induced_protected_nodes)
        num_of_protected_vec.append(num_of_protected)
        protected_portion_in_sub = num_of_protected/len(super_greedy_pp_R[0])
        protected_portion_in_sub_vec.append(protected_portion_in_sub)
        protected_portion_in_prot = num_of_protected/len(protected_nodes)
        protected_portion_in_prot_vec.append(protected_portion_in_prot)


        r = utils.compute_r(super_greedy_pp_R[0], protected_nodes_all, induced_protected_nodes, formulation=formulation)
        if formulation==1:
            Psi = r-alpha
        elif formulation==2:
            Psi = r+alpha


        if formulation==1:
            if abs(Psi)<epsilon:
                break
        elif Psi>0:
            lam_max_ = lam_mid
            # lam_min_ = lam_min_
        elif Psi<0:
            # lam_max_ = lam_max_
            lam_min_ = lam_mid
        Psi_vec.append(Psi)
        r_vec.append(r)

    
    # Save Variables in Dictionary
    r_0=[]
    r_max=[]
    variables_dict = {
        'alpha': alpha,
        'r_0': r_0,
        'r_max': r_max,
        'lam': lam_vec,
        'lam_mid': lam_mid_vec,
        'induced': super_greedy_pp_R,
        'induced_protected': induced_protected_nodes,
        'protected': protected_nodes,
        'Psi': Psi_vec,
        'r': r_vec,
        'max_iters': max_iters,
        'lam': lam_vec,
        'induced': super_greedy_pp_R,
        'num_of_nodes': num_of_nodes,
        'density': density_vec,
        'num_of_protected': num_of_protected_vec,
        'protected_portion_in_sub': protected_portion_in_sub_vec,
        'protected_portion_in_prot': protected_portion_in_prot_vec,
        'PoF': PoF_vec,
        'fairness': fairness_vec,
        'LB_UB': LB_UB_vec
    }

    if "amazon" in dataset_name:
        save_folder = 'logs/bisection_/'+'amazon'
        save_path = save_folder+'/'+dataset_name+'_'+str(formulation)+'_'+str(alpha)+'_log.npy'
    elif "lastfm" in dataset_name:
        save_folder = 'logs/bisection_/'+dataset_name
        save_path = save_folder+'/'+dataset_name+'_'+str(formulation)+'_'+str(protected_class)+'_'+str(alpha)+'_log.npy'
    else:
        save_folder = 'logs/bisection_/'+dataset_name
        save_path = save_folder+'/'+dataset_name+'_'+str(formulation)+'_'+str(alpha)+'_log.npy'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    np.save(save_path, variables_dict)


if __name__ == '__main__':
    main()