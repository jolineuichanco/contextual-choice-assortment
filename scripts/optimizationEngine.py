'''
optimizationEngine.py

MILP solvers for assortment optimization under MNL and CC choice models.

This module implements:
- solve_MNL_assortment_MIP: Solves the MNL assortment problem (polynomial size)
- solve_CC_assortment_MIP_levels_preprocess: Solves the CC assortment problem
  with preprocessing to reduce the number of reference lists

The MILP formulations use the Charnes-Cooper transformation to linearize
the fractional objective. See Section 4 of the paper for the CC formulation.

Key insight: Under CC, the optimal assortment depends on which products serve
as "reference points" (minimum attribute values). The preprocessing algorithm
(findReferenceList) enumerates only the feasible reference lists, which is
much smaller than the naive N^K enumeration.

Requirements:
- IBM CPLEX Python API (cplex)

Author: Joline Uichanco
Created: May 10, 2023
'''

import cplex
import itertools
import time
import copy
import numpy as np

# =============================================================================
# PREPROCESSING: Find Feasible Reference Lists
# =============================================================================

# Global variable to store found reference lists (used by recursive function)
found_references = []

def findReferenceList(product_dict, J_dict, K):
    '''
    Find all feasible reference lists for the CC assortment problem.
    
    This preprocessing step enumerates reference lists (zeta) that could
    potentially be optimal. A reference list specifies:
    - zeta[0]: the product with minimum price in the assortment
    - zeta[k] for k >= 1: the minimum level for attribute k in the assortment
    
    The key insight is that not all N * L^(K-1) combinations are feasible.
    By ranking products by price and iterating through possible reference
    levels, we can prune dominated combinations.
    
    See Section 4.2 of the paper (FindReferenceLists algorithm).
    
    Parameters
    ----------
    product_dict : dict
        Dictionary mapping product IDs to Product objects
    J_dict : dict
        Dictionary mapping (k, m) to the set of products with attribute k at level m
        J_dict[(k,m)] = {j : theta_{jk} = m}
    K : int
        Number of attributes (including price)
        
    Returns
    -------
    found_references : list
        List of dictionaries, each containing:
        - "ref": tuple representing the reference list (zeta)
        - "excluded": set of products that cannot be in the assortment
                     if this reference list is active
    '''
    global found_references
    found_references = []
    
    # Rank products by price (descending order)
    # We iterate from highest to lowest price as potential price references
    product_list = list(product_dict.keys())
    price_list = [product_dict[j].theta[0] for j in product_list]
    rank_by_price = [x for _,x in sorted(zip(price_list, product_list), reverse=True)]
    
    # For each product as potential price reference
    for ind in range(len(product_list)):
        # Initialize reference list with this product as price reference
        zeta = ["NA"]*K
        zeta[0] = rank_by_price[ind] # Product ID of price reference

        # Products with lower prices cannot be in assortment (they would become reference)
        excluded = set(rank_by_price[ind+1:])
        # The price reference must be in the assortment
        included = set([zeta[0]])
        
        if K > 1:
            # Recursively find valid reference levels for non-price attributes
            findReferenceList_Iter(1, zeta, included, excluded, product_dict, J_dict, K)
        else:
            # Only price attribute, so we're done
            found_references.append({"ref": tuple(zeta), "excluded": excluded})
            
    return found_references

def findReferenceList_Iter(k_next, zeta, included, excluded, product_dict, J_dict, K):
    '''
    Recursive helper to enumerate feasible reference lists for attributes k_next to K-1.
    
    For each non-price attribute k, we determine what reference levels are feasible
    given the products already included/excluded. A reference level m for attribute k
    means some product in the assortment has theta_{jk} = m, and no product has
    theta_{jk} < m.
    
    Parameters
    ----------
    k_next : int
        Next attribute index to process (1 <= k_next < K)
    zeta : list
        Partial reference list being constructed
    included : set
        Products that must be in the assortment
    excluded : set
        Products that cannot be in the assortment
    product_dict : dict
        Dictionary mapping product IDs to Product objects
    J_dict : dict
        Dictionary mapping (k, m) to set of products at level m for attribute k
    K : int
        Total number of attributes
    '''
    
    all_products = set(product_dict.keys())
    known_products = included | excluded
    unknown_products = all_products - known_products
    
    if unknown_products:
        # There are still products whose inclusion is undetermined
        # The minimum level among included products bounds possible reference levels
        min_level = min([product_dict[j].theta[k_next] for j in included])
        
        # Try each possible reference level from min_level down to 0
        for level_next in range(min_level,-1,-1):
            # Products that could provide this reference level (not yet excluded)
            possible_refs = J_dict[(k_next,level_next)] - excluded
            
            if (possible_refs == set()):
                # No product can provide this reference level
                continue
            else:
                # Update included set if only one product can provide this level
                if len(possible_refs) == 1:
                    new_included = included | possible_refs
                else:
                    new_included = included

                new_zeta = copy.deepcopy(zeta)
                new_zeta[k_next] = level_next

                # Products with better (lower) levels must be excluded
                # (otherwise they would become the reference)
                new_excluded = excluded
                for level_better in range(level_next-1,-1,-1):
                    new_excluded = new_excluded | J_dict[(k_next,level_better)]
                
                if k_next < K-1:
                    # Continue to next attribute
                    findReferenceList_Iter(k_next+1, new_zeta, new_included,
                                           new_excluded, product_dict, J_dict, K)
                else:
                    # All attributes processed, save this reference list
                    found_references.append({"ref": tuple(new_zeta), "excluded": new_excluded})
                    
    else:
        # All products are either included or excluded
        # Fill remaining attributes with minimum levels among included products
        for k in range(k_next,K):
            min_level = min([product_dict[j].theta[k] for j in included])
            #max_rank = max([rank_by_attr[k].index(j) for j in zeta[0:k_next]])
            zeta[k] = min_level
        found_references.append({"ref": tuple(zeta), "excluded": excluded})
            
    return


# =============================================================================
# MNL ASSORTMENT OPTIMIZATION
# =============================================================================

def solve_MNL_assortment_MIP(choice_model, product_dict):
    '''
    Solve the MNL assortment optimization problem via MILP.
    
    The MNL assortment problem maximizes expected profit:
    
        max_{S} sum_{j in S} margin_j * P_j(S)
        
    where P_j(S) = v_j / (v_0 + sum_{i in S} v_i) is the MNL choice probability.
    
    We use the Charnes-Cooper transformation to linearize this fractional program:
    - Let t = 1 / (v_0 + sum_{i in S} v_i)  (normalization variable)
    - Let u_j = t * x_j  (linearized probability × selection)
    
    The MILP formulation is:
    
        max  sum_j (margin_j * v_j) * u_j
        s.t. u_j <= t                          for all j  (u=0 if not selected)
             u_j <= x_j / v_0                  for all j  (upper bound)
             u_j >= t - (1-x_j) / v_0          for all j  (lower bound)
             v_0 * t + sum_j v_j * u_j = 1     (normalization)
             x_j in {0,1}, u_j >= 0, t >= 0
    
    Parameters
    ----------
    choice_model : MNLChoiceModel
        The MNL choice model with estimated parameters (beta_0, beta_price, beta_theta)
    product_dict : dict
        Dictionary mapping product IDs to Product objects
        
    Returns
    -------
    opt_assortment : list
        List of product IDs in the optimal assortment
    solve_time : float
        Time to solve the MILP in seconds
    '''
    N = len(product_dict)
    
    # Compute MNL attraction weights v_j = exp(V_j)
    v_dict = choice_model.getAttractionWeights(product_dict)

    # Compute revenue coefficients: margin_j * v_j
    rev_dict = {}
    for j in v_dict.keys(): 
        rev_dict[j] = product_dict[j].margin * v_dict[j]
        
    print('Start of creating model')
    
    # Create a CPLEX model
    prob = cplex.Cplex()
    prob.set_problem_type(prob.problem_type.MILP)
    prob.objective.set_sense(prob.objective.sense.maximize)
    
    # -------------------------------------------------------------------------
    # Decision Variables
    # -------------------------------------------------------------------------
    
    # t: normalization variable (continuous, >= 0)
    # t = 1 / (v_0 + sum of weights in assortment)
    prob.variables.add(types = [prob.variables.type.continuous],
                       names = ["t"], lb = [0])
    
    # u_j: linearized variable (continuous, >= 0)
    # u_j = t * x_j (product of normalization and selection)
    # Objective coefficient: margin_j * v_j
    prob.variables.add(types = [prob.variables.type.continuous]*len(v_dict),
                       names = ["u_" + str(j) for j in v_dict.keys()],
                       lb = [0]*len(v_dict),
                       obj = [rev_dict[j] for j in v_dict.keys()])
    
    # x_j: binary selection variable
    # x_j = 1 if product j is in assortment, 0 otherwise
    prob.variables.add(types = [prob.variables.type.binary]*N,
                       names = ["x_" + str(j) for j in product_dict.keys()])
    
    
    # -------------------------------------------------------------------------
    # Constraints (Charnes-Cooper linearization)
    # -------------------------------------------------------------------------
    
    # Constraint: u_j <= t (if x_j = 0, then u_j = 0)
    prob.linear_constraints.add(lin_expr = [cplex.SparsePair(ind = ["u_" + str(v), "t"],
                                                             val = [1.0, -1.0]
                                                             ) for v in v_dict.keys()],
                                senses = ["L"]*len(v_dict),
                                rhs = [0]*len(v_dict),
                                names = ["c6_" + str(v) for v in v_dict.keys()])
    
    # Constraint: u_j <= x_j / v_0 (upper bound on u_j)
    prob.linear_constraints.add(lin_expr = [cplex.SparsePair(ind = ["u_" + str(v), "x_" + str(v)],
                                                             val = [1.0, -1/choice_model.v0]
                                                             ) for v in v_dict.keys()],
                                senses = ["L"]*len(v_dict),
                                rhs = [0]*len(v_dict),
                                names = ["c7_" + str(v) for v in v_dict.keys()])
    
    # Constraint: u_j >= t - (1 - x_j) / v_0 (lower bound, active when x_j = 1)
    prob.linear_constraints.add(lin_expr = [cplex.SparsePair(ind = ["u_" + str(v), "x_" + str(v), "t"],
                                                             val = [1.0, -1/choice_model.v0, -1.0]
                                                             ) for v in v_dict.keys()],
                                senses = ["G"]*len(v_dict),
                                rhs = [-1/choice_model.v0]*len(v_dict),
                                names = ["c8_" + str(v) for v in v_dict.keys()])
    
    # Constraint: v_0 * t + sum_j v_j * u_j = 1 (normalization)
    prob.linear_constraints.add(lin_expr = [cplex.SparsePair(ind = ["t"] + ["u_" + str(v) for v in v_dict.keys()],
                                                             val = [choice_model.v0] + [v_dict[v] for v in v_dict.keys()])],
                                senses = ["E"],
                                rhs = [1.0],
                                names = ["c9"])
    
    print("Number of variables:", prob.variables.get_num())
    print("Number of constraints:", prob.linear_constraints.get_num())
    
    # -------------------------------------------------------------------------
    # Solve
    # -------------------------------------------------------------------------
    print('Start of solving')
    # log time
    start = time.time()
    
    prob.solve()

    # Check solution status (1 = optimal, 101 or 102 = optimal within tolerance)
    if (prob.solution.get_status() == 1) or (prob.solution.get_status() == 101) or (prob.solution.get_status() == 102):
        solve_time = time.time() - start
    
        print ("Objective value = ", prob.solution.get_objective_value())

        # Extract optimal assortment
        x = prob.solution.get_values(["x_" + j for j in product_dict.keys()])
        print(x)
        
        opt_assortment = []
        for j in product_dict.keys():
            if abs(prob.solution.get_values("x_" + j) - 1) < 1e-4:
                opt_assortment.append(j)
        opt_assortment.sort()
        
    else:
        print ("Solution status:", prob.solution.get_status())
        solve_time = 'Time limit exceed'
        opt_assortment = []
    
    print("Time elapsed (sec):", solve_time)
    return (opt_assortment, solve_time)
    

# =============================================================================
# CC ASSORTMENT OPTIMIZATION
# =============================================================================

def solve_CC_assortment_MIP_levels_preprocess(choice_model, product_dict):
    '''
    Solve the CC assortment optimization problem via MILP with preprocessing.
    
    The CC assortment problem is more complex than MNL because attraction weights
    depend on the reference point, which depends on the assortment itself:
    
        max_{S} sum_{j in S} margin_j * P_j(S | zeta(S))
        
    where zeta(S) is the reference list determined by S.
    
    We use a two-stage approach:
    1. Preprocessing: Enumerate all feasible reference lists using findReferenceList()
    2. MILP: Jointly optimize over assortment selection and reference list selection
    
    The MILP formulation extends the MNL formulation with additional variables:
    - z_zeta: binary variable indicating if reference list zeta is active
    - w_{j,zeta}: binary variable indicating product j is selected under reference zeta
    
    Key constraints ensure consistency between assortment and reference list:
    - If z_zeta = 1, the price reference product must be in assortment
    - If z_zeta = 1, some product at each reference level must be in assortment
    - If z_zeta = 1, no product with better attributes can be in assortment
    
    See Section 4 of the paper for the full MILP formulation.
    
    Parameters
    ----------
    choice_model : ChoiceModel
        The CC choice model with parameters (gamma, eta, a_dict, b_dict)
    product_dict : dict
        Dictionary mapping product IDs to Product objects
        
    Returns
    -------
    opt_assortment : list
        List of product IDs in the optimal assortment
    solve_time : float
        Time to solve the MILP in seconds
    num_feasible_lists : int
        Number of feasible reference lists found by preprocessing
    '''
    
    K = choice_model.getNumAttributes()
    N = len(product_dict)
    M = choice_model.getNumLevels()
    
    print("Number of products after pre-processing:", N)

    # -------------------------------------------------------------------------
    # Step 1: Build J_dict and find feasible reference lists
    # -------------------------------------------------------------------------
    
    # J_dict[(k,m)] = set of products whose attribute k is at level m
    J_dict = {}
    for (k,m) in itertools.product(range(1,K), range(M)):
        J_dict[(k,m)] = set([])
    for (k,i) in itertools.product(range(1,K), product_dict.keys()):
        m = product_dict[i].theta[k]
        J_dict[(k,m)].add(i)

    # Find all feasible reference lists
    findReferenceList(product_dict, J_dict, K)
    print("Max possible references:", N*M**(K-1))
    print("Total found references:", len(found_references))

    # Extract reference lists and exclusion sets
    Lset = [] # List of feasible reference lists
    Eset = [] # Corresponding exclusion sets
    for curr_dict in found_references:
        Lset.append(curr_dict["ref"])
        Eset.append(curr_dict["excluded"])
        
    # -------------------------------------------------------------------------
    # Step 2: Compute attraction weights for each (product, reference) pair
    # -------------------------------------------------------------------------
    
    v_dict = {}     # v_dict[(j, zeta)] = attraction weight of j under reference zeta
    rev_dict = {}   # rev_dict[(j, zeta)] = margin_j * v_{j,zeta}
    
    for reflist in Lset:
        mnl_weights = choice_model.getAttractionWeights(reflist, product_dict)
        for j in mnl_weights.keys():
            v_dict[(j,reflist)] = mnl_weights[j]
            rev_dict[(j,reflist)] = product_dict[j].margin*mnl_weights[j]
            
    print('Start of creating model')

    
    # -------------------------------------------------------------------------
    # Step 3: Build CPLEX model
    # -------------------------------------------------------------------------
    
    prob = cplex.Cplex()
    prob.set_problem_type(prob.problem_type.MILP)
    prob.objective.set_sense(prob.objective.sense.maximize)

    # -------------------------------------------------------------------------
    # Decision Variables
    # -------------------------------------------------------------------------
    
    # t: normalization variable (continuous, >= 0)
    prob.variables.add(types = [prob.variables.type.continuous],
                       names = ["t"], lb = [0])
    
    # u_{j,zeta}: linearized variable for each (product, reference) pair
    # Objective coefficient: margin_j * v_{j,zeta}
    prob.variables.add(types = [prob.variables.type.continuous]*len(v_dict),
                       names = ["u_" + str(j) for j in v_dict.keys()],
                       lb = [0]*len(v_dict),
                       obj = [rev_dict[j] for j in v_dict.keys()])
    
    # x_j: binary selection variable for each product
    prob.variables.add(types = [prob.variables.type.binary]*N,
                       names = ["x_" + str(j) for j in product_dict.keys()])
    
    # z_zeta: binary variable for each reference list
    # z_zeta = 1 if reference list zeta is active
    prob.variables.add(types = [prob.variables.type.binary]*len(Lset),
                       names = ["z_" + str(reflist) for reflist in Lset])
    
    # w_{j,zeta}: binary variable linking product selection to reference list
    # w_{j,zeta} = 1 if product j is selected AND reference list zeta is active
    prob.variables.add(types = [prob.variables.type.binary]*len(v_dict),
                       names = ["w_" + str(j) for j in v_dict.keys()])
    
    print('Finished adding variables')
    
    
    # -------------------------------------------------------------------------
    # Constraints
    # -------------------------------------------------------------------------
    
    # --- Reference list validity constraints ---
    
    # Constraint (C1a): z_zeta <= x_{zeta[0]} (price reference must be in assortment)
    prob.linear_constraints.add(lin_expr = [cplex.SparsePair(ind = ["z_" + str(reflist), "x_" + str(reflist[0])],
                                                             val = [1.0, -1.0]) for reflist in Lset],
                                senses = ["L"]*len(Lset),
                                rhs = [0]*len(Lset),
                                names = ["c1_" + str(reflist) + "0" for reflist in Lset])
    
    for ind in range(len(Lset)):
        reflist = Lset[ind]
        excluded = Eset[ind]
        
        # Constraint (C1b): z_zeta <= sum_{j: theta_{jk} = zeta[k]} x_j
        # (some product at reference level must be in assortment)
        prob.linear_constraints.add(lin_expr = [cplex.SparsePair(ind = ["z_" + str(reflist)] + ["x_" + str(j) for j in J_dict[(k,reflist[k])] - excluded],
                                                                 val = [1.0] + [-1.0]*len(J_dict[(k,reflist[k])] - excluded)
                                                                 ) for k in range(1,K)],
                                    senses = ["L"]*(K-1),
                                    rhs = [0]*(K-1),
                                    names = ["c1_" + str(reflist) + str(k) for k in range(1,K)])
    
        # Constraint (C2): z_zeta + x_j <= 1 for all j in excluded set
        # (products with better attributes cannot be in assortment if zeta is active)
        if len(excluded) > 0:
            prob.linear_constraints.add(lin_expr = [cplex.SparsePair(ind = ["z_" + str(reflist), "x_" + str(j)],
                                                                     val = [1.0, 1.0]
                                                                     ) for j in excluded],
                                        senses = ["L"]*len(excluded),
                                        rhs = [1.0]*len(excluded),
                                        names = ["c2_" + str(reflist) + j for j in excluded])
    
    # Constraint (C3): sum_zeta z_zeta = 1 (exactly one reference list is active)
    prob.linear_constraints.add(lin_expr = [cplex.SparsePair(ind = ["z_" + str(reflist) for reflist in Lset],
                                                             val = [1.0]*len(Lset)
                                                             )],
                                senses = ["E"],
                                rhs = [1.0],
                                names = ["c3"])

    # --- Linking constraints between x, z, and w ---
    
    # Constraint (C4): sum_zeta w_{j,zeta} = x_j (w links x and z)
    prob.linear_constraints.add(lin_expr = [cplex.SparsePair(ind = ["w_" + str((j,reflist)) for reflist in Lset] + ["x_" + str(j)],
                                                             val = [1.0]*len(Lset) + [-1.0]
                                                             ) for j in product_dict.keys()],
                                senses = ["E"]*N,
                                rhs = [0]*N,
                                names = ["c4_" + str(j) for j in product_dict.keys()])
    
    # Constraint (C5): w_{j,zeta} <= z_zeta (can only select j under zeta if zeta is active)
    prob.linear_constraints.add(lin_expr = [cplex.SparsePair(ind = ["w_" + str((j,reflist)), "z_" + str(reflist)],
                                                             val = [1.0, -1.0]
                                                             ) for (j,reflist) in itertools.product(product_dict.keys(), Lset)],
                                senses = ["L"]*N*len(Lset),
                                rhs = [0]*N*len(Lset),
                                names = ["c5_" + str(j) + str(reflist) for (j,reflist) in itertools.product(product_dict.keys(), Lset)])

    # --- Charnes-Cooper linearization constraints (same as MNL) ---
    
    # Constraint (C6): u_{j,zeta} <= t
    prob.linear_constraints.add(lin_expr = [cplex.SparsePair(ind = ["u_" + str(v), "t"],
                                                             val = [1.0, -1.0]
                                                             ) for v in v_dict.keys()],
                                senses = ["L"]*len(v_dict),
                                rhs = [0]*len(v_dict),
                                names = ["c6_" + str(v) for v in v_dict.keys()])
    
    # Constraint (C7): u_{j,zeta} <= w_{j,zeta} / v_0
    prob.linear_constraints.add(lin_expr = [cplex.SparsePair(ind = ["u_" + str(v), "w_" + str(v)],
                                                             val = [1.0, -1/choice_model.v0]
                                                             ) for v in v_dict.keys()],
                                senses = ["L"]*len(v_dict),
                                rhs = [0]*len(v_dict),
                                names = ["c7_" + str(v) for v in v_dict.keys()])
    
    # Constraint (C8): u_{j,zeta} >= t - (1 - w_{j,zeta}) / v_0
    prob.linear_constraints.add(lin_expr = [cplex.SparsePair(ind = ["u_" + str(v), "w_" + str(v), "t"],
                                                             val = [1.0, -1/choice_model.v0, -1.0]
                                                             ) for v in v_dict.keys()],
                                senses = ["G"]*len(v_dict),
                                rhs = [-1/choice_model.v0]*len(v_dict),
                                names = ["c8_" + str(v) for v in v_dict.keys()])
    
    # Constraint (C9): v_0 * t + sum_{j,zeta} v_{j,zeta} * u_{j,zeta} = 1 (normalization)
    prob.linear_constraints.add(lin_expr = [cplex.SparsePair(ind = ["t"] + ["u_" + str(v) for v in v_dict.keys()],
                                                             val = [choice_model.v0] + [v_dict[v] for v in v_dict.keys()])],
                                senses = ["E"],
                                rhs = [1.0],
                                names = ["c9"])
    
    print("Number of variables:", prob.variables.get_num())
    print("Number of constraints:", prob.linear_constraints.get_num())
    
    # set time limit to be 1 hour
    prob.parameters.tune.timelimit.set(3600)
    
    
    # -------------------------------------------------------------------------
    # Solve
    # -------------------------------------------------------------------------
    print('Start of solving')
    # log time for Section 6.1
    start = time.time()
    
    prob.solve()

    # Check solution status (1 = optimal, 101/102 = optimal within tolerance)
    if (prob.solution.get_status() == 1) or (prob.solution.get_status() == 101) or (prob.solution.get_status() == 102):
        solve_time = time.time() - start
    
        print ("Objective value = ", prob.solution.get_objective_value())

        # Extract optimal assortment
        x = prob.solution.get_values(["x_" + j for j in product_dict.keys()])
        print(x)
        
        opt_assortment = []
        for j in product_dict.keys():
            if abs(prob.solution.get_values("x_" + j) - 1) < 1e-4:
                opt_assortment.append(j)
        opt_assortment.sort()

        # Print active reference list
        z = prob.solution.get_values(["z_" + str(reflist) for reflist in Lset])
        print(z)
        
    else:
        print ("Solution status:", prob.solution.get_status())
        solve_time = 'Time limit exceed'
    
    print("Time elapsed (sec):", solve_time)
    return (opt_assortment, solve_time, len(Lset))



