'''
main_section62_CC_vs_MNL.py

Reproducibility script for Section 6.2: CC vs Misspecified MNL Comparison

This script implements the numerical experiments comparing the Contextual 
Concavity (CC) model against a misspecified Multinomial Logit (MNL) model.
The key research question is: what is the profit loss from ignoring 
contextual effects when making assortment decisions?

EXPERIMENTAL DESIGN
-------------------
The experiment uses the Multi-attribute Loss Aversion (MLA) specification
of the CC model (η = 1 for all attributes) and varies the loss aversion
parameters (γ) across three scenarios:

  Scenario 1 (gamma_equal):    γ₀ = γ_k ∈ [1, 2]
      - Equal sensitivity to price losses and quality gains
      
  Scenario 2 (gamma0_smaller): γ₀ ∈ [0.1, 1], γ_k ∈ [1, 2]  
      - Weaker price loss aversion, stronger quality gain seeking
      - MNL misspecification is more severe
      
  Scenario 3 (gamma0_larger):  γ₀ ∈ [1, 2], γ_k ∈ [0.1, 1]
      - Stronger price loss aversion, weaker quality gain seeking
      - Closer to standard MNL behavior

For each scenario and (N, K) combination:
  - N ∈ {5, 10, 20}: number of products
  - K ∈ {1, 3, 5}: number of non-price attributes
  - L = 3: number of discrete attribute levels
  - 100 random instances per configuration

WORKFLOW
--------
1. generate_and_save_model_parameters():
   - Generate random CC model parameters (a_j, b_j, γ, η)
   - Simulate 1000 choice observations from the true CC model
   - Estimate MNL parameters via Maximum Likelihood (pylogit)
   - Save both CC and MNL parameters to CSV files

2. optimize_CC_model():
   - Read CC parameters and solve CC-optimal assortment via MILP
   - Compute expected profit under true CC model
   - Save results including optimal assortment and profit

3. optimize_MNL_model():
   - Read MNL parameters and solve MNL-optimal assortment via MILP
   - Evaluate MNL-optimal assortment under TRUE CC model
   - Save results for profit gap computation

PROFIT GAP CALCULATION
----------------------
The profit gap measures the loss from using the misspecified MNL model:

    Profit Gap = (CC_profit - MNL_profit) / CC_profit × 100%

where:
  - CC_profit: expected profit from CC-optimal assortment under true CC model
  - MNL_profit: expected profit from MNL-optimal assortment under true CC model

See Section 6.2 of the paper for detailed results and discussion.

Author: Joline Uichanco
Created: Apr 9, 2025
'''

import wrapperClass as wc
import numpy as np
import optimizationEngine as oe
from scipy import stats
import os, csv, shutil, itertools, random
import pandas as pd
import pylogit as pl
from collections import OrderedDict


# =============================================================================
# CONFIGURATION
# =============================================================================

# The three gamma scenarios to test
GAMMA_SCENARIOS = ['gamma_equal', 'gamma0_smaller', 'gamma0_larger']


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_gamma_vec(scenario_name, K):
    """
    Generate gamma and eta vectors for a given scenario.
    
    Parameters:
    -----------
    scenario_name : str
        One of 'gamma_equal', 'gamma0_smaller', 'gamma0_larger'
    K : int
        Number of non-price attributes
    
    Returns:
    --------
    gamma_vec : list
        Context sensitivity parameters [gamma_0, gamma_1, ..., gamma_K]
    eta_vec : list
        Curvature parameters [eta_0, eta_1, ..., eta_K], all ones for MLA
    """
    if scenario_name == 'gamma_equal':
        # Scenario 1: Equal sensitivity to price and quality
        # γ₀ = γ_k ∈ [1, 2]
        gamma_val = np.random.uniform(1, 2)
        gamma_vec = [gamma_val] * (K + 1)
    elif scenario_name == 'gamma0_smaller':
        # Scenario 2: Weaker price loss aversion, stronger quality gain seeking
        # γ₀ ∈ [0.1, 1], γ_k ∈ [1, 2]
        # This creates larger MNL misspecification
        gamma_vec = [np.random.uniform(0.1, 1)] + [np.random.uniform(1, 2) for _ in range(K)]
    elif scenario_name == 'gamma0_larger':
        # Scenario 3: Stronger price loss aversion, weaker quality gain seeking
        # γ₀ ∈ [1, 2], γ_k ∈ [0.1, 1]
        # This is closer to standard MNL behavior
        gamma_vec = [np.random.uniform(1, 2)] + [np.random.uniform(0.1, 1) for _ in range(K)]
    else:
        raise ValueError(f"Unknown scenario: {scenario_name}")

    # MLA specification: η = 1 for all attributes (linear contextual effects)
    eta_vec = [1] * (K + 1)
    
    return gamma_vec, eta_vec


# =============================================================================
# STEP 1: GENERATE PARAMETERS AND ESTIMATE MNL
# =============================================================================

def generate_and_save_model_parameters():
    '''
    Generate CC model parameters and estimate MNL parameters for all scenarios.
    
    This function implements the data generation process:
    
    1. For each instance, generate random CC model parameters:
       - Prices uniformly spaced in [0.1, 1.0]
       - Costs yielding margins between 25% and 90%
       - Non-price attributes correlated with price (higher price → higher quality)
       - Utility intercepts a_j ~ U[0.2, 0.5]
       - Price sensitivities b_j ~ U[2.0, 5.0]
       - Gamma and eta parameters based on scenario
    
    2. Simulate choice data from the true CC model:
       - 1000 choice observations per instance
       - Random choice sets of size 3 to N
       - Randomized prices and attributes for estimation
       - Choice probabilities computed from CC model
    
    3. Estimate MNL parameters via Maximum Likelihood:
       - Uses pylogit package for MLE
       - Estimates intercept, price coefficient, and attribute coefficients
       - Saves estimated parameters for later optimization
    
    Output files (per scenario):
    - section62_instances/{scenario}/CC_parameters_{N}prod_{K}attr.csv
    - section62_instances/{scenario}/MNL_parameters_{N}prod_{K}attr.csv
    '''

    # Experimental parameters
    num_instances = 100     # Number of random instances per (N, K) combination
    Ns = [5, 10, 20]        # Number of products
    Ks = [1, 3, 5]          # Number of non-price attributes
    L = 3                   # Number of discrete attribute levels (0, 0.5, 1)
    
    for scenario_name in GAMMA_SCENARIOS:
        print(f"\n{'='*60}")
        print(f"Processing scenario: {scenario_name}")
        print(f"{'='*60}")
        
        # Create scenario directories if they don't exist
        os.makedirs(f'./section62_instances/{scenario_name}', exist_ok=True)
        
        for (N, K) in itertools.product(Ns, Ks):
            print(f"\n--- N={N}, K={K} ---")

            # Output file paths
            fname = f'./section62_instances/{scenario_name}/CC_parameters_{str(N).zfill(2)}prod_{K}attr.csv'
            fname_mnl = f'./section62_instances/{scenario_name}/MNL_parameters_{str(N).zfill(2)}prod_{K}attr.csv'
            
            with open(fname, "w") as f1, open(fname_mnl, "w") as f2:
                writer1 = csv.writer(f1)
                writer2 = csv.writer(f2)

                # ---------------------------------------------------------
                # CSV Headers
                # ---------------------------------------------------------
                # CC parameters header
                cc_header = ['Instance', 'N', 'K', 'L'] + \
                            ['c.P' + str(j+1) for j in range(N)] +         # Costs
                            ['p.P' + str(j+1) for j in range(N)] +          # Prices
                            ['theta' + str(k+1) + '.P' + str(j+1)           # Non-price attributes
                             for (k,j) in itertools.product(range(K),range(N))] +
                            ['a.P' + str(j+1) for j in range(N)] +          # Utility intercepts
                            ['b.' + str(j+1) for j in range(N)] +           # Price sensitivities
                            ['gamma0'] + ['gamma' + str(k+1) for k in range(K)] +   # Gamma parameters
                            ['eta0'] + ['eta' + str(k+1) for k in range(K)]         #Eta parameters
                writer1.writerow(cc_header)
                
                mnl_header = ['Instance', 'N', 'K', 'L'] + \
                             ['c.P' + str(j+1) for j in range(N)] +     # Costs
                             ['p.P' + str(j+1) for j in range(N)] +     # Prices
                             ['theta' + str(k+1) + '.P' + str(j+1)      # Non-price attributes
                              for (k,j) in itertools.product(range(K),range(N))] +
                             ['beta_0'] + ['beta_p'] +                  # Intercept and price coefficients
                             ['beta_theta' + str(k+1) for k in range(K)]    # Attribute coefficients
                writer2.writerow(mnl_header)

                # ---------------------------------------------------------
                # Generate instances
                # ---------------------------------------------------------
                
                curr_ins = 0
                while curr_ins < num_instances:

                    # =====================================================
                    # Part A: Generate CC model parameters
                    # =====================================================
                    
                    # Generate prices uniformly spaced in [0.1, 1.0]
                    price_list = np.linspace(0.1, 1, num=N).tolist()
                    
                    # Generate costs yielding margins between 25% and 90%
                    cost_vec = [min(price_list[j]*.9, price_list[j]*np.random.uniform(0.25,0.9)) for j in range(N)]
                    margin_list = [price_list[j] - cost_vec[j] for j in range(N)]
                    
                    # Sort products by decreasing margin (higher margin = lower index)
                    Z = sorted(zip(margin_list, price_list), reverse=True)
                    price_vec = [0]*N
                    for j in range(N):
                        (margin, price) = Z[j]
                        cost_vec[j] = price - margin
                        price_vec[j] = price 
                    
                    # Generate non-price attributes with price-quality correlation
                    # Higher-priced products are more likely to have higher quality
                    theta_vec = []
                    for k in range(K):
                        for j in range(N):
                            xk = np.linspace(0, 1.0, num=L).tolist()    # Levels: [0, 0.5, 1]
                            # Probability weights: higher price → higher quality
                            pk = [0.5*(1-price_vec[j]), 0.5, 0.5*price_vec[j]]
                            custm = stats.rv_discrete(name='choice', values=([1,2,3], pk))
                            ind = custm.rvs()
                            theta_vec.append(xk[ind-1])
                    
                    # Generate utility intercepts a_j ~ U[0.2, 0.5]
                    a_vec = np.random.uniform(0.2, 0.5, N).tolist()

                    # Generate price sensitivities b_j ~ U[2.0, 5.0]
                    b_vec = np.random.uniform(2.0, 5.0, N).tolist()
                    
                    # Generate gamma and eta based on scenario
                    gamma_vec, eta_vec = get_gamma_vec(scenario_name, K)

                    # Assemble CC parameter row
                    row = [curr_ins, N, K, L] + cost_vec + price_vec + theta_vec + a_vec + b_vec + gamma_vec + eta_vec
                    
                    
                    # =====================================================
                    # Part B: Simulate choice data from true CC model
                    # =====================================================
                    
                    num_individuals = 1000  # Number of choice observations

                    # Temporary file for choice data (used by pylogit)
                    choice_data_file = './section62_instances/CC_model_choice_historical_data.csv'
                    
                    with open(choice_data_file, 'w', newline='') as fchoice:
                        writer_choice = csv.writer(fchoice)

                        # Choice data header
                        row_choice = ['id', 'alt', 'choice', 'price'] + ['theta' + str(k+1) for k in range(K)]
                        writer_choice.writerow(row_choice)
                    
                        curr_indiv = 0
                        no_choice_chosen = False    # Track if no-purchase was ever chosen
                        
                        while True:
                            # Generate random choice set (subset of products)
                            choice_set = random.sample(range(1, N+1), random.randint(3, N))
                            
                            # Randomize prices and attributes for this observation
                            # (This creates variation for MNL estimation)
                            price_vec_rnd = np.random.choice(price_list, N).tolist()
                            theta_vec_rnd = np.random.choice(np.linspace(0, 1, num=L).tolist(), N*K).tolist()
                    
                            # Compute reference point (minimum values in choice set)
                            ref_list = [min([price_vec_rnd[j-1] for j in choice_set])] + \
                                       [min([theta_vec_rnd[k*N + j-1] for j in choice_set]) for k in range(K)]
                    
                            # Compute CC choice probabilities
                            v0 = 1  # No-purchase utility
                            v_set = []
                            for j in choice_set:
                                # Contextual utility: a_j - b_j*p_j + M_j
                                # where M_j = -γ₀(p_j - p_min)^η₀ + Σ_k γ_k(θ_jk - θ_min,k)^η_k
                                utility = a_vec[j-1] - b_vec[j-1]*price_vec_rnd[j-1] - \
                                          gamma_vec[0]*(price_vec_rnd[j-1] - ref_list[0])**eta_vec[0] + \
                                          np.sum([gamma_vec[k+1]*(theta_vec_rnd[k*N + j-1] - ref_list[k+1])**eta_vec[k+1] for k in range(K)])
                                v_set.append(np.exp(utility))

                            # Choice probabilities (MNL form with context-dependent utilities)
                            pk = ([v0] + v_set)/(v0 + np.sum(v_set))

                            # Simulate choice
                            xk = list(range(1, len(pk)+1))
                            custm = stats.rv_discrete(name='choice', values=(xk, pk))
                            ind = custm.rvs()
                            if ind > 1:
                                choice = choice_set[ind-2] # Chosen product
                            else:
                                choice = 0 # No-purchase
                                no_choice_chosen = True

                            # Write choice data (one row per alternative in choice set)
                            # Row for no-purchase option
                            writer_choice.writerow([curr_indiv, 0, int(0==choice)] + [0]*(K+1))

                            # Rows for products in choice set
                            for j in choice_set:
                                row_choice = [curr_indiv, j, int(j==choice), price_vec_rnd[j-1]] + \
                                             [theta_vec_rnd[k*N + j-1] for k in range(K)]
                                writer_choice.writerow(row_choice)
                    
                            # Continue until we have enough observations AND at least one no-purchase
                            if (curr_indiv >= num_individuals) and no_choice_chosen:
                                break
                    
                        print(f'Instance {curr_ins}: {curr_indiv} individuals')

                    # =====================================================
                    # Part C: Estimate MNL parameters via MLE
                    # =====================================================
                    
                    df = pd.read_csv("./section62_instances/CC_model_choice_historical_data.csv")
                    
                    # Get alternative IDs from data
                    alt_ids = sorted(df["alt"].unique())
                    ref_alt = 0  # Reference alternative (no-purchase)

                    # Create dummy columns for alternative-specific intercepts
                    for j in alt_ids:
                        if j != ref_alt:
                            df[f"intercept_{j}"] = (df["alt"] == j).astype(int)
                    
                    # Build MNL specification using pylogit
                    # Specification: V_j = β₀ + β_p * price + Σ_k β_k * θ_jk
                    cc_spec = OrderedDict()
                    
                    # Alternative-specific intercepts (relative to no-purchase)
                    for j in alt_ids:
                        if j != ref_alt:
                            cc_spec[f"intercept_{j}"] = [j]
                            
                    # Price coefficient (generic across alternatives)
                    cc_spec["price"] = alt_ids

                    # Attribute coefficients (generic across alternatives)
                    for k in range(1, K + 1):
                        cc_spec[f"theta{k}"] = [alt_ids]
                    
                    # Create and estimate MNL model
                    mnl_model = pl.create_choice_model(
                        data=df,
                        alt_id_col="alt",
                        obs_id_col="id",
                        choice_col="choice",
                        specification=cc_spec,
                        model_type="MNL"
                    )
                    
                    try:
                        # Fit MNL via maximum likelihood estimation
                        mnl_model.fit_mle(np.zeros(len(cc_spec)))
                        mnl_model.print_summaries()

                        # Extract estimated coefficients
                        # Note: We use a single intercept (average of alt-specific ones)
                        # and generic price/attribute coefficients
                        row_mnl = [curr_ins, N, K, L] + cost_vec + price_vec + theta_vec + \
                                  [mnl_model.coefs[0], mnl_model.coefs[1]] + \  # β₀, β_p
                                  [mnl_model.coefs[2+k] for k in range(K)]      # β_θ
                        writer2.writerow(row_mnl)
                    
                        # Save MNL parameters (only if MNL estimation succeeded)
                        writer1.writerow(row)
                        curr_ins += 1
                        
                    except Exception as e:
                        print(f"MLE failed for instance {curr_ins}: {e}")
                        # Skip this instance and try again
                        continue
    return


# =============================================================================
# STEP 2: OPTIMIZE CC MODEL
# =============================================================================


def optimize_CC_model():
    '''
    Solve CC assortment optimization for all scenarios.
    
    For each instance:
    1. Read CC parameters from section62_instances/{scenario}/
    2. Construct CC choice model
    3. Solve CC assortment optimization via MILP
    4. Evaluate optimal assortment to get expected profit
    5. Save results to section62_results/{scenario}/
    
    Output columns:
    - Instance, N, K, L: Instance identifiers
    - Exp Profit: Expected profit under CC model
    - x1, x2, ...: Binary indicators for products in optimal assortment
    - Cardinality: Number of products in optimal assortment
    - Consumer Surplus: Expected consumer surplus
    - Num Feasible Lists: Number of reference lists after preprocessing
    - Num Max Lists: Maximum possible reference lists (N × L^K)
    - Num Ref Products: Number of "reference-only" products (included for context)
    - Ref for Attr0, ...: Reference product for each attribute
    '''
    
    Ns = [5, 10, 20]  # number of products
    Ks = [1, 3, 5]    # number of non-price attributes
    L = 3             # number of levels
    
    for scenario_name in GAMMA_SCENARIOS:
        print(f"\n{'='*60}")
        print(f"Optimizing CC model for scenario: {scenario_name}")
        print(f"{'='*60}")
        
        # Create results directory if it doesn't exist
        os.makedirs(f'./section62_results/{scenario_name}', exist_ok=True)
        
        for (N, K) in itertools.product(Ns, Ks):
            print(f"\n--- N={N}, K={K} ---")

            # File paths
            fname = f'./section62_instances/{scenario_name}/CC_parameters_{str(N).zfill(2)}prod_{K}attr.csv'
            fname_out = f'./section62_results/{scenario_name}/CC_opt_assortment_{str(N).zfill(2)}prod_{K}attr.csv'
            
            with open(fname, 'r') as f1, open(fname_out, 'w') as f2:
                
                writer = csv.writer(f2)
                
                # Results header
                row_out = ['Instance', 'N', 'K', 'L', 'Exp Profit'] + \
                          ['x' + str(j+1) for j in range(N)] + \
                          ['Cardinality', 'Consumer Surplus', 'Num Feasible Lists', 'Num Max Lists', 'Num Ref Products'] + \
                          ['Ref for Attr' + str(k) for k in range(K+1)]
                writer.writerow(row_out)
                
                csvreader = csv.reader(f1)
                header = next(csvreader)  # Skip header
                
                # Process each instance
                for row in csvreader:

                    # -------------------------------------------------
                    # Construct product dictionary and choice model
                    # -------------------------------------------------
                    
                    product_dict = {}
                    a_dict = {}
                    b_dict = {}
                    
                    for j in range(N):
                        prod_id = "P" + str(j+1).zfill(2)

                        # Extract attributes: theta[0] = price, theta[k] = level for attr k
                        theta = [0]*(K+1)
                        theta[0] = float(row[N+4+j])    # Price
                        for k in range(K):
                            # Convert continuous [0, 0.5, 1] to discrete levels [0, 1, 2] (to be normalized by max_level)
                            theta[k+1] = int(float(row[2*N+4+k*N+j])*2)

                        # Margin = price - cost
                        margin = theta[0] - float(row[4+j])
                        product_dict[prod_id] = wc.Product(prod_id, margin, theta)

                        # Utility parameters
                        a_dict[prod_id] = float(row[2*N + 4 + K*N + j + 1])
                        b_dict[prod_id] = float(row[2*N + 4 + K*N + N + j + 1])
                    
                    # CC model parameters
                    v0 = 1  # No-purchase weight
                    max_level = {k+1: 0.5 for k in range(K)}  # Normalization for levels

                    # Extract gamma and eta vectors
                    gamma = [float(row[2*N + 4 + K*N + 2*N])] + \
                            [float(row[2*N + 4 + K*N + 2*N + k + 1]) for k in range(K)]
                    eta = [float(row[2*N + 4 + K*N + 2*N + K + 1])] + \
                          [float(row[2*N + 4 + K*N + 2*N + K + 1 + k + 1]) for k in range(K)]

                    # Construct choice model
                    choice_model = wc.ChoiceModel(L, max_level, gamma, eta, a_dict, b_dict, v0)

                    # -------------------------------------------------
                    # Solve CC assortment optimization
                    # -------------------------------------------------
                    
                    (opt_assortment, solve_time, num_feas_lists) = \
                                     oe.solve_CC_assortment_MIP_levels_preprocess(
                                         choice_model, product_dict)
                    
                    print(f"Instance {row[0]}: {opt_assortment}")

                    # -------------------------------------------------
                    # Analyze optimal assortment
                    # -------------------------------------------------
                    
                    # Count "reference-only" products
                    # These are products not in the contiguous high-margin set
                    # but included to improve the reference point
                    continuous_set = True
                    num_ref_products = 0
                    avg_price_refs = 0
                    
                    for j in range(N):
                        if continuous_set:
                            if ('P' + str(j+1).zfill(2) in opt_assortment):
                                continue
                            else:
                                continuous_set = False
                        else:
                            if ('P' + str(j+1).zfill(2) in opt_assortment):
                                num_ref_products += 1
                                avg_price_refs += product_dict['P' + str(j+1).zfill(2)].margin
                                
                    if avg_price_refs > 0: 
                        avg_price_refs = avg_price_refs/num_ref_products

                    # Evaluate optimal assortment
                    (exp_profit, prob_dict, consumer_surplus, ref_products) = \
                        choice_model.evaluateAssortment(opt_assortment, product_dict)

                    # -------------------------------------------------
                    # Write results
                    # -------------------------------------------------
                    
                    row_out = row[0:4] + [exp_profit]

                    # Binary indicators for products in assortment
                    for j in range(N):
                        prod_id = "P" + str(j+1).zfill(2)
                        if prod_id in opt_assortment:
                            row_out.append(1)
                        else:
                            row_out.append(0)
                            
                    row_out = row_out + [len(opt_assortment), consumer_surplus, num_feas_lists, 
                                         N*(L**K), num_ref_products] + ref_products
                    writer.writerow(row_out)


# =============================================================================
# STEP 3: OPTIMIZE MNL MODEL AND EVALUATE UNDER TRUE CC
# =============================================================================

def optimize_MNL_model():
    '''
    Solve MNL assortment optimization and evaluate under true CC model.
    
    This is the key comparison step. For each instance:
    1. Read estimated MNL parameters
    2. Solve MNL-optimal assortment via MILP
    3. Evaluate MNL-optimal assortment under the TRUE CC model
    4. Save results for profit gap computation
    
    The profit gap is computed as:
        Gap = (CC_profit - MNL_profit) / CC_profit × 100%
    
    where MNL_profit is the expected profit when using the MNL-optimal
    assortment but customers actually follow the CC model.
    
    Output columns:
    - Instance, N, K, L: Instance identifiers
    - Exp Profit: Expected profit of MNL-optimal assortment under TRUE CC model
    - x1, x2, ...: Binary indicators for products in MNL-optimal assortment
    - Cardinality: Number of products in optimal assortment
    - Consumer Surplus: Expected consumer surplus under true CC model
    '''
    
    Ns = [5, 10, 20]  # number of products
    Ks = [1, 3, 5]    # number of non-price attributes
    L = 3             # number of levels
    
    for scenario_name in GAMMA_SCENARIOS:
        print(f"\n{'='*60}")
        print(f"Optimizing MNL model for scenario: {scenario_name}")
        print(f"{'='*60}")
        
        # Create results directory if it doesn't exist
        os.makedirs(f'./section62_results/{scenario_name}', exist_ok=True)
        
        for (N, K) in itertools.product(Ns, Ks):
            print(f"\n--- N={N}, K={K} ---")
            
            # -------------------------------------------------
            # Load true CC models for evaluation
            # -------------------------------------------------
            fname_cc = f'./section62_instances/{scenario_name}/CC_parameters_{str(N).zfill(2)}prod_{K}attr.csv'
            true_CC_models = {} # Will map instance ID to CC model
            
            with open(fname_cc, 'r') as f1:
                csvreader = csv.reader(f1)
                header = next(csvreader)
                
                for row in csvreader:
                    a_dict = {}
                    b_dict = {}
                    for j in range(N):
                        prod_id = "P" + str(j+1).zfill(2)
                        theta = [0]*(K+1)
                        theta[0] = float(row[N+4+j])
                        for k in range(K):
                            theta[k+1] = int(float(row[2*N+4+k*N+j])*2)
                        margin = theta[0] - float(row[4+j])
                        
                        a_dict[prod_id] = float(row[2*N + 4 + K*N + j + 1])
                        b_dict[prod_id] = float(row[2*N + 4 + K*N + N + j + 1])
                    
                    v0 = 1
                    max_level = {}
                    for k in range(K):
                        max_level[k+1] = 0.5
                    gamma = [float(row[2*N + 4 + K*N + 2*N])] + \
                            [float(row[2*N + 4 + K*N + 2*N + k + 1]) for k in range(K)]
                    eta = [float(row[2*N + 4 + K*N + 2*N + K + 1])] + \
                          [float(row[2*N + 4 + K*N + 2*N + K + 1 + k + 1]) for k in range(K)]

                    # Store true CC model indexed by instance ID
                    true_CC_models[row[0]] = wc.ChoiceModel(L, max_level, gamma, eta, a_dict, b_dict, v0)
            
            # -------------------------------------------------
            # Optimize MNL and evaluate under true CC
            # -------------------------------------------------
            
            fname_mnl = f'./section62_instances/{scenario_name}/MNL_parameters_{str(N).zfill(2)}prod_{K}attr.csv'
            fname_out = f'./section62_results/{scenario_name}/MNL_opt_assortment_{str(N).zfill(2)}prod_{K}attr.csv'
            
            with open(fname_mnl, 'r') as f1, open(fname_out, 'w') as f2:
                
                writer = csv.writer(f2)
                # Results header
                row_out = ['Instance', 'N', 'K', 'L', 'Exp Profit'] + \
                          ['x' + str(j+1) for j in range(N)] + \
                          ['Cardinality', 'Consumer Surplus']
                writer.writerow(row_out)
                
                csvreader = csv.reader(f1)
                header = next(csvreader)
                
                for row in csvreader:
                    
                    # -------------------------------------------------
                    # Construct product dictionary
                    # -------------------------------------------------
                    
                    product_dict = {}
                    for j in range(N):
                        prod_id = "P" + str(j+1).zfill(2)
                        theta = [0]*(K+1)
                        theta[0] = float(row[N+4+j])
                        for k in range(K):
                            theta[k+1] = float(row[2*N+4+k*N+j])
                        margin = theta[0] - float(row[4+j])
                        product_dict[prod_id] = wc.Product(prod_id, margin, theta)

                    # -------------------------------------------------
                    # Construct MNL choice model with estimated parameters
                    # -------------------------------------------------
                    
                    v0 = 1
                    beta_0 = float(row[4 + 2*N + K*N])
                    beta_price = float(row[5 + 2*N + K*N])
                    beta_theta = [0]*K
                    for k in range(K):
                        beta_theta[k] = float(row[6 + 2*N + K*N + k])
                    choice_model = wc.MNLChoiceModel(beta_0, beta_price, beta_theta, v0)

                    # -------------------------------------------------
                    # Solve MNL-optimal assortment
                    # -------------------------------------------------
                    
                    (opt_assortment, solve_time) = oe.solve_MNL_assortment_MIP(choice_model, product_dict)
                    print(f"Instance {row[0]}: {opt_assortment}")
                    
                    # -------------------------------------------------
                    # Evaluate MNL-optimal assortment under TRUE CC model
                    # -------------------------------------------------
                    
                    # This is the key step: what profit do we get if we use
                    # the MNL-recommended assortment but customers actually
                    # behave according to the CC model?
                    (exp_profit, prob_dict, consumer_surplus, ref_products) = \
                        true_CC_models[row[0]].evaluateAssortment(opt_assortment, product_dict)


                    # -------------------------------------------------
                    # Write results
                    # -------------------------------------------------
                                        
                    row_out = row[0:4] + [exp_profit]
                    for j in range(N):
                        prod_id = "P" + str(j+1).zfill(2)
                        if prod_id in opt_assortment:
                            row_out.append(1)
                        else:
                            row_out.append(0)
                    row_out = row_out + [len(opt_assortment), consumer_surplus]
                    writer.writerow(row_out)

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    '''
    Run the complete Section 6.2 experiment pipeline.
    
    Expected runtime: 7-8 hours for all three scenarios
    
    After completion, compute profit gaps by comparing:
    - section62_results/{scenario}/CC_opt_assortment_*.csv (CC_profit)
    - section62_results/{scenario}/MNL_opt_assortment_*.csv (MNL_profit)
    
    Profit Gap = (CC_profit - MNL_profit) / CC_profit × 100%
    '''
    
    # Step 1: Generate CC parameters and estimate MNL
    generate_and_save_model_parameters()

    # Step 2: Solve CC-optimal assortment
    optimize_CC_model()

    # Step 3: Solve MNL-optimal and evaluate under true CC
    optimize_MNL_model()
