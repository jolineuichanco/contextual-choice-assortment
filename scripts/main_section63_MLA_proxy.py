'''
main_section63_MLA_proxy.py

Reproducibility script for Section 6.3: MLA as a Proxy for General CC

This script tests whether the Multi-attribute Loss Aversion (MLA) structural
property (η = 1) serves as a good heuristic when the true model exhibits
diminishing sensitivity (η < 1).

MOTIVATION
----------
The MLA special case (η = 1) has attractive structural properties:
- The optimal assortment has a "quasi-nested" structure (Theorem 1)
- This enables efficient preprocessing and faster optimization

However, empirical evidence suggests consumers often exhibit diminishing
sensitivity (η < 1), meaning the marginal impact of attribute differences
decreases as the differences grow larger.

RESEARCH QUESTION
-----------------
If we optimize using the MLA assumption (η = 1) but consumers actually
have diminishing sensitivity (η < 1), how much profit do we lose?

EXPERIMENTAL DESIGN
-------------------
Unlike Section 6.2 (which involves estimation), this is a pure structural
comparison with known parameters:

  - γ₀ ∈ [0.1, 0.2]: Weak price loss aversion
  - γ_k ∈ [1, 2]: Stronger quality gain seeking
  - η ∈ {0.01, 0.1, 0.5}: True diminishing sensitivity parameter

For each instance and η value:
  1. CC-Optimal: Solve with TRUE η, get optimal profit
  2. MLA-Heuristic: Solve with η = 1, evaluate under TRUE η

The profit gap measures the cost of using the MLA approximation:

    Profit Gap = (CC_profit - MLA_profit) / CC_profit × 100%

WORKFLOW
--------
1. generate_and_save_model_parameters():
   - Generate 100 random instances per (N, K) combination
   - Save CC parameters (γ values, but NOT η since it's varied later)

2. optimize_CC_model():
   - For each η ∈ {0.01, 0.1, 0.5}, solve CC-optimal assortment
   - This gives the "best possible" profit for each η

3. optimize_MLA_heuristic():
   - Solve assortment using η = 1 (MLA assumption)
   - Evaluate under TRUE η to get actual profit
   - Compare to CC-optimal to measure profit loss

KEY DIFFERENCE FROM SECTION 6.2
-------------------------------
Section 6.2: Compares CC vs misspecified MNL (involves MLE estimation)
Section 6.3: Compares true CC vs MLA heuristic (structural comparison only)

In Section 6.3, we know the true parameters—we're testing whether the
MLA structure provides a good approximation, not whether we can estimate well.

See Section 6.3 of the paper for detailed results and discussion.

Author: Joline Uichanco
Created: Apr 9, 2025
'''

import wrapperClass as wc
import numpy as np
import optimizationEngine as oe
from scipy import stats
import os, csv, itertools

# =============================================================================
# STEP 1: GENERATE CC MODEL PARAMETERS
# =============================================================================


def generate_and_save_model_parameters():
    '''
    Generate CC model parameters for Section 6.3 experiments.
    
    This function generates random instances with:
    - Prices uniformly spaced in [0.1, 1.0]
    - Costs yielding margins between 25% and 90%
    - Non-price attributes correlated with price
    - Utility parameters a_j ~ U[0.2, 0.5], b_j ~ U[2.0, 5.0]
    - Context sensitivity: γ₀ ∈ [0.1, 0.2], γ_k ∈ [1, 2]
    
    NOTE: η values are NOT stored since they are varied at optimization time.
    The same instance is solved multiple times with different η values.
    
    Output files:
    - section63_instances/CC_parameters_{N}prod_{K}attr.csv
    '''
    
    num_instances = 100 # Number of random instances per (N, K) combination
    Ns = [5, 10, 20]  # number of products
    Ks = [1, 3, 5]    # number of non-price attributes
    L = 3             # number of levels
    
    # Create instances directory if it doesn't exist
    os.makedirs('./section63_instances', exist_ok=True)
    
    for (N, K) in itertools.product(Ns, Ks):
        print(f"\n--- Generating N={N}, K={K} ---")
    
        fname = f'./section63_instances/CC_parameters_{str(N).zfill(2)}prod_{K}attr.csv'
        
        with open(fname, "w") as f1:
            writer1 = csv.writer(f1)

            # ---------------------------------------------------------
            # CSV Header
            # ---------------------------------------------------------
            # Note: No η columns since η is varied at optimization time
            
            writer1.writerow(['Instance', 'N', 'K', 'L'] + 
                             ['c.' + str(j+1) for j in range(N)] + 
                             ['p.' + str(j+1) for j in range(N)] +
                             ['theta' + str(k+1) + '.' + str(j+1) for (k,j) in itertools.product(range(K),range(N))] + 
                             ['a.' + str(j+1) for j in range(N)] + 
                             ['b.' + str(j+1) for j in range(N)] +
                             ['gamma0'] + ['gamma' + str(k+1) for k in range(K)]
                            )

            # ---------------------------------------------------------
            # Generate instances
            # ---------------------------------------------------------
            
            for curr_ins in range(num_instances):
                
                # generate price vector between $0.1 to $1
                price_list = np.linspace(0.1, 1, num=N).tolist()
                
                # Generate costs yielding margins between 25% and 90%
                # Costs are rounded to nearest 0.25 for cleaner values
                cost_vec = [min(price_list[j]*.9, round(price_list[j]*np.random.uniform(0.25,0.9)*4)/4) for j in range(N)]
                margin_list = [price_list[j] - cost_vec[j] for j in range(N)]
                
                # sort in decreasing margin
                Z = sorted(zip(margin_list, price_list), reverse=True)
                price_vec = [0]*N
                for j in range(N):
                    (margin, price) = Z[j]
                    cost_vec[j] = price - margin
                    price_vec[j] = price 
                
                # generate non-price attributes, either 0, 0.5 or 1
                # Higher-priced products probabilistically have better attributes
                theta_vec = []
                for k in range(K):
                    for j in range(N):
                        xk = np.linspace(0, 1.0, num=L).tolist()
                        pk = [0.5*(1-price_vec[j]), 0.5, 0.5*price_vec[j]]
                        custm = stats.rv_discrete(name='choice', values=([1,2,3], pk))
                        ind = custm.rvs()
                        theta_vec.append(xk[ind-1])
                
                # Generate utility intercepts a_j ~ U[0.2, 0.5]
                a_vec = np.random.uniform(0.2, 0.5, N).tolist()

                # Generate price sensitivities b_j ~ U[2.0, 5.0]
                b_vec = np.random.uniform(2.0, 5.0, N).tolist()
                
                # Generate γ parameters (weaker loss aversion than Section 6.2)
                # γ₀ ∈ [0.1, 0.2]: Weak price loss aversion
                # γ_k ∈ [1, 2]: Stronger quality gain seeking
                gamma_vec = [np.random.uniform(0.1, 0.2)] + [np.random.uniform(1, 2) for _ in range(K)]

                # Assemble row (no η values)
                row = [curr_ins, N, K, L] + cost_vec + price_vec + theta_vec + a_vec + b_vec + gamma_vec
                
                writer1.writerow(row)
                
        print(f"Saved {num_instances} instances to {fname}")
    
    return

# =============================================================================
# STEP 2: SOLVE CC-OPTIMAL WITH TRUE η
# =============================================================================


def optimize_CC_model():
    '''
    Solve CC assortment optimization with true η values.
    
    For each η ∈ {0.01, 0.1, 0.5}:
    - Constructs CC model with the TRUE diminishing sensitivity parameter
    - Solves for optimal assortment via MILP
    - Records optimal expected profit
    
    This gives the "best possible" profit that could be achieved if we
    knew the true η value and optimized accordingly.
    
    Output files (one per η value):
    - section63_results/CC_opt_assortment_{N}prod_{K}attr_{η}eta.csv
    
    Output columns:
    - Instance, N, K, L: Instance identifiers
    - Exp Profit: Expected profit from CC-optimal assortment
    - x1, x2, ...: Binary indicators for products in optimal assortment
    - Cardinality: Number of products in optimal assortment
    - Consumer Surplus: Expected consumer surplus
    '''

    # Experimental parameters
    Ns = [5, 10, 20]  # number of products
    Ks = [1, 3, 5]    # number of non-price attributes
    etas = [0.01, 0.1, 0.5]  # diminishing sensitivity parameters
    L = 3             # number of levels
    
    # Create results directory if it doesn't exist
    os.makedirs('./section63_results', exist_ok=True)
    
    for (N, K, eta_val) in itertools.product(Ns, Ks, etas):
        print(f"\n--- CC Optimal: N={N}, K={K}, η={eta_val} ---")
        
        fname = f'./section63_instances/CC_parameters_{str(N).zfill(2)}prod_{K}attr.csv'
        fname_out = f'./section63_results/CC_opt_assortment_{str(N).zfill(2)}prod_{K}attr_{eta_val}eta.csv'
        
        with open(fname, 'r') as f1, open(fname_out, 'w') as f2:
            
            writer = csv.writer(f2)

            # Results header
            row_out = ['Instance', 'N', 'K', 'L', 'Exp Profit'] + \
                      ['x' + str(j+1) for j in range(N)] + \
                      ['Cardinality', 'Consumer Surplus']
            writer.writerow(row_out)
            
            csvreader = csv.reader(f1)
            header = next(csvreader)

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

                    # Extract attributes
                    theta = [0]*(K+1)
                    theta[0] = float(row[N+4+j]) # Price
                    for k in range(K):
                        # Convert continuous [0, 0.5, 1] to discrete levels [0, 1, 2] (normalized using max_level)
                        theta[k+1] = int(float(row[2*N+4+k*N+j])*2)
                        
                    margin = theta[0] - float(row[4+j])
                    product_dict[prod_id] = wc.Product(prod_id, margin, theta)

                    # Utility parameters
                    a_dict[prod_id] = float(row[2*N + 4 + K*N + j])
                    b_dict[prod_id] = float(row[2*N + 4 + K*N + N + j])
                
                # CC model parameters
                v0 = 1
                max_level = {k+1: 0.5 for k in range(K)}

                # Extract γ from CSV
                gamma = [float(row[2*N + 4 + K*N + 2*N])] + \
                        [float(row[2*N + 4 + K*N + 2*N + k + 1]) for k in range(K)]

                # Use TRUE η value (the parameter we're testing)
                eta = [eta_val] * (K+1)

                # Construct CC choice model with true parameters
                choice_model = wc.ChoiceModel(L, max_level, gamma, eta, a_dict, b_dict, v0)

                # -------------------------------------------------
                # Solve CC-optimal assortment
                # -------------------------------------------------
                
                (opt_assortment, solve_time, num_feas_lists) = oe.solve_CC_assortment_MIP_levels_preprocess(
                    choice_model, product_dict)
                print(f"Instance {row[0]}: {opt_assortment}")

                # Evaluate optimal assortment
                (exp_profit, prob_dict, consumer_surplus, ref_products) = \
                    choice_model.evaluateAssortment(opt_assortment, product_dict)

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
# STEP 3: SOLVE MLA HEURISTIC AND EVALUATE UNDER TRUE CC
# =============================================================================

def optimize_MLA_heuristic():
    '''
    Solve MLA heuristic (η = 1) and evaluate under true CC model.
    
    This is the key experiment. For each true η ∈ {0.01, 0.1, 0.5}:
    
    1. OPTIMIZATION: Solve assortment problem using MLA assumption (η = 1)
       - Uses the tractable MLA structure for optimization
       - Ignores the true diminishing sensitivity
    
    2. EVALUATION: Compute profit of MLA-optimal assortment under TRUE model
       - Customers actually behave with diminishing sensitivity (η < 1)
       - This profit will be ≤ the CC-optimal profit
    
    The profit gap measures the cost of the MLA approximation:
    
        Profit Gap = (CC_profit - MLA_profit) / CC_profit × 100%
    
    If the gap is small, the MLA structural property provides a good
    heuristic even when diminishing sensitivity is present.
    
    Output files (one per true η value):
    - section63_results/MLA_heuristic_assortment_{N}prod_{K}attr_{η}eta.csv
    
    Output columns:
    - Instance, N, K, L: Instance identifiers
    - Exp Profit: Expected profit of MLA-optimal assortment under TRUE CC model
    - x1, x2, ...: Binary indicators for products in MLA-optimal assortment
    - Cardinality: Number of products in optimal assortment
    - Consumer Surplus: Expected consumer surplus under true CC model
    '''

    # Experimental parameters
    Ns = [5, 10, 20]  # number of products
    Ks = [1, 3, 5]    # number of non-price attributes
    etas = [0.01, 0.1, 0.5]  # true η values (MLA uses η=1)
    L = 3             # number of levels
    
    # Create results directory if it doesn't exist
    os.makedirs('./section63_results', exist_ok=True)
    
    for (N, K, eta_val) in itertools.product(Ns, Ks, etas):
        print(f"\n--- MLA Heuristic: N={N}, K={K}, true η={eta_val} ---")

        # File paths
        fname = f'./section63_instances/CC_parameters_{str(N).zfill(2)}prod_{K}attr.csv'
        fname_out = f'./section63_results/MLA_heuristic_assortment_{str(N).zfill(2)}prod_{K}attr_{eta_val}eta.csv'
        
        with open(fname, 'r') as f1, open(fname_out, 'w') as f2:
            
            writer = csv.writer(f2)

            # Results header
            row_out = ['Instance', 'N', 'K', 'L', 'Exp Profit'] + \
                      ['x' + str(j+1) for j in range(N)] + \
                      ['Cardinality', 'Consumer Surplus']
            writer.writerow(row_out)
            
            csvreader = csv.reader(f1)
            header = next(csvreader)

            # Process each instance
            for row in csvreader:
                
                # -------------------------------------------------
                # Construct product dictionary
                # -------------------------------------------------
                
                product_dict = {}
                a_dict = {}
                b_dict = {}
                for j in range(N):
                    prod_id = "P" + str(j+1).zfill(2)
                    theta = [0]*(K+1)
                    theta[0] = float(row[N+4+j])
                    for k in range(K):
                        theta[k+1] = int(float(row[2*N+4+k*N+j])*2)
                    margin = theta[0] - float(row[4+j])
                    product_dict[prod_id] = wc.Product(prod_id, margin, theta)
                    
                    a_dict[prod_id] = float(row[2*N + 4 + K*N + j])
                    b_dict[prod_id] = float(row[2*N + 4 + K*N + N + j])
                
                # -------------------------------------------------
                # Construct TWO choice models
                # -------------------------------------------------
                
                v0 = 1
                max_level = {k+1: 0.5 for k in range(K)}
                    
                gamma = [float(row[2*N + 4 + K*N + 2*N])] + \
                        [float(row[2*N + 4 + K*N + 2*N + k + 1]) for k in range(K)]
                
                # MODEL 1: MLA model (η = 1) for OPTIMIZATION
                # This is the heuristic - we pretend η = 1 to get tractable structure
                eta_mla = [1] * (K+1)
                mla_choice_model = wc.ChoiceModel(L, max_level, gamma, eta_mla, a_dict, b_dict, v0)
                
                # MODEL 2: True CC model (η < 1) for EVALUATION
                # This is how customers actually behave
                eta_true = [eta_val] * (K+1)
                true_choice_model = wc.ChoiceModel(L, max_level, gamma, eta_true, a_dict, b_dict, v0)
                
                # -------------------------------------------------
                # Solve using MLA heuristic (η = 1)
                # -------------------------------------------------
                
                (opt_assortment, solve_time, num_feas_lists) = oe.solve_CC_assortment_MIP_levels_preprocess(
                    mla_choice_model, product_dict)
                print(f"Instance {row[0]}: {opt_assortment}")
                
                # -------------------------------------------------
                # Evaluate MLA-optimal assortment under TRUE CC model
                # -------------------------------------------------
                
                # Key step: The assortment was chosen assuming η = 1,
                # but customers actually have η < 1 (diminishing sensitivity)
                (exp_profit, prob_dict, consumer_surplus, ref_products) = \
                    true_choice_model.evaluateAssortment(opt_assortment, product_dict)

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
    Run the complete Section 6.3 experiment pipeline.
    
    Expected runtime: 7-8 hours for all η values
    
    After completion, compute profit gaps by comparing:
    - section63_results/CC_opt_assortment_*_{η}eta.csv (CC_profit with true η)
    - section63_results/MLA_heuristic_assortment_*_{η}eta.csv (MLA_profit under true η)
    
    For each instance and η:
        Profit Gap = (CC_profit - MLA_profit) / CC_profit × 100%
    
    Small gaps indicate MLA is a good heuristic despite diminishing sensitivity.
    '''
    
    # Step 1: Generate CC parameters (without η)
    generate_and_save_model_parameters()

    # Step 2: Solve CC-optimal with true η values
    optimize_CC_model()

    # Step 3: Solve MLA heuristic (η=1) and evaluate under true η
    optimize_MLA_heuristic()
