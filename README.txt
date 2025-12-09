================================================================================
REPRODUCIBILITY PACKAGE
Assortment and Price Optimization Under a Multi-Attribute (Contextual) Choice Model
================================================================================

This package contains the Python scripts and data structures needed to 
reproduce the numerical experiments in Sections 6.2 and 6.3 of the paper.

================================================================================
1. SOFTWARE REQUIREMENTS
================================================================================

Python Version: 3.8 or higher

Required Python Packages:
- numpy (>=1.20)
- scipy (>=1.7)
- pandas (>=1.3)
- pylogit (>=1.0)

Optimization Solver:
- IBM CPLEX (version 12.10 or higher)
- CPLEX Python API (cplex package)

Note: CPLEX requires a valid license. Academic licenses are available free 
of charge from IBM. See: https://www.ibm.com/academic/

To install Python dependencies (excluding CPLEX):
    pip install numpy scipy pandas pylogit

================================================================================
2. FOLDER STRUCTURE
================================================================================

reproducibility_package/
├── README.txt
├── requirements.txt
├── scripts/
│   ├── main_section62_CC_vs_MNL.py     # Section 6.2: CC vs MNL comparison
│   ├── main_section63_MLA_proxy.py     # Section 6.3: MLA proxy experiments
│   ├── wrapperClass.py                 # Product and choice model classes
│   └── optimizationEngine.py           # MILP solvers using CPLEX
│
├── section62_instances/                # Section 6.2 parameter files
│   ├── gamma_equal/                    # Scenario (i): γ₀ = γ_k
│   ├── gamma0_smaller/                 # Scenario (ii): γ₀ < γ_k
│   └── gamma0_larger/                  # Scenario (iii): γ₀ > γ_k
│
├── section62_results/                  # Section 6.2 output files
│   ├── gamma_equal/
│   ├── gamma0_smaller/
│   └── gamma0_larger/
│
├── section63_instances/                # Section 6.3 parameter files
│
├── section63_results/                  # Section 6.3 output files
│
└── ecompanion_R/                       # E-companion experiments (R scripts)
    ├── README_ecompanion.txt           # Documentation for R scripts
    ├── GridSearch.R                    # Section EC5.5: Algorithm 2
    ├── ComparePrices*.R                # Section EC6: Pricing comparisons
    ├── Analyzing_Outputs*.R            # Summary table generation
    └── Results*/                       # Output folders

================================================================================
3. SECTION 6.2: CC vs MNL COMPARISON
================================================================================

Purpose: Compare CC-optimal assortments against MNL-optimal assortments
         to measure the profit loss from ignoring contextual effects.

Script: main_section62_CC_vs_MNL.py

GAMMA SCENARIOS (η = 1 for all, i.e., MLA specification):
- gamma_equal:     γ₀ = γ_k ∈ [1,2]
- gamma0_smaller:  γ₀ ∈ [0.1,1], γ_k ∈ [1,2]
- gamma0_larger:   γ₀ ∈ [1,2], γ_k ∈ [0.1,1]

WORKFLOW:
1. generate_and_save_model_parameters()
   - Generates 100 instances per (N,K) combination for each scenario
   - N ∈ {5, 10, 20}, K ∈ {1, 3, 5}
   - Simulates 1000 choice observations from true CC model
   - Estimates MNL parameters via MLE using pylogit

2. optimize_CC_model()
   - Solves CC assortment optimization via MILP

3. optimize_MNL_model()
   - Solves MNL assortment optimization
   - Evaluates MNL-optimal assortment under TRUE CC model

TO RUN:
    cd scripts/
    python main_section62_CC_vs_MNL.py

PROFIT GAP CALCULATION:
    profit_gap = (CC_profit - MNL_profit) / CC_profit × 100%

================================================================================
4. SECTION 6.3: MLA PROXY FOR GENERAL CC
================================================================================

Purpose: Test whether the MLA structural property (η=1) serves as a good
         heuristic when the true model has diminishing sensitivity (η < 1).

Script: main_section63_MLA_proxy.py

KEY INSIGHT: This is a STRUCTURAL comparison, not an estimation exercise.
- We do NOT estimate MLA parameters from data
- We fix CC parameters and vary only η
- Compare: (i) true CC-optimal vs (ii) best MLA-structured assortment

PARAMETERS:
- N ∈ {5, 10, 20}, K ∈ {1, 3, 5}
- γ₀ ∈ [0.1, 0.2], γ_k ∈ [1, 2]
- η ∈ {0.01, 0.1, 0.5} (true diminishing sensitivity)

WORKFLOW:
1. generate_and_save_model_parameters()
   - Generates 100 instances per (N,K) combination
   - η values NOT stored (varied at optimization time)

2. optimize_CC_model()
   - Solves CC assortment with true η ∈ {0.01, 0.1, 0.5}
   - Records optimal profit under true CC model

3. optimize_MLA_heuristic()
   - Solves assortment assuming η=1 (MLA structure)
   - Evaluates MLA-optimal assortment under TRUE CC model (η < 1)

TO RUN:
    cd scripts/
    python main_section63_MLA_proxy.py

PROFIT GAP CALCULATION:
    profit_gap = (CC_profit - MLA_profit) / CC_profit × 100%

Where:
- CC_profit: from CC_opt_assortment_*_[eta]eta.csv
- MLA_profit: from MLA_heuristic_assortment_*_[eta]eta.csv

================================================================================
5. PARAMETER CONFIGURATIONS
================================================================================

SECTION 6.2 Product Parameters:
- Prices: Equally spaced in [0.1, 1.0]
- Margins: Random in [25%, 90%] of price
- Products indexed in decreasing margin order
- Non-price attributes: θ_jk ∈ {0, 0.5, 1} (L=3 levels)
- a_j ~ Uniform[0.2, 0.5]
- b_j ~ Uniform[2.0, 5.0]
- v₀ = 1 (no-purchase utility)

SECTION 6.3 Product Parameters:
- Same as Section 6.2
- γ₀ ∈ [0.1, 0.2] (weaker loss aversion range)
- γ_k ∈ [1, 2]

================================================================================
6. OUTPUT FILE FORMAT
================================================================================

SECTION 6.2 FILES:

CC_opt_assortment_*.csv:
- Instance, N, K, L: Identifiers
- Exp Profit: Expected profit under CC-optimal assortment
- x1, x2, ...: Binary indicators for products in assortment
- Cardinality, Consumer Surplus, Num Feasible Lists, etc.

MNL_opt_assortment_*.csv:
- Exp Profit: MNL assortment profit evaluated under TRUE CC model

SECTION 6.3 FILES:

CC_opt_assortment_*_[eta]eta.csv:
- True CC-optimal assortment and profit for given η

MLA_heuristic_assortment_*_[eta]eta.csv:
- MLA-optimal assortment (solved with η=1)
- Exp Profit: Evaluated under true CC model (with η < 1)

================================================================================
7. EXPECTED RUNTIME
================================================================================

On a standard workstation (Intel i7, 16GB RAM):

SECTION 6.2 (all three scenarios):
- Parameter generation + MNL estimation: ~2-3 hours
- CC optimization: ~3-5 hours
- MNL optimization: ~1-2 hours
- Total: ~7-8 hours

SECTION 6.3 (all η values):
- Parameter generation: ~15 minutes
- CC optimization (3 η values): ~3-4 hours
- MLA heuristic (3 η values): ~3-4 hours
- Total: ~7-8 hours

================================================================================
8. TROUBLESHOOTING
================================================================================

CPLEX not found:
- Ensure CPLEX is installed and the Python API is configured
- Add CPLEX to your Python path

pylogit estimation fails (Section 6.2 only):
- Occasionally MLE may not converge
- The script catches exceptions and retries

Running a subset of experiments:
- Section 6.2: Edit GAMMA_SCENARIOS list (line 26)
- Section 6.3: Edit etas list in optimize_CC_model() and optimize_MLA_heuristic()

================================================================================
9. E-COMPANION EXPERIMENTS (R SCRIPTS)
================================================================================

The ecompanion_R/ folder contains R scripts for experiments in Sections EC5.5 
and EC6 of the E-companion. See ecompanion_R/README_ecompanion.txt for details.

SECTION EC5.5: Algorithm 2 Performance (Tables EC.4 and EC.5)
- Script: GridSearch.R
- Tests grid search algorithm across different precision values (ε)

SECTION EC6: Joint Assortment and Pricing Comparisons (Tables EC.7-EC.9)
- Scripts: ComparePrices.R, ComparePrices2.R, ComparePrices3.R
- Compare CC vs MNL pricing optimization
- IMPORTANT: These scripts require instance files generated by the Python 
  scripts in this package (Section 6.2)

DATA PIPELINE (Python → R):
The R pricing scripts read from the same instance folders used by Section 6.2:
- section62_instances/gamma_equal/      →  Instances/
- section62_instances/gamma0_smaller/   →  Instances2gamma0lessgammak/
- section62_instances/gamma0_larger/    →  Instances3gamma0greatergammak/

Before running R scripts, create symbolic links in ecompanion_R/:
    cd ecompanion_R/
    ln -s ../section62_instances/gamma_equal Instances
    ln -s ../section62_instances/gamma0_smaller Instances2gamma0lessgammak
    ln -s ../section62_instances/gamma0_larger Instances3gamma0greatergammak

================================================================================
10. CONTACT
================================================================================

For questions about this reproducibility package, please contact the authors.

================================================================================
