# Assortment and Price Optimization Under a Multi-Attribute (Contextual) Choice Model

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the code and data required to reproduce the numerical experiments in:

> **"Assortment and Price Optimization Under a Multi-Attribute (Contextual) Choice Model"**  
> Sajjad Najafi, Stefanus Jasin, Joline Uichanco, and Jinglong Zhao  

## Overview

This paper studies assortment and pricing optimization under a contextual choice (CC) model where customer preferences depend on the relative positioning of products across multiple attributes. The repository includes:

- **Section 6.2**: Comparison of CC-optimal vs MNL-optimal assortments
- **Section 6.3**: MLA proxy heuristic for general CC models
- **Section EC5.5**: Algorithm 2 (Grid Search) performance analysis
- **Section EC6**: Joint assortment and pricing comparisons

## Repository Structure

```
├── README.md                     # This file
├── README.txt                    # Detailed documentation
├── requirements.txt              # Python dependencies
├── LICENSE                       # MIT License
│
├── scripts/                      # Python scripts
│   ├── main_section62_CC_vs_MNL.py
│   ├── main_section63_MLA_proxy.py
│   ├── wrapperClass.py
│   └── optimizationEngine.py
│
├── section62_instances/          # Section 6.2 parameter files
│   ├── gamma_equal/
│   ├── gamma0_smaller/
│   └── gamma0_larger/
│
├── section62_results/            # Section 6.2 output files
│   ├── gamma_equal/
│   ├── gamma0_smaller/
│   └── gamma0_larger/
│
├── section63_instances/          # Section 6.3 parameter files
│
├── section63_results/            # Section 6.3 output files
│
└── ecompanion_R/                 # E-companion R scripts
    ├── README_ecompanion.txt
    ├── GridSearch.R
    ├── ComparePrices.R
    ├── ComparePrices2.R
    ├── ComparePrices3.R
    ├── Analyzing_Outputs.R
    ├── Analyzing_Outputs2.R
    ├── Analyzing_Outputs3.R
    └── reference_outputs/
```

## Requirements

### Python (Sections 6.2 and 6.3)

- Python 3.8+
- NumPy, SciPy, Pandas, pylogit
- IBM CPLEX 12.10+ (requires license; [academic licenses available](https://www.ibm.com/academic/))

```bash
pip install numpy scipy pandas pylogit
```

### R (E-companion)

- R 4.0+
- Packages: `pracma`, `tidyverse`

```r
install.packages(c("pracma", "tidyverse"))
```

## Quick Start

### Section 6.2: CC vs MNL Comparison

```bash
cd scripts/
python main_section62_CC_vs_MNL.py
```

This generates instances and compares CC-optimal vs MNL-optimal assortments across three γ scenarios.

### Section 6.3: MLA Proxy Heuristic

```bash
cd scripts/
python main_section63_MLA_proxy.py
```

This tests the MLA structural approximation for CC models with diminishing sensitivity (η < 1).

### E-companion Experiments (R)

```bash
cd ecompanion_R/

# Create symbolic links to instance files
ln -s ../section62_instances/gamma_equal Instances
ln -s ../section62_instances/gamma0_smaller Instances2gamma0lessgammak
ln -s ../section62_instances/gamma0_larger Instances3gamma0greatergammak

# Run in R/RStudio
source("GridSearch.R")           # Section EC5.5
source("ComparePrices.R")        # Section EC6 (run with SGE_TASK_ID=1-9)
source("Analyzing_Outputs.R")    # Generate summary tables
```

## Output Files

| Experiment | Output | Paper Reference |
|------------|--------|-----------------|
| Section 6.2 (γ₀ = γₖ) | `section62_results/gamma_equal/*.csv` | Table EC.2 |
| Section 6.2 (γ₀ > γₖ) | `section62_results/gamma0_larger/*.csv` | Table EC.3 |
| Section 6.2 (γ₀ < γₖ) | `section62_results/gamma0_smaller/*.csv` | Table EC.4 |
| Section 6.3 | `section63_results/*_[eta]eta.csv` | Table EC.5 |
| Section EC5.5 | Console output (GridSearch.R) | Tables EC.7, EC.8 |
| Section EC6 (γ₀ = γₖ) | `ecompanion_R/reference_outputs/summary_table.csv` | Table EC.9 |
| Section EC6 (γ₀ > γₖ) | `ecompanion_R/reference_outputs/summary_table3.csv` | Table EC.10 |
| Section EC6 (γ₀ < γₖ) | `ecompanion_R/reference_outputs/summary_table2.csv` | Table EC.11 |

## Citation

If you use this code, please cite:

```bibtex
@article{najafi2025assortment,
  title={Assortment and Price Optimization Under a Multi-Attribute (Contextual) Choice Model},
  author={Najafi, Sajjad and Jasin, Stefanus and Uichanco, Joline and Zhao, Jinglong},
  journal={Operations Research},
  year={2025},
  note={Forthcoming}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions about this repository, please open an issue or contact Joline Uichanco (joline.uichanco@nyu.edu).
