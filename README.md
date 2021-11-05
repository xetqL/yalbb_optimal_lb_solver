# YALBB Experiments Repository

This repository contains examples and real experiments carried out with the [YALBB](https://github.com/xetqL/yalbb) framework. 
It should be use as a starting point to build new experiments or to reproduce the experiments in these scientific papers:
- Submitted to JPDC: https://arxiv.org/pdf/2104.01688.pdf
- Submitted to JOCS: https://arxiv.org/pdf/2108.11099

# Dependencies
- YALBB
- NORCB (for paper#2 experiments)
- Zoltan 
- C++ >= 17
- MPI 
- cmake

# Download and install 
```shell 
git clone https://github.com/xetqL/yalbb_optimal_lb_solver --recurse-submodules && cd yalbb_optimal_lb_solver && mkdir build && cd build && cmake -DCMAKE_BUILD_TYPE=Release ..
```

# Structure
This repository is structured as follows:
- `experiments` contains actual experiments (either tutorial or paper experiments). 
  - `JOCS` contains the experiments that appears in the paper submitted to Journal of Computational Science
  - `JPDC` contains the experiments that have been carried out for the paper submitted to the Journal of Parallel and Distributed Computing
- `src` contains the functions defined and being used in YALBB (ex: load balancing functions, initial conditions, etc.)

