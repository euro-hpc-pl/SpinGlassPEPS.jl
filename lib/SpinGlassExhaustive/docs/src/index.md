
```@meta
Author = "Dariusz Kurzyk, Łukasz Pawela"
```

# Home

A julia package providing algorithm for solving the problem of finding the ground state of a spin glass system based on exhaustive search by GPU. 

The package includes algorithms:
- naive brute-force exhaustive search by GPU
- brute-force exhaustive search with bucket selection by GPU
- brute-force exhaustive search returning partial results by one kernel in GPU

## [References](@id refs)

[1] [Jałowiecki, K., Rams, M. M., & Gardas, B. (2021). Brute-forcing spin-glass problems with CUDA. Computer Physics Communications, 260, 107728.](https://arxiv.org/pdf/1904.03621.pdf)

[2] [Tao, M., Nakano, K., Ito, Y., Yasudo, R., Tatekawa, M., Katsuki, R., ... & Inaba, Y. (2020, May). A Work-Time Optimal Parallel Exhaustive Search Algorithm for the QUBO and the Ising model, with GPU implementation. In 2020 IEEE International Parallel and Distributed Processing Symposium Workshops (IPDPSW) (pp. 557-566). IEEE.](https://sci-hub.se/10.1109/ipdpsw50202.2020.00098)

[3] [Cook, C., Zhao, H., Sato, T., Hiromoto, M., & Tan, S. X. D. (2018). GPU based parallel Ising computing for combinatorial optimization problems in VLSI physical design. arXiv preprint arXiv:1807.10750.](https://arxiv.org/pdf/1807.10750.pdf)