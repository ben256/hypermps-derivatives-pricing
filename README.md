# hypermps-derivatives-pricing

Main code in experiments/hypernetwork2 and experiments/black_scholes2

> **_NOTE (11/08/25):_** Check `experiments/nerual_netowrks/README.md` for in-depth instructions on the NN part of things

> **_NOTE:_** I've modified the cross function in TNTorch to implement early stopping if the error stagnates, so to run most of the code that will need to be changed. Haven't got around to forking the repo and doing it properly, so currently doing it the badly and replacing the module code after installing. The updated function can be found in `utils/mods` and it needs to go in `~/python3.9/site-packages/tntorch/cross.py`.