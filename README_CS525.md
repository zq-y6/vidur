### Project-Specific Files

- sarathi-serve
- data/profiling/network/a100_hgx

### Profiling ALL-REDUCE on A Real Machine

- Install Pytorch and Ray
- Pull the Sarathi submodule
- Start Ray by `ray start --head`
- Run `./cc_profile.sh`
- stop Ray by `ray stop`
