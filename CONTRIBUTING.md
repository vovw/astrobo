# Contributing

Install the package with `python -m pip install -e '.[test]'` and run
`python -m pytest tests -q` from this repository.

Package checks run without CUDA dependencies. Motion contract tests require torch
and the compatible cuRobo checkout described in [the usage guide](docs/usage.md).
Those tests use mocks and CPU tensors; passing them does not validate GPU motion quality.

Keep the public API small. Include focused tests for behavior changes, preserve
joint names and units, and never return trajectories from unsuccessful solver results.
Do not commit credentials, camera recordings, or machine-specific rig configuration.
