# Release verification

The following checks were performed on the minimal core release, independently
of the original simulator and experiment directories.

## Environment

- Python 3.9.25
- NumPy 1.26.4
- PyTorch 2.4.0+cu121
- Windows; all verification computations ran on CPU with one PyTorch thread

These versions identify the tested environment. CUDA is not required by the
demo, and dependency ranges in `pyproject.toml` allow other supported builds.

## Checks

| Check | Result |
| --- | --- |
| Build a wheel without downloading dependencies | Passed |
| Install the wheel to a separate directory | Passed |
| Execute the test suite against the separately installed package | 7 tests passed |
| One-step training followed by joint multi-stage training | Passed |
| Save and reload a trained checkpoint | Passed |
| Multi-stage forecast with finite outputs | Passed |
| Closed-loop SABS decisions across all six toy stages | Passed |
| Next-budget feasibility and conditional support checks | Passed |
| Fixed-seed search repeatability | Passed |
| Fallback on invalid model predictions | Passed |
| Reject inconsistent data and overlapping train/validation days | Passed |
| Command-line forecast evaluation and online budget search | Passed |

The demo was run both from the source directory and from the separately
installed wheel, with seed 7 and 20 optimizer steps per phase. Both runs produced
the same training losses and toy decisions in this environment. This is a
software execution check, not an experimental result from the paper.

Verification-generated logs and checkpoints are not included in the release.
