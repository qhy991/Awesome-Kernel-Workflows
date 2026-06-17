# WarpSpeed correctness harness (project-owned)

WarpSpeed scaffolded this directory into your project because no harness
existed yet. **The search will not start until a human completes it.** The
harness is the single correctness ground truth for every experiment; agents
may run it but never modify it.

## Completion steps (human, once)

1. `problem_shapes.json` — replace the TODO example shapes with the canonical
   shapes for your kernel (used for both correctness and benchmarking).
2. `tolerances.json` — review every per-dtype `rtol`/`atol`, especially FP8.
   These values decide what counts as "correct" for the entire search.
3. `correctness.py` — implement `make_inputs()`, `run_reference()`,
   `run_impl()` (and `compare()` if the default allclose is wrong for your
   kernel). Keep the CLI contract and the deterministic `SEED` untouched.
4. Verify by hand on the baseline kernel binary:
   `python3 harness/correctness.py --impl <binary>` → exit 0 + JSON report,
   twice, identical results.
5. Lock it: `chmod -R a-w harness/` (WarpSpeed init re-applies this).

## Contracts the rest of WarpSpeed relies on

### Correctness CLI (this directory)

```
python3 harness/correctness.py --impl <binary> [--shape <name>]
```
- exit 0 iff all shapes pass, 1 on any failure, 2 on usage error
- single-line JSON report on stdout
- deterministic: same impl ⇒ same verdict, every time (seeded inputs)

### Benchmark binary (your build output, `binary_path`)

Every binary produced by `build_command` must self-time and support:

```
<binary> --reps N --warmup W [--shape <name>]
```
- prints exactly one `LAT_US=<float>` line per measured rep to stdout
- does its own warmup (W reps, unprinted) before the N measured reps
- `--shape` selects a shape from problem_shapes.json (default: `default`)

This contract is consumed by `bench_screen.sh`, `bench_confirm.sh`,
`calibrate.sh`, and `ncu_profile.sh`. GPU access always goes through
`gpu_run` — never run the binary bare.
