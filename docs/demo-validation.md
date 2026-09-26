# Demo validation

The release checks focus on functionality and portable installation. They do
not reproduce benchmark scores or establish retrieval accuracy.

## Release check on 2026-09-26

On Linux with Python 3.11.15, all 33 core and CLI tests passed. The same 33 tests
also passed against the installed wheel from outside the source checkout.
Both a new wheel-only environment and a separate source-distribution environment
ran the demo and portable four-query verification successfully.

Neither clean environment needed Torch, Transformers, or the original integration
repository. The generated synthetic memory JSON matched the previous core
implementation byte for byte. Runtime versions are recorded in
`constraints-demo.txt`. These results validate this small functional fixture,
not arbitrary videos or paper benchmark performance.

## Automated coverage

- Motion and static intervals, missing observations, and last-seen semantics.
- Five-view references, stored visual crops, and feature-space mismatch rejection.
- Time, scene, and spatial-region filters, plus trajectory compression.
- Portable memory reload without source observations or model files.
- JSON schema, invalid geometry, and asset path traversal rejection.
- Command-line demo, inspect, build, query, and verify entry points.
- Refusal to overwrite an existing demo or memory output directory.
- Import and execution without Torch or the original integration repository.

The standalone demo retains the existing LTE, query, storage, and five-view
algorithms. The command-line wrapper, package layout, and minimal frame contracts
are adapted for this repository. Generated memory is compared against the
previous implementation's synthetic fixture as part of release preparation.

## Clean-install check

Build a wheel with `python -m build`, install it into a new virtual environment,
and run the following from a directory outside the source checkout:

```bash
stmem demo --output demo
stmem verify --memory demo/memory/memory.json
```

The expected verification result is `portable_reload: passed`, with nonempty
results from all four query paths. A separate check installs from the source
distribution. CI repeats installation, core tests, and the wheel smoke test.

Only synthetic inputs are part of this release. No model weights, private
footage, cached research outputs, or benchmark evaluation assets are included.
