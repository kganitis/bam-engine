# Contributing to BAM Engine

Thank you for your interest in BAM Engine! Contributions are welcome. Bug
fixes, documentation, tests, performance work, new extensions, and help with
the known limitations on the [roadmap](https://bamengine.org/roadmap/) are all
appreciated.

BAM Engine is a research framework validated against quantitative targets from
Delli Gatti et al. (2011), and its API is still stabilizing ahead of v1.0, so a
little coordination goes a long way.

## Before you start

- **Small, self-contained fixes** (bugs, docs, typos): open a pull request
  directly.
- **Larger or model-behavior changes** (new features, market mechanics,
  anything that could shift simulation output against the validation targets or
  change the public API): please
  [open an issue](https://github.com/kganitis/bam-engine/issues) first so we can
  align on the approach before you invest time.

## Ways to contribute without writing code

- **Bug reports & feature requests**: open an issue on the
  [issue tracker](https://github.com/kganitis/bam-engine/issues).
- **Documentation feedback**: spotted an error, unclear explanation, or missing
  topic? Let us know via an issue.
- **Testing & reporting**: run simulations and report unexpected behavior.
- **Questions & discussions**: ask on the issue tracker.
- **Spread the word**: star the repository or share the project with colleagues.

## Development setup

```bash
git clone https://github.com/kganitis/bam-engine.git
cd bam-engine
pip install -e ".[dev]"
```

Verify your environment before making changes:

```bash
pytest
ruff format . && ruff check --fix . && mypy
```

## Quality bar

Every pull request is expected to keep the test suite and checks green:

- **Tests pass**: `pytest`. The suite maintains **99% coverage**, so new code
  needs tests.
- **Formatting and linting**: `ruff format . && ruff check --fix .`
- **Type checking**: `mypy` runs clean.
- **Docstrings**: NumPy-style (see the
  [development guide](https://bam-engine.readthedocs.io/en/latest/development/index.html)).
- **Determinism**: all randomness must go through `sim.rng`, never `np.random`
  directly, so results stay reproducible across seeds.

## Pull request process

1. Fork the repository and create a feature branch off `main`
   (`git checkout -b feature/my-feature`).
1. Make your change and add tests.
1. Run the full test suite and the quality checks above.
1. Commit with clear, descriptive messages.
1. Push and open a pull request against `main`, linking the related issue.

A maintainer will review as time allows. BAM Engine is maintained by a single
person on a best-effort basis, so please be patient with review turnaround.

## Code of Conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md). By
participating, you are expected to uphold it.

## Licensing

BAM Engine is released under the [MIT License](LICENSE). By submitting a
contribution, you agree that your work is licensed under the same terms. No
separate contributor agreement is required.
