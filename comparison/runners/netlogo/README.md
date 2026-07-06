# NetLogo BAM Runner (Plan E)

Drives the third-party Platas BAM model as a **non-blocking cross-language
reference** in the `comparison/` benchmark. NetLogo is the most widely used ABM
teaching platform; this runner measures how a real-world NetLogo BAM behaves and
scales beside bamengine. Its gate result is informational (see
`comparison/equivalence/gate.py`, `NON_BLOCKING`).

The runner drives NetLogo from Python via [pyNetLogo](https://pynetlogo.readthedocs.io)
(a JPype bridge to the JVM), reads per-tick reporters, and derives the six
comparison series identically to every other runner. It never edits the model:
agent quantities are read via inline reporter strings.

## Third-party model (not committed)

- Model: *BAM: The Bottom-up Adaptive Macroeconomics Model*, Alejandro
  Platas-Lopez and Alejandro Guerra-Hernandez (2020). File `DelliBAM_.nlogo`.
- Source: https://github.com/alexplatasl/BAMmodel (CoMSES DOI 10.25937/tfx7-y446).
- License: GPL-2.0. Declared NetLogo version: 6.1.1.
- The model is fetched locally and is **never committed or pushed**. It lives
  under `model/` (gitignored) and is obtained via `fetch_model.sh`, which pins a
  SHA-256 of the exact file used.

## Toolchain (temporary: install, run, delete)

Everything here is removed afterward by `teardown.sh` plus the manual uninstall
lines it prints.

### 1. NetLogo 6.4.0 (not 7.x)

Use NetLogo **6.4.0**, the last 6.x release. The model is a 6.1.x file and
pyNetLogo's supported baseline is NetLogo 6.x. NetLogo 7.x changed the file
format and API and is not used here. Homebrew's `netlogo` cask ships 7.x, so
download 6.4.0 directly instead:

```bash
# Cross-platform tgz (the NetLogo jars are pure Java, so this works on macOS).
curl -fL -o NetLogo-6.4.0-64.tgz \
  https://github.com/NetLogo/NetLogo/releases/download/v6.4.0/NetLogo-6.4.0-64.tgz
tar xzf NetLogo-6.4.0-64.tgz     # -> "NetLogo-6.4.0-64/"
```

`NETLOGO_HOME` must point at the directory holding the core jars, which for this
tgz is `NetLogo-6.4.0-64/lib/app` (a flat directory containing
`netlogo-6.4.0.jar` and the correct `asm-9.4.jar`). Do NOT point it at the
install root: pyNetLogo globs jars recursively, and the bundled `vid` extension
ships an old `asm-4.0.jar` that shadows `asm-9.4` and breaks NetLogo's bytecode
compiler with a `NoSuchMethodError`.

### 2. A JVM whose architecture matches your Python

On Apple Silicon the JVM arch must match the Python arch, so install an arm64
JDK and point pyNetLogo at its `libjvm`:

```bash
brew install openjdk@17     # arm64 formula, no admin password required
# libjvm: /opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home/lib/server/libjvm.dylib
```

### 3. The bridge venv

```bash
python3.12 -m venv comparison/runners/netlogo/.venv-nl
comparison/runners/netlogo/.venv-nl/bin/pip install -r \
    comparison/runners/netlogo/requirements.txt   # pynetlogo + jpype1
```

### 4. A curated extensions directory

The model declares `extensions [palette array]`. Build a clean extensions dir
that contains only those two (excluding `vid`, whose old asm breaks compilation):

```bash
NLH="<path>/NetLogo-6.4.0-64"
mkdir -p nlext
ln -sfn "$NLH/extensions/.bundled/palette" nlext/palette
ln -sfn "$NLH/extensions/.bundled/array"   nlext/array
```

### 5. Fetch the model

```bash
bash comparison/runners/netlogo/fetch_model.sh
```

## Running

The runner reads its toolchain wiring from environment variables (documented in
`run.py`). Export them, then run the orchestrator; the values propagate to the
runner subprocess:

```bash
export NETLOGO_HOME="<path>/NetLogo-6.4.0-64/lib/app"
export NETLOGO_JVM_PATH="/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home/lib/server/libjvm.dylib"
export NETLOGO_EXT_DIR="<path>/nlext"
export NETLOGO_VERSION="6.4.0"

python -m comparison.orchestrator.run --frameworks netlogo
```

| Variable           | Purpose                                            |
| ------------------ | -------------------------------------------------- |
| `NETLOGO_HOME`     | Dir with the core jars (`.../lib/app`)             |
| `NETLOGO_JVM_PATH` | `libjvm` of an arch-matching JDK (Apple Silicon)   |
| `NETLOGO_EXT_DIR`  | Curated extensions dir (palette + array only)      |
| `NETLOGO_VERSION`  | Reported as `language_version` (default `unknown`) |

Each timing run boots a fresh JVM, so NetLogo is by far the slowest framework and
self-caps early under the harness's 120 s adaptive budget. That is expected and
recorded in `results/skips.json`, not a failure.

## Parameter and series mapping

The runner overrides only `number-of-firms`, `labor-market-M`,
`credit-market-H`, `goods-market-Z`, and the seed (`random-seed`). Workers and
banks are derived by the model (5 x firms; max(H+1, firms/10)), matching the
harness ratios natively at the gate size (100 firms -> 500 workers, 10 banks).

Other scalar parameters stay at Platas defaults. Most equal bamengine's; a few
differ and are reported as deviations rather than corrected, for example the
dividend share (Platas 15% versus bamengine `delta = 0.10`).

The six series are derived identically to every other runner (`build_series`),
from these reporters (confirmed against the model):

| Series           | NetLogo reporter                                           |
| ---------------- | ---------------------------------------------------------- |
| unemployment     | `fn-unemployment-rate`                                     |
| price_inflation  | `annualized-inflation - 1` (gross ratio to net YoY rate)   |
| wage_inflation   | growth of `mean [wage-offered-Wb] of firms`                |
| log_gdp          | `ln real-GDP`                                              |
| vacancy_rate     | `sum [number-of-vacancies-offered-V] of firms` / n_workers |
| production_final | `[production-Y] of firms` (final tick)                     |

## Teardown

```bash
bash comparison/runners/netlogo/teardown.sh   # removes model/ and .venv-nl/
brew uninstall openjdk@17                      # remove the JDK (manual)
rm -rf "<path>/NetLogo-6.4.0-64" "<path>/nlext"  # remove NetLogo + curated exts
```
