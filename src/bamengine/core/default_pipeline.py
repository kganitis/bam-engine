"""Default BAM simulation pipeline."""

from importlib import resources
from pathlib import Path

from bamengine.core.pipeline import Pipeline


def create_default_pipeline(max_M: int, max_H: int, max_Z: int) -> Pipeline:
    """
    Create default BAM simulation event pipeline.

    Loads the pipeline from default_pipeline.yml and substitutes
    market round parameters (max_M, max_H, max_Z).

    Parameters
    ----------
    max_M : int
        Number of job application rounds.
    max_H : int
        Number of loan application rounds.
    max_Z : int
        Number of shopping rounds.

    Returns
    -------
    Pipeline
        Default BAM pipeline with all events in correct order.

    Notes
    -----
    This function creates the "canonical" BAM pipeline. Users can modify
    it using insert_after(), remove(), replace() methods, or create their
    own pipeline from a custom YAML file using Pipeline.from_yaml().

    Market rounds are explicitly interleaved: send-hire-send-hire pattern
    for labor market, same for credit market and goods market.
    """
    # Locate the default pipeline YAML file
    try:
        # Python 3.9+: importlib.resources.files() returns a Traversable.
        # Use as_file() to obtain a real filesystem Path (mypy-compatible).
        traversable = resources.files("bamengine") / "default_pipeline.yml"
        with resources.as_file(traversable) as yaml_fs_path:
            return Pipeline.from_yaml(
                Path(yaml_fs_path),
                max_M=max_M,
                max_H=max_H,
                max_Z=max_Z,
            )
    except AttributeError:
        # Fallback for Python 3.8 where resources.files() is unavailable.
        import bamengine

        yaml_path = Path(bamengine.__file__).parent / "default_pipeline.yml"
        return Pipeline.from_yaml(yaml_path, max_M=max_M, max_H=max_H, max_Z=max_Z)
