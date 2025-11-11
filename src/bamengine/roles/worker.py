from bamengine.core.decorators import role
from bamengine.typing import Bool1D, Float1D, Idx1D, Idx2D, Int1D


@role
class Worker:
    """
    Worker role for agents.

    Represents and entity that is able to provide labor for an offered wage.
    """

    employer: Idx1D
    employer_prev: Idx1D
    wage: Float1D
    periods_left: Int1D
    contract_expired: Bool1D
    fired: Bool1D

    # Scratch queues
    job_apps_head: Idx1D
    job_apps_targets: Idx2D  # shape (n_households, M)

    @property
    def employed(self) -> Bool1D:
        """
        Compute employment status from employer ID.

        Returns true for workers with employer >= 0, false otherwise.
        This is a computed property derived from the employer array.
        """
        return self.employer >= 0
