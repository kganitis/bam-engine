>>> from bamengine.scheduler import Scheduler
>>> sch = Scheduler.init(n_firms=10, n_households=50, seed=42)
>>> for _ in range(5):
...     sch.step()
...     print(f"mean Yd = {sch.mean_Yd:.2f}")
