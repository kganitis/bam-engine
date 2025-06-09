Tell me what you know about my progress 
in the ECS implementation of the BAM model. 
What we're doing and where exactly are we.

---

I see that you do not keep any memories from our other conversation in this project,
so let me give you a summary of my progress so far.


I'm programming a simulation of the BAM 
(Bottom-Up Adaptive Macroeconomics) model,
which is an agent-based model.
I initially programmed it in OOP 
and now I'm implementing it step by step in ECS using Python.


My priorities for the project are:
1. Performance (first speed, then memory - speed is more important)
2. Testability (we must be SURE that it works as intended)
3. Usage as a library by researchers (ease of use and parameterization)
4. Scalability, modularity and maintainability.


Each step or sub-step of the simulation corresponds to one system in ECS.
The agent types are Firm, Household and Bank. 
The components correspond to "roles" like:
`Employer`, `Producer`, `Worker`, `Consumer`, `Borrower`, `Lender` etc.


So far I've implemented the first 4 events:
planning, labor market, credit market, production,
and the components needed for them: 
`Employer`, `Producer`, `Worker`, `Consumer`, `Borrower`, `Lender`.
Each component corresponds to one of the possible roles and agent can have.


`Scheduler` is the class that owns the components,
wires together and runs sequentially all the different systems.


I've written unit tests for all the systems,
integration tests for the events as a whole,
and for the scheduler as well


Before moving on, let me give you my `ROADMAP.md` below
and the current code in the next prompt.


Do not make any comments on them yet, just hold them for reference.