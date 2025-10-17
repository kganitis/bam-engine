# BAM Engine

### What's this project?
This is a Python implementation of the BAM model (Delli Gatti et al., 2008). Below is a short outline of the model (in greek)

**Περιγραφή του μοντέλου**

Το πιο σημαντικό μοντέλο στην κατηγορία των CATS (Complex Adaptive Trivial Systems), μια οικογένεια μακροοικονομικών μοντέλων βασισμένων σε πράκτορες. 
Αποτελεί τη βάση και τον πυρήνα μιας σειράς από πολλά μοντέλα που ακολούθησαν, όπως το CC-MABM *(Macroeconomic Agent-Based Model with Capital & Credit) (Assenza et al., 2015)*. 
Περιλαμβάνει τρεις βασικούς τύπους πρακτόρων: νοικοκυριά, επιχειρήσεις και τράπεζες, οι οποίοι αλληλεπιδρούν μεταξύ τους σε τρεις διαφορετικές αγορές: εργασίας, πίστωσης και καταναλωτικών αγαθών. 
Το μοντέλο καταγράφει το πώς οι διάφορες δυναμικές στις αγορές μπορούν και αναδύονται ενδογενώς μέσα από τις αλληλεπιδράσεις μεταξύ ετερογενών (και όχι ομοιογενών) πρακτόρων, και χωρίς την επιβολή εξωτερικά προκαθορισμένων ισορροπιών.

### What's its architecture?
It's almost fully vectorized, with mostly numpy operations, where for loops are only used to iterate through market trials and implement the market queuing system.

### Pros and cons of this approach
This version works perfectly, as intended, but it's not very modular. It's a monolith, just in separate files. So it's hard to modify or extend any part of the BAM framework.

### Future work & research extensions

#### Low-priority housekeeping

* Use consistent naming: e.g. when to use `firm` vs `borrower`/`employer`
* How to deal with _EPS values: where to use them, how often they actually occur

#### Research Extensions

* Implement "growth+" model with R&D and productivity
* Consumption and buffer shock
* Exploration of the parameter space
* Preferential attachment in consumption and the entry mechanism
* Add reinforcement-learning agents for policy search experiments
* Investigate GPU or distributed back-ends once single-node scaling limits are reached