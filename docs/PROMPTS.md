It's time to implement the next event: goods market.
I will provide you with the respective code from my older OOP implementation.
I want you to write an implementation based on my current structure and logic.
Your output should be the necessary new/updated components and new systems.
We should use and update the existing `Economy`, `Consumer`, `Producer` components.
We won't need new ones.
For the systems write a new `goods_market.py` file.

Give extra attention to the goods market rounds
and how the existing credit and labor markets handle things.
You should use the same logic (queues, shocks, scratch buffers etc).

``` 
# Event 5 - Goods Market
'''
Firms post their offer price, and consumers contact a given number of randomly
chosen firms to purchase goods, starting from the one which posts the lowest price.
'''

class Firm():
...
@property
def break_even_price(self) -> float:
    # TODO avoid division by zero (should never happen)
    return (self.wage_bill + self.total_interest_amount) / self.production
        
def decide_price(self, p_avg: float, h_eta: float) -> None:
    shock = random.uniform(0, h_eta)
    if self.inventory == 0 and self.price < p_avg:
        self.price = max(self.break_even_price, self.price * (1 + shock))
    elif self.inventory > 0 and self.price >= p_avg:
        self.price = max(self.break_even_price, self.price * (1 - shock))


class Household():
...
def _propensity_to_consume(self, avg_sav: float, beta: float) -> float:
    return 1 / (1 + (math.tanh(self.savings / avg_sav)) ** beta)

def decide_income_to_spend(self, avg_sav: float, beta: float) -> None:
    wealth = self.savings + self.income
    prop = self._propensity_to_consume(avg_sav, beta)
    self.remaining_income_to_spend = prop * wealth
    self.savings = wealth - self.remaining_income_to_spend

def decide_firms_to_visit(
    self,
    n_visits: int,
    firms_with_inventory: List["Firm"]
) -> None:
    self.firms_to_visit.clear()
    if not firms_with_inventory:
        return

    n_visits = min(n_visits, len(firms_with_inventory))
    if (self.largest_prod_prev and self.largest_prod_prev in firms_with_inventory
            and not self.largest_prod_prev.bankrupt):
        self.firms_to_visit.append(self.largest_prod_prev)
        n_visits -= 1

    other = [f for f in firms_with_inventory if f != self.largest_prod_prev]
    candidates = random.sample(other, n_visits)
    candidates.sort(key=lambda f: f.price)
    self.firms_to_visit.extend(candidates)

def visit_next_firm(self) -> None:
    if not self.remaining_income_to_spend or not self.firms_to_visit:
        self.savings += self.remaining_income_to_spend
        self.remaining_income_to_spend = 0.0
        return

    firm = self.firms_to_visit.pop(0)
    spent = firm.sell_good(self.remaining_income_to_spend)
    self.remaining_income_to_spend -= spent

    if (self.largest_prod_prev is None
            or firm.production_Y > self.largest_prod_prev.production_Y):
        self.largest_prod_prev = firm


class Scheduler():
...
self.prices_max_growth_rate_H_eta = 0.1
self.goods_market_n_trials = 2
self.beta = 0.87

@property
def average_market_price(self):
    return statistics.mean([f.price for f in self.firms])
    
@property
def average_consumer_savings(self):
    return statistics.mean([h.savings for h in self.households])
    
@property
def firms_with_inventory(self):
    return [f for f in self.firms if f.inventory > 0]

average_market_price = self.average_market_price
for f in self.firms:
    f.decide_price(average_market_price, self.prices_max_growth_rate_H_eta)
self.average_market_price_history.append(self.average_market_price)

average_consumer_savings = self.average_consumer_savings
for c in self.consumers:
    c.decide_firms_to_visit(self.goods_market_n_trials_Z, self.firms_with_inventory)
    c.decide_income_to_spend(average_consumer_savings, self.beta)
for _ in range(self.goods_market_n_trials_Z):
    for c in self.consumers:
        c.visit_next_firm()
```