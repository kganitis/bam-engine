
- When to call breakeven price calculation
  - `During planning`
    - Wage Bill:
      - Use wage bill after firing
      - `Use wage bill after contracts expiration`
    - Interest Rate:
      - `Delete interest after debt repayment --> 0 interest carried over`
      - Do NOT delete interest, carry over to t+1 to use for breakeven
  - After credit market (use current labor + interest + projected production)

- Use breakeven during price adjustment
  - When cutting
    - `Allow increase due to breakeven being higher than old price`
    - Do NOT allow
  - When increasing
    - `Cap to breakeven`
      - `Allow it to be higher than P * (1 + shock)`
        - Set upper limit
        - `Let extreme jumps be possible`
      - Cap it to P * (1 + shock)
    - Don't cap to breakeven

- Contracts duration
  - θ
  - `θ + Poisson(10)`

- When to calc unemployment rate
  - After firing (actual labor used for production)
  - `After contracts expiration (more unemployment)`

- Providing loans priority
  - `Sort applicants by net worth`
  - Provide in order of appearance

- Financial fragility
  - `Apply cap to avoid extreme interest jumps`
    - `Cap to Β`
    - Cap to another value 
    - Cap interest rate instead
  - Allow extreme interest jumps
    - Cap interest rate instead

- Firing workers method
  - Fire random workers
  - `Fire most expensive`
  - Fire recent
  - Fire older

- Handle loanbook rows after repayment
  - `Clear after repayment (this will affect breakeven calc)`
  - Carry over to t+1, clear when about to be used

- Handle firms with zero production
  - `Mark them as bankrupt`
  - Handle them in other way (e.g don't reset their inventory history)

- New firms entry
  - Size factor
  - Production
  - Wage offer
  - Price

- Parameter calibration
  - beta
  - delta
  - v
  - r_bar
  - min_wage
  - net_worth_init
  - production_init
  - price_init
  - savings_init
  - wage_offer_init
  - equity_base_init