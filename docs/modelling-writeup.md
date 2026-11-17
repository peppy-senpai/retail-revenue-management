# Retail Revenue Management Simulation: Model Write-up (draft)

Draft text describing how the system is modelled, organised to drop into the report structure
proposed in Part A. Section numbers in Part B match the report outline.

---

## Part A: Suggested report structure

| # | Section | What goes in it | Source in repo |
|---|---|---|---|
| 1 | **Introduction** | The problem is joint inventory, sourcing and pricing for a multi-department grocery/general retailer with perishable stock, uncertain demand and uncertain lead times. State the research question, e.g. *how much does demand-aware pricing add on top of a newsvendor replenishment policy?* | — |
| 2 | **Data and exploratory analysis** | M5 dataset, scope (stores CA_1 + CA_2, department grain), aggregation, WTP curves, ABC-XYZ classification, shelf-life proxy | `code/work/work.ipynb`, `abc-xyz-analysis.ipynb` |
| 3 | **Simulation model** | The "world": clock and events, demand generator, customer response to price, inventory, suppliers, forecast. Include one architecture diagram and one daily-loop flowchart | `code/sim/sim.py`, the setup cells of each notebook |
| 4 | **Decision policies** | The shared replenishment core (reorder point, newsvendor, sourcing MILP), then the four pricing variants as a ladder of increasing sophistication | `Policy` class in each `newswendor_*.ipynb` |
| 5 | **Experimental design** | Epochs, horizon, seeding and common random numbers, KPIs and how each is computed | `compare_performance.ipynb` |
| 6 | **Results** | Headline comparison table, per-department breakdown, wastage and stockout trade-off | `results/model comparison.xlsx` |
| 7 | **Discussion and limitations** | Why each policy behaves the way it does, threats to validity (see Part C) | — |
| 8 | **Conclusion and future work** | Multi-supplier splitting, demand-index stress tests, pricing-aware ordering, holding costs | — |
| A | **Appendix** | Parameter tables (supplier config, MAPE, shelf lives, markups), full LP/MILP formulations | Part B tables below |

Two figures to include: (1) a **component diagram** showing Env (clock + calendar + RNG streams)
at the centre, with Demand, Inventory, Supplier and Policy around it and arrows labelled with the
events (`stockup`, `expire`, `price_change`); (2) a **daily-loop flowchart** (see §3.7).

---

## Part B: Model description

### 2. Data

**Source.** The M5 Forecasting dataset (Walmart daily unit sales, 1,941 days) contributes three
tables: `sales_train_evaluation.csv` (units per item per day), `sell_prices.csv` (weekly price per
item and store) and `calendar.csv` (maps day index `d` to Walmart week `wm_yr_wk`).

**Scope and grain.** Stores CA_1 and CA_2 are pooled, and items are rolled up to the
**department** level: FOODS_1–3, HOBBIES_1–2, HOUSEHOLD_1–2 (7 departments). For each
department and day:

- demand $D_{j,t} = \sum_i \text{units}_{i,t}$
- unit-weighted average price $\bar p_{j,t} = \sum_i \text{units}_{i,t}\, p_{i,t} \,/\, D_{j,t}$

Item-weeks with no listed price are dropped by an inner join. Department-days with zero demand
have an undefined average price and are excluded from the price table. In the demand matrix those
days are zero-filled.

**ABC-XYZ.** Departments are ranked by cumulative revenue share (A ≤ 50%, B ≤ 90%, C the rest)
and by the coefficient of variation of daily revenue (XYZ). A: FOODS_3, HOUSEHOLD_1.
B: HOBBIES_1, HOUSEHOLD_2, FOODS_2. C: FOODS_1, HOBBIES_2. *(Add the XYZ split and the chart from the
notebook.)*

**Shelf-life proxy.** M5 has no shelf-life field, so a holding horizon is inferred from how fast
each department's demand autocorrelation decays. FOODS_2 and FOODS_3 lose memory within about
50–150 days, while non-food departments persist far longer. These readings inform the shelf lives
and decision horizons in the appendix table.

### 3. Simulation model

The simulator is a **discrete-time, event-calendar** model with a 1-day step. All components share
a single environment object.

#### 3.1 Environment (`Env`)

- `clock`: the current day index.
- `event_calendar`: `{day: [events]}`. Components book future events and the main loop drains the
  current day's events before trading. Event types: `stockup` (a delivery arrives), `expire` (a lot
  reaches its expiry) and `price_change` (a scheduled markdown, used by the two-price policy only).
- **Randomness.** One seed is split with `SeedSequence.spawn` into three independent streams:
  demand, supplier (quote prices and realised lead times) and forecast error. Each epoch $k$ is
  reset with seed $k$, so every policy faces **identical demand paths** (common random numbers).
  This is why `total_demand` is identical across policies in the results.

#### 3.2 Demand generator (`Demand`)

Synthetic demand paths are produced by **perturbing the historical department series in PCA
space**:

1. Form $X \in \mathbb{R}^{7 \times T}$ (departments × days) and standardise each department's
   series to zero mean and unit variance, so FOODS_3 (thousands of units/day) does not dominate.
2. Fit PCA treating departments as samples and days as features, keeping enough components to
   explain 95% of variance. The components are **temporal patterns shared across departments**
   (trend, seasonality, holiday effects), and each department is a weighted mix of them (its scores).
3. Perturb the scores: $\tilde z_{j,k} = z_{j,k} + \varepsilon_{j,k}$, with
   $\varepsilon_{j,k} \sim \mathcal N(0,\ (0.8)^2 \lambda_k)$, where $\lambda_k$ is component $k$'s
   explained variance. Dominant patterns therefore move more than minor ones.
4. Invert the PCA and the scaling, round, and clip at 0.

Each epoch therefore sees a plausible variant of history that keeps the cross-department
correlation structure.

#### 3.3 Customer response to price (willingness to pay)

For each department, a weighted Gaussian KDE $f_j$ is fitted to the historical daily average
prices $\bar p_{j,t}$ with weights $D_{j,t}$. Its CDF $F_j$ defines the **share of potential
demand that buys at price $p$**:

$$S_j(p) = 1 - F_j(p)$$

Realised demand on day $t$ is $\text{round}\big(D_{j,t}\, S_j(p_{j,t})\big)$. This is the only
place price enters the model, and it is what makes pricing decisions matter. $S_j(p)$ is cached
per (department, price).

#### 3.4 Inventory (`Inventory`)

- Stock for each department is a **min-heap of lots** `[expiry_day, quantity]`, so the soonest-expiring
  lot is always at the front and depletion is **FEFO** (first-expired, first-out).
- On delivery, a lot is pushed with expiry $= t + \text{shelf life}_j$, and an `expire` event is
  booked for that day.
- On an `expire` event, every lot with expiry ≤ today is written off as **wastage**. The handler
  reads the remaining quantity at that moment, so partially sold lots are handled correctly and a
  repeated event writes off nothing.
- **Selling.** Price-adjusted demand is served from the front of the heap. Unmet demand is **lost**
  (no backorders). A department with no price set sells nothing.

#### 3.5 Suppliers (`Supplier`, `SupplierInfo`, `Quote`)

Each department has **two suppliers** that trade cost against speed:

| | "low" supplier | "premium" supplier |
|---|---|---|
| Unit cost | cheaper (larger retailer margin) | dearer |
| Lead time | $L \sim \text{Exp}(\text{loc}=L_{\max},\ \text{scale}=L_{\max})$ | $L \sim \text{Exp}(\text{loc}=L_{\min},\ \text{scale}=L_{\min})$ |

- **Volume price breaks.** Each supplier sells three discrete lot sizes. The unit cost at tier $\tau$
  is $c_\tau \sim \mathcal N\big(\mu_\tau,\ (\text{CV}\cdot\mu_\tau)^2\big)$ with
  $\mu_\tau = \bar p_j (1 - m_s)\cdot \{1.0, 0.5, 0.2\}_\tau$, where $\bar p_j$ is the historical
  mean retail price and $m_s$ the supplier's margin. The CV is held constant across tiers.
- **Quotes.** `order_proposal` draws a fresh price for each tier on every request and quotes the
  lead time's minimum ($\text{loc}$).
- **Purchase.** `order_purchase` validates lot sizes, draws a realised lead time
  $\max\big(1, \text{round}(L \cdot (1 + \delta))\big)$, where $\delta$ is a market-congestion index
  (0 in all runs), and books one `stockup` event. A multi-lot order arrives as a single delivery at
  its unit-weighted average price.

#### 3.6 Demand forecast

The forecast is a **controlled-error oracle**. It takes the true future path and adds
multiplicative errors $\hat D_t = D_t (1 \pm e_t)$ with random signs. The magnitudes are rescaled
so that the window's MAPE equals a target drawn per call from
$\mathcal N(\text{MAPE}_j, 0.03^2)$. Department targets range from 10% (FOODS_3, HOUSEHOLD_1) to
25% (HOBBIES_2). This isolates **decision quality** from forecasting-model quality, because
forecast accuracy is an experimental parameter.

#### 3.7 Daily loop

```
for each epoch k = 0..99:
    reset Env(seed=k), Inventory, Policy; draw demand path
    opening stock = newsvendor quantity per department; set opening prices
    for t = 0..1900:
        1. process today's events  (stockup → receive lot + log lead time;
                                    expire → write off; price_change → markdown)
        2. sell today's demand at current prices (WTP-adjusted, FEFO)
        3. policy reviews every department; may order (and, by variant, reprice)
        4. log: demand, sold, price, wastage, order qty/cost, stock
        t += 1
```

### 4. Decision policies

All variants share the same **replenishment core** and differ only in **pricing**.

#### 4.1 Replenishment core (shared)

**Continuous review with a reorder point.** Each day, for each department:

- inventory position $= \text{on hand} + \text{on order}$
- reorder point $s = \bar d\, \mathbb E[L] + z\,\sigma_{LT}$, with
  $\sigma_{LT} = \sqrt{\mathbb E[L]\,\sigma_d^2 + \bar d^2 \sigma_L^2}$ and $z = 1.65$ (≈95% cycle
  service)
- $\bar d, \sigma_d$ come from the forecast over the decision horizon $H_j$
- $\mathbb E[L], \sigma_L$ come from the **last 5 realised lead times** (pooled across suppliers).
  Before 5 deliveries have been observed, the prior is an equal mixture of the two suppliers'
  distributions.

When position ≤ $s$, the policy places an order.

**Order quantity: newsvendor.** With $\bar c$ the highest quoted unit cost and $m_j$ the
department markup, the underage cost is $c_u = \bar c\, m_j$ and the overage cost is $c_o = \bar c$:

$$\text{CR} = \frac{c_u}{c_u + c_o} = \frac{m_j}{1 + m_j}, \qquad Q^* = F^{-1}_{D_H}(\text{CR})$$

$F_{D_H}$ is the **empirical distribution of total demand over non-overlapping $H_j$-day windows**
in the history observed so far. If less than one full window exists, the policy falls back to the
forecast total.

**Sourcing: mixed-integer program.** Given the quotes, the policy chooses one supplier and how many
lots of each size to buy:

$$
\min_{x, y, u, o}\ \sum_{s,\tau} c_{s\tau}\, q_{s\tau}\, x_{s\tau}
\;+\; m\,\sum_s \text{ES}_s\, y_s \;+\; m\,u \;+\; c_o\, o
$$
$$
\text{s.t.}\ \sum_s y_s = 1,\quad x_{s\tau} \le 100\, y_s,\quad
\sum_{s,\tau} q_{s\tau} x_{s\tau} + u - o = Q^*,\quad
x \in \mathbb Z_{\ge 0},\ y \in \{0,1\}
$$

$\text{ES}_s = \sigma_{L,s}\big[\phi(k_s) - k_s(1 - \Phi(k_s))\big]$ is the expected units short
while waiting for supplier $s$'s delivery (normal loss function, with $k_s$ = (on hand − lead-time
demand)/σ). The objective therefore trades **cheaper, slower** suppliers against **lost sales
during the lead time**, and trades lot-size rounding against the newsvendor target.
Solver: HiGHS via cvxpy.

#### 4.2 Pricing variants

| Policy | Results label | Pricing rule |
|---|---|---|
| **P0. Constant price** | `ConstantPrice` | Fixed at each department's historical mean price for the whole run. |
| **P1. Cost-plus** | `Margin` | On every order, price ← $(1 + m_j)\times$ that order's unit cost. |
| **P2. Bid-price, single price** | `DP-Single-Price` | On every restock: solve the time-at-price LP below, take the **dual of the stock constraint** as the bid price $\pi$, and set $p = \arg\max_{p \ge c}\,(p - \pi)\,S_j(p)$ on a 1-cent grid. The price holds until the next restock. |
| **P3. Two-price schedule** | `DP-2-price` | Same LP, but the **primal** solution is implemented: charge the higher price for $t_{\text{high}}$ days, then switch to the lower price via a scheduled `price_change` event. Plan IDs cancel stale switches if a new restock re-plans first. |

**Time-at-price LP (P2, P3).** Two candidate prices: cost-plus $p_1 = (1+m)c$ and the
revenue-maximising price $p_2 = \arg\max p\,S(p)$ over the historical price range. Candidates below
unit cost are dropped. With $r_i = S(p_i)\,\bar d$ as the expected daily sales at price $p_i$:

$$\max_{t \ge 0}\ \sum_i p_i r_i t_i \quad \text{s.t.}\quad \sum_i r_i t_i \le \text{stock},\quad \sum_i t_i \le H$$

With only two constraints, an optimal solution uses either a single price or both prices with both
constraints binding. P3 exploits this and solves the LP in closed form by checking those three
vertices. The bid price $\pi$ is the marginal revenue of one more unit of stock. When stock is
plentiful, $\pi = 0$ and P2 charges the revenue-peak price. When stock is scarce, $\pi$ rises and
pushes the price up.

### 5. Experimental design

- **100 epochs × 1,901 days**, one demand path per epoch, with common random numbers across
  policies.
- **KPIs** per epoch and department (`compare_performance.ipynb`): units sold; revenue
  $= \sum \text{sold} \times \text{price}$; wastage units; order quantity and cost; stockout days
  (days ending with zero stock); lost units on stockout days (demand × the observed buy share at that
  price, minus units sold); underage cost (lost units × unit margin); and **profit = revenue −
  purchase cost**.
- Report the mean and standard deviation across epochs.

### 6. Results (mean over 100 epochs, summed over 7 departments)

| Policy | Units sold | Revenue | Purchase cost | Profit | Wastage (units) | Stockout days | Lost units |
|---|---:|---:|---:|---:|---:|---:|---:|
| P1 Cost-plus | 23.8 M | 18.6 M | 9.66 M | 8.98 M | 351 k | 266 | 453 k |
| P2 Bid-price, single | 18.1 M | 49.8 M | 7.40 M | 42.4 M | 916 k | 83 | 177 k |
| P3 Two-price | 24.3 M | 62.6 M | 9.99 M | **52.6 M** | **298 k** | 226 | 394 k |

Points worth discussing:

- **P3 dominates.** It sells as many units as cost-plus at roughly 3.3× the revenue, and it
  has the lowest wastage. Starting high and then marking down captures high-WTP customers early and
  clears stock before it expires.
- **P2 trades volume for price.** It has the fewest stockouts (83 days), but sells 25% fewer
  units and has the **highest wastage**, almost all in FOODS_3 (899 k units, 12-day shelf life).
- **P1's low revenue** (about 0.78 per unit sold, against historical prices of 2.3–5.2) comes from
  applying the markup to tier-discounted unit costs. See C2 before interpreting it.

---

## Part C: Issues to fix or disclose before submitting

These are places where the code and the description above diverge, or where a reviewer would push
back. Fix them, or state them explicitly as limitations.

1. **Ordering ignores price.** The newsvendor and the reorder point use *potential* demand, but
   realised demand is $S(p) \times$ potential. Any policy that raises prices therefore over-orders
   by about $1/S(p)$. This likely explains P2's FOODS_3 wastage. Either use $S(p)\,D$ in the
   newsvendor or present it as a limitation.
2. **The cost-plus baseline is a weak comparator.** Tier multipliers {1.0, 0.5, 0.2} mean the largest
   lot costs 20% of the base price, so cost-plus prices fall far below the historical range, where
   $S(p) \approx 1$. P1 versus P3 then mostly reflects the supplier configuration. Consider
   anchoring cost-plus to the base-tier cost, or justify the 80% volume discount.
3. **P0 (constant price) is missing from the comparison.** The notebook writes
   `Newsvendor_SingleSupplier_ConstantPrice.parquet`, but the file in `results/` is the older
   `Newswendor_...` one and `model comparison.xlsx` has no P0 row. Re-run P0 and include it as the
   true "no pricing" baseline.
4. **Different horizons between runs.** The constant-price notebook uses a 100-day decision
   horizon and 1000-day shelf life for HOBBIES/HOUSEHOLD, while the other three use 500/400. The
   replenishment core is therefore not identical across all variants. Align the settings or report
   the difference.
5. **WTP identification.** The KDE describes the prices *at which units sold* (shaped by Walmart's
   own pricing), not customers' reservation prices. Treating $1 - F(p)$ as purchase probability is a
   modelling assumption, and the report should name it.
6. **The demand generator drops idiosyncratic noise.** The 5% of variance discarded by PCA is the
   day-level noise, so synthetic paths are smoother than reality, and event timing is the same in
   every epoch. Mention this, or add a residual-bootstrap term.
7. **Common random numbers hold for demand only.** Policies that request different numbers of
   quotes or forecasts consume the supplier and forecast streams at different rates, so lead-time
   and price draws differ between policies. Say so in §5.
8. **Profit omits holding cost**, and lost sales are not charged in the profit figure (they are
   reported separately as underage cost). State this definition clearly.
9. **The forecast is an oracle with noise**, not a fitted model. Frame this as a deliberate design
   choice (it isolates decision quality) and state it in §3.6.
10. **Minor code/docstring mismatch.** The P3 `plan_prices` docstring says the horizon is "days until
    newest lot expires, capped at the decision horizon", but the code uses the decision horizon
    only.

---

## Appendix: parameter tables

**Supplier configuration** (price_mean / price_std = historical retail mean and sd; lead times in days)

| Dept | price mean | price sd | $L_{\min}$ | $L_{\max}$ | low: lots / margin | premium: lots / margin |
|---|---:|---:|---:|---:|---|---|
| FOODS_1 | 2.30 | 0.20 | 5 | 10 | 25/85/250 · 0.45 | 25/80/230 · 0.30 |
| FOODS_2 | 3.31 | 0.27 | 3 | 7 | 17/60/170 · 0.30 | 17/55/170 · 0.20 |
| FOODS_3 | 2.33 | 0.16 | 1 | 4 | 65/215/640 · 0.35 | 60/200/610 · 0.20 |
| HOBBIES_1 | 3.63 | 0.76 | 5 | 10 | 30/100/300 · 0.40 | 20/75/250 · 0.30 |
| HOBBIES_2 | 2.19 | 0.41 | 7 | 14 | 20/100/200 · 0.50 | 20/100/200 · 0.35 |
| HOUSEHOLD_1 | 3.87 | 0.27 | 7 | 14 | 25/85/250 · 0.38 | 25/75/230 · 0.28 |
| HOUSEHOLD_2 | 5.18 | 0.23 | 5 | 10 | 50/170/500 · 0.30 | 50/160/480 · 0.20 |

**Policy and product parameters** (P1–P3 runs)

| Dept | Shelf life (days) | Decision horizon $H$ | Markup $m$ | Forecast MAPE |
|---|---:|---:|---:|---:|
| FOODS_1 | 270 | 60 | 0.40 | 0.20 |
| FOODS_2 | 120 | 120 | 0.25 | 0.15 |
| FOODS_3 | 12 | 10 | 0.30 | 0.10 |
| HOBBIES_1 | 500 | 500 | 0.35 | 0.15 |
| HOBBIES_2 | 500 | 500 | 0.40 | 0.25 |
| HOUSEHOLD_1 | 400 | 400 | 0.32 | 0.10 |
| HOUSEHOLD_2 | 400 | 400 | 0.25 | 0.15 |

Other settings: $z = 1.65$; lead-time window = 5 deliveries; PCA variance kept = 95%; demand
noise damping = 0.8; congestion index $\delta = 0$; at most 100 lots per tier per order.
