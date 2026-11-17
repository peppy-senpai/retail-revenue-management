# Environment Design

## 1. Overview

The environment is a simulated store that sells products, holds stock and buys from suppliers.
Each day, customers arrive, buy what they can afford from what is on the shelf, and leave. Each
day, the store also checks its stock, decides whether to reorder, and sets its prices.

The store itself, meaning everything that happens *to* it, is kept separate from the decisions
it makes. The environment covers demand, deliveries, expiry and customer behaviour. The
**policy** is the decision-maker that plugs into it. Because of this split, different policies
can be tested on exactly the same world and compared fairly.

The model is built as a **discrete-event simulation (DES)**. The system changes only at specific
moments (events), such as a delivery arriving or a batch of stock expiring. Time moves forward one
day at a time. Events planned for the future are kept on a **future event list (FEL)**, a calendar
the simulation checks each morning.

**Scope**
- **Store:** stores CA_1 and CA_2 from the M5 (Walmart) dataset, combined into one.
- **Products:** 7 departments, each treated as a single product (a **stock-keeping unit**, or SKU).
  The departments are FOODS_1–3, HOBBIES_1–2 and HOUSEHOLD_1–2.
- **Suppliers:** two per department.
- **Run length:** 1,901 days per run, repeated 100 times (**replications**) with different random
  draws.

---

## 2. System State

The **state** of the system is a snapshot of everything needed to carry the simulation forward
from today. If the simulation were paused and restarted from this snapshot, it would continue
exactly as before. Each day, the state is updated by three things: the **decisions** the policy
makes, the **random events** the environment produces, and a set of rules that combine the two
(the **transition function**).

### 2.1 State variables

For each department, the simulation tracks:

| State variable | What it holds | Where it lives |
|---|---|---|
| **Clock** $t$ | Today's day number | `Env.clock` |
| **Stock on hand** | A list of **lots**, each with a quantity and an expiry day | `Inventory.stock[dept].sku` |
| **Selling price** $p_t$ | Today's shelf price (unset until the policy first prices the department) | `Inventory.stock[dept].price` |
| **Event calendar** | Future events, by the day they happen: deliveries, expiries, price changes | `Env.event_calendar` |
| **Stock on order** | Units ordered but not yet delivered (the **pipeline inventory**) | `Policy.on_order` |
| **Lead-time history** | The last 5 actual delivery times for the department | `Policy.lead_obs` |

**Stock is stored by age, not as a single number.** Two departments could each hold 500 units,
but if one department's units expire tomorrow and the other's expire next month, the two stores
are in very different situations. Holding each lot separately lets the model track **product
age**, which is what makes perishability and FEFO (first-expired, first-out) selling possible.
The lots are kept in a **priority queue** (min-heap) sorted by expiry day, so the oldest lot is
always at the front.

### 2.2 Derived quantities

Some useful numbers are calculated from the state rather than stored:

- **Total stock on hand** = the sum of all lot quantities.
- **Inventory position** = stock on hand + stock on order. This is the number the policy compares
  with its **reorder point** when deciding whether to order.
- **Stockout** = total stock on hand equals zero at the end of the day.

### 2.3 Entities and their attributes

In DES terms, **entities** are the things that move through the system. There are two kinds:

| Entity | Attributes | Life cycle |
|---|---|---|
| **Purchase order** | supplier, department, quantity, unit cost, order day | Created when the policy orders. It waits on the calendar for the lead time, then becomes a lot. |
| **Lot** | department, quantity, expiry day | Created on delivery. It shrinks as customers buy, and leaves the system when it sells out or expires. |

Customers are **not** modelled as individual entities. Each day's demand arrives as a single
number per department, and the WTP curve decides what share of it buys.

### 2.4 Environment state vs. policy memory

The state is split between the two sides of the model:

- **Environment state** is the physical world: the clock, the lots on the shelf, the prices and
  the event calendar. It is the same whichever policy is running.
- **Policy memory** is what the decision-maker keeps track of itself: stock on order, past lead
  times, the cost of the last order and the current pricing plan. Different policies can keep
  different memories.

Stock on order is held by the policy. The environment already holds each pending delivery as an
event on its calendar, but it does not add them up; the policy keeps its own running total to
calculate the inventory position.

### 2.5 What is not state

Two things are needed to run the model but are **not** part of the state:

- **Parameters** are fixed for the whole run: the WTP curves, the supplier terms (lot sizes, price
  and lead-time distributions), shelf lives, markups and forecast accuracy.
- **Exogenous information** arrives from outside each day and the store cannot control it:
  customer demand, supplier quotes and actual lead times. Each run's full demand path is generated
  at the start, but the policy only sees it one day at a time (plus the noisy forecast).

### 2.6 Initial and reset state

At the start of each run:
- the clock is set to day 0 and the event calendar is cleared;
- all stock and orders in transit are removed;
- an **opening stock** is placed for every department (sized by the newsvendor rule), and its
  expiry events are booked;
- opening prices are set;
- the random streams are re-seeded with the run number.

Past lead-time observations are **kept** between runs, so the policy carries what it has learned
about suppliers from one run to the next.

### 2.7 State update each day

$$S_{t+1} = f\big(S_t,\ x_t,\ W_{t+1}\big)$$

- $S_t$ = today's state
- $x_t$ = today's decisions (what to order, from whom, at what price)
- $W_{t+1}$ = new random information (demand, quotes, delivery times)
- $f$ = the **transition function**: the event rules plus the stock balance
  $I_{t+1} = I_t + R_t - \text{Sold}_t - W^{\text{waste}}_t$

---

## 3. Events and the daily cycle

### 3.1 Event types

| Event | When it is booked | What happens when it fires |
|---|---|---|
| **Delivery** (`stockup`) | When an order is placed, for day *today + lead time* | A new lot joins the stock, and the actual lead time is recorded |
| **Expiry** (`expire`) | When a lot arrives, for the day it goes bad | Whatever is left of that lot is thrown away (**wastage**) |
| **Price change** (`price_change`) | When a two-price plan is made | The price drops to the planned markdown price |

Two events happen every day without being booked: **customer demand** and the **policy review**.

### 3.2 Order of a day

1. **Process events:** receive deliveries, remove expired stock and apply price changes.
2. **Sell:** customers buy at today's price from today's stock.
3. **Review:** the policy checks stock, may place orders and may set new prices.
4. **Record:** log demand, sales, price, wastage, orders and stock.
5. **Advance the clock** by one day.

The order matters. Because deliveries are processed before selling, stock that arrives today can
be sold today.

### 3.3 Handling outdated events

Some events become irrelevant before they fire. For example, a new delivery may replace an old
pricing plan. The simulation handles this safely in two ways:
- Each price plan has an **ID**, and a price-change event is ignored if its plan has since been
  replaced.
- An expiry event checks what is *actually* left of the lot when it fires. If the lot has already
  sold out, nothing is thrown away.

---

## 4. Demand: how customers behave

Demand is modelled in two layers: **how many people show up**, and **how many of them buy at the
current price**.

### 4.1 Potential demand: how many people show up

**Potential demand** is the number of units customers would buy if price were no obstacle. It is
**exogenous**, meaning the store cannot influence it.

To create many realistic but different futures, historical demand is reshaped with **principal
component analysis (PCA)**:

1. Each department's daily sales history is standardised (scaled to mean 0, spread 1) so that big
   departments do not drown out small ones.
2. PCA finds a few **common patterns** over time that all departments share, such as seasons,
   trends and holiday spikes. Each department is described as a mix of these patterns.
3. A small amount of random noise is added to each department's mix. Patterns that matter more get
   more noise.
4. The result is converted back into units per day, and negative values are set to zero.

Each run therefore gets a demand path that looks like real history but is not identical to it. It
keeps the links between departments; for example, a busy week for one food department tends to be
a busy week for the others.

### 4.2 Willingness to pay: how many of them buy

Every customer has a **reservation price**, the most they are willing to pay. If the shelf price
is at or below it, they buy; if it is above it, they walk away.

The spread of reservation prices across customers is the **willingness-to-pay (WTP)
distribution**. It is estimated from history with a **kernel density estimate (KDE)**, a smooth
curve fitted to the prices at which units actually sold, with each day weighted by how many units
sold that day.

From this curve, the share of customers who buy at price $p$ is:

$$S(p) = 1 - F(p)$$

$F(p)$ is the share of customers whose reservation price is below $p$. Expected sales are
therefore:

$$\text{sales} = \text{potential demand} \times S(p)$$

This is the store's **demand curve**. Raising the price lowers $S(p)$, so fewer people buy. Price
is the only lever the store has over demand, which makes demand **endogenous** to the pricing
decision.

### 4.3 Useful results from the demand curve

- **Revenue per customer** is $p \times S(p)$. It rises at first as the price goes up, then falls
  as too many customers leave. The peak is the **revenue-maximising price** $p^*$.
- **Price elasticity** measures how sensitive customers are to price. Revenue peaks where
  elasticity equals −1, the point where a 1% price rise loses exactly 1% of buyers.

### 4.4 When there is not enough stock

- If demand is higher than stock, the extra customers are **lost**. They do not wait or come back
  later (**lost-sales** assumption, no backorders).
- Customers are **myopic**: they buy now or not at all, and never wait for a discount.
- Because sales are capped by stock, recorded sales can understate true demand. This is called
  **censored demand**, and lost sales have to be estimated afterwards.

---

## 5. Supply: how the store buys stock

### 5.1 Two suppliers per department (dual sourcing)

| | Budget supplier | Premium supplier |
|---|---|---|
| **Cost** | Cheaper | More expensive |
| **Delivery** | Slower | Faster |

This is the classic **cost vs. speed trade-off**. The store can save money or get stock sooner,
but not both.

### 5.2 Bulk discounts (quantity discounts)

Each supplier sells only in **fixed lot sizes** (for example 25, 85 or 250 units). Bigger lots get
a lower unit price through **price breaks**: the medium and large lots cost 50% and 20% of the
small-lot unit price.

### 5.3 Uncertain prices and delivery times

- **Supplier prices** change each time the store asks for a quote. Each quote is a random draw
  around the supplier's usual price.
- **Lead time**, the time from placing an order to receiving it, is random and follows a
  **shifted exponential distribution**. There is a guaranteed minimum delay, and usually extra
  delay on top.
- The supplier **quotes** the minimum lead time, but the **actual** lead time is drawn when the
  order is placed. Deliveries are therefore often later than promised, as in real life.
- A **congestion factor** can stretch all lead times to simulate a busy or disrupted market. It is
  set to zero in the current runs.

---

## 6. Stock: how inventory changes

### 6.1 Perishable stock

Each department has a **shelf life**, ranging from 12 days for FOODS_3 to 500 days for HOBBIES.
Each delivery becomes its own lot with an expiry date. Stock past its expiry date is thrown away;
this loss is called **outdating** or wastage.

### 6.2 Selling oldest first (FEFO)

Customers are served from the lot that expires soonest, a rule called **first-expired,
first-out (FEFO)**. This keeps waste as low as possible. Lots are stored in a **priority queue**
(a heap) sorted by expiry date, so the next one to sell is always at the front.

### 6.3 Stock balance

For each department, tomorrow's stock is:

$$I_{t+1} = I_t + R_t - \text{Sold}_t - W_t$$

where
- $I_t$ = stock today
- $R_t$ = units delivered today
- $\text{Sold}_t = \min\big(I_t + R_t,\ \text{potential demand}_t \times S(p_t)\big)$, so the store
  cannot sell more than it has
- $W_t$ = units thrown away today

### 6.4 Inventory position

When deciding whether to reorder, the store looks at its **inventory position**:

$$\text{inventory position} = \text{stock on hand} + \text{stock on order}$$

Counting stock on order prevents the store from ordering again for something that is already on
its way.

---

## 7. What the store can and cannot see (information structure)

| The store **can** see | The store **cannot** see |
|---|---|
| Its own stock, on hand and on order | The true future demand |
| Supplier quotes (price and promised lead time) | The true lead-time distributions |
| Past demand | Future supplier prices |
| Actual lead times of past deliveries | |
| A demand forecast | |

**The forecast.** The forecast is built from the true future demand with a **controlled error**
added. Each department has a set accuracy level, measured as **mean absolute percentage error
(MAPE)**, ranging from 10% to 25% depending on the department. Fixing forecast accuracy this way
means that any differences between policies come from *how good the decisions are*, not from how
good a forecasting model happens to be.

---

## 8. Randomness and fair comparison

### 8.1 Sources of randomness

| Source | How it is modelled |
|---|---|
| Customer demand | PCA patterns + normal (Gaussian) noise |
| Customer WTP | Kernel density estimate from historical prices |
| Supplier prices | Normal distribution around the usual price |
| Lead times | Shifted exponential distribution |
| Forecast error | Normal distribution around each department's MAPE |

### 8.2 Separate random streams

One master **seed** is split into three independent **random number streams**, one each for
demand, suppliers and forecasts. Drawing more numbers from one stream therefore never shifts the
others.

### 8.3 Common random numbers

Run *k* always uses seed *k*, so every policy faces **exactly the same customer demand** in each
run. This technique is called **common random numbers (CRN)**. It removes luck from the
comparison: if one policy beats another, the cause is the policy and not easier demand.

*Note:* this match is exact for demand only. Policies that ask for more quotes or forecasts use up
the supplier and forecast streams at different rates, so their lead times and supplier prices can
differ slightly.

---

## 9. How performance is measured

| Measure | Meaning |
|---|---|
| **Revenue** | Units sold × price |
| **Purchase cost** | What the store paid suppliers |
| **Profit** | Revenue − purchase cost |
| **Wastage** | Units thrown away after expiry |
| **Stockout days** | Days that ended with an empty shelf |
| **Lost sales** | Estimated customers turned away on stockout days |
| **Underage cost** | Profit lost on those turned-away customers |

Each measure is averaged over the 100 runs and reported with its standard deviation.

---

## 10. Checking the model

**Verification** (the code does what it is meant to):
- Expiry events are safe to process twice, so there is no double counting.
- Orders are rejected if the lot size is one the supplier does not sell.
- The shortcut pricing calculation matches a full optimisation solver (cvxpy) to within 0.0000001.

**Validation** (the model behaves like the real world):
- Generated demand should match the historical averages, spread and links between departments.
- WTP curves should match the historical relationship between price and units sold.

---

## 11. Assumptions and limitations

- **WTP is estimated from past sale prices.** It reflects the prices Walmart chose to charge, not
  necessarily what customers would truly pay.
- **Generated demand is smoother than reality.** PCA keeps the main patterns but drops some
  day-to-day noise, and holidays fall on the same days in every run.
- **No holding cost.** Keeping stock on the shelf costs nothing apart from the risk of it expiring.
- **Unlimited supplier capacity.** Suppliers can always fill an order.
- **No substitution.** A customer who cannot find FOODS_1 does not switch to FOODS_2.
- **Daily time steps.** Events within a day are not timed more precisely than "today".
