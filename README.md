# Retail Revenue Management Simulation

A discrete-event simulation of a retail store that has to decide **how much stock to buy, from
which supplier, and at what price to sell it**. Demand is uncertain, delivery times are uncertain
and some stock perishes. The project builds the simulated store from the M5 (Walmart) dataset and
uses it to compare inventory and pricing policies on exactly the same demand.

**Research question:** how much does demand-aware (dynamic) pricing add on top of a newsvendor
replenishment policy?

## Project scope

| | |
|---|---|
| **Data** | M5 Forecasting dataset: Walmart daily unit sales, weekly prices and calendar (1,941 days) |
| **Store** | Stores CA_1 and CA_2, pooled into one store |
| **Products** | 7 departments, each treated as one product: FOODS_1–3, HOBBIES_1–2, HOUSEHOLD_1–2 |
| **Suppliers** | Two per department: a cheaper, slower one and a dearer, faster one. Each offers three lot sizes with bulk discounts, random prices and random lead times |
| **Customers** | Potential demand comes from a PCA-based generator fitted to the history. The share who buy at a given price comes from a willingness-to-pay (WTP) curve estimated from historical prices |
| **Stock** | Held as dated lots, sold oldest first (FEFO). Lots expire after a shelf life that varies by department |
| **Experiment** | 100 epochs (replications) × 1,901 days. Seeded random streams give every policy the same demand paths (common random numbers) |

### What the simulation covers

- **Environment** (`code/sim/sim.py`): the daily clock and event calendar (deliveries, expiries),
  the demand generator, the inventory with expiry, and the supplier book with quotes and orders.
- **Policy** (the `Policy` class in each simulation notebook): everything the store decides.
  - **Replenishment, shared by all policies:**
    - a reorder point from forecast demand and the last 5 actual lead times;
    - a newsvendor order quantity;
    - a mixed-integer program that picks the supplier and lot sizes, trading price against
      lost sales during the lead time.
  - **Pricing, which is what differs between policies:**

| Policy | Pricing rule | Notebook | Results file |
|---|---|---|---|
| P0. Constant price | Each department's historical mean price, fixed for the whole run | `newswendor_constant_price.ipynb` | `Newsvendor_SingleSupplier_ConstantPrice.parquet` |
| P1. Cost-plus | On every order, price = unit cost × (1 + markup) | `newswendor_apply_margin.ipynb` | `Newsvendor_SingleSupplier_MarginPrice.parquet` |
| P2. Bid price, single price | On every restock, a time-at-price LP gives the bid price π (the shadow price of stock), then price = argmax (p − π) × share who buy at p | `newswendor_DP_ConstantPrice.ipynb` | `Newsvendor_SingleSupplier_DP_SinglePrice.parquet` |
| P3. Two-price schedule | Same LP, but charge the higher price for its planned days, then switch to the lower price through a calendar event | `newswendor_DynamicPricing.ipynb` | `Newsvendor_SingleSupplier_DynamicPrice.parquet` |

### Performance measures

`compare_performance.ipynb` summarises each run per epoch and department, then reports the
mean and standard deviation over epochs. The measures are:

- demand, units sold and revenue;
- wastage, order quantity and purchase cost;
- stockout days, estimated lost units and underage cost.

## Folder structure

```
retail-revenue-management/
├── code/
│   ├── sim/                          # simulation
│   │   ├── sim.py                    # environment: Env, Demand, Inventory, Supplier, SupplierInfo, Quote
│   │   ├── newswendor_constant_price.ipynb   # P0: constant price
│   │   ├── newswendor_apply_margin.ipynb     # P1: cost-plus price
│   │   ├── newswendor_DP_ConstantPrice.ipynb # P2: bid-price LP, one price per restock
│   │   ├── newswendor_DynamicPricing.ipynb   # P3: bid-price LP, two-price schedule
│   │   └── compare_performance.ipynb         # KPIs per policy, mean/std over epochs, Excel export
│   └── work/                         # exploratory analysis
│       ├── work.ipynb                # data prep, WTP/demand curves, demand generator, shelf-life proxy
│       └── abc-xyz-analysis.ipynb    # ABC-XYZ classification of departments
├── data/                             # M5 raw data (not in git, see "Setup")
│   ├── calendar.csv
│   ├── sales_train_evaluation.csv
│   ├── sales_train_validation.csv
│   ├── sell_prices.csv
│   ├── sample_submission.csv
│   └── work/supplier_config.xlsx     # supplier parameters
├── results/                          # simulation output (not in git)
│   ├── *.parquet                     # daily log per policy: one row per epoch × day × department
│   ├── performance_*.xlsx            # per-epoch and summary KPIs from compare_performance
│   └── model comparison.xlsx         # side-by-side comparison of the policies
├── docs/
│   ├── environment-design.md         # environment design: state, events, demand, supply, randomness
│   └── modelling-writeup.md          # model write-up draft: data, simulation, policies, experiment
├── report/                           # report drafts (PDF / Word)
├── .gitignore
└── README.md
```

Each simulation notebook is self-contained. It loads the data, builds the environment, defines
its `Policy`, runs all epochs and writes its daily log to `results/`. Apart from the policy, the
notebooks share the same setup cells.

### Daily log columns (`results/*.parquet`)

`epoch`, `seed`, `day`, `product`, `demand` (potential customers), `sold`, `price` (price sold at
that day), `wastage`, `order_qty` and `order_cost` (empty on days with no order), `stock` (end of
day).

## Setup

1. **Data:** download the [M5 Forecasting – Accuracy](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data)
   dataset from Kaggle and unzip the CSVs into `data/`. The notebooks read
   `calendar.csv`, `sell_prices.csv` and `sales_train_evaluation.csv` from there.
2. **Environment:** Python 3.13 (the conda environment `dev_env` was used) with:
   ```
   polars numpy scipy scikit-learn cvxpy matplotlib seaborn tqdm ipywidgets xlsxwriter
   ```
   cvxpy needs the HiGHS and Clarabel solvers. Recent cvxpy versions install both.
3. **Run:** open a simulation notebook in `code/sim/` and run all cells. Paths are relative to
   the notebook, so run it from that folder. A full run (100 epochs × 1,901 days) takes about
   25 minutes. For quick tests, lower `EPOCH` and `RUN_DAYS` in the *Simulation* cell.
4. **Compare:** run `code/sim/compare_performance.ipynb`. It reads every `.parquet` in
   `results/` and writes the KPI workbooks there.

## Reproducibility

- `Env(seed=...)` splits one seed into separate random streams for demand, suppliers and forecasts.
- Each epoch is reseeded from `epoch_seeds`, so any single epoch can be replayed with
  `sim.reset(seed=epoch_seeds[k])`.
- Every policy uses the same epoch seeds, so they all see the same demand paths.

## Notes

- The notebook files keep the original spelling `newswendor_*`. The method names and the newer
  output files use *newsvendor*.
- `data/` and `results/` are ignored by git: the raw data is large, and results can be
  regenerated.
