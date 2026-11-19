"""Importable simulation environment with the configuration of newswendor_DynamicPricing.ipynb.

Everything the notebook builds in cells (data, WTP curves, suppliers, forecast, replenishment
policy, two-price dynamic pricing, daily loop) lives here as functions and classes, so it can be
imported to run baselines and to train / test RL pricing policies.

Three layers:

* load_data()      M5 data -> (wide_df, ca_1, wtp_kde), cached under data/cache/.
* RetailWorld      the store: Env, Demand, Inventory, suppliers and the replenishment Policy.
                   pricing='dp2price' reproduces the notebook (two-price LP schedule);
                   pricing='external' leaves prices to the caller (an RL agent, or fixed prices).
* PricingEnv       one-department RL environment: reset() / step(action) / action_masks(),
                   where the action is one of n_prices price levels.

Usage, from a notebook in code/work/:

    import sys; sys.path.insert(0, '../sim')
    import set_enviroment_a1 as rre

    data = rre.load_data()                                   # ~30 s first time, cached after
    world = rre.RetailWorld(data)                            # notebook policy
    log = world.run(seeds=range(100))                        # same output as the notebook

    env = rre.PricingEnv('FOODS_3', data=data)               # RL training
    obs, _ = env.reset()
    obs, reward, terminated, truncated, info = env.step(5)

    test_world = rre.RetailWorld(data, pricing='external')   # RL evaluation, all departments
    log = test_world.run(seeds=range(100), price_fn=rre.rl_price_fn(agent_act))
"""

import sys
from collections import defaultdict, deque
from pathlib import Path

import cvxpy as cp
import numpy as np
import polars as pl
import scipy
import scipy.stats
from scipy.optimize import minimize_scalar

sys.path.insert(0, str(Path(__file__).resolve().parent))      # sim.py lives next to this file
from sim import Demand, Env, Inventory, Supplier, SupplierInfo  # noqa: E402

try:
    from tqdm.auto import tqdm as _progress
except ImportError:                                            # progress bar is optional
    _progress = None


# =============================================================================================
# Configuration (as in newswendor_DynamicPricing.ipynb)
# =============================================================================================

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_FOLDER = REPO_ROOT / 'data'
RESULTS_FOLDER = REPO_ROOT / 'results'

STORE = ['CA_1', 'CA_2']
SEED = 42
EPOCH = 100
RUN_DAYS = 1900

# Opening prices and the shelf life of the opening lots. Quantities are not used: opening
# stock is the day-0 newsvendor quantity.
INITIAL_ORDER = {
    'FOODS_1':     {'quantity': 270,  'price': 2.5, 'shelf_life': 60},
    'FOODS_2':     {'quantity': 5000, 'price': 3.3, 'shelf_life': 120},
    'FOODS_3':     {'quantity': 5000, 'price': 2.3, 'shelf_life': 10},
    'HOBBIES_1':   {'quantity': 5000, 'price': 3.7, 'shelf_life': 1000},
    'HOBBIES_2':   {'quantity': 5000, 'price': 2.3, 'shelf_life': 1000},
    'HOUSEHOLD_1': {'quantity': 5000, 'price': 3.9, 'shelf_life': 1000},
    'HOUSEHOLD_2': {'quantity': 5000, 'price': 5.2, 'shelf_life': 1000},
}

# Shelf life of lots delivered by suppliers (days).
SHELF_LIFE = {
    'FOODS_1': 270, 'FOODS_2': 120, 'FOODS_3': 12,
    'HOBBIES_1': 500, 'HOBBIES_2': 500,
    'HOUSEHOLD_1': 400, 'HOUSEHOLD_2': 400,
}

# Supplier terms per department: a cheaper slower 'low' supplier and a dearer faster 'prem' one.
DEPT_DATA: dict[str, dict] = {
    'FOODS_3':     dict(price_mean=2.33, price_std=0.16, lead_min=1, lead_max=4,
                        low=dict(q=[65, 215, 640], margin=0.35),
                        prem=dict(q=[60, 200, 610], margin=0.20)),
    'FOODS_1':     dict(price_mean=2.30, price_std=0.20, lead_min=5, lead_max=10,
                        low=dict(q=[25, 85, 250], margin=0.45),
                        prem=dict(q=[25, 80, 230], margin=0.30)),
    'FOODS_2':     dict(price_mean=3.31, price_std=0.27, lead_min=3, lead_max=7,
                        low=dict(q=[17, 60, 170], margin=0.30),
                        prem=dict(q=[17, 55, 170], margin=0.20)),
    'HOBBIES_1':   dict(price_mean=3.63, price_std=0.76, lead_min=5, lead_max=10,
                        low=dict(q=[30, 100, 300], margin=0.40),
                        prem=dict(q=[20, 75, 250], margin=0.30)),
    'HOBBIES_2':   dict(price_mean=2.19, price_std=0.41, lead_min=7, lead_max=14,
                        low=dict(q=[20, 100, 200], margin=0.50),
                        prem=dict(q=[20, 100, 200], margin=0.35)),
    'HOUSEHOLD_1': dict(price_mean=3.87, price_std=0.27, lead_min=7, lead_max=14,
                        low=dict(q=[25, 85, 250], margin=0.38),
                        prem=dict(q=[25, 75, 230], margin=0.28)),
    'HOUSEHOLD_2': dict(price_mean=5.18, price_std=0.23, lead_min=5, lead_max=10,
                        low=dict(q=[50, 170, 500], margin=0.30),
                        prem=dict(q=[50, 160, 480], margin=0.20)),
}
TIER_MULTIPLIERS = [1.0, 0.5, 0.2]   # unit price of each lot size relative to the smallest lot

# Policy settings.
MARKUP = {
    'FOODS_1': 0.4, 'FOODS_2': 0.25, 'FOODS_3': 0.3,
    'HOBBIES_1': 0.35, 'HOBBIES_2': 0.4,
    'HOUSEHOLD_1': 0.32, 'HOUSEHOLD_2': 0.25,
}
# Decision horizon: days an order should cover, and the pricing horizon.
DECISION_HORIZON = {
    'FOODS_1': 60, 'FOODS_2': 120, 'FOODS_3': 10,
    'HOBBIES_1': 500, 'HOBBIES_2': 500,
    'HOUSEHOLD_1': 400, 'HOUSEHOLD_2': 400,
}
LEAD_WINDOW = 5      # realised lead times kept per product, pooled over its suppliers
Z_SAFETY = 1.65      # reorder-point safety factor (~95% of lead-time cycles)

# Forecast accuracy: each call draws a fresh MAPE from these.
FORECAST_WINDOW = 30
FORECAST_MAPE_BY_DEPT = {
    'FOODS_1':     scipy.stats.norm(loc=0.20, scale=0.03),
    'FOODS_2':     scipy.stats.norm(loc=0.15, scale=0.03),
    'FOODS_3':     scipy.stats.norm(loc=0.10, scale=0.03),
    'HOBBIES_1':   scipy.stats.norm(loc=0.15, scale=0.03),
    'HOBBIES_2':   scipy.stats.norm(loc=0.25, scale=0.03),
    'HOUSEHOLD_1': scipy.stats.norm(loc=0.10, scale=0.03),
    'HOUSEHOLD_2': scipy.stats.norm(loc=0.15, scale=0.03),
}

LOG_SCHEMA = {
    'epoch': pl.Int32, 'seed': pl.Int64, 'day': pl.Int32, 'product': pl.Utf8,
    'demand': pl.Int64, 'sold': pl.Int64, 'price': pl.Float64, 'wastage': pl.Int64,
    'order_qty': pl.Int64, 'order_cost': pl.Float64, 'stock': pl.Int64,
}


# =============================================================================================
# Data
# =============================================================================================

def load_data(data_folder=DATA_FOLDER, use_cache=True):
    """M5 data for stores CA_1 + CA_2 at department grain.

    Returns (wide_df, ca_1, wtp_kde):
      wide_df  one row per department, one column per day (d_1 ... d_1941) of units sold
      ca_1     one row per department-day: total_demand, total_weighted_price, price_avg, day_n
      wtp_kde  department -> demand-weighted KDE of daily average price (willingness to pay)

    The two tables are cached as parquet in data/cache/ after the first build (~30 s).
    """
    data_folder = Path(data_folder)
    cache = data_folder / 'cache'
    wide_path, ca_1_path = cache / 'wide_df.parquet', cache / 'ca_1.parquet'

    if use_cache and wide_path.exists() and ca_1_path.exists():
        wide_df, ca_1 = pl.read_parquet(wide_path), pl.read_parquet(ca_1_path)
    else:
        wide_df, ca_1 = _build_tables(data_folder)
        if use_cache:
            cache.mkdir(exist_ok=True)
            wide_df.write_parquet(wide_path)
            ca_1.write_parquet(ca_1_path)

    wtp_kde = {
        df['dept_id'][0]: scipy.stats.gaussian_kde(df['price_avg'], weights=df['total_demand'])
        for df in ca_1.partition_by('dept_id', maintain_order=True)
    }
    return wide_df, ca_1, wtp_kde


def _build_tables(data_folder):
    calendar_lf = pl.scan_csv(data_folder / 'calendar.csv')
    prices_lf = pl.scan_csv(data_folder / 'sell_prices.csv')
    sales_lf = pl.scan_csv(data_folder / 'sales_train_evaluation.csv')

    day_cols = [c for c in sales_lf.collect_schema().names() if c.startswith('d_')]
    day_order = sorted(day_cols, key=lambda c: int(c.removeprefix('d_')))

    merged_lf = (
        sales_lf
        .filter(pl.col('store_id').is_in(STORE))
        .select(['item_id', *day_cols])
        .unpivot(index='item_id', on=day_cols, variable_name='d', value_name='demand')
        .join(calendar_lf.select('d', 'wm_yr_wk'), on='d', how='inner')
        .join(prices_lf.filter(pl.col('store_id').is_in(STORE))
              .select('item_id', 'wm_yr_wk', 'sell_price'),
              on=['item_id', 'wm_yr_wk'], how='inner')        # drops unpriced item-weeks
        .with_columns(
            pl.col('item_id').str.split('_').list.slice(0, 2).list.join('_').alias('dept_id'),
            pl.col('d').str.split('_').list.get(-1).cast(pl.Int32).alias('day_n'),
        )
    )

    # department x day: units and unit-weighted average price
    ca_1 = (
        merged_lf
        .with_columns((pl.col('demand') * pl.col('sell_price')).alias('total_price'))
        .group_by('d', 'dept_id')
        .agg(pl.col('demand').sum().alias('total_demand'),
             pl.col('total_price').sum().alias('total_weighted_price'))
        .with_columns(
            (pl.col('total_weighted_price') / pl.col('total_demand')).round(3).alias('price_avg'),
            pl.col('d').str.split('_').list.get(-1).cast(pl.Int32).alias('day_n'),
        )
        .drop_nans(subset='price_avg')                          # zero-demand days give 0/0
        .sort('dept_id', 'day_n')                               # deterministic order
        .collect()
    )

    day_spine = pl.LazyFrame({'day_n': [int(c.removeprefix('d_')) for c in day_order]},
                             schema={'day_n': pl.Int32})
    dept_day_lf = ca_1.lazy().select('dept_id', 'day_n', 'total_demand')
    wide_df = (
        dept_day_lf.select('dept_id').unique()
        .join(day_spine, how='cross')                           # every dept x every day
        .join(dept_day_lf, on=['dept_id', 'day_n'], how='left')
        .with_columns(pl.col('total_demand').fill_null(0))
        .group_by('dept_id')
        .agg(pl.col('total_demand').sort_by('day_n'))
        .with_columns(pl.col('total_demand').list.to_struct(fields=day_order))
        .unnest('total_demand')
        .sort('dept_id')
        .collect()
    )
    return wide_df, ca_1


def build_supplier_book():
    """Two suppliers per department, three lot sizes each with a volume price break."""
    def price_dists(price_mean, price_std, margin):
        cv = price_std / price_mean                             # same CV across tiers
        lot_price = price_mean * (1 - margin)
        return [scipy.stats.norm(loc=lot_price * m, scale=lot_price * m * cv) for m in TIER_MULTIPLIERS]

    book = Supplier()
    for dept, cfg in DEPT_DATA.items():
        book.add_supplier(
            SupplierInfo(name=f'{dept}_low', product=dept,
                         price=price_dists(cfg['price_mean'], cfg['price_std'], cfg['low']['margin']),
                         lead_time=scipy.stats.expon(loc=cfg['lead_max'], scale=cfg['lead_max']),
                         order_quantity=cfg['low']['q']),
            SupplierInfo(name=f'{dept}_premium', product=dept,
                         price=price_dists(cfg['price_mean'], cfg['price_std'], cfg['prem']['margin']),
                         lead_time=scipy.stats.expon(loc=cfg['lead_min'], scale=cfg['lead_min']),
                         order_quantity=cfg['prem']['q']),
        )
    return book


# =============================================================================================
# Forecast
# =============================================================================================

def _draw_mape(value, rng):
    """A frozen scipy distribution becomes a draw (floored at 0); a number passes through."""
    if hasattr(value, 'rvs'):
        return max(0.0, float(value.rvs(random_state=rng)))
    return float(value)


def forecast(env, data, product, products, mape=FORECAST_MAPE_BY_DEPT, window=FORECAST_WINDOW,
             seed=None, history=False):
    """Forecast `product`'s daily demand for the next `window` days.

    `data` is (days, products) actual demand with columns in `products` order. Per-day relative
    errors are rescaled so their mean (the MAPE) equals a draw from `mape`. Draws come from
    env.forecast_rng unless `seed` is given. Returns (past, forecast); `past` is all actual
    demand before today if `history`, else empty.
    """
    rng = np.random.default_rng(env.forecast_rng if seed is None else seed)
    data = np.asarray(data)
    n_days = data.shape[0]
    col = list(products).index(product)

    start = min(env.clock + 1, n_days)
    end = min(start + window, n_days)
    y_true = data[start:end, col].astype(float)

    if isinstance(mape, dict):
        mape = mape[product]
    target = _draw_mape(mape, rng)

    scored = y_true > 0
    magnitudes = rng.uniform(0.1, 1.0, size=y_true.size)
    if scored.any():
        magnitudes = magnitudes * (target / magnitudes[scored].mean())

    signs = rng.choice([-1.0, 1.0], size=y_true.size)
    y_pred = y_true * (1 + signs * magnitudes)

    past = data[:env.clock, col] if history else data[:0, col]
    return past, np.round(y_pred)


# =============================================================================================
# Policy: replenishment + two-price dynamic pricing (as in the notebook)
# =============================================================================================

class Policy:
    """Reorder point + newsvendor quantity + sourcing MILP; optional two-price LP pricing.

    Reads the world's state (clock, demand path, inventory, suppliers) through `world`.
    """

    def __init__(self, world, z=Z_SAFETY):
        self.world = world
        self.z = z
        self.reorder_point = {p: 0 for p in world.products}
        self.on_order = defaultdict(int)                                # product -> units in transit
        self.lead_obs = defaultdict(lambda: deque(maxlen=LEAD_WINDOW))  # product -> last lead times
        self.markup = dict(MARKUP)
        self.shelf_life = dict(DECISION_HORIZON)                        # decision horizon

        self.price_range = {
            dept: (lo, hi) for dept, lo, hi in
            world.ca_1.group_by('dept_id')
            .agg(pl.col('price_avg').min().alias('lo'), pl.col('price_avg').max().alias('hi'))
            .iter_rows()
        }
        self.wtp = world.inventory.wtp
        self.peak_price = {}
        for product in world.products_all:
            res = minimize_scalar(lambda x: -x * self.buy_share(product, x),
                                  bounds=self.price_range[product], method='bounded')
            self.peak_price[product] = round(float(res.x), 2)

        # Lead-time prior until LEAD_WINDOW deliveries are seen: equal mix of the suppliers.
        self.lead_prior = {}
        for product in world.products_all:
            dists = [s.lead_time for s in world.supplier_book.by_product[product]]
            means = np.array([d.mean() for d in dists])
            variances = np.array([d.var() for d in dists])
            self.lead_prior[product] = (means.mean(), np.sqrt(variances.mean() + means.var()))

        self.unit_cost = {}   # product -> unit cost of the latest order
        self.price_plan = {}  # product -> {id, high, low, days_high, days_low, switch_day}
        self.plan_seq = 0

    # ---- bookkeeping -------------------------------------------------------------------------
    def reset_epoch(self):
        """Clear stock in transit and plans; lead-time history is kept across epochs."""
        self.on_order.clear()
        self.unit_cost.clear()
        self.price_plan.clear()

    def record_order(self, placed):
        for o in placed:
            self.on_order[o['product']] += o['quantity']

    def record_arrival(self, product, quantity, lead_time):
        self.on_order[product] -= quantity
        self.lead_obs[product].append(lead_time)

    def _forecast(self, product, window, history=False):
        w = self.world
        return forecast(w.sim, w.demand_data, product, w.products_all, window=window, history=history)

    # ---- replenishment -----------------------------------------------------------------------
    def lead_stats(self, product):
        obs = self.lead_obs[product]
        if len(obs) == LEAD_WINDOW:
            return np.mean(obs), np.std(obs, ddof=1)
        return self.lead_prior[product]

    def update_reorder_point(self, product):
        """d*E[L] + z*sigma_LT, with demand and lead-time uncertainty."""
        _, fc = self._forecast(product, self.shelf_life[product])
        d, sigma = np.mean(fc), np.std(fc)
        L, sL = self.lead_stats(product)
        sigma_lt = np.sqrt(L * sigma**2 + d**2 * sL**2)
        self.reorder_point[product] = d * L + self.z * sigma_lt

    def newsvendor(self, product, quotes):
        """Critical-ratio quantile of demand over the decision horizon."""
        decision_period = self.shelf_life[product]
        max_cost = np.max([x for q in quotes for x in q.price])
        max_price = max_cost * (1 + self.markup[product])
        historical, fc = self._forecast(product, decision_period, history=True)

        cost_u, cost_o = (max_price - max_cost), max_cost
        critical_ratio = cost_u / (cost_u + cost_o)

        n_windows = len(historical) // decision_period
        if n_windows == 0:
            return np.round(np.sum(fc))      # less than one window of history: forecast total
        historical = np.asarray(historical[:n_windows * decision_period])
        window_demand = historical.reshape(-1, decision_period).sum(axis=1)
        return np.round(np.percentile(window_demand, critical_ratio * 100))

    def order_lp(self, product, quotes, order_qty):
        """MILP: one supplier and lots per size, trading price against lead-time lost sales.

        Returns (quote, lots) with lots[t] lots of quote.order_quantity[t], or None.
        """
        _, fc = self._forecast(product, self.shelf_life[product])
        d, sigma = np.mean(fc), max(np.std(fc), 1e-6)

        cost = np.array([q.price for q in quotes], dtype=float)
        lot = np.array([q.order_quantity for q in quotes], dtype=float)
        lead_mean = np.array([q.supplier.lead_time.mean() for q in quotes], dtype=float)
        lead_std = np.array([q.supplier.lead_time.std() for q in quotes], dtype=float)

        margin = np.max(cost) * self.markup[product]
        cost_o = np.max(cost)

        d_l = d * lead_mean
        sigma_l = np.sqrt(lead_mean * sigma**2 + d**2 * lead_std**2)
        position = self.world.inventory.inventory_status().get(product, {'quantity': 0})['quantity']
        k = (position - d_l) / sigma_l
        shortfall = sigma_l * (scipy.stats.norm.pdf(k) - k * scipy.stats.norm.sf(k))

        x = cp.Variable(lot.shape, integer=True, nonneg=True)
        y = cp.Variable(len(quotes), boolean=True)
        under = cp.Variable(integer=True, nonneg=True)
        over = cp.Variable(integer=True, nonneg=True)
        qty = cp.sum(cp.multiply(lot, x))

        constraints = [
            cp.sum(y) == 1,
            x <= 100 * cp.reshape(y, (len(quotes), 1), order='C'),
            qty + under - over == order_qty,
        ]
        obj = (cp.sum(cp.multiply(cost * lot, x)) + margin * (shortfall @ y)
               + margin * under + cost_o * over)
        prob = cp.Problem(cp.Minimize(obj), constraints)
        prob.solve(solver=cp.HIGHS)
        if prob.status != cp.OPTIMAL:
            return None
        s = int(np.argmax(y.value))
        return quotes[s], np.round(x.value[s]).astype(int)

    def decision(self, inventory, **kwargs):
        """Reorder any product at or below its reorder point; returns the orders placed today."""
        w = self.world
        placed_today = []
        status = inventory.inventory_status()
        for product in self.reorder_point:
            on_hand = status.get(product, {'quantity': 0})['quantity']
            position = on_hand + self.on_order[product]

            self.update_reorder_point(product)
            if position <= self.reorder_point[product]:
                quotes = w.supplier_book.order_proposal(product, rng=w.sim.supplier_rng)
                order_qty = self.newsvendor(product, quotes)
                result = self.order_lp(product, quotes, order_qty)
                if result is None:
                    continue
                quote, lots = result
                placed = w.supplier_book.order_purchase(
                    w.sim, {quote.supplier: {'lots': [int(n) for n in lots], 'price': quote.price}},
                    demand_index=0,
                )
                self.record_order(placed)
                placed_today.extend(placed)
                if placed:
                    self.unit_cost[product] = placed[0]['unit_price']
        return placed_today

    # ---- two-price dynamic pricing (pricing='dp2price') --------------------------------------
    def buy_share(self, product, price):
        """Share of potential demand still willing to buy at `price` (1 - WTP CDF)."""
        return float(np.clip(1 - self.wtp[product].integrate_box_1d(-np.inf, price), 0.0, 1.0))

    @staticmethod
    def solve_time_lp(prices, sales_rate, stock, horizon):
        """Days at each of two prices maximising revenue s.t. stock and horizon (exact)."""
        p, r = np.asarray(prices, float), np.asarray(sales_rate, float)
        best_t, best_rev = np.zeros(2), 0.0
        for i in range(2):
            if r[i] > 0:
                t = np.zeros(2)
                t[i] = min(horizon, stock / r[i])
                if p[i] * r[i] * t[i] > best_rev:
                    best_t, best_rev = t, p[i] * r[i] * t[i]
        if r[0] != r[1]:
            t0 = (stock - r[1] * horizon) / (r[0] - r[1])
            t = np.array([t0, horizon - t0])
            if (t >= 0).all() and (p * r) @ t > best_rev:
                best_t, best_rev = t, (p * r) @ t
        return best_t

    def plan_prices(self, inventory, product):
        """On restock: cost-plus vs revenue-peak LP; charge the higher price first, then
        switch to the lower one via a 'price_change' event."""
        w = self.world
        on_hand = inventory.inventory_status().get(product, {'quantity': 0})['quantity']
        stock_info = inventory.stock.get(product)
        self.price_plan.pop(product, None)
        if on_hand <= 0 or stock_info is None or not stock_info.sku:
            return None

        horizon = self.shelf_life[product]
        _, fc = self._forecast(product, horizon)
        d = float(np.mean(fc)) if len(fc) else 0.0
        if d <= 0:
            return None

        cost = self.unit_cost.get(product)
        cost_plus = cost * (1 + self.markup[product]) if cost is not None else inventory.stock[product].price
        prices = np.array([round(float(cost_plus), 2), self.peak_price[product]])
        prices = prices[np.isfinite(prices)]
        if cost is not None:
            prices = prices[prices >= cost]
        if len(prices) == 0:
            return None
        prices = np.unique(prices)

        sales_rate = np.array([self.buy_share(product, x) for x in prices]) * d
        if len(prices) == 1:
            days = np.array([min(horizon, on_hand / sales_rate[0]) if sales_rate[0] > 0 else 0.0])
        else:
            days = self.solve_time_lp(prices, sales_rate, on_hand, horizon)
        used = days > 1e-6
        if not used.any():
            return None

        high, low = float(prices[used].max()), float(prices[used].min())
        days_high = float(days[used][prices[used].argmax()])
        self.plan_seq += 1
        plan = {
            'id': self.plan_seq, 'high': high, 'low': low,
            'days_high': days_high, 'days_low': float(days[used].sum() - days_high),
            'switch_day': w.sim.clock + round(days_high) if low < high else None,
        }
        self.price_plan[product] = plan

        if plan['switch_day'] is not None and plan['switch_day'] <= w.sim.clock:
            inventory.set_price({product: {'price': low}})
        else:
            inventory.set_price({product: {'price': high}})
            if plan['switch_day'] is not None:
                w.sim.add_event(plan['switch_day'], ('price_change', product, low, plan['id']))
        return plan

    def on_price_change(self, inventory, product, price, plan_id):
        plan = self.price_plan.get(product)
        if plan is not None and plan['id'] == plan_id:
            inventory.set_price({product: {'price': price}})


# =============================================================================================
# World: the store and its daily loop
# =============================================================================================

class RetailWorld:
    """The simulated store with the notebook's configuration.

    pricing='dp2price'  the notebook's two-price LP schedule (planned on every restock)
    pricing='external'  prices only change when the caller passes them to finish_day / run
                        (opening prices come from INITIAL_ORDER, so no price_fn = constant price)

    products: subset of departments to simulate (default all 7). Demand is still generated
    for all departments jointly, so a department sees the same demand path either way.
    """

    def __init__(self, data=None, products=None, pricing='dp2price', seed=SEED):
        if pricing not in ('dp2price', 'external'):
            raise ValueError("pricing must be 'dp2price' or 'external'")
        self.wide_df, self.ca_1, self.wtp_kde = data if data is not None else load_data()
        self.demand = Demand(self.wide_df, self.wtp_kde)
        self.products_all = list(self.demand.product)
        self.products = list(products) if products is not None else list(self.products_all)
        self.col = {p: i for i, p in enumerate(self.products_all)}
        self.pricing = pricing

        self.sim = Env(seed=seed)
        self.inventory = Inventory(self.sim, self.demand.wtp)
        self.supplier_book = build_supplier_book()
        self.policy = Policy(self)

        self.demand_data = np.zeros((0, len(self.products_all)), dtype=int)   # set by reset()
        self.wastage = []
        self.day_wastage = defaultdict(int)
        self.last_restock = {}
        self.epoch_seed = None
        self.hist_mean_demand = dict(
            self.ca_1.group_by('dept_id').agg(pl.col('total_demand').mean()).iter_rows()
        )

    # ---- epoch ---------------------------------------------------------------------------------
    def reset(self, seed):
        """Start an epoch from `seed`: new demand path, opening stock = day-0 newsvendor qty."""
        self.epoch_seed = int(seed)
        self.sim.reset(seed=self.epoch_seed)
        self.inventory.reset(keep_price=False)
        self.policy.reset_epoch()
        self.wastage = []
        self.day_wastage = defaultdict(int)

        self.demand_data = self.demand.generate_demand(seed=self.sim.demand_rng).T.astype(int)

        opening_order = {
            p: {
                'quantity': int(self.policy.newsvendor(
                    p, self.supplier_book.order_proposal(p, rng=self.sim.supplier_rng))),
                'price': INITIAL_ORDER[p]['price'],
                'shelf_life': INITIAL_ORDER[p]['shelf_life'],
            }
            for p in self.products
        }
        self.inventory.stockup(opening_order)
        self.inventory.set_price(opening_order)
        self.last_restock = {p: 0 for p in self.products}
        if self.pricing == 'dp2price':
            for p in self.products:
                self.policy.plan_prices(self.inventory, p)

    # ---- one day, in two halves so an agent can act in between -----------------------------------
    def begin_day(self):
        """Process today's calendar events: deliveries, expiries (and price switches)."""
        self.day_wastage = defaultdict(int)
        clock = self.sim.clock
        if clock in self.sim.event_calendar:
            for event in self.sim.event_calendar[clock]:
                flushout = self._process_event(clock, event)
                if event[0] == 'expire':
                    for product, units in zip(*flushout):
                        self.day_wastage[product] += units
            self.sim.remove_event(clock)

    def finish_day(self, prices=None):
        """Apply `prices` ({product: price}, optional), sell today's demand, let the policy
        reorder, advance the clock. Returns {product: today's record}."""
        if prices:
            self.inventory.set_price({p: {'price': float(v)} for p, v in prices.items()})

        day = self.sim.clock
        todays_demand = self.demand_data[day]
        todays_price = {p: info['price'] for p, info in self.inventory.inventory_status().items()}
        sold = self.inventory.sell_stock({p: int(todays_demand[self.col[p]]) for p in self.products})

        order_qty, order_cost = defaultdict(int), defaultdict(float)
        for order in self.policy.decision(self.inventory):
            order_qty[order['product']] += order['quantity']
            order_cost[order['product']] += order['cost']

        status = self.inventory.inventory_status()
        record = {}
        for p in self.products:
            ordered = p in order_qty
            record[p] = {
                'day': day,
                'demand': int(todays_demand[self.col[p]]),
                'sold': sold.get(p, 0),
                'price': float(todays_price.get(p, np.nan)),
                'wastage': self.day_wastage[p],
                'order_qty': order_qty[p] if ordered else None,
                'order_cost': float(order_cost[p]) if ordered else None,
                'stock': status.get(p, {}).get('quantity', 0),
            }
        self.sim.clock_step()
        return record

    def _process_event(self, time, event):
        if event[0] == 'stockup':
            _, product, quantity, price, supplier_name, ordered_on = event
            self.inventory.stockup({product: {'quantity': quantity, 'shelf_life': SHELF_LIFE[product]}})
            self.policy.record_arrival(product, quantity, lead_time=time - ordered_on)
            self.last_restock[product] = time
            if self.pricing == 'dp2price':
                self.policy.plan_prices(self.inventory, product)
            return None
        if event[0] == 'price_change':
            _, product, price, plan_id = event
            self.policy.on_price_change(self.inventory, product, price, plan_id)
            return None
        if event[0] == 'expire':
            flushout = self.inventory.flushout_expired(event[1])
            self.wastage.append((time, flushout))
            return flushout

    # ---- full runs ---------------------------------------------------------------------------------
    def run(self, seeds=range(EPOCH), days=RUN_DAYS, price_fn=None, progress=True):
        """Simulate one epoch per seed, days 0..`days`, and return the daily log.

        price_fn(world) -> {product: price} or None is called every day after the morning
        events, before selling (use rl_price_fn(...) for a trained agent). The log has the
        notebook's columns, so compare_performance.ipynb can read it.
        """
        seeds = list(seeds)
        log = defaultdict(list)
        bar = _progress(seeds, desc='epochs') if (progress and _progress) else seeds
        for epoch, seed in enumerate(bar):
            self.reset(seed)
            while self.sim.clock <= days:
                self.begin_day()
                record = self.finish_day(price_fn(self) if price_fn else None)
                for p in self.products:
                    r = record[p]
                    log['epoch'].append(epoch)
                    log['seed'].append(int(seed))
                    log['product'].append(p)
                    for k in ('day', 'demand', 'sold', 'price', 'wastage', 'order_qty', 'order_cost', 'stock'):
                        log[k].append(r[k])
        return (pl.DataFrame(log, schema=LOG_SCHEMA)
                .select(list(LOG_SCHEMA))
                .with_columns(pl.col('price').fill_nan(None)))

    # ---- helpers for agents --------------------------------------------------------------------------
    def unit_cost(self, product):
        """Unit cost of the latest order; before any order, the expected unit cost of the
        cheap supplier's largest (cheapest) lot."""
        cost = self.policy.unit_cost.get(product)
        if cost is not None:
            return cost
        cfg = DEPT_DATA[product]
        return cfg['price_mean'] * (1 - cfg['low']['margin']) * TIER_MULTIPLIERS[-1]


def save_log(log, name):
    """Write a daily log to results/<name>.parquet (compare_performance reads every parquet there)."""
    RESULTS_FOLDER.mkdir(exist_ok=True)
    path = RESULTS_FOLDER / f'{name}.parquet'
    log.write_parquet(path)
    return path


