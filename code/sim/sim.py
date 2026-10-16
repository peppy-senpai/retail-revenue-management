"""Simulation components for the retail revenue-management model.

Three pieces, meant to be driven by a policy loop:

* ``Demand``    - generates synthetic daily demand paths by perturbing the real
                  department series in PCA space.
* ``Env``       - the simulation clock everything else reads its "today" from.
* ``Inventory`` - per-department stock held as expiry-ordered lots (FEFO).

Built against the department-grain ``wide_df`` produced in ``sim.ipynb``:
one row per department, one column per day.
"""

import heapq
from collections import defaultdict
from typing import NamedTuple

import numpy as np
import polars as pl
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class Demand:
    """Synthetic demand generator fitted on the historical department series."""

    def __init__(self, wide_df, wtp, n_component=0.95):
        self.wide_df = wide_df
        # Row order of wide_df == row order of every matrix below. Callers rely on
        # this to map a row index back to a department name.
        self.product = self.wide_df['dept_id'].to_list()
        self.n_component = n_component
        self.wtp = wtp                  # dept -> scipy gaussian_kde of willingness to pay
        self.get_pca()

    def get_pca(self):
        # x: (n_departments, n_days) matrix of daily unit demand.
        x = self.wide_df.select(
            pl.exclude('dept_id')
        ).to_numpy()

        # Standardise *per department*, not per day. StandardScaler works column-wise, so x is
        # transposed in and back out: each department's series gets its own mean and sd. Without
        # this, FOODS_3 (thousands of units/day) would swamp HOUSEHOLD_2 (hundreds) in every
        # component.
        self.scaler = StandardScaler()
        x_scaled = self.scaler.fit_transform(x.T).T

        # n_components=0.95 -> keep however many components explain 95% of the variance.
        # With only 7 departments there are at most 7 components to choose from.
        self.pca = PCA(n_components=self.n_component)
        self.X_reduced = self.pca.fit_transform(x_scaled).round(0)

    def generate_demand(self, seed=None, damp=0.8):
        """Draw one synthetic demand path, shaped like ``X_reduced``'s source matrix.

        Jitters each department's PCA scores by noise scaled to each component's own
        standard deviation, so dominant components move more than minor ones and the
        perturbation stays consistent with the data's covariance structure. ``damp``
        shrinks that noise; 1.0 would add as much variance again as the component carries.

        ``seed`` is an int or a ``np.random.Generator`` — pass ``env.demand_rng`` so each
        epoch draws a new path from the simulation's seed. A fixed int returns the same
        path on every call.
        """
        rng = np.random.default_rng(seed)
        noise = rng.normal(0, damp * np.sqrt(self.pca.explained_variance_), size=self.X_reduced.shape)
        sampled_score = self.X_reduced + noise

        # Invert both transforms: PCA space -> z-scores -> units.
        sampled_scaled = self.pca.inverse_transform(sampled_score)
        sampled_demand = self.scaler.inverse_transform(sampled_scaled.T).T.round(0)

        # The inverse transform is linear and unconstrained, so it can go negative.
        return np.clip(sampled_demand, 0, None)


class Env:
    """Simulation clock. Every component reads ``clock`` for the current day index.

    ``seed`` is the single source of randomness for a run. It is split into independent
    streams, one per source of uncertainty, so changing how often one is drawn (e.g. a
    policy that requests more quotes) does not shift the others — every policy sees the
    same demand paths for the same seed:

    * ``demand_rng``   - synthetic demand paths (``Demand.generate_demand``)
    * ``supplier_rng`` - quoted prices and realised lead times (``Supplier``)
    * ``forecast_rng`` - forecast errors

    ``seed=None`` draws fresh entropy, so runs are not reproducible.
    """

    def __init__(self, seed=None):
        self._seed_streams(seed)
        self._reset_state()

    def _seed_streams(self, seed):
        self.seed = seed
        demand_ss, supplier_ss, forecast_ss = np.random.SeedSequence(seed).spawn(3)
        self.demand_rng = np.random.default_rng(demand_ss)
        self.supplier_rng = np.random.default_rng(supplier_ss)
        self.forecast_rng = np.random.default_rng(forecast_ss)

    def _reset_state(self):
        self.clock = 0
        # Pending events keyed by the day they fire: {day: payload}. Supplier.order_purchase
        # books 'stockup' deliveries here via add_event; the policy loop drains them with
        # remove_event(self.clock) once the clock reaches that day.
        self.event_calendar = defaultdict(list)

    def add_event(self, time, event):
        self.event_calendar[time].append(event)

    def remove_event(self, time):
        res = self.event_calendar.pop(time, None)
        return res

    def clock_step(self, stepup=1):
        self.clock += stepup

    def reset(self, seed=None):
        """Clear the clock and calendar for a new epoch.

        With ``seed``, the random streams are re-seeded from it, so this epoch can be
        reproduced on its own by resetting with the same seed. Without it, the streams
        carry on rather than restarting: each epoch gets new draws while the run as a
        whole stays reproducible from the seed given to ``Env``.
        """
        if seed is not None:
            self._seed_streams(seed)
        self._reset_state()


class StockInfo:
    """Stock held for one department.

    ``sku`` is a min-heap of ``[expiry_day, quantity]`` lots, so the lot that expires
    soonest is always at the front — first-expired-first-out depletion.
    """

    def __init__(self):
        self.sku = []
        self.price = np.nan


class Inventory:

    def __init__(self, env, wtp):
        self.stock = defaultdict(StockInfo)
        self.env = env
        # {product: scipy gaussian_kde} of willingness to pay. Held once here rather than
        # passed on every sell_stock call, so it cannot drift from the curves Demand uses.
        self.wtp = wtp

    def reset(self, keep_price=False):
        """Drop every lot, ready for a fresh epoch. Pairs with Env.reset.

        Without this, stock and pending expiry events carry over between epochs and each run
        starts from whatever the previous one left behind.

        ``env`` and ``wtp`` survive — they are configuration, not state. Prices are state, so
        they reset to NaN by default and the epoch must call set_price again; pass
        ``keep_price=True`` to hold pricing fixed and re-initialise only the stock.
        """
        prices = {p: info.price for p, info in self.stock.items()} if keep_price else {}

        self.stock = defaultdict(StockInfo)

        for product, price in prices.items():
            self.stock[product].price = price

    def stockup(self, order):
        """Receive an order. ``order``: {product: {'quantity': int, 'shelf_life': int, ...}}."""
        for product in order:
            # shelf_life is a duration; adding the clock turns it into an absolute
            # expiry day, which is what the heap orders on.
            expiry = order[product]['shelf_life'] + self.env.clock
            quantity = order[product]['quantity']

            # A list, not a tuple, because sell_stock decrements the quantity in place
            # on partial consumption.
            heapq.heappush(self.stock[product].sku, [expiry, quantity])

            # Book the write-off on the day this lot goes bad; the policy loop pops it and
            # calls flushout_expired(product). No quantity in the payload — sell_stock will
            # have eaten into the lot by then, so the handler reads what is left off the heap.
            self.env.add_event(expiry, ('expire', product))

    def set_price(self, order):
        """Set the current sell price per department. ``order``: {product: {'price': float, ...}}."""
        for product in order:
            self.stock[product].price = order[product]['price']

    def inventory_status(self):
        """Return {product: {'quantity': total units on hand, 'price': current price}}."""
        status = {}

        for product_select, info in self.stock.items():
            # Total on hand is the sum across every lot, regardless of expiry.
            status[product_select] = {
                'quantity': sum(lot[1] for lot in info.sku),
                'price': info.price,
            }

        return status

    def flushout_expired(self, product=None):
        """Write off every lot that has reached its expiry day.

        ``product``: one department, matching the payload of an 'expire' calendar event, or
        None to sweep the whole inventory. Returns parallel lists of (product, units lost).

        Cheap because sku is expiry-ordered: the loop stops at the first lot still in date,
        so only genuinely expired lots are ever touched.

        Reads the surviving quantity off the heap rather than trusting anything recorded when
        the lot arrived — sell_stock will have eaten into it in the meantime. That also makes
        the call idempotent, so a duplicate or late 'expire' event simply writes off nothing.
        """
        if product is None:
            selected = list(self.stock)
        else:
            # Not self.stock[product]: it is a defaultdict, and indexing a product that was
            # never stocked would insert an empty entry as a side effect of the lookup.
            selected = [product] if product in self.stock else []

        flushed = []
        wasted = []

        for product_select in selected:
            sku = self.stock[product_select].sku
            product_wasted = 0

            # <= clock, not <: a lot expiring today is already unsellable today.
            while sku and sku[0][0] <= self.env.clock:
                product_wasted += heapq.heappop(sku)[1]

            flushed.append(product_select)
            wasted.append(product_wasted)

        return [flushed, wasted]

    def sell_stock(self, order):
        """Consume price-adjusted demand from stock, oldest lot first.

        ``order``: {product: units} — *potential* demand, before customers react to price.
        Prices come from ``set_price`` and the WTP curves from ``self.wtp``.

        Returns {product: units actually sold}. Unmet demand is
        ``realised demand - sold`` and is not recorded here.
        """
        sold = {}

        for product_select in order:
            product_price = self.stock[product_select].price

            # A product that has never been priced cannot be sold. Without this guard the
            # KDE integral returns NaN and round(NaN) raises ValueError — reachable as soon
            # as a supplier 'stockup' event delivers something set_price has not covered.
            if np.isnan(product_price):
                sold[product_select] = 0
                continue

            # Walk the WTP demand curve to turn potential demand into realised demand.
            # integrate_box_1d(-inf, p) is the CDF at p — the share of demand that would
            # only buy below p — so 1 - that is the share still willing to transact at p.
            # Raising the price shrinks demand_percent; this is what makes pricing bite.
            demand_percent = 1 - self.wtp[product_select].integrate_box_1d(-np.inf, product_price)

            # Round to whole units; sub-unit demand disappears rather than accumulating.
            demand = round(order[product_select] * demand_percent)

            sku = self.stock[product_select].sku
            product_sold = 0

            # sku stays a valid heap throughout: partial consumption only edits the
            # quantity at index 1, never the expiry key the heap orders on.
            while sku and demand > 0:
                lot = heapq.heappop(sku)
                in_stock = lot[1]

                if in_stock > demand:
                    # Lot outlasts demand: put the remainder back and stop.
                    lot[1] = in_stock - demand
                    heapq.heappush(sku, lot)
                    product_sold += demand
                    demand = 0
                else:
                    # Lot is fully consumed; carry the shortfall to the next one.
                    product_sold += in_stock
                    demand -= in_stock

            sold[product_select] = product_sold

        return sold


class SupplierInfo:
    """One supplier's terms for one product.

    ``lead_time`` is a frozen scipy distribution (e.g. ``norm(loc=3, scale=1)``), not a
    scalar — each purchase draws from it, so the delivery date varies.

    ``order_quantity`` is the discrete set of lot sizes this supplier will sell, e.g.
    ``[1000, 5000, 10000]``. Purchases must pick one of them; ``order_purchase`` rejects
    anything else.

    ``price`` is a *volume price break*: one frozen distribution per lot size, in the same
    order as ``order_quantity``, so bigger lots can carry a lower unit cost. The two must
    be the same length.
    """

    def __init__(self, name, product, price, lead_time, order_quantity):
        # Parallel sequences are easy to get out of step, so refuse a mismatch up front
        # rather than letting it surface as a wrong unit price much later.
        if len(price) != len(order_quantity):
            raise ValueError(
                f'{name}/{product}: {len(price)} price tiers for '
                f'{len(order_quantity)} lot sizes — they must correspond one to one'
            )

        self.name = name
        self.product = product      # single product code, e.g. 'FOODS_1'
        self.lead_time = lead_time  # frozen distribution over delivery delay in days
        # Stored as tuples so a caller cannot mutate the supplier's terms through the
        # lists they passed in, and so Quote stays hashable.
        self.price = tuple(price)                   # unit-cost distribution per lot size
        self.order_quantity = tuple(order_quantity)


class Quote(NamedTuple):
    """One supplier's offer for one product, as returned by ``Supplier.order_proposal``.

    A named tuple rather than parallel lists: fields stay attached to each other, so
    callers cannot misalign them by index. Still unpacks like a tuple if wanted.

    ``price`` holds one drawn unit price per lot size, aligned with ``order_quantity``.
    Use ``price_for(quantity)`` rather than indexing the two by hand.
    """

    supplier: SupplierInfo
    product: str
    price: tuple
    lead_time: float
    order_quantity: tuple

    def price_for(self, quantity):
        """Quoted unit price for one of the offered lot sizes."""
        return self.price[self.order_quantity.index(quantity)]

    def cost_for(self, quantity):
        """Total quoted cost of ordering that lot size."""
        return self.price_for(quantity) * quantity


class Supplier:
    """Supplier book: who can supply what, and on what terms."""

    def __init__(self):
        self.supplier = []                    # every SupplierInfo, in registration order
        # Index for order_proposal. Kept in sync by add_supplier so quoting is a lookup
        # rather than a scan of the whole book.
        self.by_product = defaultdict(list)   # product code -> [SupplierInfo]

    def add_supplier(self, *supplier_info):
        """Register one or more SupplierInfo, updating both the book and the index."""
        for info in supplier_info:
            self.supplier.append(info)
            self.by_product[info.product].append(info)

    def order_proposal(self, product, rng=None):
        """Quote every supplier that carries any of the products in ``product``.

        Returns a list of Quote. Cost is O(len(product) + matches) via the index, rather
        than O(len(self.supplier)) per call.

        Note the two quoted figures are produced differently: price is a fresh random draw,
        while lead time reads the distribution's ``loc`` parameter directly.

        ``rng`` drives the price draws — pass ``env.supplier_rng`` for reproducible quotes.
        """
        if isinstance(product, str):    # a single code, not a collection of them
            product = [product]

        quotes = []
        for code in product:
            # .get, not [code] — self.by_product is a defaultdict, and indexing it would
            # insert an empty list for every unknown code that gets asked about.
            for select in self.by_product.get(code, ()):
                quotes.append(Quote(
                    supplier=select,
                    product=code,
                    # One draw per price break, so each lot size gets its own unit price.
                    price=tuple(round(tier.rvs(random_state=rng), 2) for tier in select.price),
                    lead_time=select.lead_time.kwds['loc'],
                    order_quantity=select.order_quantity,   # lot sizes on offer
                ))

        return quotes

    def order_purchase(self, env, supplier, demand_index):
        """Place orders and schedule their arrival.

        ``supplier``: {SupplierInfo: terms} — the accepted subset of a proposal, keyed by
        the supplier object itself. ``terms`` takes one of two forms:

        * ``{'quantity': n, 'price': p}`` — a single lot. ``n`` must be one of the lot sizes
          that supplier offers, and ``p`` its unit price: ``q.price_for(n)``.
        * ``{'lots': [n0, n1, ...], 'price': [p0, p1, ...]}`` — any mix of lot sizes at once.
          ``lots[t]`` is how many lots of ``order_quantity[t]`` to buy and ``price[t]`` that
          tier's unit price, both aligned with ``order_quantity``, so a quote's own
          ``q.price`` fits directly: ``{q.supplier: {'lots': lots, 'price': q.price}}``.

        Anything that does not match the supplier's lot sizes is a ValueError rather than a
        silently impossible order.

        Each order becomes one future 'stockup' event on the shared calendar rather than
        landing in inventory now; the policy loop pops it once the clock reaches that day.
        A multi-lot order arrives as a single delivery at its unit-weighted average price.
        The event payload is ``('stockup', product, quantity, price, supplier_name,
        ordered_on)``.

        Returns a list of the orders placed, one dict per supplier with ``supplier``,
        ``product``, ``quantity``, ``unit_price``, ``cost`` (quantity x unit price) and
        ``ordered_on`` — enough to track stock on order and spend.
        """
        placed = []
        for select, terms in supplier.items():
            if 'lots' in terms:
                lots, prices = list(terms['lots']), list(terms['price'])
                if len(lots) != len(select.order_quantity) or len(prices) != len(lots):
                    raise ValueError(
                        f'{select.name}: lots and price need one entry per lot size '
                        f'{list(select.order_quantity)}; got lots={lots}, price={prices}'
                    )
                if any(n < 0 for n in lots):
                    raise ValueError(f'{select.name}: negative lot count in {lots}')

                quantity = sum(n * size for n, size in zip(lots, select.order_quantity))
                if quantity == 0:
                    continue    # nothing ordered from this supplier

                price = sum(n * size * p for n, size, p
                            in zip(lots, select.order_quantity, prices)) / quantity
            else:
                quantity, price = terms['quantity'], terms['price']
                if quantity not in select.order_quantity:
                    raise ValueError(
                        f'{select.name} does not sell {select.product} in lots of {quantity}; '
                        f'offers {list(select.order_quantity)}'
                    )

            # Actual delivery delay is drawn here, independent of the lead time quoted above,
            # then stretched by how busy the market is. Floored at 1 so a negative draw or a
            # collapsed market cannot schedule an arrival on or before today.
            lead_time = max(1, round(select.lead_time.rvs(random_state=env.supplier_rng)
                                     * (1 + demand_index)))

            # The arrival day is the calendar key, so the payload carries the delivery plus
            # who shipped it and when it was ordered — the policy measures the realised lead
            # time on arrival rather than peeking at it here. add_event appends, so several
            # orders can land on the same day.
            env.add_event(env.clock + lead_time,
                          ('stockup',
                           select.product,
                           quantity,
                           price,
                           select.name,
                           env.clock))
            placed.append({'supplier': select.name, 'product': select.product,
                           'quantity': quantity, 'unit_price': price,
                           'cost': quantity * price, 'ordered_on': env.clock})

        return placed
    

    