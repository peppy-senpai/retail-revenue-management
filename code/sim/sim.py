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

    def generate_demand(self, seed=42, damp=0.8):
        """Draw one synthetic demand path, shaped like ``X_reduced``'s source matrix.

        Jitters each department's PCA scores by noise scaled to each component's own
        standard deviation, so dominant components move more than minor ones and the
        perturbation stays consistent with the data's covariance structure. ``damp``
        shrinks that noise; 1.0 would add as much variance again as the component carries.
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
    """Simulation clock. Every component reads ``clock`` for the current day index."""

    def __init__(self):
        self.clock = 0
        # Min-heap of pending events ordered by the day they fire:
        # (day, event_type, *payload). Supplier.order_purchase pushes 'stockup' here.
        # No heapify needed — an empty list is already a valid heap.
        self.event_calendar = []

    def clock_step(self, stepup=1):
        self.clock += stepup

    def reset(self):
        self.__init__()


class StockInfo:
    """Stock held for one department.

    ``sku`` is a min-heap of ``[expiry_day, quantity]`` lots, so the lot that expires
    soonest is always at the front — first-expired-first-out depletion.
    """

    def __init__(self):
        self.sku = []
        self.price = np.nan


class Inventory:

    def __init__(self, env):
        self.stock = defaultdict(StockInfo)
        self.env = env

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

    def set_price(self, order):
        """Set the current sell price per department. ``order``: {product: {'price': float, ...}}."""
        for product in order:
            self.stock[product].price = order[product]['price']

    def inventory_status(self):
        """Return parallel lists of (product, total units on hand, current price)."""
        product = []
        quantity = []
        price = []

        for product_select, info in self.stock.items():
            # Total on hand is the sum across every lot, regardless of expiry.
            product.append(product_select)
            quantity.append(sum(lot[1] for lot in info.sku))
            price.append(info.price)

        return [product, quantity, price]

    def sell_stock(self, order):
        """Consume demand from stock, oldest lot first.

        ``order``: {product: units_demanded}. Returns parallel lists of (product, units
        actually sold). Unmet demand is ``demanded - sold`` and is not recorded here.
        """
        product = []
        sold = []

        for product_select in order:
            demand = order[product_select]
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

            product.append(product_select)
            sold.append(product_sold)

        return [product, sold]


class SupplierInfo:
    """One supplier's terms for one product.

    ``price`` and ``lead_time`` are frozen scipy distributions (e.g. ``norm(loc=3, scale=1)``),
    not scalars — each purchase draws from them, so quoted cost and delivery date vary.

    ``order_quantity`` is the discrete set of lot sizes this supplier will sell, e.g.
    ``[1000, 5000, 10000]``. Purchases must pick one of them; ``order_purchase`` rejects
    anything else.
    """

    def __init__(self, name, product, price, lead_time, order_quantity):
        self.name = name
        self.product = product      # single product code, e.g. 'FOODS_1'
        self.price = price          # frozen distribution over unit cost
        self.lead_time = lead_time  # frozen distribution over delivery delay in days
        # Stored as a tuple so a caller cannot mutate the supplier's terms through the
        # list they passed in, and so Quote stays hashable.
        self.order_quantity = tuple(order_quantity)


class Quote(NamedTuple):
    """One supplier's offer for one product, as returned by ``Supplier.order_proposal``.

    A named tuple rather than parallel lists: fields stay attached to each other, so
    callers cannot misalign them by index. Still unpacks like a tuple if wanted.

    ``price`` is per unit; multiply by the chosen ``order_quantity`` entry for order value.
    """

    supplier: SupplierInfo
    product: str
    price: float
    lead_time: float
    order_quantity: tuple


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

    def order_proposal(self, product):
        """Quote every supplier that carries any of the products in ``product``.

        Returns a list of Quote. Cost is O(len(product) + matches) via the index, rather
        than O(len(self.supplier)) per call.

        Note the two quoted figures are produced differently: price is a fresh random draw,
        while lead time reads the distribution's ``loc`` parameter directly.
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
                    price=round(select.price.rvs(), 2),
                    lead_time=select.lead_time.kwds['loc'],
                    order_quantity=select.order_quantity,   # lot sizes on offer
                ))

        return quotes

    def order_purchase(self, env, supplier):
        """Place orders and schedule their arrival.

        ``supplier``: {SupplierInfo: {'quantity': int, 'price': float}} — the accepted
        subset of a proposal, keyed by the supplier object itself. Build it from quotes
        with e.g. ``{q.supplier: {'quantity': q.order_quantity[0], 'price': q.price}}``.

        ``quantity`` must be one of the lot sizes that supplier offers; anything else is a
        ValueError rather than a silently impossible order.

        Each order becomes a future 'stockup' event on the shared calendar rather than
        landing in inventory now; the policy loop pops it once the clock reaches that day.
        """
        for select, terms in supplier.items():
            quantity = terms['quantity']
            if quantity not in select.order_quantity:
                raise ValueError(
                    f'{select.name} does not sell {select.product} in lots of {quantity}; '
                    f'offers {list(select.order_quantity)}'
                )

            # Actual delivery delay is drawn here, independent of the lead time quoted above.
            lead_time = round(select.lead_time.rvs())
            heapq.heappush(env.event_calendar,
                           (env.clock+lead_time,
                            'stockup',
                            select.product,
                            quantity,
                            terms['price']))
    

    