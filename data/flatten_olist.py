"""
flatten_olist.py

Stage 0 of the capstone project: this is a ONE-TIME, human-run data prep
script. It is NOT one of the AI agents. Its only job is to take the 9 raw
Olist relational tables and join them into a single flat, order-level CSV.

Once this script runs, the platform (all the agents) will treat the output
file (data/processed/olist_flattened.csv) exactly like any CSV a user might
upload -- the agents have no idea it came from multiple joined tables.

Target variable created here: `is_late_delivery`
  1 = order was delivered after its estimated delivery date
  0 = order was delivered on or before its estimated delivery date
  (orders with no delivered date, i.e. never delivered/cancelled, are
   dropped from the modeling target but kept in an audit file for
   transparency)

LEAKAGE POLICY (read before changing feature columns)
-------------------------------------------------------
This dataset answers a real business question: "at the moment a customer
places an order, can we flag delivery risk?" The FEATURE set must only
contain information actually known at purchase/checkout time -- not
anything only known later in the order lifecycle, because that wouldn't
exist yet in a real, live prediction scenario. Including it would make
the model look artificially accurate during training while being useless
in production.

Excluded from the feature set for this reason (kept only for building the
label or for audit purposes, never passed to the model):
  - review_score                     (written AFTER delivery; a late
                                       delivery makes a bad review more
                                       likely -- the outcome leaking
                                       into a "predictor")
  - order_approved_at                (a downstream processing timestamp)
  - order_delivered_carrier_date     (only known once already shipped)
  - order_delivered_customer_date    (this IS the outcome -- used only
                                       to compute the label)
  - order_status                     (describes the order's post-purchase
                                       lifecycle state, not something known
                                       at checkout. In this snapshot it's
                                       99.99% "delivered" -- since only
                                       delivered orders keep a row here --
                                       so it currently carries no signal and
                                       DataCleaningAgent's low-variance
                                       dropper happens to remove it anyway.
                                       But that's an accident of this one
                                       dataset's distribution, not a
                                       deliberate policy -- excluded here
                                       explicitly so the leakage-free
                                       contract doesn't depend on a
                                       variance threshold.)

Kept as legitimate purchase-time features:
  - order_purchase_timestamp, order_estimated_delivery_date (both known
    the instant the order is placed)
  - customer location, product category, seller location
  - item count, price, freight, product weight/photos
  - payment method/installments (submitted at checkout)
"""

import pandas as pd
import os

RAW_DIR = os.path.join(os.path.dirname(__file__), "raw")
OUT_DIR = os.path.join(os.path.dirname(__file__), "processed")
os.makedirs(OUT_DIR, exist_ok=True)


def load_raw():
    orders = pd.read_csv(os.path.join(RAW_DIR, "olist_orders_dataset.csv"))
    customers = pd.read_csv(os.path.join(RAW_DIR, "olist_customers_dataset.csv"))
    items = pd.read_csv(os.path.join(RAW_DIR, "olist_order_items_dataset.csv"))
    payments = pd.read_csv(os.path.join(RAW_DIR, "olist_order_payments_dataset.csv"))
    reviews = pd.read_csv(os.path.join(RAW_DIR, "olist_order_reviews_dataset.csv"))
    products = pd.read_csv(os.path.join(RAW_DIR, "olist_products_dataset.csv"))
    sellers = pd.read_csv(os.path.join(RAW_DIR, "olist_sellers_dataset.csv"))
    category_translation = pd.read_csv(
        os.path.join(RAW_DIR, "product_category_name_translation.csv")
    )
    return orders, customers, items, payments, reviews, products, sellers, category_translation


def aggregate_items(items, products, sellers, category_translation):
    """Order-items is many-rows-per-order (one per line item). Aggregate to
    one row per order, and bring in product + seller context."""
    products = products.merge(category_translation, on="product_category_name", how="left")
    items_full = items.merge(products, on="product_id", how="left")
    items_full = items_full.merge(sellers, on="seller_id", how="left")

    agg = items_full.groupby("order_id").agg(
        n_items=("order_item_id", "count"),
        n_distinct_sellers=("seller_id", "nunique"),
        n_distinct_products=("product_id", "nunique"),
        total_price=("price", "sum"),
        total_freight=("freight_value", "sum"),
        avg_product_weight_g=("product_weight_g", "mean"),
        avg_product_photos_qty=("product_photos_qty", "mean"),
        primary_seller_state=("seller_state", lambda x: x.mode().iloc[0] if not x.mode().empty else None),
    ).reset_index()

    # Most common product category for the order (orders can have multiple items)
    top_category = (
        items_full.groupby("order_id")["product_category_name_english"]
        .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else "unknown")
        .reset_index()
        .rename(columns={"product_category_name_english": "product_category"})
    )
    agg = agg.merge(top_category, on="order_id", how="left")
    return agg


def aggregate_payments(payments):
    """One row per order: total paid, installments, primary payment method."""
    agg = payments.groupby("order_id").agg(
        total_payment_value=("payment_value", "sum"),
        max_installments=("payment_installments", "max"),
        n_payment_methods=("payment_type", "nunique"),
        primary_payment_type=("payment_type", lambda x: x.mode().iloc[0] if not x.mode().empty else None),
    ).reset_index()
    return agg


def aggregate_reviews(reviews):
    """One row per order. If an order somehow has multiple reviews, keep the latest."""
    reviews_sorted = reviews.sort_values("review_answer_timestamp")
    latest = reviews_sorted.drop_duplicates(subset="order_id", keep="last")
    return latest[["order_id", "review_score"]]


def build_flat_dataset():
    orders, customers, items, payments, reviews, products, sellers, category_translation = load_raw()

    df = orders.merge(customers, on="customer_id", how="left")

    items_agg = aggregate_items(items, products, sellers, category_translation)
    df = df.merge(items_agg, on="order_id", how="left")

    payments_agg = aggregate_payments(payments)
    df = df.merge(payments_agg, on="order_id", how="left")

    reviews_agg = aggregate_reviews(reviews)
    df = df.merge(reviews_agg, on="order_id", how="left")

    # ---- Feature engineering that requires the raw timestamps (do it here,
    # since after this point the platform just sees a generic flat CSV) ----
    date_cols = [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]
    for c in date_cols:
        df[c] = pd.to_datetime(df[c], errors="coerce")

    df["purchase_to_estimated_days"] = (
        df["order_estimated_delivery_date"] - df["order_purchase_timestamp"]
    ).dt.days

    # ---- Target variable ----
    # Only defined for orders that actually have a delivered date.
    delivered_mask = df["order_delivered_customer_date"].notna()
    df["is_late_delivery"] = None
    df.loc[delivered_mask, "is_late_delivery"] = (
        df.loc[delivered_mask, "order_delivered_customer_date"]
        > df.loc[delivered_mask, "order_estimated_delivery_date"]
    ).astype(int)

    # Audit split: undelivered orders kept separately for transparency,
    # not silently dropped.
    undelivered = df[~delivered_mask].copy()
    full_delivered_df = df[delivered_mask].copy()

    # ---- Leakage-free MODELING file: only purchase-time-known columns ----
    leaky_or_id_cols = [
        "customer_id",
        "review_score",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",  # outcome itself, only used for label above
        "order_status",  # post-purchase lifecycle state, not known at checkout
    ]
    modeling_df = full_delivered_df.drop(
        columns=[c for c in leaky_or_id_cols if c in full_delivered_df.columns]
    )

    # ---- Separate REVIEW ANALYSIS file: keeps review_score, for optional
    # exploratory work (e.g. "what drives bad reviews"), clearly labeled so
    # nobody accidentally feeds it to the late-delivery model. ----
    review_analysis_df = full_delivered_df[
        ["order_id", "review_score", "is_late_delivery", "product_category",
         "total_price", "total_freight", "purchase_to_estimated_days"]
    ].copy()

    modeling_path = os.path.join(OUT_DIR, "olist_flattened.csv")
    undelivered_path = os.path.join(OUT_DIR, "olist_undelivered_orders_audit.csv")
    review_path = os.path.join(OUT_DIR, "olist_review_analysis_only.csv")

    modeling_df.to_csv(modeling_path, index=False)
    undelivered.to_csv(undelivered_path, index=False)
    review_analysis_df.to_csv(review_path, index=False)

    print(f"Modeling dataset (leakage-free): {modeling_df.shape} -> {modeling_path}")
    print(f"Undelivered/audit dataset: {undelivered.shape} -> {undelivered_path}")
    print(f"Review-analysis-only dataset: {review_analysis_df.shape} -> {review_path}")
    print(f"Late delivery rate: {modeling_df['is_late_delivery'].mean():.3%}")
    print(f"\nModeling feature columns: {list(modeling_df.columns)}")

    return modeling_df


if __name__ == "__main__":
    build_flat_dataset()
