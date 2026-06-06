"""
Полусеместровый контроль №4: поиск аномальных респондентов
Автор: Криницкий, группа ШЦТ-111
Date: 2026-06-06
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

# ==================== CONFIGURATION ====================
try:
    SCRIPT_DIR = Path(__file__).parent.absolute()
except NameError:
    SCRIPT_DIR = Path.cwd()
DATA_DIR = SCRIPT_DIR / "data"
OUTPUT_DIR = SCRIPT_DIR / "output"
PLOTS_DIR = OUTPUT_DIR / "plots"

PERCENTILE = 99.0
MIN_DAILY_OBS = 5
MIN_GLOBAL_OBS = 20

# Настройка русских шрифтов для графиков
plt.rcParams['font.family'] = 'DejaVu Sans'
# Если в системе есть русские шрифты, можно использовать:
# plt.rcParams['font.family'] = 'Arial'
# Для Windows можно попробовать 'Segoe UI', 'Microsoft Sans Serif'
# Чтобы точно работало, закомментируйте следующую строку, если шрифт не найден
try:
    plt.rcParams['font.family'] = 'Segoe UI'
except:
    pass

# ==================== COLUMN AUTO-DETECTION ====================
COLUMN_MAP = {
    "gender": ["Пол", "пол", "Gender"],
    "age": ["Возраст", "возраст", "Age"],
    "region": ["Регион", "регион", "Region"],
    "district": ["Федеральный_округ", "федеральный_округ", "FederalDistrict"],
    "children": ["Количество_детей", "количество_детей", "ChildrenCount"],
    "employment": ["Занятость", "занятость", "Employment"],
    "income": ["Доход", "доход", "Income"],
    "ResourceName": ["ResourceName"],
    "ResourceType": ["ResourceType"],
    "UseType": ["UseType"],
    "Platform": ["Platform"],
    "Category1": ["Category1"],
    "Category2": ["Category2"],
    "Category3": ["Category3"],
    "CategoryDelivery": ["CategoryNameDelivery", "CategoryDelivery"],
}

# Сопоставление ключей с русскими названиями для графиков
RUSSIAN_NAMES = {
    "gender": "Пол",
    "age": "Возраст",
    "region": "Регион",
    "district": "Федеральный округ",
    "children": "Количество детей",
    "employment": "Занятость",
    "income": "Доход",
    "ResourceName": "Ресурс",
    "ResourceType": "Тип ресурса",
    "UseType": "Тип использования",
    "Platform": "Платформа",
    "Category1": "Категория 1",
    "Category2": "Категория 2",
    "Category3": "Категория 3",
    "CategoryDelivery": "Категория поставки",
}

def find_columns(df):
    real_names = {}
    for key, variants in COLUMN_MAP.items():
        for var in variants:
            if var in df.columns:
                real_names[key] = var
                break
    required = ["CategoryDelivery"]
    for req in required:
        if req not in real_names:
            raise KeyError(f"Required column {req} not found. Available columns: {df.columns.tolist()}")
    return real_names

# ==================== DATA LOADING (RECURSIVE) ====================
def load_data():
    print(f"Looking for data in: {DATA_DIR}")
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Data folder not found: {DATA_DIR}")
    files = list(DATA_DIR.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No .parquet files found in {DATA_DIR} or its subfolders.")
    print(f"Found {len(files)} parquet file(s).")
    df_list = []
    for f in files:
        rel_path = f.relative_to(DATA_DIR) if DATA_DIR in f.parents else f.name
        print(f"  Reading {rel_path} ...")
        try:
            df_list.append(pd.read_parquet(f))
        except Exception as e:
            print(f"    ERROR reading {f}: {e}")
            continue
    if not df_list:
        raise ValueError("No data could be read from parquet files.")
    df = pd.concat(df_list, ignore_index=True)
    print(f"Total rows loaded: {len(df)}")
    return df

def preprocess(df, col_names):
    if "BrandinDelivery" not in df.columns:
        raise KeyError("Column 'BrandinDelivery' not found.")
    before = len(df)
    df = df[df["BrandinDelivery"] == 1.0].copy()
    print(f"Rows after BrandinDelivery==1: {len(df)} (removed {before - len(df)})")

    cat_col = col_names["CategoryDelivery"]
    before = len(df)
    df = df[df[cat_col].notna() & (df[cat_col].str.strip() != "")].copy()
    print(f"Rows after removing empty categories: {len(df)} (removed {before - len(df)})")

    for wcol in ["Weight", "week_weight", "month_weight"]:
        if wcol in df.columns:
            df[wcol] = pd.to_numeric(df[wcol], errors="coerce")
            print(f"Converted {wcol} to numeric")

    df["researchdate"] = pd.to_datetime(df["researchdate"], errors="coerce")
    before = len(df)
    df = df.dropna(subset=["researchdate"])
    print(f"Rows after dropping invalid dates: {len(df)} (removed {before - len(df)})")

    df["BrandID"] = df["BrandID"].astype(str)
    return df

def compute_daily_ots(df, cat_col):
    weight_day = df.groupby(["SubjectID", "researchdate"])["Weight"].first().reset_index()
    weight_day.rename(columns={"Weight": "daily_weight"}, inplace=True)
    count_df = df.groupby(["SubjectID", "researchdate", "BrandID", cat_col], as_index=False).size()
    count_df.rename(columns={"size": "count_rows"}, inplace=True)
    merged = count_df.merge(weight_day, on=["SubjectID", "researchdate"], how="left")
    merged["daily_ots"] = merged["daily_weight"] * merged["count_rows"]
    result = merged[["SubjectID", "researchdate", "BrandID", cat_col, "daily_ots"]].copy()
    result.rename(columns={cat_col: "CategoryDelivery"}, inplace=True)
    print(f"Computed daily_ots for {len(result)} (respondent, day, brand, category) combinations.")
    return result

# ==================== OPTIMIZED ANOMALY DETECTION ====================
def detect_anomalies(ots_df):
    print("  Precomputing global percentiles for brands...")
    global_stats = ots_df.groupby(["BrandID", "CategoryDelivery"]).agg(
        n=("daily_ots", "size"),
        global_threshold=("daily_ots", lambda x: x.quantile(PERCENTILE/100.0) if len(x) >= MIN_GLOBAL_OBS else None)
    ).reset_index()
    global_stats = global_stats[global_stats["global_threshold"].notna()].copy()
    global_threshold_dict = dict(zip(zip(global_stats["BrandID"], global_stats["CategoryDelivery"]), global_stats["global_threshold"]))
    print(f"  Computed global thresholds for {len(global_threshold_dict)} brand-category pairs.")
    
    anomalies_list = []
    grouped = ots_df.groupby(["researchdate", "BrandID", "CategoryDelivery"])
    total_groups = len(grouped)
    print(f"  Analyzing {total_groups} (day, brand, category) groups...")
    
    processed = 0
    for (date, brand, cat), group in grouped:
        processed += 1
        if processed % max(1, total_groups // 10) == 0:
            print(f"    Progress: {processed}/{total_groups} groups...")
        n = len(group)
        if n >= MIN_DAILY_OBS:
            threshold = group["daily_ots"].quantile(PERCENTILE / 100.0)
            reason = f"Exceeds {PERCENTILE}th percentile for this day (n={n})"
        else:
            key = (brand, cat)
            if key in global_threshold_dict:
                threshold = global_threshold_dict[key]
                reason = f"Exceeds {PERCENTILE}th percentile for brand (few daily obs, n={n})"
            else:
                continue
        anom = group[group["daily_ots"] > threshold].copy()
        if not anom.empty:
            anom["threshold"] = threshold
            anom["reason"] = reason
            anom["score"] = anom["daily_ots"]
            anomalies_list.append(anom)
    
    print(f"  Finished analyzing {processed} groups.")
    if not anomalies_list:
        print("  No anomalies found.")
        return pd.DataFrame(columns=list(ots_df.columns) + ["threshold", "reason", "score"])
    anomalies = pd.concat(anomalies_list, ignore_index=True)
    print(f"  Found {len(anomalies)} anomaly triggers.")
    return anomalies

def get_anomaly_pairs(anomalies_df):
    pairs = anomalies_df[["SubjectID", "researchdate"]].drop_duplicates()
    print(f"Unique (SubjectID, researchdate) pairs to remove: {len(pairs)}")
    return pairs

def clean_data(original_df, pairs):
    df = original_df.copy()
    df["_key"] = df["SubjectID"].astype(str) + "_" + df["researchdate"].dt.strftime("%Y-%m-%d")
    remove_keys = set(pairs["SubjectID"].astype(str) + "_" + pairs["researchdate"].dt.strftime("%Y-%m-%d"))
    before = len(df)
    df_clean = df[~df["_key"].isin(remove_keys)].copy()
    df_clean.drop(columns=["_key"], inplace=True)
    print(f"Cleaned data: {before} -> {len(df_clean)} rows (removed {before - len(df_clean)})")
    return df_clean

def total_ots_by_day(df):
    daily_ots = df.groupby("researchdate").apply(
        lambda g: (g.groupby("SubjectID")["Weight"].first() * g.groupby("SubjectID").size()).sum()
    ).reset_index(name="total_ots")
    return daily_ots

def ots_by_category(df, cat_col):
    temp = df.groupby(["SubjectID", "researchdate", cat_col]).size().reset_index(name="cnt")
    weight_day = df.groupby(["SubjectID", "researchdate"])["Weight"].first().reset_index()
    temp = temp.merge(weight_day, on=["SubjectID", "researchdate"])
    temp["ots"] = temp["cnt"] * temp["Weight"]
    cat_ots = temp.groupby(cat_col)["ots"].sum().reset_index(name="total_ots")
    return cat_ots

# ==================== PLOTS WITH RUSSIAN LABELS ====================
def plot_total_ots_before_after(before, after, out_path):
    plt.figure(figsize=(12, 5))
    plt.plot(before["researchdate"], before["total_ots"], label="До очистки", marker='o')
    plt.plot(after["researchdate"], after["total_ots"], label="После очистки", marker='s', linestyle='--')
    plt.xlabel("Дата")
    plt.ylabel("Суммарный OTS")
    plt.title("Изменение общего OTS по дням")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved plot: {out_path}")

def plot_category_ots_change(before, after, cat_name, out_path):
    merged = before.merge(after, on=cat_name, suffixes=("_before", "_after"), how="outer").fillna(0)
    merged["change_pct"] = (merged["total_ots_after"] - merged["total_ots_before"]) / merged["total_ots_before"] * 100
    merged = merged.sort_values("change_pct")
    if len(merged) > 20:
        merged = merged.iloc[:20]
    plt.figure(figsize=(10, 6))
    plt.barh(merged[cat_name].astype(str), merged["change_pct"], color='skyblue')
    plt.xlabel("Изменение OTS (%)")
    plt.title(f"Изменение OTS по категориям: {cat_name}")
    plt.axvline(x=0, color='red', linestyle='--')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved plot: {out_path}")

def plot_daily_anomaly_count(anomalies_df, out_path):
    daily = anomalies_df[["SubjectID", "researchdate"]].drop_duplicates().groupby("researchdate").size().reset_index(name="count")
    plt.figure(figsize=(10, 5))
    plt.bar(daily["researchdate"].astype(str), daily["count"], color='salmon')
    plt.xlabel("Дата")
    plt.ylabel("Количество аномальных респондентов")
    plt.title("Аномальные респонденты по дням")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved plot: {out_path}")

def plot_before_after_by_feature(original, cleaned, feature_key, col_names, out_dir):
    """Строит график 'до/после' для признака, используя русские подписи."""
    if feature_key not in col_names:
        return
    real_col = col_names[feature_key]
    if real_col not in original.columns:
        return
    # Получаем русское название для заголовка
    title_ru = RUSSIAN_NAMES.get(feature_key, real_col)
    
    def ots_by_feat(df, col):
        temp = df.groupby(["SubjectID", "researchdate", col]).size().reset_index(name="cnt")
        w = df.groupby(["SubjectID", "researchdate"])["Weight"].first().reset_index()
        temp = temp.merge(w, on=["SubjectID", "researchdate"])
        temp["ots"] = temp["cnt"] * temp["Weight"]
        return temp.groupby(col)["ots"].sum().reset_index(name="total_ots")
    
    before = ots_by_feat(original, real_col)
    after = ots_by_feat(cleaned, real_col)
    merged = before.merge(after, on=real_col, suffixes=("_before", "_after"), how="outer").fillna(0)
    merged["share_before"] = merged["total_ots_before"] / merged["total_ots_before"].sum() * 100
    merged["share_after"] = merged["total_ots_after"] / merged["total_ots_after"].sum() * 100
    
    plt.figure(figsize=(10, 6))
    x = np.arange(len(merged))
    width = 0.35
    plt.bar(x - width/2, merged["share_before"], width, label="До очистки")
    plt.bar(x + width/2, merged["share_after"], width, label="После очистки")
    plt.ylabel("Доля в суммарном OTS (%)")
    plt.title(f"Распределение OTS по признаку: {title_ru}")
    plt.xticks(x, merged[real_col].astype(str), rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    # Имя файла используем английский ключ, чтобы не было кракозябр
    out_path = out_dir / f"before_after_{feature_key}.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved plot: {out_path}")

def show_query_text(original_df, subject_id, date):
    rows = original_df[(original_df["SubjectID"] == subject_id) & (original_df["researchdate"] == pd.Timestamp(date))]
    if rows.empty:
        print(f"No data for SubjectID={subject_id}, date={date}")
    else:
        print(f"\nSearch queries for respondent {subject_id} on {date}:")
        for q in rows["QueryText"].unique():
            print(f"  • {q}")

def plot_brand_ots_dynamics(original, cleaned, brand_id, out_dir):
    def brand_ots(df, bid):
        brand_df = df[df["BrandID"] == str(bid)]
        if brand_df.empty:
            return pd.DataFrame(columns=["researchdate", "ots"])
        daily = brand_df.groupby("researchdate").apply(
            lambda g: (g.groupby("SubjectID")["Weight"].first() * g.groupby("SubjectID").size()).sum()
        ).reset_index(name="ots")
        return daily
    before = brand_ots(original, brand_id)
    after = brand_ots(cleaned, brand_id)
    if before.empty and after.empty:
        print(f"No data for brand {brand_id}")
        return
    merged = before.merge(after, on="researchdate", suffixes=("_before", "_after"), how="outer").fillna(0)
    plt.figure(figsize=(10, 5))
    plt.plot(merged["researchdate"], merged["ots_before"], label="До очистки", marker='o')
    plt.plot(merged["researchdate"], merged["ots_after"], label="После очистки", marker='s', linestyle='--')
    plt.xlabel("Дата")
    plt.ylabel("OTS бренда")
    plt.title(f"Динамика OTS бренда ID={brand_id}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    out_path = out_dir / f"brand_{brand_id}_ots.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved plot: {out_path}")

# ==================== MAIN ====================
def main():
    print("="*60)
    print("Anomaly Detection in Search Query Activity (SoS)")
    print("="*60)

    OUTPUT_DIR.mkdir(exist_ok=True)
    PLOTS_DIR.mkdir(exist_ok=True)

    print("\n[1] Loading data...")
    try:
        df_raw = load_data()
    except Exception as e:
        print(f"ERROR loading data: {e}")
        sys.exit(1)

    print("\n[2] Detecting column names...")
    try:
        col_names = find_columns(df_raw)
        print(f"  Identified columns: {col_names}")
    except KeyError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    print("\n[3] Preprocessing...")
    df = preprocess(df_raw, col_names)
    if df.empty:
        print("No data left after preprocessing. Exiting.")
        empty_df = pd.DataFrame(columns=["SubjectID", "researchdate"])
        empty_df.to_csv(OUTPUT_DIR / "anomalies.csv", index=False)
        pd.DataFrame(columns=["SubjectID", "researchdate", "BrandID", "CategoryDelivery", "daily_ots", "score", "threshold", "reason"]).to_csv(OUTPUT_DIR / "anomaly_reasons.csv", index=False)
        print("Created empty output files.")
        return

    print("\n[4] Computing daily_ots...")
    ots = compute_daily_ots(df, col_names["CategoryDelivery"])
    if ots.empty:
        print("No daily_ots records. Exiting.")
        return

    print("\n[5] Detecting anomalies...")
    anomalies = detect_anomalies(ots)
    if anomalies.empty:
        print("No anomalies found. Creating empty output files.")
        empty_pairs = pd.DataFrame(columns=["SubjectID", "researchdate"])
        empty_pairs.to_csv(OUTPUT_DIR / "anomalies.csv", index=False)
        empty_reasons = pd.DataFrame(columns=["SubjectID", "researchdate", "BrandID", "CategoryDelivery", "daily_ots", "score", "threshold", "reason"])
        empty_reasons.to_csv(OUTPUT_DIR / "anomaly_reasons.csv", index=False)
        print("Saved empty anomalies.csv and anomaly_reasons.csv")
        print("Exiting.")
        return

    # Save anomaly_reasons.csv
    reasons = anomalies[["SubjectID", "researchdate", "BrandID", "CategoryDelivery", "daily_ots", "score", "threshold", "reason"]].copy()
    brand_names = df[["BrandID", "Brand"]].drop_duplicates()
    reasons = reasons.merge(brand_names, on="BrandID", how="left")
    reasons.to_csv(OUTPUT_DIR / "anomaly_reasons.csv", index=False)
    print("Saved anomaly_reasons.csv")

    # Save anomalies.csv
    pairs = get_anomaly_pairs(anomalies)
    pairs.to_csv(OUTPUT_DIR / "anomalies.csv", index=False)
    print("Saved anomalies.csv")

    print("\n[6] Cleaning data and recalculating metrics...")
    df_clean = clean_data(df, pairs)

    # Total OTS by day
    ots_day_before = total_ots_by_day(df)
    ots_day_after = total_ots_by_day(df_clean)
    plot_total_ots_before_after(ots_day_before, ots_day_after, PLOTS_DIR / "total_ots_before_after.png")

    # OTS by delivery category
    cat_before = ots_by_category(df, col_names["CategoryDelivery"])
    cat_after = ots_by_category(df_clean, col_names["CategoryDelivery"])
    plot_category_ots_change(cat_before, cat_after, col_names["CategoryDelivery"], PLOTS_DIR / "category_ots_change.png")

    # Anomaly count per day
    plot_daily_anomaly_count(anomalies, PLOTS_DIR / "daily_anomaly_count.png")

    print("\n[7] Generating analytical before/after plots (Russian labels)...")
    features_to_plot = ["gender", "age", "region", "district", "children", "employment", "income",
                        "ResourceName", "ResourceType", "UseType", "Platform",
                        "Category1", "Category2", "Category3"]
    for feat in features_to_plot:
        plot_before_after_by_feature(df, df_clean, feat, col_names, PLOTS_DIR)

    print("\n[8] Demo: show queries for first anomalous respondent and brand OTS dynamics")
    first = pairs.iloc[0]
    show_query_text(df_raw, first["SubjectID"], first["researchdate"])
    top_brand = anomalies["BrandID"].value_counts().index[0]
    plot_brand_ots_dynamics(df, df_clean, top_brand, PLOTS_DIR)

    print("\n" + "="*60)
    print("SUCCESS! Results saved in:")
    print(f"  {OUTPUT_DIR}")
    print("  - anomalies.csv")
    print("  - anomaly_reasons.csv")
    print("  - plots/ (all charts with Russian labels)")
    print("="*60)

if __name__ == "__main__":
    main()