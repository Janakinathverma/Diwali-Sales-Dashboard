import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os
import matplotlib
import numpy as np

matplotlib.use('Agg')
matplotlib.rcParams['figure.max_open_warning'] = 0

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Diwali Sales Dashboard",
    page_icon="🪔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=DM+Sans:wght@400;500;600&display=swap');

/* Root theme */
:root {
    --gold:    #D4A017;
    --saffron: #FF6B2B;
    --crimson: #C0392B;
    --deep:    #1A0A00;
    --card-bg: rgba(255,255,255,0.03);
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Gradient background */
.stApp {
    background: linear-gradient(135deg, #1A0A00 0%, #2D1200 40%, #1A0505 100%);
    color: #F5E6C8;
}

/* Header */
.main-header {
    font-family: 'Playfair Display', serif;
    font-size: 2.8rem;
    font-weight: 700;
    background: linear-gradient(90deg, #D4A017, #FF6B2B, #D4A017);
    background-size: 200%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-align: center;
    padding: 1rem 0 0.25rem;
    letter-spacing: -0.5px;
    animation: shimmer 3s infinite linear;
}
@keyframes shimmer {
    0%   { background-position: 0%   }
    100% { background-position: 200% }
}

.sub-header {
    text-align: center;
    color: #B8966E;
    font-size: 1rem;
    margin-bottom: 2rem;
    letter-spacing: 2px;
    text-transform: uppercase;
}

/* KPI cards */
.kpi-card {
    background: linear-gradient(135deg, rgba(212,160,23,0.12), rgba(255,107,43,0.08));
    border: 1px solid rgba(212,160,23,0.3);
    border-radius: 16px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    transition: transform 0.2s, box-shadow 0.2s;
}
.kpi-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 24px rgba(212,160,23,0.2);
}
.kpi-value {
    font-family: 'Playfair Display', serif;
    font-size: 1.9rem;
    color: #D4A017;
    font-weight: 700;
}
.kpi-label {
    font-size: 0.78rem;
    color: #B8966E;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-top: 0.2rem;
}

/* Section headers */
.section-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.4rem;
    color: #D4A017;
    border-left: 4px solid #FF6B2B;
    padding-left: 0.75rem;
    margin: 1.5rem 0 1rem;
}

/* Divider */
hr { border-color: rgba(212,160,23,0.2) !important; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1A0A00, #2D1200);
    border-right: 1px solid rgba(212,160,23,0.2);
}
section[data-testid="stSidebar"] * { color: #F5E6C8 !important; }

/* Insight boxes */
.insight-box {
    background: rgba(212,160,23,0.08);
    border: 1px solid rgba(212,160,23,0.25);
    border-radius: 12px;
    padding: 0.9rem 1.2rem;
    margin-top: 0.5rem;
    font-size: 0.88rem;
    color: #E8D5A3;
    line-height: 1.6;
}
.insight-box strong { color: #D4A017; }

/* Tabs */
button[data-baseweb="tab"] {
    color: #B8966E !important;
    font-family: 'DM Sans', sans-serif;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #D4A017 !important;
    border-bottom: 2px solid #D4A017 !important;
}
</style>
""", unsafe_allow_html=True)

# ─── Matplotlib / Seaborn global theme ──────────────────────────────────────────
DARK_BG   = "#1A0A00"
CARD_BG   = "#2D1200"
GOLD      = "#D4A017"
SAFFRON   = "#FF6B2B"
CRIMSON   = "#C0392B"
TEXT      = "#F5E6C8"
MUTED     = "#B8966E"
PALETTE   = [GOLD, SAFFRON, CRIMSON, "#8E44AD", "#2980B9", "#27AE60",
             "#E67E22", "#1ABC9C", "#E91E63", "#00BCD4"]

SPINE_COLOR = (0.831, 0.627, 0.090, 0.30)   # gold @30% alpha — matplotlib tuple
GRID_COLOR  = (0.831, 0.627, 0.090, 0.10)   # gold @10% alpha

def style_axes(ax, fig, title=""):
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(CARD_BG)
    ax.set_title(title, color=GOLD, fontsize=13, fontweight='bold', pad=12)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    for spine in ax.spines.values():
        spine.set_edgecolor(SPINE_COLOR)
        spine.set_linewidth(0.8)
    ax.grid(axis='y', color=GRID_COLOR, linewidth=0.6)
    ax.set_axisbelow(True)

# ─── File & Data ────────────────────────────────────────────────────────────────
FILE_NAME = 'Diwali Sales Data.csv'

@st.cache_data
def load_and_clean_data(file):
    df = pd.read_csv(file, encoding='unicode_escape')
    cols_to_drop = [c for c in ['Status', 'Unnamed: 0'] if c in df.columns]
    if cols_to_drop:
        df.drop(cols_to_drop, axis=1, inplace=True)
    df.dropna(subset=['Amount'], inplace=True)
    df['Amount'] = df['Amount'].astype('int')
    return df

# ─── Header ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">🪔 Diwali Sales Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Festive Season Revenue Intelligence</div>', unsafe_allow_html=True)

if not os.path.exists(FILE_NAME):
    st.error(f"❌ '{FILE_NAME}' not found. Please place the CSV in the same directory.")
    st.stop()

df_raw = load_and_clean_data(FILE_NAME)

# ─── Sidebar Filters ────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔍 Filters")

    genders = ["All"] + sorted(df_raw['Gender'].dropna().unique().tolist())
    sel_gender = st.selectbox("Gender", genders)

    if 'Age Group' in df_raw.columns:
        age_groups = ["All"] + sorted(df_raw['Age Group'].dropna().unique().tolist())
        sel_age = st.selectbox("Age Group", age_groups)
    else:
        sel_age = "All"

    states = ["All"] + sorted(df_raw['State'].dropna().unique().tolist())
    sel_state = st.multiselect("State(s)", states[1:], placeholder="All States")

    if 'Product_Category' in df_raw.columns:
        cats = ["All"] + sorted(df_raw['Product_Category'].dropna().unique().tolist())
        sel_cat = st.selectbox("Product Category", cats)
    else:
        sel_cat = "All"

    st.markdown("---")
    st.markdown("### 📊 Dataset Info")
    st.metric("Total Rows", f"{len(df_raw):,}")
    st.metric("Total Revenue", f"₹{df_raw['Amount'].sum():,.0f}")

# ─── Apply Filters ───────────────────────────────────────────────────────────────
df = df_raw.copy()
if sel_gender != "All":
    df = df[df['Gender'] == sel_gender]
if sel_age != "All" and 'Age Group' in df.columns:
    df = df[df['Age Group'] == sel_age]
if sel_state:
    df = df[df['State'].isin(sel_state)]
if sel_cat != "All" and 'Product_Category' in df.columns:
    df = df[df['Product_Category'] == sel_cat]

# ─── KPI Row ─────────────────────────────────────────────────────────────────────
st.markdown("---")
k1, k2, k3, k4, k5 = st.columns(5)

total_rev   = df['Amount'].sum()
total_orders= df['Orders'].sum() if 'Orders' in df.columns else len(df)
avg_order   = total_rev / max(total_orders, 1)
unique_buyers = df['User_ID'].nunique() if 'User_ID' in df.columns else len(df)
top_state   = df.groupby('State')['Amount'].sum().idxmax() if len(df) else "—"

kpi_data = [
    (f"₹{total_rev/1e6:.1f}M",   "Total Revenue"),
    (f"{total_orders:,}",         "Total Orders"),
    (f"₹{avg_order:,.0f}",        "Avg Order Value"),
    (f"{unique_buyers:,}",        "Unique Buyers"),
    (top_state,                   "Top State"),
]
for col, (val, lbl) in zip([k1,k2,k3,k4,k5], kpi_data):
    with col:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{val}</div>
            <div class="kpi-label">{lbl}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─── Tabs ────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(
    ["👤 Demographics", "🗺️ Geography", "📦 Products", "📈 Deep Dive"]
)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — DEMOGRAPHICS
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-title">Gender Analysis</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        fig, ax = plt.subplots(figsize=(5.5, 4))
        counts = df['Gender'].value_counts()
        bars = ax.bar(counts.index, counts.values,
                      color=[GOLD, SAFFRON], width=0.5, edgecolor=DARK_BG, linewidth=1.5)
        ax.bar_label(bars, fmt='%d', color=TEXT, fontsize=10, padding=4)
        style_axes(ax, fig, "Gender Distribution")
        fig.tight_layout()
        st.pyplot(fig); plt.close(fig)

    with c2:
        fig, ax = plt.subplots(figsize=(5.5, 4))
        sales_gen = df.groupby('Gender')['Amount'].sum().reset_index()
        bars = ax.bar(sales_gen['Gender'], sales_gen['Amount'] / 1e6,
                      color=[GOLD, SAFFRON], width=0.5, edgecolor=DARK_BG, linewidth=1.5)
        ax.bar_label(bars, fmt='₹%.2fM', color=TEXT, fontsize=9, padding=4)
        ax.set_ylabel("Revenue (₹M)", color=MUTED)
        style_axes(ax, fig, "Revenue by Gender")
        fig.tight_layout()
        st.pyplot(fig); plt.close(fig)

    # Gender insight
    if len(sales_gen):
        top_g = sales_gen.loc[sales_gen['Amount'].idxmax(), 'Gender']
        top_pct = sales_gen['Amount'].max() / sales_gen['Amount'].sum() * 100
        st.markdown(f"""<div class="insight-box">
        💡 <strong>{top_g}</strong> shoppers drive <strong>{top_pct:.1f}%</strong>
        of total Diwali revenue in the filtered dataset.
        </div>""", unsafe_allow_html=True)

    if 'Age Group' in df.columns:
        st.markdown('<div class="section-title">Age Group Analysis</div>', unsafe_allow_html=True)
        c3, c4 = st.columns(2)

        with c3:
            fig, ax = plt.subplots(figsize=(6, 4))
            age_counts = df['Age Group'].value_counts().sort_index()
            ax.bar(age_counts.index, age_counts.values,
                   color=PALETTE[:len(age_counts)], edgecolor=DARK_BG, linewidth=1)
            ax.bar_label(ax.containers[0], fmt='%d', color=TEXT, fontsize=9, padding=3)
            plt.xticks(rotation=30)
            style_axes(ax, fig, "Buyers by Age Group")
            fig.tight_layout()
            st.pyplot(fig); plt.close(fig)

        with c4:
            fig, ax = plt.subplots(figsize=(6, 4))
            age_rev = df.groupby('Age Group')['Amount'].sum().sort_index()
            ax.bar(age_rev.index, age_rev.values / 1e6,
                   color=PALETTE[:len(age_rev)], edgecolor=DARK_BG, linewidth=1)
            ax.set_ylabel("Revenue (₹M)", color=MUTED)
            plt.xticks(rotation=30)
            style_axes(ax, fig, "Revenue by Age Group")
            fig.tight_layout()
            st.pyplot(fig); plt.close(fig)

    if 'Marital_Status' in df.columns:
        st.markdown('<div class="section-title">Marital Status vs Spending</div>', unsafe_allow_html=True)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.patch.set_facecolor(DARK_BG)

        ms_rev = df.groupby(['Marital_Status', 'Gender'])['Amount'].sum().reset_index()
        ms_map = {0: 'Single', 1: 'Married'}
        ms_rev['Status'] = ms_rev['Marital_Status'].map(ms_map)

        for i, ax in enumerate(axes):
            ax.set_facecolor(CARD_BG)
            ax.tick_params(colors=MUTED, labelsize=9)
            for spine in ax.spines.values():
                spine.set_edgecolor(SPINE_COLOR)

        # Pie
        ms_totals = ms_rev.groupby('Status')['Amount'].sum()
        axes[0].pie(ms_totals, labels=ms_totals.index, autopct='%1.1f%%',
                    colors=[GOLD, SAFFRON], textprops={'color': TEXT},
                    wedgeprops={'edgecolor': DARK_BG, 'linewidth': 2})
        axes[0].set_title("Marital Split by Revenue", color=GOLD, fontsize=12, fontweight='bold')

        # Grouped bar
        pivot = ms_rev.pivot(index='Status', columns='Gender', values='Amount').fillna(0) / 1e6
        x = np.arange(len(pivot))
        w = 0.35
        for j, (col, clr) in enumerate(zip(pivot.columns, [GOLD, SAFFRON])):
            bars = axes[1].bar(x + j*w, pivot[col], w, label=col, color=clr,
                               edgecolor=DARK_BG, linewidth=1)
            axes[1].bar_label(bars, fmt='₹%.1fM', color=TEXT, fontsize=8, padding=2)
        axes[1].set_xticks(x + w/2)
        axes[1].set_xticklabels(pivot.index, color=MUTED)
        axes[1].set_ylabel("Revenue (₹M)", color=MUTED)
        axes[1].set_title("Revenue by Status & Gender", color=GOLD, fontsize=12,
                           fontweight='bold')
        axes[1].legend(facecolor=CARD_BG, labelcolor=TEXT, edgecolor=GOLD)
        axes[1].grid(axis='y', color=GRID_COLOR, linewidth=0.6)
        axes[1].set_axisbelow(True)

        fig.tight_layout()
        st.pyplot(fig); plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — GEOGRAPHY
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-title">Top 10 States — Orders & Revenue</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        if 'Orders' in df.columns:
            top_orders = (df.groupby('State', as_index=False)['Orders']
                          .sum().sort_values('Orders', ascending=False).head(10))
            fig, ax = plt.subplots(figsize=(6, 5))
            colors = [GOLD if i == 0 else SAFFRON if i == 1 else CRIMSON if i == 2
                      else "#8E7040" for i in range(len(top_orders))]
            bars = ax.barh(top_orders['State'][::-1], top_orders['Orders'][::-1],
                           color=colors[::-1], edgecolor=DARK_BG, linewidth=1)
            ax.bar_label(bars, fmt='%d', color=TEXT, fontsize=8, padding=3)
            style_axes(ax, fig, "Top 10 States by Orders")
            ax.set_xlabel("Orders", color=MUTED)
            fig.tight_layout()
            st.pyplot(fig); plt.close(fig)

    with c2:
        top_rev = (df.groupby('State', as_index=False)['Amount']
                   .sum().sort_values('Amount', ascending=False).head(10))
        fig, ax = plt.subplots(figsize=(6, 5))
        colors = [GOLD if i == 0 else SAFFRON if i == 1 else CRIMSON if i == 2
                  else "#8E7040" for i in range(len(top_rev))]
        bars = ax.barh(top_rev['State'][::-1], top_rev['Amount'][::-1] / 1e6,
                       color=colors[::-1], edgecolor=DARK_BG, linewidth=1)
        ax.bar_label(bars, fmt='₹%.1fM', color=TEXT, fontsize=8, padding=3)
        style_axes(ax, fig, "Top 10 States by Revenue")
        ax.set_xlabel("Revenue (₹M)", color=MUTED)
        fig.tight_layout()
        st.pyplot(fig); plt.close(fig)

    # Zone analysis
    if 'Zone' in df.columns:
        st.markdown('<div class="section-title">Zone-wise Performance</div>', unsafe_allow_html=True)
        zone_data = df.groupby('Zone').agg(
            Revenue=('Amount', 'sum'),
            Orders=('Orders', 'sum') if 'Orders' in df.columns else ('Amount', 'count')
        ).reset_index()

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.patch.set_facecolor(DARK_BG)
        for ax in axes:
            ax.set_facecolor(CARD_BG)
            ax.tick_params(colors=MUTED, labelsize=9)
            for spine in ax.spines.values():
                spine.set_edgecolor(SPINE_COLOR)

        axes[0].pie(zone_data['Revenue'], labels=zone_data['Zone'],
                    autopct='%1.1f%%', colors=PALETTE[:len(zone_data)],
                    textprops={'color': TEXT}, wedgeprops={'edgecolor': DARK_BG, 'linewidth': 2})
        axes[0].set_title("Revenue Share by Zone", color=GOLD, fontsize=12,
                           fontweight='bold')

        bars = axes[1].bar(zone_data['Zone'], zone_data['Revenue'] / 1e6,
                           color=PALETTE[:len(zone_data)], edgecolor=DARK_BG, linewidth=1)
        axes[1].bar_label(bars, fmt='₹%.1fM', color=TEXT, fontsize=9, padding=3)
        axes[1].set_ylabel("Revenue (₹M)", color=MUTED)
        axes[1].set_title("Revenue by Zone", color=GOLD, fontsize=12,
                           fontweight='bold')
        axes[1].grid(axis='y', color=GRID_COLOR, linewidth=0.6)
        axes[1].set_axisbelow(True)

        fig.tight_layout()
        st.pyplot(fig); plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — PRODUCTS
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    if 'Product_Category' in df.columns:
        st.markdown('<div class="section-title">Category Performance</div>', unsafe_allow_html=True)

        cat_rev = (df.groupby('Product_Category', as_index=False)['Amount']
                   .sum().sort_values('Amount', ascending=False).head(10))

        fig, ax = plt.subplots(figsize=(12, 5))
        clrs = [GOLD, SAFFRON] + [CRIMSON] * (len(cat_rev) - 2)
        bars = ax.bar(cat_rev['Product_Category'], cat_rev['Amount'] / 1e6,
                      color=clrs, edgecolor=DARK_BG, linewidth=1)
        ax.bar_label(bars, fmt='₹%.1fM', color=TEXT, fontsize=8, padding=3, rotation=45)
        plt.xticks(rotation=35, ha='right')
        ax.set_ylabel("Revenue (₹M)", color=MUTED)
        style_axes(ax, fig, "Top 10 Product Categories by Revenue")
        fig.tight_layout()
        st.pyplot(fig); plt.close(fig)

        # Category + Gender heatmap
        st.markdown('<div class="section-title">Category × Gender Heatmap</div>', unsafe_allow_html=True)
        top_cats = cat_rev['Product_Category'].head(8).tolist()
        heat_df  = df[df['Product_Category'].isin(top_cats)]
        heat_pivot = heat_df.pivot_table(
            index='Product_Category', columns='Gender',
            values='Amount', aggfunc='sum', fill_value=0
        ) / 1e6

        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor(DARK_BG)
        ax.set_facecolor(CARD_BG)
        sns.heatmap(heat_pivot, ax=ax, annot=True, fmt='.1f',
                    cmap='YlOrBr', linewidths=0.5, linecolor=DARK_BG,
                    annot_kws={'color': '#1A0A00', 'size': 9})
        ax.set_title("Revenue (₹M) — Category × Gender", color=GOLD,
                     fontsize=12, fontweight='bold', pad=12)
        ax.tick_params(colors=MUTED, labelsize=9)
        ax.set_xlabel("Gender", color=MUTED)
        ax.set_ylabel("Category", color=MUTED)
        fig.tight_layout()
        st.pyplot(fig); plt.close(fig)

    if 'Product_ID' in df.columns:
        st.markdown('<div class="section-title">Top 10 Products by Revenue</div>', unsafe_allow_html=True)
        prod_rev = (df.groupby('Product_ID', as_index=False)['Amount']
                    .sum().sort_values('Amount', ascending=False).head(10))
        fig, ax = plt.subplots(figsize=(12, 4))
        bars = ax.bar(prod_rev['Product_ID'].astype(str), prod_rev['Amount'] / 1e3,
                      color=PALETTE[:len(prod_rev)], edgecolor=DARK_BG, linewidth=1)
        ax.bar_label(bars, fmt='₹%.0fK', color=TEXT, fontsize=8, padding=3)
        plt.xticks(rotation=30)
        ax.set_ylabel("Revenue (₹K)", color=MUTED)
        style_axes(ax, fig, "Top 10 Products by Revenue")
        fig.tight_layout()
        st.pyplot(fig); plt.close(fig)

    else:
        st.info("Product_Category column not found in data.")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — DEEP DIVE
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-title">Occupation Analysis</div>', unsafe_allow_html=True)

    if 'Occupation' in df.columns:
        occ_rev = (df.groupby('Occupation', as_index=False)['Amount']
                   .sum().sort_values('Amount', ascending=False))

        fig, ax = plt.subplots(figsize=(12, 4))
        clrs = [GOLD if i == 0 else SAFFRON if i < 3 else CRIMSON
                for i in range(len(occ_rev))]
        bars = ax.bar(occ_rev['Occupation'], occ_rev['Amount'] / 1e6,
                      color=clrs, edgecolor=DARK_BG, linewidth=1)
        ax.bar_label(bars, fmt='₹%.1fM', color=TEXT, fontsize=8, padding=3)
        plt.xticks(rotation=35, ha='right')
        ax.set_ylabel("Revenue (₹M)", color=MUTED)
        style_axes(ax, fig, "Revenue by Occupation")
        fig.tight_layout()
        st.pyplot(fig); plt.close(fig)

        top_occ = occ_rev.iloc[0]['Occupation']
        top_occ_rev = occ_rev.iloc[0]['Amount']
        st.markdown(f"""<div class="insight-box">
        💡 <strong>{top_occ}</strong> professionals are the biggest spenders,
        contributing <strong>₹{top_occ_rev/1e6:.2f}M</strong> in total revenue.
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-title">Revenue Distribution</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        fig, ax = plt.subplots(figsize=(5.5, 4))
        ax.hist(df['Amount'], bins=40, color=SAFFRON, edgecolor=DARK_BG, linewidth=0.6, alpha=0.85)
        ax.axvline(df['Amount'].mean(), color=GOLD, linewidth=2, linestyle='--',
                   label=f"Mean ₹{df['Amount'].mean():,.0f}")
        ax.axvline(df['Amount'].median(), color=CRIMSON, linewidth=2, linestyle=':',
                   label=f"Median ₹{df['Amount'].median():,.0f}")
        ax.legend(facecolor=CARD_BG, labelcolor=TEXT, edgecolor=GOLD, fontsize=8)
        ax.set_xlabel("Amount (₹)", color=MUTED)
        ax.set_ylabel("Frequency", color=MUTED)
        style_axes(ax, fig, "Spend Distribution")
        fig.tight_layout()
        st.pyplot(fig); plt.close(fig)

    with c2:
        if 'Age Group' in df.columns:
            fig, ax = plt.subplots(figsize=(5.5, 4))
            fig.patch.set_facecolor(DARK_BG)
            ax.set_facecolor(CARD_BG)
            age_order = sorted(df['Age Group'].dropna().unique())
            sns.boxplot(data=df, x='Age Group', y='Amount', order=age_order,
                        palette={g: PALETTE[i] for i, g in enumerate(age_order)},
                        ax=ax, linewidth=0.8,
                        medianprops={'color': GOLD, 'linewidth': 2},
                        whiskerprops={'color': MUTED},
                        capprops={'color': MUTED},
                        flierprops={'marker': 'o', 'color': SAFFRON, 'alpha': 0.4, 'markersize': 3})
            plt.xticks(rotation=30)
            ax.set_xlabel("Age Group", color=MUTED)
            ax.set_ylabel("Amount (₹)", color=MUTED)
            ax.set_title("Spend Spread by Age Group", color=GOLD, fontsize=12,
                         fontweight='bold', pad=12)
            ax.tick_params(colors=MUTED, labelsize=8)
            for spine in ax.spines.values():
                spine.set_edgecolor(SPINE_COLOR)
            ax.grid(axis='y', color=GRID_COLOR, linewidth=0.6)
            ax.set_axisbelow(True)
            fig.tight_layout()
            st.pyplot(fig); plt.close(fig)

    # Correlation heatmap
    st.markdown('<div class="section-title">Numeric Correlation Matrix</div>', unsafe_allow_html=True)
    num_cols = df.select_dtypes(include='number').columns.tolist()
    if len(num_cols) > 1:
        corr = df[num_cols].corr()
        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor(DARK_BG)
        ax.set_facecolor(CARD_BG)
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, ax=ax, annot=True, fmt='.2f',
                    cmap='YlOrBr', linewidths=0.5, linecolor=DARK_BG,
                    annot_kws={'color': '#1A0A00', 'size': 9},
                    vmin=-1, vmax=1)
        ax.set_title("Correlation Matrix", color=GOLD, fontsize=12,
                     fontweight='bold', pad=12)
        ax.tick_params(colors=MUTED, labelsize=9)
        fig.tight_layout()
        st.pyplot(fig); plt.close(fig)

# ─── Footer ──────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#B8966E; font-size:0.8rem; letter-spacing:1.5px; text-transform:uppercase; padding: 0.5rem 0;'>
    🪔 Diwali Sales Intelligence Dashboard &nbsp;|&nbsp; Built with Streamlit
</div>
""", unsafe_allow_html=True)