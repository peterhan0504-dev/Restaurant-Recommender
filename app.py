"""
🍽️ AI-Powered Restaurant Recommendation Engine
Streamlit application — Content-Based + LLM-Semantic approaches
"""

import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🍽️ AI Restaurant Finder",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Imports ──────────────────────────────────────────────────────────────────
from recommender import (
    load_restaurants,
    content_based_recommend,
    llm_semantic_recommend,
    hybrid_recommend,
    get_embeddings,
    CUISINES,
)
from evaluation import (
    precision_at_k,
    recall_at_k,
    ndcg_at_k,
    intra_list_diversity,
    category_coverage,
    average_rating,
    simulate_evaluation,
)

# ─── CSS Styling ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global ── */
body { font-family: 'Segoe UI', sans-serif; }

/* ── Restaurant Card ── */
.rec-card {
    background: #ffffff;
    border-radius: 14px;
    border: 1px solid #e8ecf0;
    padding: 20px 22px;
    margin-bottom: 16px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    transition: box-shadow 0.2s;
}
.rec-card:hover { box-shadow: 0 4px 18px rgba(0,0,0,0.12); }

/* ── Card Header ── */
.card-title {
    font-size: 1.18rem;
    font-weight: 700;
    color: #1a1a2e;
    margin-bottom: 2px;
}
.card-meta {
    font-size: 0.82rem;
    color: #6c757d;
    margin-bottom: 8px;
}
.card-desc {
    font-size: 0.88rem;
    color: #444;
    line-height: 1.5;
    margin-bottom: 10px;
}

/* ── Tags ── */
.tag {
    display: inline-block;
    background: #f0f4ff;
    color: #3d5af1;
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.75rem;
    margin-right: 4px;
    margin-bottom: 4px;
    font-weight: 500;
}
.tag-green { background: #eafaf1; color: #1a7a4a; }
.tag-orange { background: #fff3e0; color: #e65100; }
.tag-purple { background: #f3e5f5; color: #6a1b9a; }

/* ── Explanation Box ── */
.explanation {
    background: #f8f9ff;
    border-left: 3px solid #3d5af1;
    border-radius: 0 8px 8px 0;
    padding: 8px 12px;
    font-size: 0.83rem;
    color: #333;
    margin-top: 8px;
    font-style: italic;
}

/* ── Score Badge ── */
.score-badge {
    float: right;
    background: linear-gradient(135deg, #3d5af1, #6c63ff);
    color: white;
    border-radius: 20px;
    padding: 3px 10px;
    font-size: 0.75rem;
    font-weight: 600;
}

/* ── Method Badge ── */
.method-cb   { background:#e8f5e9; color:#2e7d32; border-radius:10px; padding:2px 8px; font-size:0.72rem; font-weight:600; }
.method-llm  { background:#e3f2fd; color:#1565c0; border-radius:10px; padding:2px 8px; font-size:0.72rem; font-weight:600; }
.method-hyb  { background:#fce4ec; color:#880e4f; border-radius:10px; padding:2px 8px; font-size:0.72rem; font-weight:600; }

/* ── Metrics Table ── */
.metric-row {
    display:flex; justify-content:space-between;
    padding: 6px 0; border-bottom: 1px solid #f0f0f0;
    font-size: 0.88rem;
}

/* ── Section Header ── */
.section-header {
    font-size: 1.05rem; font-weight: 700; color: #1a1a2e;
    margin: 16px 0 8px 0; padding-bottom: 4px;
    border-bottom: 2px solid #3d5af1;
}
</style>
""", unsafe_allow_html=True)

# ─── Session State Init ────────────────────────────────────────────────────────
if "feedback" not in st.session_state:
    st.session_state.feedback = {}          # {id: 'like'|'dislike'}
if "ratings" not in st.session_state:
    st.session_state.ratings = {}           # {id: 1-5}
if "liked_ids" not in st.session_state:
    st.session_state.liked_ids = []
if "last_recs" not in st.session_state:
    st.session_state.last_recs = []
if "eval_results" not in st.session_state:
    st.session_state.eval_results = None

# ─── Data Load ────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading restaurants & embeddings…")
def load_all():
    restaurants = load_restaurants()
    embeddings = get_embeddings(restaurants)
    id_to_idx = {r["id"]: i for i, r in enumerate(restaurants)}
    return restaurants, embeddings, id_to_idx

restaurants, embeddings, id_to_idx = load_all()

CUISINE_LIST = sorted(set(r["cuisine"] for r in restaurants))
price_label = {"$": "Budget ($)", "$$": "Moderate ($$)", "$$$": "Upscale ($$$)", "$$$$": "Fine Dining ($$$$)"}

# ─── Helpers ──────────────────────────────────────────────────────────────────
def star_str(rating: float) -> str:
    full = int(rating)
    half = 1 if rating - full >= 0.5 else 0
    empty = 5 - full - half
    return "★" * full + "½" * half + "☆" * empty

def price_color(p: str) -> str:
    return {"$": "tag-green", "$$": "tag", "$$$": "tag-orange", "$$$$": "tag-purple"}.get(p, "tag")

def method_badge(m: str) -> str:
    cls = {"content-based": "method-cb", "llm-semantic": "method-llm", "hybrid": "method-hyb"}.get(m, "method-cb")
    label = {"content-based": "🔍 Content-Based", "llm-semantic": "🤖 LLM-Semantic", "hybrid": "⚡ Hybrid"}.get(m, m)
    return f'<span class="{cls}">{label}</span>'

def render_restaurant_card(r: dict, show_feedback: bool = True, idx: int = 0):
    """Render a restaurant recommendation card."""
    dishes_preview = ", ".join(r["popular_dishes"][:3])
    dietary_tags = "".join(f'<span class="tag tag-green">{d}</span>' for d in r.get("dietary_options", []))
    feature_tags = "".join(f'<span class="tag">{f}</span>' for f in r.get("features", [])[:4])
    score = r.get("_score", 0)
    expl = r.get("_explanation", "")
    method = r.get("_method", "")

    html = f"""
    <div class="rec-card">
      <span class="score-badge">Match {score:.0%}</span>
      <div class="card-title">🍽️ {r['name']}</div>
      <div class="card-meta">
        {method_badge(method)} &nbsp;|&nbsp;
        <b>{r['cuisine']}</b> &nbsp;|&nbsp;
        {star_str(r['rating'])} {r['rating']} ({r['review_count']:,} reviews) &nbsp;|&nbsp;
        <span class="{price_color(r['price_range'])}">{r['price_range']}</span> &nbsp;|&nbsp;
        📍 {r['location']} &nbsp;|&nbsp; 🪑 {r['ambiance']}
      </div>
      <div class="card-desc">{r['description'][:200]}…</div>
      <div><b>🍴 Popular dishes:</b> <span style="color:#555">{dishes_preview}</span></div>
      <div style="margin-top:6px">{dietary_tags}{feature_tags}</div>
      <div class="explanation">💡 {expl}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

    if show_feedback:
        col_l, col_d, col_r, col_spacer = st.columns([1, 1, 2, 6])
        rid = r["id"]
        current_feedback = st.session_state.feedback.get(rid, "")
        current_rating = st.session_state.ratings.get(rid, 0)

        with col_l:
            liked = current_feedback == "like"
            if st.button("👍 Like" if not liked else "✅ Liked", key=f"like_{rid}_{idx}", type="secondary"):
                st.session_state.feedback[rid] = "like"
                if rid not in st.session_state.liked_ids:
                    st.session_state.liked_ids.append(rid)
                st.rerun()
        with col_d:
            disliked = current_feedback == "dislike"
            if st.button("👎 Dislike" if not disliked else "❌ Disliked", key=f"dislike_{rid}_{idx}", type="secondary"):
                st.session_state.feedback[rid] = "dislike"
                if rid in st.session_state.liked_ids:
                    st.session_state.liked_ids.remove(rid)
                st.rerun()
        with col_r:
            rating = st.select_slider(
                "Rate", options=[0, 1, 2, 3, 4, 5],
                format_func=lambda x: ["–", "⭐", "⭐⭐", "⭐⭐⭐", "⭐⭐⭐⭐", "⭐⭐⭐⭐⭐"][x],
                value=current_rating,
                key=f"rate_{rid}_{idx}",
                label_visibility="collapsed",
            )
            if rating != current_rating:
                st.session_state.ratings[rid] = rating
                if rating >= 4 and rid not in st.session_state.liked_ids:
                    st.session_state.liked_ids.append(rid)
                elif rating <= 2 and rid in st.session_state.liked_ids:
                    st.session_state.liked_ids.remove(rid)


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/restaurant.png", width=72)
    st.title("🍽️ Restaurant Finder")
    st.caption("AI-powered • Personalized • Explained")

    st.divider()
    st.markdown("### ⚙️ Settings")

    approach = st.radio(
        "Recommendation approach",
        ["🤖 LLM-Semantic (Query)", "🔍 Content-Based (Liked items)", "⚡ Hybrid (Both)"],
        index=0,
    )

    top_k = st.slider("Number of recommendations", 3, 20, 8)

    st.divider()
    st.markdown("### 🏷️ Filter by Cuisine")
    selected_cuisines = st.multiselect(
        "Select cuisines (optional)",
        options=CUISINE_LIST,
        placeholder="Any cuisine…",
    )

    st.divider()
    st.markdown("### 📊 Your Activity")
    n_liked = len(st.session_state.liked_ids)
    n_rated = len(st.session_state.ratings)
    n_feedback = len(st.session_state.feedback)
    st.metric("Items Liked", n_liked)
    st.metric("Items Rated", n_rated)
    st.metric("Feedback Given", n_feedback)

    if st.button("🔄 Reset Session", type="secondary"):
        for key in ["feedback", "ratings", "liked_ids", "last_recs", "eval_results"]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

# ─── Main Content ─────────────────────────────────────────────────────────────
st.title("🍽️ AI Restaurant Recommendation Engine")
st.caption("Powered by sentence-transformers embeddings · Content-Based & LLM-Semantic approaches")

tab_discover, tab_browse, tab_mylist, tab_metrics, tab_about = st.tabs([
    "🔍 Discover", "🗂️ Browse All", "❤️ My List", "📊 Metrics", "ℹ️ About"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — DISCOVER
# ══════════════════════════════════════════════════════════════════════════════
with tab_discover:
    st.markdown("## Find Your Perfect Restaurant")

    # ── Input Methods ─────────────────────────────────────────────────────────
    input_col1, input_col2 = st.columns([3, 2])

    with input_col1:
        st.markdown('<div class="section-header">💬 Describe What You\'re Looking For</div>', unsafe_allow_html=True)
        query = st.text_area(
            "Natural language query",
            placeholder="e.g. 'I want cozy Italian food with outdoor seating under $30' or 'spicy Asian, vegan-friendly, downtown'",
            height=90,
            label_visibility="collapsed",
        )

    with input_col2:
        st.markdown('<div class="section-header">⭐ Rate a Restaurant to Personalise</div>', unsafe_allow_html=True)
        browse_name = st.selectbox(
            "Pick a restaurant you know",
            options=["— Select one —"] + [r["name"] for r in sorted(restaurants, key=lambda x: x["name"])],
            label_visibility="collapsed",
        )
        if browse_name != "— Select one —":
            chosen = next((r for r in restaurants if r["name"] == browse_name), None)
            if chosen:
                quick_rating = st.select_slider(
                    "Your rating",
                    options=[1, 2, 3, 4, 5],
                    format_func=lambda x: "⭐" * x,
                    value=st.session_state.ratings.get(chosen["id"], 3),
                    key="quick_rate",
                )
                if st.button("✅ Save rating", key="save_quick"):
                    st.session_state.ratings[chosen["id"]] = quick_rating
                    if quick_rating >= 4 and chosen["id"] not in st.session_state.liked_ids:
                        st.session_state.liked_ids.append(chosen["id"])
                    st.success(f"Saved {quick_rating}★ for {chosen['name']}!")

    # ── Recommend Button ──────────────────────────────────────────────────────
    st.markdown("")
    get_recs = st.button("🚀 Get Recommendations", type="primary", use_container_width=True)

    if get_recs or st.session_state.last_recs:
        exclude = set(st.session_state.liked_ids)

        if get_recs:
            with st.spinner("Computing personalised recommendations…"):
                liked_ids = st.session_state.liked_ids or None
                rated_items = {k: v for k, v in st.session_state.ratings.items() if v > 0} or None

                if "LLM" in approach:
                    if not query and not selected_cuisines:
                        st.warning("Please enter a query or select cuisines for LLM-Semantic recommendations.")
                        st.stop()
                    recs = llm_semantic_recommend(
                        restaurants, query or " ".join(selected_cuisines),
                        selected_categories=selected_cuisines or None,
                        exclude_ids=exclude, top_k=top_k,
                    )
                elif "Content" in approach:
                    if not liked_ids and not rated_items:
                        st.info("ℹ️ Rate or like some restaurants first for content-based recommendations. Showing top-rated instead.")
                        recs = sorted(restaurants, key=lambda x: -x["rating"])[:top_k]
                        for r in recs:
                            r["_score"] = r["rating"] / 5
                            r["_method"] = "content-based"
                            r["_explanation"] = f"Highly rated {r['cuisine']} restaurant ({r['rating']}★) in {r['location']}."
                    else:
                        recs = content_based_recommend(
                            restaurants, liked_ids=liked_ids, rated_items=rated_items,
                            exclude_ids=exclude, top_k=top_k,
                        )
                else:  # Hybrid
                    recs = hybrid_recommend(
                        restaurants, liked_ids=liked_ids, rated_items=rated_items,
                        query=query or None, selected_categories=selected_cuisines or None,
                        exclude_ids=exclude, top_k=top_k,
                    )

            st.session_state.last_recs = recs

        recs = st.session_state.last_recs

        if recs:
            st.success(f"✨ Found {len(recs)} recommendations")
            st.divider()

            for i, r in enumerate(recs):
                render_restaurant_card(r, show_feedback=True, idx=i)
        else:
            st.info("No recommendations found. Try a different query or like some restaurants first.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — BROWSE ALL
# ══════════════════════════════════════════════════════════════════════════════
with tab_browse:
    st.markdown("## 🗂️ Browse Restaurant Database")
    st.caption(f"Explore all {len(restaurants)} restaurants in the dataset")

    # Filters
    bcol1, bcol2, bcol3, bcol4 = st.columns(4)
    with bcol1:
        f_cuisine = st.multiselect("Cuisine", CUISINE_LIST, key="b_cuisine")
    with bcol2:
        f_price = st.multiselect("Price", ["$", "$$", "$$$", "$$$$"], key="b_price")
    with bcol3:
        f_ambiance = st.multiselect("Ambiance", sorted(set(r["ambiance"] for r in restaurants)), key="b_amb")
    with bcol4:
        f_sort = st.selectbox("Sort by", ["Rating ↓", "Name A-Z", "Price ↑", "Price ↓"], key="b_sort")

    filtered = restaurants
    if f_cuisine:
        filtered = [r for r in filtered if r["cuisine"] in f_cuisine]
    if f_price:
        filtered = [r for r in filtered if r["price_range"] in f_price]
    if f_ambiance:
        filtered = [r for r in filtered if r["ambiance"] in f_ambiance]

    if f_sort == "Rating ↓":
        filtered = sorted(filtered, key=lambda x: -x["rating"])
    elif f_sort == "Name A-Z":
        filtered = sorted(filtered, key=lambda x: x["name"])
    elif f_sort == "Price ↑":
        filtered = sorted(filtered, key=lambda x: len(x["price_range"]))
    elif f_sort == "Price ↓":
        filtered = sorted(filtered, key=lambda x: -len(x["price_range"]))

    st.caption(f"Showing {len(filtered)} restaurants")

    # Paginate
    page_size = 12
    n_pages = max(1, (len(filtered) + page_size - 1) // page_size)
    page = st.number_input("Page", 1, n_pages, 1, key="b_page") - 1
    page_items = filtered[page * page_size: (page + 1) * page_size]

    # Table view
    df = pd.DataFrame([{
        "Name": r["name"],
        "Cuisine": r["cuisine"],
        "Rating": r["rating"],
        "Price": r["price_range"],
        "Ambiance": r["ambiance"],
        "Location": r["location"],
        "Dishes": ", ".join(r["popular_dishes"][:2]),
    } for r in page_items])
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Quick like from browse
    st.markdown("**Like a restaurant from this page:**")
    browse_like = st.selectbox("Select restaurant to like", ["—"] + [r["name"] for r in page_items], key="bl_select")
    if browse_like != "—":
        if st.button("👍 Like this restaurant", key="bl_btn"):
            r = next((x for x in page_items if x["name"] == browse_like), None)
            if r and r["id"] not in st.session_state.liked_ids:
                st.session_state.liked_ids.append(r["id"])
                st.session_state.feedback[r["id"]] = "like"
                st.success(f"Added {r['name']} to your liked list!")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — MY LIST
# ══════════════════════════════════════════════════════════════════════════════
with tab_mylist:
    st.markdown("## ❤️ My Liked & Rated Restaurants")

    id_to_r = {r["id"]: r for r in restaurants}

    liked_restaurants = [id_to_r[rid] for rid in st.session_state.liked_ids if rid in id_to_r]
    rated_restaurants = [(id_to_r[rid], rating) for rid, rating in st.session_state.ratings.items() if rid in id_to_r and rating > 0]

    if liked_restaurants or rated_restaurants:
        if liked_restaurants:
            st.markdown("### 👍 Liked Restaurants")
            for r in liked_restaurants:
                feedback = st.session_state.feedback.get(r["id"], "like")
                rating = st.session_state.ratings.get(r["id"], 0)
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    st.markdown(f"**{r['name']}** — {r['cuisine']} | {star_str(r['rating'])} | {r['price_range']}")
                with col2:
                    st.caption(f"Your rating: {'⭐' * rating if rating else '—'}")
                with col3:
                    if st.button("🗑️ Remove", key=f"rem_{r['id']}"):
                        st.session_state.liked_ids.remove(r["id"])
                        if r["id"] in st.session_state.feedback:
                            del st.session_state.feedback[r["id"]]
                        st.rerun()

        if rated_restaurants:
            st.markdown("### ⭐ Your Ratings")
            rated_df = pd.DataFrame([{
                "Restaurant": r["name"],
                "Cuisine": r["cuisine"],
                "Your Rating": "⭐" * rating,
                "Stars": rating,
                "Price": r["price_range"],
            } for r, rating in rated_restaurants])
            st.dataframe(rated_df[["Restaurant", "Cuisine", "Your Rating", "Price"]], use_container_width=True, hide_index=True)
    else:
        st.info("You haven't liked or rated any restaurants yet. Use the Discover tab to explore!")
        st.markdown("**Quick start:** Browse the database and like a few restaurants, then come back here.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — METRICS
# ══════════════════════════════════════════════════════════════════════════════
with tab_metrics:
    st.markdown("## 📊 Evaluation Metrics")
    st.caption("Simulated evaluation across 50 synthetic user profiles with known preferences")

    k_eval = st.slider("Evaluate at K =", 5, 20, 10, key="k_eval")

    run_eval = st.button("▶️ Run Evaluation", type="primary")

    if run_eval:
        with st.spinner("Running simulated evaluation across 50 users…"):
            cb_metrics = simulate_evaluation(
                restaurants,
                recommender_fn=lambda **kw: content_based_recommend(**kw),
                n_users=50,
                k=k_eval,
                seed=42,
            )
            llm_metrics = simulate_evaluation(
                restaurants,
                recommender_fn=lambda **kw: llm_semantic_recommend(
                    restaurants=kw["restaurants"],
                    query="",
                    exclude_ids=kw.get("exclude_ids"),
                    top_k=kw.get("top_k", 10),
                ),
                n_users=50,
                k=k_eval,
                seed=42,
            )
        st.session_state.eval_results = (cb_metrics, llm_metrics, k_eval)

    if st.session_state.eval_results:
        cb_m, llm_m, k_used = st.session_state.eval_results

        st.markdown(f"### Results at K = {k_used}")
        col_cb, col_llm = st.columns(2)

        def metric_card(title: str, metrics: dict):
            st.markdown(f"**{title}**")
            for name, val in metrics.items():
                if name == "N Users Evaluated":
                    st.metric(name, int(val))
                elif "Coverage" in name or "Ratio" in name:
                    st.metric(name, f"{val:.1%}")
                elif "Rating" in name:
                    st.metric(name, f"{val:.2f} ★")
                else:
                    st.metric(name, f"{val:.3f}")

        with col_cb:
            metric_card("🔍 Content-Based Filtering", cb_m)

        with col_llm:
            metric_card("🤖 LLM-Semantic Search", llm_m)

        # Comparison chart
        st.markdown("### 📈 Metric Comparison")
        common_keys = [k for k in cb_m if k in llm_m and k != "N Users Evaluated"]
        chart_df = pd.DataFrame({
            "Metric": common_keys,
            "Content-Based": [cb_m[k] for k in common_keys],
            "LLM-Semantic": [llm_m[k] for k in common_keys],
        }).set_index("Metric")
        st.bar_chart(chart_df)

        st.divider()
        st.markdown("### 🔬 Metric Definitions")
        st.markdown("""
| Metric | Description |
|---|---|
| **Precision@K** | Fraction of top-K recommendations that are relevant (same cuisine as user preference) |
| **Recall@K** | Fraction of all relevant restaurants found in the top-K list |
| **NDCG@K** | Discounted Cumulative Gain — rewards relevant items ranked higher |
| **Intra-List Diversity** | Average pairwise embedding distance among recommendations (higher = more diverse) |
| **Category Coverage** | Fraction of cuisine categories represented in the recommendation list |
| **Avg Recommended Rating** | Mean star rating of recommended restaurants (proxy for quality) |
        """)

    # ── Live session metrics ───────────────────────────────────────────────────
    st.divider()
    st.markdown("### 🎯 Your Live Session Metrics")

    if st.session_state.last_recs and st.session_state.feedback:
        recs = st.session_state.last_recs
        rec_ids = [r["id"] for r in recs]
        liked_set = {rid for rid, v in st.session_state.feedback.items() if v == "like"}
        disliked_set = {rid for rid, v in st.session_state.feedback.items() if v == "dislike"}

        feedback_on_recs = {rid: v for rid, v in st.session_state.feedback.items() if rid in set(rec_ids)}

        lc1, lc2, lc3, lc4 = st.columns(4)
        with lc1:
            total_feedback = len(feedback_on_recs)
            likes_on_recs = sum(1 for v in feedback_on_recs.values() if v == "like")
            st.metric("Like Ratio (your session)", f"{likes_on_recs/total_feedback:.0%}" if total_feedback else "—")
        with lc2:
            st.metric("Precision@K (liked)", f"{precision_at_k(rec_ids, liked_set, len(recs)):.1%}")
        with lc3:
            ild = intra_list_diversity(rec_ids, embeddings, id_to_idx) if rec_ids else 0
            st.metric("Diversity (ILD)", f"{ild:.3f}")
        with lc4:
            cov = category_coverage(recs, restaurants)
            st.metric("Category Coverage", f"{cov:.1%}")
    else:
        st.info("Get recommendations and provide feedback to see your live metrics here.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — ABOUT
# ══════════════════════════════════════════════════════════════════════════════
with tab_about:
    st.markdown("## ℹ️ About This System")

    st.markdown("""
### 🏗️ Architecture

This recommendation engine combines two complementary approaches:

**1. Content-Based Filtering**
- Each restaurant is represented as a rich text document (name + description + cuisine + dishes + features)
- Documents are embedded using `sentence-transformers/all-MiniLM-L6-v2` (a lightweight BERT-based model)
- User preferences are encoded as a **weighted average** of liked-item embeddings (with star ratings as weights)
- Recommendations are ranked by **cosine similarity** to the preference vector

**2. LLM-Semantic Search**
- User's natural language query is embedded using the same sentence-transformer model
- Structured preferences (cuisine, price, ambiance, dietary) are extracted via keyword parsing
- A **composite score** blends semantic similarity (80%) with structured attribute matching boosts (20%)
- Explanations cite the specific attributes that drove each recommendation

**3. Hybrid Mode**
- Blends content-based and LLM-semantic scores using a configurable weight (default 50/50)
- Best used when the user has both liked items AND a natural language query

---

### 📊 Dataset

- **300 restaurants** spanning **20 cuisines** (15 per cuisine)
- Each restaurant has: name, description, cuisine, price range, rating, ambiance, location, dietary options, popular dishes, features
- Synthetically generated using realistic templates and variety

---

### 🧠 Explanation Generation

Every recommendation includes a **specific, attribute-driven explanation**:
- Content-based: cites shared cuisine, price point, ambiance with the user's liked items
- LLM-semantic: cites which query terms matched which restaurant attributes

---

### 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Embeddings | `sentence-transformers` (all-MiniLM-L6-v2) |
| Similarity | Cosine similarity (numpy) |
| UI | Streamlit |
| Data | Synthetic JSON (300 restaurants) |
| Metrics | Custom implementation (Precision, Recall, NDCG, ILD) |

---
*Built as part of an AI Recommendation System project.*
    """)

    st.markdown("### 📈 Dataset Statistics")
    cuisine_counts = pd.Series([r["cuisine"] for r in restaurants]).value_counts()
    price_counts = pd.Series([r["price_range"] for r in restaurants]).value_counts().sort_index()
    ambiance_counts = pd.Series([r["ambiance"] for r in restaurants]).value_counts()

    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        st.markdown("**Cuisine Distribution**")
        st.bar_chart(cuisine_counts)
    with sc2:
        st.markdown("**Price Distribution**")
        st.bar_chart(price_counts)
    with sc3:
        st.markdown("**Ambiance Distribution**")
        st.bar_chart(ambiance_counts)
