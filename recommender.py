"""
Restaurant Recommendation Engine
Implements two approaches:
  1. Content-Based Filtering — sentence-transformer embeddings + cosine similarity
  2. LLM-Powered Semantic Search — natural language query parsing + semantic matching

Author: AI Recommendation System Project
"""
from __future__ import annotations  # makes all annotations strings → works on Python 3.9+

import json
import re
import math
from pathlib import Path
from typing import Any, Optional

import numpy as np

# ─── Data Loading ────────────────────────────────────────────────────────────

DATA_PATH = Path(__file__).parent / "restaurants.json"

def load_restaurants() -> list:
    with open(DATA_PATH) as f:
        return json.load(f)


# ─── Text Preparation ─────────────────────────────────────────────────────────

def build_rich_text(r: dict) -> str:
    """Concatenate all meaningful fields into a single text for embedding."""
    parts = [
        r["name"],
        r["description"],
        f"Cuisine: {r['cuisine']}",
        f"Ambiance: {r['ambiance']}",
        f"Price range: {r['price_range']}",
        f"Location: {r['location']}",
        "Popular dishes: " + ", ".join(r["popular_dishes"]),
        "Dietary options: " + (", ".join(r["dietary_options"]) if r["dietary_options"] else "None specified"),
        "Features: " + ", ".join(r["features"]),
    ]
    return " | ".join(parts)


# ─── Embedding Engine ─────────────────────────────────────────────────────────

_model = None
_embeddings: Optional[np.ndarray] = None
_restaurants: Optional[list] = None


def get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def get_embeddings(restaurants: list) -> np.ndarray:
    """Compute (and cache) embeddings for all restaurants."""
    global _embeddings, _restaurants
    if _embeddings is None or _restaurants is not restaurants:
        model = get_model()
        texts = [build_rich_text(r) for r in restaurants]
        _embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        _restaurants = restaurants
    return _embeddings


def embed_query(text: str) -> np.ndarray:
    model = get_model()
    return model.encode([text], normalize_embeddings=True)[0]


def cosine_similarity_matrix(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Dot product suffices when both sides are L2-normalised."""
    return matrix @ query_vec


# ─── 1. Content-Based Recommender ────────────────────────────────────────────

def content_based_recommend(
    restaurants: list,
    liked_ids: Optional[list] = None,
    rated_items: Optional[dict] = None,
    exclude_ids: Optional[set] = None,
    top_k: int = 10,
) -> list:
    """
    Recommend restaurants based on user's liked/rated items.

    Strategy: compute a weighted average of liked-item embeddings (using
    ratings as weights when available), then rank by cosine similarity.
    """
    if not liked_ids and not rated_items:
        return []

    embeddings = get_embeddings(restaurants)
    id_to_idx = {r["id"]: i for i, r in enumerate(restaurants)}

    # Build weighted preference vector
    vectors, weights = [], []
    if rated_items:
        for rid, rating in rated_items.items():
            if rid in id_to_idx:
                vectors.append(embeddings[id_to_idx[rid]])
                weights.append(max(0.1, (rating - 2.5) / 2.5))  # centre around 2.5
    if liked_ids:
        for rid in liked_ids:
            if rid in id_to_idx and rid not in (rated_items or {}):
                vectors.append(embeddings[id_to_idx[rid]])
                weights.append(1.0)

    if not vectors:
        return []

    weights = np.array(weights)
    if weights.sum() <= 0:
        weights = np.ones_like(weights)
    preference_vec = np.average(vectors, axis=0, weights=weights)
    norm = np.linalg.norm(preference_vec)
    if norm > 0:
        preference_vec /= norm

    sims = cosine_similarity_matrix(preference_vec, embeddings)

    exclude = set(liked_ids or []) | set((rated_items or {}).keys()) | (exclude_ids or set())
    ranked = sorted(
        [(i, s) for i, s in enumerate(sims) if restaurants[i]["id"] not in exclude],
        key=lambda x: -x[1],
    )

    results = []
    for idx, score in ranked[:top_k]:
        r = dict(restaurants[idx])
        r["_score"] = float(np.clip(score, 0.0, 1.0))
        r["_method"] = "content-based"
        r["_explanation"] = generate_content_explanation(r, restaurants, liked_ids, rated_items)
        results.append(r)
    return results


# ─── 2. LLM-Powered Semantic Recommender ──────────────────────────────────────

# Keyword maps for structured parsing of natural language queries
CUISINE_KEYWORDS: dict = {
    "Italian":        ["italian", "pizza", "pasta", "risotto", "tiramisu", "rome", "naples", "trattoria"],
    "Chinese":        ["chinese", "dim sum", "peking duck", "dumplings", "wonton", "cantonese", "sichuan", "beijing"],
    "Japanese":       ["japanese", "sushi", "ramen", "tempura", "sashimi", "udon", "miso", "izakaya", "wagyu"],
    "Mexican":        ["mexican", "tacos", "burrito", "enchilada", "guacamole", "salsa", "mole", "tequila"],
    "Indian":         ["indian", "curry", "biryani", "tikka", "naan", "masala", "tandoor", "spice", "spicy"],
    "Thai":           ["thai", "pad thai", "green curry", "tom yum", "satay", "lemongrass", "pad see ew"],
    "French":         ["french", "bistro", "croissant", "coq au vin", "bouillabaisse", "baguette", "paris", "brasserie"],
    "American":       ["american", "burger", "bbq", "barbecue", "fried chicken", "steak", "wings", "mac and cheese"],
    "Mediterranean":  ["mediterranean", "mezze", "hummus", "falafel", "grilled", "olive oil", "pita", "aegean"],
    "Korean":         ["korean", "bbq", "bibimbap", "kimchi", "bulgogi", "k-food", "galbi"],
    "Vietnamese":     ["vietnamese", "pho", "banh mi", "spring rolls", "bun bo", "saigon", "hanoi"],
    "Greek":          ["greek", "souvlaki", "gyro", "spanakopita", "moussaka", "taverna", "athens"],
    "Spanish":        ["spanish", "tapas", "paella", "sangria", "chorizo", "patatas bravas", "barcelona"],
    "Middle Eastern": ["middle eastern", "shawarma", "kebab", "falafel", "hummus", "levant", "arabic"],
    "Ethiopian":      ["ethiopian", "injera", "doro wat", "berbere", "east african", "addis"],
    "Brazilian":      ["brazilian", "churrasco", "caipirinha", "feijoada", "samba"],
    "Peruvian":       ["peruvian", "ceviche", "lomo saltado", "pisco sour", "andes", "lima"],
    "Lebanese":       ["lebanese", "kibbeh", "labneh", "tabbouleh", "beirut", "levant"],
    "Turkish":        ["turkish", "kebab", "baklava", "pide", "istanbul", "ottoman"],
    "Caribbean":      ["caribbean", "jerk", "jerk chicken", "plantains", "reggae", "island", "rum"],
}

# Flat list of all supported cuisines (exported for use in app.py)
CUISINES = list(CUISINE_KEYWORDS.keys())

PRICE_KEYWORDS: dict = {
    "$":    ["cheap", "affordable", "budget", "inexpensive", "under $15", "under 15", "student", "value"],
    "$$":   ["moderate", "mid-range", "reasonable", "average price", "mid price", "under $30", "under 30"],
    "$$$":  ["upscale", "nice", "elevated", "premium", "date night", "anniversary", "classy", "fine"],
    "$$$$": ["luxury", "fine dining", "splurge", "expensive", "michelin", "tasting menu", "high-end"],
}

AMBIANCE_KEYWORDS: dict = {
    "Casual":           ["casual", "relaxed", "laid-back", "chill", "informal", "quick"],
    "Fine Dining":      ["fine dining", "formal", "elegant", "upscale", "white tablecloth"],
    "Family-Friendly":  ["family", "kids", "children", "family-friendly", "kid-friendly"],
    "Romantic":         ["romantic", "date night", "date", "intimate", "candlelit", "cozy dinner"],
    "Trendy":           ["trendy", "hip", "cool", "instagrammable", "modern", "chic", "fashionable"],
    "Cozy":             ["cozy", "warm", "comfortable", "snug", "homey", "intimate"],
    "Lively":           ["lively", "fun", "vibrant", "energetic", "bustling", "loud", "festive"],
    "Quiet":            ["quiet", "peaceful", "serene", "calm", "tranquil", "work-friendly"],
}

DIETARY_KEYWORDS: dict = {
    "Vegetarian-friendly": ["vegetarian", "veggie", "no meat", "plant"],
    "Vegan options":       ["vegan", "plant-based", "dairy-free vegan"],
    "Gluten-free options": ["gluten-free", "gluten free", "celiac", "no gluten"],
    "Halal":               ["halal"],
    "Kosher":              ["kosher"],
    "Dairy-free options":  ["dairy-free", "dairy free", "no dairy", "lactose"],
}

FEATURE_KEYWORDS: dict = {
    "Outdoor seating":       ["outdoor", "patio", "terrace", "outside", "al fresco"],
    "Delivery":              ["delivery", "deliver"],
    "Takeout":               ["takeout", "take-out", "take out", "to-go", "to go"],
    "Bar":                   ["bar", "drinks", "cocktails", "wine bar"],
    "Live music":            ["live music", "music", "entertainment", "jazz", "band"],
    "Private dining":        ["private", "private room", "special event"],
    "Parking available":     ["parking", "park"],
    "Reservations accepted": ["reservation", "reserve", "book a table"],
    "Pet-friendly":          ["pet", "dog", "dog-friendly"],
    "Happy hour":            ["happy hour", "happy hr", "specials"],
}


def parse_query(query: str) -> dict:
    """Extract structured preferences from a natural language query."""
    q = query.lower()
    prefs: dict = {
        "cuisines": [],
        "price_ranges": [],
        "ambiances": [],
        "dietary": [],
        "features": [],
        "raw_query": query,
    }

    for cuisine, kws in CUISINE_KEYWORDS.items():
        if any(kw in q for kw in kws):
            prefs["cuisines"].append(cuisine)

    for price, kws in PRICE_KEYWORDS.items():
        if any(kw in q for kw in kws):
            prefs["price_ranges"].append(price)

    for amb, kws in AMBIANCE_KEYWORDS.items():
        if any(kw in q for kw in kws):
            prefs["ambiances"].append(amb)

    for diet, kws in DIETARY_KEYWORDS.items():
        if any(kw in q for kw in kws):
            prefs["dietary"].append(diet)

    for feat, kws in FEATURE_KEYWORDS.items():
        if any(kw in q for kw in kws):
            prefs["features"].append(feat)

    # Extract rating preference
    m = re.search(r"(above|over|at least|minimum)\s*(\d+\.?\d*)\s*star", q)
    if m:
        prefs["min_rating"] = float(m.group(2))

    return prefs


def llm_semantic_recommend(
    restaurants: list,
    query: str,
    selected_categories: Optional[list] = None,
    exclude_ids: Optional[set] = None,
    top_k: int = 10,
) -> list:
    """
    Recommend restaurants using semantic query embedding + structured filtering.

    Approach:
      1. Parse the natural language query for hard filters (cuisine, price, dietary…)
      2. Embed the query with the same sentence-transformer model
      3. Rank by cosine similarity (semantic matching)
      4. Apply soft boosts for matching structured attributes
      5. Return top-K with attribute-specific explanations
    """
    prefs = parse_query(query)

    # Apply selected category override
    if selected_categories:
        prefs["cuisines"] = list(set(prefs["cuisines"]) | set(selected_categories))

    # Embed query
    query_vec = embed_query(query)
    embeddings = get_embeddings(restaurants)
    sims = cosine_similarity_matrix(query_vec, embeddings)

    exclude = exclude_ids or set()
    scored = []
    for i, r in enumerate(restaurants):
        if r["id"] in exclude:
            continue

        base = float(sims[i])

        # Structured attribute boosts
        boost = 0.0
        if prefs["cuisines"] and r["cuisine"] in prefs["cuisines"]:
            boost += 0.15
        if prefs["price_ranges"] and r["price_range"] in prefs["price_ranges"]:
            boost += 0.08
        if prefs["ambiances"] and r["ambiance"] in prefs["ambiances"]:
            boost += 0.06
        for diet in prefs.get("dietary", []):
            if diet in r["dietary_options"]:
                boost += 0.05
        for feat in prefs.get("features", []):
            if feat in r["features"]:
                boost += 0.03
        if "min_rating" in prefs and r["rating"] >= prefs["min_rating"]:
            boost += 0.04

        scored.append((i, base + boost, base))

    scored.sort(key=lambda x: -x[1])

    # Normalise composite scores so the display stays ≤ 100%
    max_score = scored[0][1] if scored else 1.0
    if max_score <= 0:
        max_score = 1.0

    results = []
    for idx, composite, base_sim in scored[:top_k]:
        r = dict(restaurants[idx])
        r["_score"] = float(np.clip(composite / max_score, 0.0, 1.0))
        r["_base_sim"] = base_sim
        r["_method"] = "llm-semantic"
        r["_prefs"] = prefs
        r["_explanation"] = generate_llm_explanation(r, prefs, composite)
        results.append(r)
    return results


# ─── Explanation Generation ───────────────────────────────────────────────────

def generate_content_explanation(
    restaurant: dict,
    all_restaurants: list,
    liked_ids: Optional[list],
    rated_items: Optional[dict],
) -> str:
    """Generate a specific, feature-driven explanation for content-based recs."""
    parts = []

    # Find reference restaurant (highest-rated liked item)
    ref_restaurant = None
    best_rating = -1
    id_to_r = {r["id"]: r for r in all_restaurants}

    if rated_items:
        for rid, rating in rated_items.items():
            if rating > best_rating and rid in id_to_r:
                best_rating = rating
                ref_restaurant = id_to_r[rid]
    if liked_ids and ref_restaurant is None:
        for rid in liked_ids[:1]:
            if rid in id_to_r:
                ref_restaurant = id_to_r[rid]

    if ref_restaurant:
        if restaurant["cuisine"] == ref_restaurant["cuisine"]:
            parts.append(f"Same **{restaurant['cuisine']}** cuisine as \"{ref_restaurant['name']}\" which you enjoyed")
        else:
            parts.append(f"Similar style to \"{ref_restaurant['name']}\"")

        if restaurant["price_range"] == ref_restaurant["price_range"]:
            price_labels = {"$": "budget-friendly", "$$": "moderately priced", "$$$": "upscale", "$$$$": "fine-dining"}
            parts.append(f"matching {price_labels.get(restaurant['price_range'], restaurant['price_range'])} price point ({restaurant['price_range']})")

        if restaurant["ambiance"] == ref_restaurant["ambiance"]:
            parts.append(f"{restaurant['ambiance'].lower()} atmosphere like restaurants you prefer")

    # Rating highlight
    if restaurant["rating"] >= 4.5:
        parts.append(f"exceptional rating of {restaurant['rating']}★ ({restaurant['review_count']:,} reviews)")
    elif restaurant["rating"] >= 4.0:
        parts.append(f"strong rating of {restaurant['rating']}★")

    # Signature dish
    if restaurant["popular_dishes"]:
        parts.append(f"known for {restaurant['popular_dishes'][0]}")

    if not parts:
        parts.append(f"{restaurant['cuisine']} restaurant with {restaurant['rating']}★ rating in {restaurant['location']}")

    return "Recommended because: " + "; ".join(parts[:4]) + "."


def generate_llm_explanation(restaurant: dict, prefs: dict, score: float) -> str:
    """Generate a query-specific, attribute-driven explanation for semantic recs."""
    matched_attrs = []

    if prefs["cuisines"] and restaurant["cuisine"] in prefs["cuisines"]:
        matched_attrs.append(f"**{restaurant['cuisine']}** cuisine matches your request")

    if prefs["price_ranges"] and restaurant["price_range"] in prefs["price_ranges"]:
        price_label = {"$": "budget-friendly", "$$": "moderately priced", "$$$": "upscale", "$$$$": "fine-dining"}
        matched_attrs.append(f"price range **{restaurant['price_range']}** ({price_label.get(restaurant['price_range'], '')})")

    if prefs["ambiances"] and restaurant["ambiance"] in prefs["ambiances"]:
        matched_attrs.append(f"**{restaurant['ambiance']}** ambiance as you specified")

    diet_matches = [d for d in prefs.get("dietary", []) if d in restaurant.get("dietary_options", [])]
    if diet_matches:
        matched_attrs.append(f"offers {', '.join(diet_matches)}")

    feat_matches = [f for f in prefs.get("features", []) if f in restaurant.get("features", [])]
    if feat_matches:
        matched_attrs.append(f"has {', '.join(feat_matches[:2])}")

    if restaurant["popular_dishes"]:
        matched_attrs.append(f"features **{restaurant['popular_dishes'][0]}** and other dishes relevant to your search")

    if restaurant["rating"] >= 4.5:
        matched_attrs.append(f"highly rated at {restaurant['rating']}★")
    elif restaurant["rating"] >= 4.0:
        matched_attrs.append(f"rated {restaurant['rating']}★")

    if matched_attrs:
        return "Matches your query because: " + "; ".join(matched_attrs[:5]) + "."

    return (
        f"Semantically similar to your description — a {restaurant['ambiance'].lower()} "
        f"{restaurant['cuisine']} restaurant rated {restaurant['rating']}★ "
        f"in {restaurant['location']} at {restaurant['price_range']} pricing."
    )


# ─── Hybrid Blend ─────────────────────────────────────────────────────────────

def hybrid_recommend(
    restaurants: list,
    liked_ids: Optional[list] = None,
    rated_items: Optional[dict] = None,
    query: Optional[str] = None,
    selected_categories: Optional[list] = None,
    exclude_ids: Optional[set] = None,
    top_k: int = 10,
    cb_weight: float = 0.5,
) -> list:
    """
    Blend content-based and LLM-semantic scores when both inputs are available.
    """
    cb_results: dict = {}
    llm_results: dict = {}

    if liked_ids or rated_items:
        for r in content_based_recommend(restaurants, liked_ids, rated_items, exclude_ids, top_k * 2):
            cb_results[r["id"]] = r

    if query or selected_categories:
        for r in llm_semantic_recommend(restaurants, query or "", selected_categories, exclude_ids, top_k * 2):
            llm_results[r["id"]] = r

    if not cb_results and not llm_results:
        # Fallback: return top-rated with basic metadata
        sorted_r = sorted(restaurants, key=lambda x: -x["rating"])[:top_k]
        for r in sorted_r:
            r["_score"] = r["rating"] / 5.0
            r["_method"] = "content-based"
            r["_explanation"] = f"Highly rated {r['cuisine']} restaurant ({r['rating']}★) in {r['location']}."
        return sorted_r

    all_ids = set(cb_results) | set(llm_results)
    blended = []
    for rid in all_ids:
        if exclude_ids and rid in exclude_ids:
            continue
        cb_score = cb_results[rid]["_score"] if rid in cb_results else 0.0
        llm_score = llm_results[rid]["_score"] if rid in llm_results else 0.0

        if cb_results and llm_results:
            final_score = cb_weight * cb_score + (1 - cb_weight) * llm_score
            method = "hybrid"
        elif cb_results:
            final_score = cb_score
            method = "content-based"
        else:
            final_score = llm_score
            method = "llm-semantic"

        base = cb_results.get(rid) or llm_results.get(rid)
        r = dict(base)
        r["_score"] = float(np.clip(final_score, 0.0, 1.0))
        r["_method"] = method
        if method == "hybrid":
            r["_explanation"] = (
                cb_results[rid]["_explanation"]
                if rid in cb_results
                else llm_results[rid]["_explanation"]
            )
        blended.append(r)

    blended.sort(key=lambda x: -x["_score"])
    return blended[:top_k]
