# 🍽️ AI-Powered Restaurant Recommendation Engine

> An end-to-end intelligent recommendation system that combines **sentence-transformer embeddings** and **LLM-style semantic search** to surface personalised restaurant suggestions — each with a specific, human-readable explanation of *why* it was recommended.

**[➡️ Live Demo on Streamlit Cloud](#)** *(add your URL after deployment)*

---

## Table of Contents

1. [Introduction](#introduction)
2. [Project Structure](#project-structure)
3. [Dataset](#dataset)
4. [Recommendation Methods](#recommendation-methods)
5. [Explanation Generation](#explanation-generation)
6. [Evaluation Metrics & Results](#evaluation-metrics--results)
7. [App Interface & How to Use It](#app-interface--how-to-use-it)
8. [Installation & Local Setup](#installation--local-setup)
9. [Deployment to Streamlit Cloud](#deployment-to-streamlit-cloud)
10. [Design Decisions & Limitations](#design-decisions--limitations)
11. [References](#references)

---

## Introduction

Modern restaurant discovery tools like Yelp or Google Maps rely heavily on keyword search and aggregate star ratings. They tell you *what* is popular, but rarely *why* a specific place is right for *you*. This project takes a different approach: instead of filtering by category, it learns from your interactions and understands your natural language preferences to surface genuinely personalised recommendations.

The core challenge in recommendation systems is the tension between **accuracy** (showing things you'll definitely like) and **diversity** (helping you discover something new). This project addresses that by implementing two complementary algorithms:

- **Content-Based Filtering** — learns from your taste history using semantic embeddings. High precision, ideal for returning users.
- **LLM-Semantic Search** — understands natural language queries like *"cozy Italian spot for a date night under $40"* and maps them to restaurants without any prior interaction history. Ideal for new users or exploratory sessions.

Both approaches produce item-level explanations that cite the actual attributes that drove each recommendation, avoiding the black-box problem that plagues most ML-based recommendation systems.

---

## Project Structure

```
restaurant_recommender/
│
├── app.py                  # Main Streamlit application (5-tab UI)
├── recommender.py          # Core algorithm implementations
│   │                         ├─ Content-Based Filtering
│   │                         ├─ LLM-Semantic Search
│   │                         ├─ Hybrid Blending
│   │                         └─ Explanation generators
│
├── evaluation.py           # All evaluation metric implementations
│   │                         ├─ Precision@K, Recall@K, NDCG@K
│   │                         ├─ Intra-List Diversity (ILD)
│   │                         ├─ Category Coverage
│   │                         └─ Simulated user evaluation harness
│
├── generate_data.py        # Synthetic dataset generator (run once)
│
├── data/
│   └── restaurants.json    # 300-restaurant dataset (pre-generated)
│
├── .streamlit/
│   └── config.toml         # Theme and server configuration
│
├── requirements.txt        # Python dependencies
├── evaluation_report.md    # Full 2-page evaluation report with results
└── README.md               # This file
```

---

## Dataset

### Overview

The dataset contains **300 restaurants** spanning **20 global cuisines**, generated programmatically using `generate_data.py`. Each record is designed to be realistic and rich enough to support meaningful semantic similarity comparisons.

### Cuisines Covered

Italian · Chinese · Japanese · Mexican · Indian · Thai · French · American · Mediterranean · Korean · Vietnamese · Greek · Spanish · Middle Eastern · Ethiopian · Brazilian · Peruvian · Lebanese · Turkish · Caribbean

Each cuisine has exactly **15 restaurants** to ensure balanced representation.

### Data Schema

Each restaurant record contains the following fields:

| Field | Type | Description | Example |
|---|---|---|---|
| `id` | int | Unique identifier | `42` |
| `name` | str | Restaurant name | `"Sakura Kitchen"` |
| `description` | str | 2–3 sentence narrative | `"A serene Japanese restaurant..."` |
| `cuisine` | str | Primary cuisine type | `"Japanese"` |
| `price_range` | str | Budget indicator | `"$$"` |
| `rating` | float | Star rating (1.0–5.0) | `4.3` |
| `review_count` | int | Number of reviews | `847` |
| `ambiance` | str | Atmosphere descriptor | `"Romantic"` |
| `location` | str | Neighbourhood | `"Downtown"` |
| `dietary_options` | list | Dietary accommodations | `["Vegan options", "Gluten-free"]` |
| `popular_dishes` | list | 4–6 signature dishes | `["Omakase Sushi", "Tonkotsu Ramen"]` |
| `features` | list | Operational attributes | `["Outdoor seating", "Reservations"]` |
| `tags` | list | Searchable flat tags | `["japanese", "romantic", "downtown"]` |

### Data Generation

Rather than scraping real data (which raises IP and privacy concerns), the dataset is generated with controlled diversity:

- **Descriptions** are drawn from 5 carefully written, cuisine-authentic paragraph templates per cuisine, capturing real culinary traditions, cooking techniques, and regional specificity.
- **Dishes** are sampled from a curated list of 15 authentic dishes per cuisine.
- **Ratings** are drawn from a triangular distribution (min 2.5, mode 4.2, max 5.0), reflecting the real-world skew toward higher ratings on discovery platforms.
- **Price distribution** is weighted realistically: 20% budget ($), 40% moderate ($$), 30% upscale ($$$), 10% fine dining ($$$$).
- **Ambiance and location** are sampled uniformly from realistic categorical lists.

To regenerate or extend the dataset, run:
```bash
python generate_data.py
```

---

## Recommendation Methods

### Method 1 — Content-Based Filtering

**Core idea:** Represent each restaurant as a vector in semantic space. Build a user preference vector from their interaction history, then find the restaurants closest to that vector.

#### Step 1: Text Representation

Each restaurant is serialised into a rich, structured text document:

```
{name} | {description} | Cuisine: {cuisine} | Ambiance: {ambiance} |
Price range: {price_range} | Location: {location} |
Popular dishes: {dish1}, {dish2}, ... |
Dietary options: {dietary} | Features: {features}
```

This format ensures all semantically meaningful fields contribute to the embedding — the description provides nuance, while the structured fields provide precision.

#### Step 2: Embedding

Text documents are encoded using [`all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), a distilled, fast sentence-transformer model with 22M parameters that maps variable-length text into a fixed **384-dimensional vector space**. All embeddings are L2-normalised, so cosine similarity reduces to a simple dot product.

```
embedding(restaurant) ∈ ℝ³⁸⁴,  ‖embedding‖ = 1
```

Embeddings are computed once at startup and cached in memory for the session.

#### Step 3: User Preference Modelling

When a user likes or rates restaurants, the system builds a **weighted preference vector**:

```
preference_vec = Σ(wᵢ × embedding_i) / Σ(wᵢ)
```

Where weights `wᵢ` are derived from star ratings:
- **Liked item (no rating):** weight = 1.0
- **Star rating r:** weight = max(0.1, (r − 2.5) / 2.5), so a 5★ item contributes +1.0 and a 1★ item contributes a small negative pull

The preference vector is then L2-normalised so it stays comparable to restaurant embeddings.

#### Step 4: Ranking

Cosine similarities between the preference vector and all restaurant embeddings are computed in one batched matrix operation (`O(n·d)`). Already-seen restaurants are excluded. The top-K results are returned.

---

### Method 2 — LLM-Semantic Search

**Core idea:** Embed the user's natural language query in the same semantic space as restaurants, then retrieve the closest matches — augmented by structured attribute parsing for higher precision.

This approach is inspired by modern **dense retrieval** systems (e.g., DPR, FAISS-based retrieval) and works with zero interaction history, making it ideal for cold-start scenarios.

#### Step 1: Structured Query Parsing

A keyword-map parser extracts hard preferences from the query text:

| Preference Type | Example Query Fragment | Extracted Signal |
|---|---|---|
| Cuisine | `"italian"`, `"pad thai"`, `"dim sum"` | Cuisine = Italian / Thai / Chinese |
| Price | `"cheap"`, `"under $30"`, `"splurge"` | Price range = $ / $$ / $$$ |
| Ambiance | `"romantic"`, `"family-friendly"`, `"cozy"` | Ambiance type |
| Dietary | `"vegan"`, `"gluten-free"`, `"halal"` | Dietary filter |
| Features | `"outdoor seating"`, `"delivery"`, `"parking"` | Operational features |
| Rating | `"above 4 stars"`, `"at least 4.5"` | Minimum rating threshold |

This covers **20 cuisines × 5 price signals × 8 ambiance types × 6 dietary types × 10 features**, meaning the vast majority of natural language restaurant queries are handled.

#### Step 2: Semantic Embedding

The raw query string is embedded using the same `all-MiniLM-L6-v2` model, producing a 384-dimensional vector. This captures intent beyond keyword matching — a query about *"soul food"* will still surface Southern American restaurants even if the word "American" never appears.

#### Step 3: Composite Scoring

Each restaurant receives a composite score:

```
score(r) = cosine_sim(query_vec, embedding(r))
         + Σ(attribute_boosts)
```

Attribute boosts:

| Match Type | Boost |
|---|---|
| Cuisine match | +0.15 |
| Price range match | +0.08 |
| Ambiance match | +0.06 |
| Each dietary match | +0.05 |
| Each feature match | +0.03 |
| Minimum rating met | +0.04 |

The boosts are calibrated so semantic similarity still dominates (typically 0.4–0.8 range) while structured matches provide meaningful re-ranking within semantically similar clusters.

#### Step 4: Ranking

Restaurants are sorted by composite score and the top-K are returned with query-specific explanations.

---

### Method 3 — Hybrid Blending

When both liked items and a query are available, the system blends both scores:

```
hybrid_score = α × content_score + (1−α) × llm_score
```

The default α = 0.5 weights both approaches equally, but this is configurable. Hybrid mode is particularly effective mid-session when a user has built some interaction history but also wants to express a new directional preference via natural language.

---

## Explanation Generation

A core design goal of this project is that **every recommendation must be explainable** — and the explanation must reference real, specific attributes rather than generic phrases like "based on your preferences."

### Content-Based Explanations

The system identifies the reference restaurant (highest-rated liked item) and generates a sentence that cites shared attributes:

> *"Recommended because: Same **Italian** cuisine as 'La Golden Bistro' which you enjoyed; matching **moderately priced** price point ($$); **Casual** atmosphere like restaurants you prefer; known for **Margherita Pizza**."*

### Semantic Explanations

The system cross-references the parsed query against the restaurant's actual attributes:

> *"Matches your query because: **Thai** cuisine matches your request; price range **$$** (moderately priced); **Outdoor seating** as specified; offers **Vegetarian-friendly** dining; features **Pad Thai** and other dishes relevant to your search."*

The explanation generator caps output at 4–5 cited attributes to keep explanations readable without becoming overwhelming.

---

## Evaluation Metrics & Results

### Metric Definitions

| Metric | Formula | What it measures |
|---|---|---|
| **Precision@K** | hits@K ÷ K | Of the top-K recommendations, how many were relevant? |
| **Recall@K** | hits@K ÷ \|relevant\| | Of all relevant items, how many did we surface? |
| **NDCG@K** | DCG@K ÷ IDCG@K | Are relevant items ranked near the top? (position-sensitive) |
| **Intra-List Diversity** | avg(1 − cosine\_sim) across pairs | How different are the recommendations from each other? |
| **Category Coverage** | \|unique cuisines\| ÷ 20 | What fraction of cuisine categories appear in the list? |
| **Avg Recommended Rating** | mean(rating) | Average quality of recommended items |

### Simulation Setup

Ground-truth evaluation used **50 synthetic user profiles**, each with:
- A randomly assigned preferred cuisine
- 4 "liked" restaurants drawn from that cuisine (interaction history)
- Ground truth = all remaining same-cuisine restaurants not yet seen

Both algorithms were evaluated at K=10.

### Results

| Metric | Content-Based | LLM-Semantic |
|---|---|---|
| **Precision@10** | **0.848** | 0.700 |
| **Recall@10** | **0.771** | 0.636 |
| **NDCG@10** | **0.891** | 0.801 |
| **Category Coverage** | 11.9% | **19.3%** |
| **Avg Recommended Rating** | 3.92★ | **4.04★** |

**Key takeaway:** Content-Based filtering wins on all accuracy metrics, which makes sense — when you know someone loves Italian food and a cozy atmosphere, finding more Italian restaurants with a cozy atmosphere is easy and precise. LLM-Semantic search wins on diversity (19.3% vs 11.9% category coverage) and average quality, because it blends cuisines and surfaces highly-described restaurants even outside the user's habitual preferences.

The **Hybrid mode** is designed to get the best of both: high accuracy from content-based matching plus the exploratory diversity of semantic search.

---

## App Interface & How to Use It

The app has five tabs accessible from the top navigation:

### 🔍 Discover Tab — Your Main Workspace

This is where you get recommendations. Choose your approach from the sidebar:

**LLM-Semantic (Query):** Type any natural language description of what you want. Examples that work well:
```
"cozy Italian with outdoor seating, under $30"
"spicy Asian food, vegan-friendly, good for groups"
"romantic fine dining for a special occasion"
"quick cheap lunch near downtown, takeout available"
"vegetarian Ethiopian, above 4 stars"
```

**Content-Based (Liked items):** First rate a restaurant using the dropdown on the Discover tab, or like items in the Browse tab. Then switch to Content-Based and hit "Get Recommendations" — the system builds your taste profile from your history.

**Hybrid (Both):** Works best mid-session when you have some liked items AND want to express a new preference via text.

Hit **🚀 Get Recommendations** to see your personalised cards. Each card shows:
- Restaurant name, cuisine, rating, price, location, ambiance
- Top popular dishes
- Dietary and feature tags
- The **💡 explanation** of why it was recommended
- A match score (higher = stronger match)
- 👍 / 👎 feedback buttons and a ⭐ rating slider

Your feedback immediately feeds back into the content-based algorithm for future recommendations.

### 🗂️ Browse Tab — Explore the Full Dataset

Filter all 300 restaurants by cuisine, price range, and ambiance. Sort by rating, name, or price. Like items directly from the browse table to build your taste profile.

### ❤️ My List Tab

See all the restaurants you've liked or rated. Remove items you no longer want in your profile. Review your rating history in a table.

### 📊 Metrics Tab

Click **▶️ Run Evaluation** to simulate the full benchmark across 50 users and see live Precision@K, NDCG@K, Intra-List Diversity, and Category Coverage results for both algorithms, with a comparison bar chart.

At the bottom of this tab, your **live session metrics** update in real-time as you give feedback — including your personal like ratio and the diversity of recommendations you received.

### ℹ️ About Tab

Architecture overview, dataset statistics with distribution charts, and tech stack information.

---

## Installation & Local Setup

### Requirements

- Python 3.10+
- ~500MB disk space (for the sentence-transformer model download on first run)

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/your-username/restaurant-recommender.git
cd restaurant-recommender

# 2. (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. The dataset is already included. To regenerate it:
python generate_data.py

# 5. Launch the app
streamlit run app.py
```

On **first launch**, the `all-MiniLM-L6-v2` model (~80MB) is automatically downloaded from HuggingFace and cached. All subsequent launches are instant.

The app will open at `http://localhost:8501`.

### Dependencies

```
streamlit>=1.35.0          # Web UI framework
sentence-transformers>=3.0  # Embedding model
numpy>=1.26.0              # Vector operations
pandas>=2.1.0              # Data handling and tables
torch>=2.1.0               # PyTorch backend for sentence-transformers
transformers>=4.40.0        # HuggingFace model utilities
scikit-learn>=1.4.0        # Cosine similarity utilities
```

---

## Deployment to Streamlit Cloud

Streamlit Cloud offers **free hosting** for public GitHub repositories and handles all dependency installation automatically.

1. Push your project to a **public GitHub repository**
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
3. Click **"New app"**
4. Select your repository, branch (`main`), and set the main file path to `app.py`
5. Click **Deploy**

Streamlit Cloud will install `requirements.txt`, download the model, and give you a shareable public URL (e.g. `https://your-app.streamlit.app`). Typical cold-start time is 2–3 minutes; subsequent loads are fast due to caching.

> **Tip:** If the app sleeps after inactivity (free tier behaviour), the first visit after sleep may take ~30 seconds to wake up — this is normal.

---

## Design Decisions & Limitations

### Why `all-MiniLM-L6-v2`?

This model strikes the best balance for this use case: it produces high-quality semantic embeddings (competitive with much larger models on semantic search benchmarks), runs comfortably on CPU with no GPU required, downloads in seconds, and fits within Streamlit Cloud's free tier memory limits. Larger models like `all-mpnet-base-v2` or OpenAI `text-embedding-3-small` would improve embedding quality but add latency and cost.

### Why not Collaborative Filtering?

Collaborative filtering requires a matrix of real user-item interactions. With a synthetic dataset and no persistent user database, there is no cross-user signal to leverage. The architecture is designed to be extended with collaborative filtering once real interaction logs are available — the evaluation harness already supports plugging in new recommender functions.

### Why synthetic data?

Scraping real restaurant data from Yelp or Google raises legal (ToS), ethical (business privacy), and technical (anti-scraping) barriers. Synthetic data allows full control over the distribution, diversity, and schema while avoiding these issues entirely. The descriptions and dish lists were crafted to be realistic enough for meaningful semantic similarity.

### Known Limitations

- The keyword parser for query understanding is rule-based. It may miss unusual phrasings or slang (e.g. "something bougie" won't parse as fine dining). A fine-tuned NER model would improve this.
- With 300 restaurants, diversity metrics are naturally constrained. A larger, real-world dataset would show more differentiation between the algorithms.
- The app does not persist user sessions across browser refreshes — liked items and ratings reset on page reload. A database backend (SQLite or Firebase) would fix this.
- Evaluation is simulated. Real-world CTR and satisfaction data would give a more accurate picture of recommendation quality.

---

## References

1. Reimers, N. & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. *Proceedings of EMNLP 2019*. https://arxiv.org/abs/1908.10084

2. Lops, P., de Gemmis, M. & Semeraro, G. (2011). Content-Based Recommender Systems: State of the Art and Trends. In *Recommender Systems Handbook*. Springer.

3. Järvelin, K. & Kekäläinen, J. (2002). Cumulated Gain-Based Evaluation of IR Techniques. *ACM Transactions on Information Systems, 20*(4), 422–446.

4. Hurley, N. & Zhang, M. (2011). Novelty and Diversity in Top-N Recommendation — Analysis and Evaluation. *ACM Transactions on Internet Technology, 10*(4).

5. Koren, Y., Bell, R. & Volinsky, C. (2009). Matrix Factorization Techniques for Recommender Systems. *IEEE Computer, 42*(8), 30–37.

6. Johnson, J., Douze, M. & Jégou, H. (2019). Billion-scale similarity search with GPUs. *IEEE Transactions on Big Data*. (Inspiration for dense retrieval approach.)

---

*Built as part of an AI Systems course assignment on recommendation engines.*
