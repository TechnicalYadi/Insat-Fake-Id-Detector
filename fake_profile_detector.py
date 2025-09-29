"""
Fake Profile Detector (with Instagram username/URL lookup)
Single-file Streamlit app.
Save as fake_profile_detector.py and run:
    pip install streamlit scikit-learn pandas numpy instaloader pillow
    streamlit run fake_profile_detector.py
"""
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
import instaloader
from instaloader.exceptions import ProfileNotExistsException, PrivateProfileNotFollowedException, InstaloaderException
from urllib.parse import urlparse
from datetime import datetime
import io
import time
import random
import requests
from PIL import Image

# ---------- Config ----------
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
MAX_PUBLIC_LOOKUPS_DEFAULT = 2

# ---------- Synthetic dataset + model training (demo) ----------
def generate_synthetic_accounts(n=2000, seed=RANDOM_SEED):
    rng = np.random.RandomState(seed)
    followers = rng.lognormal(mean=4.0, sigma=1.8, size=n).astype(int)
    following = rng.lognormal(mean=3.6, sigma=1.7, size=n).astype(int)
    posts = rng.poisson(lam=50, size=n)
    account_age_days = rng.exponential(scale=600, size=n).astype(int)
    bio_length = rng.poisson(lam=30, size=n)
    has_profile_pic = rng.binomial(1, p=0.85, size=n)
    avg_likes = (followers * rng.uniform(0.01, 0.2, size=n)).astype(int)
    score = (
        0.4 * (np.maximum(0, following - followers) / (1 + following)) +
        0.25 * (1 - np.tanh(posts / 10)) +
        0.2 * (1 - np.tanh(account_age_days / 365)) +
        0.25 * (1 - has_profile_pic) +
        0.2 * (1 - np.tanh(avg_likes / (1 + followers)))
    )
    score = score + rng.normal(scale=0.15, size=n)
    score = (score - score.min()) / (score.max() - score.min())
    labels = (score > 0.55).astype(int)
    df = pd.DataFrame({
        "followers": followers,
        "following": following,
        "posts": posts,
        "account_age_days": account_age_days,
        "bio_length": bio_length,
        "has_profile_pic": has_profile_pic,
        "avg_likes": avg_likes,
        "is_fake": labels
    })
    return df

@st.cache_data(show_spinner=False)
def train_model(df):
    X = df[["followers", "following", "posts", "account_age_days", "bio_length", "has_profile_pic", "avg_likes"]]
    y = df["is_fake"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y)
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000))
    ])
    pipeline.fit(X_train, y_train)
    y_proba = pipeline.predict_proba(X_test)[:,1]
    y_pred = pipeline.predict(X_test)
    auc = roc_auc_score(y_test, y_proba)
    acc = accuracy_score(y_test, y_pred)
    coefs = pipeline.named_steps["clf"].coef_[0]
    feature_names = X.columns.tolist()
    feat_imp = pd.Series(coefs, index=feature_names).sort_values(key=abs, ascending=False)
    return pipeline, auc, acc, feat_imp

# ---------- Safer Instaloader fetch helper ----------
@st.cache_data(show_spinner=False)
def fetch_instagram_profile_safe(username, max_posts_for_likes=8):
    L = instaloader.Instaloader(download_pictures=False, download_videos=False, save_metadata=False, compress_json=False)
    try:
        profile = instaloader.Profile.from_username(L.context, username)
    except ProfileNotExistsException:
        raise ValueError("Profile does not exist.")
    except InstaloaderException as e:
        raise ValueError(f"Instaloader error: {e}")

    followers = int(profile.followers or 0)
    following = int(profile.followees or 0)
    posts = int(profile.mediacount or 0)
    bio_length = len(profile.biography or "")
    has_profile_pic = 1 if profile.profile_pic_url else 0

    account_age_days = None
    avg_likes = None
    accessible_posts_checked = False

    if not profile.is_private:
        accessible_posts_checked = True
        try:
            posts_iter = profile.get_posts()
            first_post = next(posts_iter, None)
            if first_post:
                account_age_days = (datetime.utcnow() - first_post.date).days
            else:
                account_age_days = 0
        except Exception:
            account_age_days = 0
        try:
            like_sum = 0
            count = 0
            for i, post in enumerate(profile.get_posts()):
                if i >= max_posts_for_likes:
                    break
                like_sum += getattr(post, "likes", 0) or 0
                count += 1
            avg_likes = int(like_sum / count) if count > 0 else 0
        except Exception:
            avg_likes = 0
    else:
        account_age_days = None
        avg_likes = None

    profile_pic_url = profile.profile_pic_url or ""
    biography = profile.biography or ""

    return {
        "username": username,
        "is_private": bool(profile.is_private),
        "followers": followers,
        "following": following,
        "posts": posts,
        "account_age_days": int(account_age_days) if account_age_days is not None else None,
        "bio_length": bio_length,
        "has_profile_pic": has_profile_pic,
        "avg_likes": int(avg_likes) if avg_likes is not None else None,
        "accessible_posts_checked": accessible_posts_checked,
        "profile_pic_url": profile_pic_url,
        "biography": biography
    }

# ---------- Predict helpers ----------
def predict_account(model, features_df):
    proba = model.predict_proba(features_df)[:,1]
    pred = (proba > 0.5).astype(int)
    return pred, proba

def explain_prediction(feat_imp):
    df = pd.DataFrame({
        "feature": feat_imp.index,
        "coefficient": feat_imp.values
    })
    df["meaning"] = df["coefficient"].apply(lambda x: "Positive -> increases fake probability" if x>0 else "Negative -> reduces fake probability")
    return df

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Fake Profile Detector", layout="wide")
st.title("Fake Profile Detector — Username lookup (Instagram public info only)")

with st.spinner("Preparing local demo model..."):
    df_synth = generate_synthetic_accounts(2000)
    model, auc, acc, feat_imp = train_model(df_synth)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Model snapshot")
    st.markdown(f"- ROC AUC: **{auc:.3f}**  \n- Accuracy (test): **{acc:.3f}**")
    st.table(feat_imp.round(3).to_frame("coef"))

with col2:
    st.subheader("Lookup by username / URL (public info only)")
    user_input = st.text_input("Enter one or multiple usernames / URLs (comma-separated).", value="", placeholder="user1, user2")
    max_public_lookups = st.number_input("Max allowed public fetches", min_value=1, max_value=10, value=MAX_PUBLIC_LOOKUPS_DEFAULT, step=1)
    lookup_btn = st.button("Start lookup(s)")

def extract_username(text):
    text = text.strip()
    if not text:
        return ""
    if text.startswith("http"):
        try:
            p = urlparse(text)
            path = p.path.strip("/")
            username = path.split("/")[0]
            return username
        except Exception:
            return text
    if text.startswith("@"): return text[1:]
    return text

if "last_results" not in st.session_state:
    st.session_state["last_results"] = []

if lookup_btn:
    raw_usernames = [u.strip() for u in user_input.split(",") if u.strip()]
    usernames = [extract_username(u) for u in raw_usernames]
    results = []
    public_fetches_done = 0

    for uname in usernames:
        if public_fetches_done >= max_public_lookups:
            st.info("Reached max public fetches.")
            break
        st.info(f"Checking @{uname} ...")
        time.sleep(random.uniform(1.0, 2.0))
        try:
            features = fetch_instagram_profile_safe(uname)
            bio_text = features.get("biography") or "No bio set"
            img_url = features.get("profile_pic_url")

            # always show card
            st.markdown(f"### Profile: @{uname}")
            if img_url:
                try:
                    resp = requests.get(img_url, timeout=10)
                    image = Image.open(io.BytesIO(resp.content))
                    st.image(image, caption=f"@{uname}", width=150)
                except Exception:
                    st.write("(Couldn't load profile picture)")
            else:
                st.write("(No profile picture available)")

            st.markdown(f"**📝 Bio:** {bio_text}")

            if features["is_private"]:
                st.warning("Private profile — limited info only")
            else:
                public_fetches_done += 1
                feat_for_model = {
                    "followers": features["followers"],
                    "following": features["following"],
                    "posts": features["posts"],
                    "account_age_days": features["account_age_days"] or 0,
                    "bio_length": features["bio_length"],
                    "has_profile_pic": features["has_profile_pic"],
                    "avg_likes": features["avg_likes"] or 0
                }
                feat_df = pd.DataFrame([feat_for_model])
                pred, proba = predict_account(model, feat_df)
                verdict = "FAKE" if int(pred[0]) else "GENUINE"
                prob = float(proba[0])
                st.success(f"Prediction: {verdict} ({prob*100:.1f}% fake probability)")

            results.append(features)
        except Exception as e:
            st.error(f"Error fetching @{uname}: {e}")

    st.session_state["last_results"] = results

with st.sidebar:
    st.header("Last run (session)")
    if st.session_state.get("last_results"):
        for r in st.session_state["last_results"]:
            st.markdown(f"**@{r['username']}** — {'PRIVATE' if r['is_private'] else 'PUBLIC'}")
            st.write({k: v for k, v in r.items() if k not in ["username", "profile_pic_url", "biography"]})
            st.markdown("---")
    else:
        st.write("No lookups yet.")

st.markdown("---")
st.table(explain_prediction(feat_imp))
