#!/usr/bin/env python3
"""
Build a balanced Spotify genre classification dataset.

Remaps ~114 subgenres in the Kaggle dataset to 10 main genres,
then augments underrepresented genres via the Spotify Search API.

Usage:
    # Remap only (no API needed):
    python build_dataset.py --remap-only

    # Remap + augment with Spotify API:
    export SPOTIPY_CLIENT_ID="your_client_id"
    export SPOTIPY_CLIENT_SECRET="your_client_secret"
    python build_dataset.py

    # Custom target count per genre:
    python build_dataset.py --target 15000
"""

import argparse
import os
import sys
import time
import math

import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

from main import GENRE_MAP

# ---------------------------------------------------------------------------
# Search queries per genre. Each query is searched via /v1/search.
# Spotify caps search at offset=1000, so each query yields up to ~1000 tracks.
# More queries per genre = more unique tracks we can collect.
# ---------------------------------------------------------------------------
SEARCH_QUERIES: dict[str, list[str]] = {
    "Hip-Hop / Rap": [
        "hip hop", "rap", "trap", "drill rap", "boom bap",
        "gangsta rap", "conscious rap", "grime", "crunk",
        "east coast hip hop", "west coast hip hop", "southern hip hop",
        "underground hip hop", "freestyle rap", "cloud rap",
        "mumble rap", "dirty south rap", "hip hop beats",
        "old school hip hop", "rap songs",
    ],
    "Jazz": [
        "jazz", "smooth jazz", "jazz fusion", "bebop", "swing jazz",
        "cool jazz", "free jazz", "latin jazz", "acid jazz",
        "contemporary jazz", "big band jazz", "jazz piano",
        "modal jazz", "post-bop", "hard bop", "jazz vocal",
        "jazz trumpet", "jazz saxophone", "jazz standards",
        "bossa nova jazz",
    ],
    "Country / Folk": [
        "country music", "folk music", "americana", "bluegrass",
        "alt country", "folk rock", "singer songwriter country",
        "traditional country", "country pop", "outlaw country",
        "country rock", "country songs", "folk songs",
        "acoustic folk", "country ballad",
    ],
    "Classical / Ambient": [
        "classical music", "ambient music", "orchestral",
        "symphony", "chamber music", "contemporary classical",
        "baroque music", "minimalist music", "neoclassical",
        "classical piano", "classical violin", "opera music",
        "ambient electronic", "meditation music", "classical symphony",
    ],
    "R&B / Soul": [
        "r&b", "soul music", "neo soul", "funk music", "motown",
        "contemporary r&b", "rhythm and blues", "quiet storm",
        "new jack swing", "gospel music", "r&b songs",
        "soul songs", "funk songs", "r&b slow jam",
        "classic soul",
    ],
    "Metal": [
        "heavy metal", "death metal", "black metal", "thrash metal",
        "power metal", "doom metal", "progressive metal", "nu metal",
        "metalcore", "deathcore", "symphonic metal",
        "metal songs", "speed metal", "groove metal",
        "folk metal",
    ],
    "Pop": [
        "pop music", "dance pop", "electro pop", "indie pop",
        "teen pop", "art pop", "bedroom pop", "pop songs",
        "synth pop", "k-pop", "j-pop", "pop hits",
    ],
    "Rock": [
        "rock music", "alternative rock", "classic rock", "punk rock",
        "progressive rock", "psychedelic rock", "garage rock",
        "rock songs", "indie rock", "hard rock", "rock hits",
        "grunge", "emo rock",
    ],
    "Electronic / Dance": [
        "electronic music", "EDM", "house music", "techno music",
        "trance music", "drum and bass", "dubstep music",
        "electronic dance", "deep house", "ambient electronic",
        "electro house", "future bass", "synthwave",
    ],
    "Latin": [
        "latin music", "reggaeton", "salsa music", "bachata",
        "cumbia", "latin pop", "latin trap", "merengue",
        "latin songs", "bossa nova", "tango music", "samba music",
        "latin hits", "musica latina",
    ],
}

AUDIO_FEATURE_COLS = [
    "danceability", "energy", "key", "loudness", "mode", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo",
    "time_signature",
]

INPUT_CSV = "spotify_kaggle_dataset.csv"
DEFAULT_OUTPUT_CSV = "spotify_remapped_balanced.csv"
PROGRESS_CSV = "spotify_augment_progress.csv"

# ---------------------------------------------------------------------------
# Spotify helpers
# ---------------------------------------------------------------------------

def init_spotify() -> spotipy.Spotify:
    """Authenticate with client-credentials flow and verify connectivity."""
    client_id = os.environ.get("SPOTIPY_CLIENT_ID")
    client_secret = os.environ.get("SPOTIPY_CLIENT_SECRET")
    if not client_id or not client_secret:
        print("Error: Spotify API credentials not found.")
        print("  export SPOTIPY_CLIENT_ID='your_client_id'")
        print("  export SPOTIPY_CLIENT_SECRET='your_client_secret'")
        print("Get credentials at https://developer.spotify.com/dashboard")
        sys.exit(1)

    sp = spotipy.Spotify(
        client_credentials_manager=SpotifyClientCredentials(
            client_id=client_id, client_secret=client_secret,
        ),
        requests_timeout=30,
        retries=3,
    )

    # Quick connectivity check with a lightweight call
    try:
        sp.search(q="test", type="track", limit=1)
        print("Spotify API connected (search OK).")
    except Exception as e:
        print(f"Error connecting to Spotify API: {e}")
        sys.exit(1)

    return sp


def check_audio_features_available(sp: spotipy.Spotify) -> bool:
    """Probe whether the audio-features endpoint works for this app."""
    try:
        result = sp.search(q="pop", type="track", limit=1)
        track_id = result["tracks"]["items"][0]["id"]
        feats = sp.audio_features([track_id])
        return feats and feats[0] is not None
    except Exception:
        return False


def fetch_audio_features(sp: spotipy.Spotify, track_ids: list[str]) -> dict:
    """Return {track_id: feature_dict} for a list of track IDs (batched)."""
    features: dict = {}
    for i in range(0, len(track_ids), 100):
        batch = track_ids[i : i + 100]
        try:
            results = sp.audio_features(batch)
            for f in results:
                if f is not None:
                    features[f["id"]] = f
        except Exception as e:
            print(f"    Warning: audio_features batch failed: {e}")
            time.sleep(5)
        time.sleep(0.05)
    return features


def fetch_via_search(
    sp: spotipy.Spotify,
    genre: str,
    count: int,
    seen_ids: set[str],
) -> list[dict]:
    """Collect up to *count* unique tracks through the search API."""
    queries = SEARCH_QUERIES.get(genre, [genre.lower()])
    tracks: list[dict] = []

    per_query = math.ceil(count / len(queries)) + 200

    for qi, query in enumerate(queries):
        if len(tracks) >= count:
            break
        query_hits = 0
        for offset in range(0, 990, 10):
            if len(tracks) >= count or query_hits >= per_query:
                break
            try:
                results = sp.search(q=query, type="track", limit=10, offset=offset)
                items = results.get("tracks", {}).get("items", [])
                if not items:
                    break
                for t in items:
                    if t["id"] not in seen_ids:
                        seen_ids.add(t["id"])
                        tracks.append(t)
                        query_hits += 1
            except Exception as e:
                print(f"    Warning: search error (q={query!r}, offset={offset}): {e}")
                time.sleep(3)
                break
            time.sleep(0.05)

        if (qi + 1) % 3 == 0 or len(tracks) >= count:
            print(f"    [{genre}] {len(tracks):>5d}/{count} tracks collected …")

    return tracks[:count]


def tracks_to_rows(
    sp: spotipy.Spotify,
    tracks: list[dict],
    genre: str,
    has_audio_features: bool,
) -> list[dict]:
    """Convert raw Spotify track objects into flat dicts matching our CSV schema."""
    if not tracks:
        return []

    features: dict = {}
    if has_audio_features:
        track_ids = [t["id"] for t in tracks]
        print(f"    Fetching audio features for {len(track_ids)} tracks …")
        features = fetch_audio_features(sp, track_ids)

    rows: list[dict] = []
    for t in tracks:
        feat = features.get(t["id"])
        if has_audio_features and feat is None:
            # Skip tracks where audio features couldn't be retrieved
            continue
        row = {
            "track_id": t["id"],
            "artists": ";".join(a["name"] for a in t["artists"]),
            "album_name": t["album"]["name"],
            "track_name": t["name"],
            "popularity": t["popularity"],
            "duration_ms": t["duration_ms"],
            "explicit": t["explicit"],
            "track_genre": genre,
        }
        if feat:
            row.update({col: feat[col] for col in AUDIO_FEATURE_COLS})
        else:
            row.update({col: None for col in AUDIO_FEATURE_COLS})
        rows.append(row)
    return rows

# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def load_and_remap() -> pd.DataFrame:
    print(f"Loading {INPUT_CSV} …")
    df = pd.read_csv(INPUT_CSV)
    if df.columns[0] in ("", "Unnamed: 0"):
        df = df.drop(columns=[df.columns[0]])
    df["track_genre"] = df["track_genre"].map(GENRE_MAP)
    print(f"Loaded and remapped {len(df)} tracks.")
    return df


def show_distribution(df: pd.DataFrame, title: str = "Genre distribution"):
    counts = df["track_genre"].value_counts()
    print(f"\n{title}:")
    for genre in counts.index:
        print(f"  {genre:<22s} {counts[genre]:>6d} tracks")
    print(f"  {'TOTAL':<22s} {len(df):>6d} tracks")


def compute_deficit(df: pd.DataFrame, target: int) -> dict[str, int]:
    counts = df["track_genre"].value_counts()
    return {
        genre: target - counts[genre]
        for genre in counts.index
        if counts[genre] < target
    }


def save_progress(rows: list[dict]):
    if rows:
        pd.DataFrame(rows).to_csv(PROGRESS_CSV, index=False)


def load_progress() -> list[dict]:
    if os.path.exists(PROGRESS_CSV):
        prev = pd.read_csv(PROGRESS_CSV)
        print(f"Resuming from progress file ({len(prev)} previously fetched rows)")
        return prev.to_dict("records")
    return []


def augment_dataset(
    sp: spotipy.Spotify,
    df: pd.DataFrame,
    target: int,
) -> list[dict]:
    deficit = compute_deficit(df, target)
    if not deficit:
        print("\nAll genres already meet the target – nothing to augment.")
        return []

    print(f"\nGenres to augment (target = {target} per genre):")
    for genre, needed in sorted(deficit.items(), key=lambda x: -x[1]):
        print(f"  {genre:<22s} needs {needed:>6d} more tracks")

    # Check audio features availability once up front
    has_audio_features = check_audio_features_available(sp)
    if has_audio_features:
        print("\nAudio-features endpoint is available.")
    else:
        print("\nWARNING: Audio-features endpoint is NOT available for this app.")
        print("New tracks will have NULL audio features. You can still train on")
        print("metadata columns (popularity, duration_ms, explicit), or backfill")
        print("audio features later with a different Spotify app.\n")

    existing_ids = set(df["track_id"].tolist())
    seen_ids = set(existing_ids)

    all_new_rows = load_progress()
    if all_new_rows:
        seen_ids.update(r["track_id"] for r in all_new_rows)
        prev_counts: dict[str, int] = {}
        for r in all_new_rows:
            prev_counts[r["track_genre"]] = prev_counts.get(r["track_genre"], 0) + 1
        for genre in list(deficit):
            deficit[genre] -= prev_counts.get(genre, 0)
            if deficit[genre] <= 0:
                del deficit[genre]

    for genre, needed in sorted(deficit.items(), key=lambda x: -x[1]):
        print(f"\n{'=' * 60}")
        print(f"  Augmenting '{genre}' – need {needed} tracks")
        print(f"{'=' * 60}")

        raw_tracks = fetch_via_search(sp, genre, needed, seen_ids)
        print(f"    Collected {len(raw_tracks)} raw tracks from search")

        rows = tracks_to_rows(sp, raw_tracks, genre, has_audio_features)
        all_new_rows.extend(rows)
        print(f"  -> {len(rows)} usable rows added for '{genre}'")
        save_progress(all_new_rows)

    return all_new_rows

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build a balanced Spotify genre classification dataset.",
    )
    parser.add_argument(
        "--remap-only", action="store_true",
        help="Only remap genres; skip Spotify API augmentation",
    )
    parser.add_argument(
        "--target", type=int, default=12_000,
        help="Target track count per genre (default: 12 000)",
    )
    parser.add_argument(
        "--output", type=str, default=DEFAULT_OUTPUT_CSV,
        help=f"Output CSV path (default: {DEFAULT_OUTPUT_CSV})",
    )
    args = parser.parse_args()

    # Step 1 – remap
    df = load_and_remap()
    show_distribution(df, "Distribution after remapping")

    if args.remap_only:
        df.to_csv(args.output, index=False)
        print(f"\nSaved remapped dataset → {args.output}")
        return

    # Step 2 – augment via Spotify API
    sp = init_spotify()
    new_rows = augment_dataset(sp, df, args.target)

    # Step 3 – combine and save
    if new_rows:
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)

    show_distribution(df, "Final distribution")
    df.to_csv(args.output, index=False)
    print(f"\nSaved balanced dataset → {args.output} ({len(df)} total tracks)")

    if os.path.exists(PROGRESS_CSV):
        os.remove(PROGRESS_CSV)
        print("Cleaned up progress file.")


if __name__ == "__main__":
    main()
