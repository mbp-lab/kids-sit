import pandas as pd
import numpy as np

def find_series(df, behavior_col):

    filtered = df.loc[df[behavior_col] != 0]

    # If no annotations, return empty DataFrame
    if filtered.empty:
        return pd.DataFrame(columns=['series_id', 'frame', behavior_col])

    # Mark behavior annotation (fill missing with 'NaN')
    annot = filtered[behavior_col].fillna('NaN')

    # check series (behavior change OR non-consecutive frame)
    change = (annot != annot.shift()) | (filtered['frame'] != filtered['frame'].shift() + 1)

    # Assign series_id
    filtered = filtered.assign(series_id=change.cumsum())

    # Group consecutive frames
    series = (
        filtered.groupby('series_id')
        .agg({
            'frame': list,
            behavior_col: 'first'
        })
        .reset_index()
    )

    return series

def compute_series_agreement(df_group, behavior, annotator_1=None, annotator_2=None):
    annotator_1 = f"{behavior}_FD" if annotator_1 is None else annotator_1 # annotator 1 column
    annotator_2 = f"{behavior}_VP" if annotator_2 is None else annotator_2 # annotator 2 column

    series_1 = find_series(df_group, annotator_1)
    series_2 = find_series(df_group, annotator_2)
    # If both are empty, no annotations at all
    if series_1.empty and series_2.empty:
        return pd.Series({
            'behavior': behavior,
            'annotator_1': annotator_1,
            'annotator_2': annotator_2,
            'agreed_series_count': 0,
            'series_1_count': 0,
            'series_2_count': 0,
            'agreement_score': np.nan
        })
    # Track which annotator 1 series are agreed
    agreed_series_1_ids = set()

    # Also track which annotator 2 series have been counted as agreed
    agreed_series_2_ids = set()

    # Compare series_1 against series_2
    for idx1, row1 in series_1.iterrows():
        Frames_1 = set(row1['frame'])

        # Look for overlap with any series from annotator 2 that hasn't been already matched
        for idx2, row2 in series_2.iterrows():
            if idx2 in agreed_series_2_ids:
                continue  # Already matched, skip it

            Frames_2 = set(row2['frame'])
            overlap = Frames_1 & Frames_2

            if len(overlap) >= 2:
                # Found an agreement, count both series as matched
                agreed_series_1_ids.add(idx1)
                agreed_series_2_ids.add(idx2)

                break  # Only count one match for this series_1 
    agreed_series_count = len(agreed_series_2_ids) #########agreed_series_1_ids

    # Agreement score calculation
    total_series = len(series_1) + len(series_2) ###### remove max
    agreement_score = (2 * agreed_series_count) / total_series if total_series > 0 else np.nan

    return pd.Series({
        'behavior': behavior,
        'annotator_1': annotator_1,
        'annotator_2': annotator_2,
        'agreed_series_count': agreed_series_count,
        'series_1_count': len(series_1),
        'series_2_count': len(series_2),
        'agreement_score': agreement_score
    })

def compute_series_agreement_3_a_plus_2b(
    df_group,
    behavior,
    annotator_1=None,
    annotator_2=None,
    annotator_3=None,
    min_overlap_frames=2
):
    """
    3-annotator analogue of 2-annotator greedy one-to-one series matching.

    We build:
      - E3: triple events (one series from each annotator) where each pair overlaps >= min_overlap_frames
      - E2: double events (one series from two annotators) where overlap >= min_overlap_frames
            using only series not already used in an E3 or another E2

    Score:
      agreement_score = (3*E3 + 2*E2) / (N1 + N2 + N3)
    """

    a1 = f"{behavior}_FD" if annotator_1 is None else annotator_1
    a2 = f"{behavior}_VP" if annotator_2 is None else annotator_2
    a3 = f"{behavior}_LK" if annotator_3 is None else annotator_3

    s1 = find_series(df_group, a1)
    s2 = find_series(df_group, a2)
    s3 = find_series(df_group, a3)

    N1, N2, N3 = len(s1), len(s2), len(s3)
    total_series = N1 + N2 + N3

    if total_series == 0:
        return pd.Series({
            "behavior": behavior,
            "annotator_1": a1,
            "annotator_2": a2,
            "annotator_3": a3,
            "E2": 0,
            "E3": 0,
            "series_1_count": 0,
            "series_2_count": 0,
            "series_3_count": 0,
            "agreement_score": np.nan
        })

    # precompute frame sets (same as your 2-annotator logic)
    frames1 = {i: set(r["frame"]) for i, r in s1.iterrows()}
    frames2 = {j: set(r["frame"]) for j, r in s2.iterrows()}
    frames3 = {k: set(r["frame"]) for k, r in s3.iterrows()}

    def ov_ok(A, B) -> bool:
        return len(A & B) >= min_overlap_frames

    # --- STEP 1: build triple agreements (E3) ---
    triple_candidates = []
    for i, A in frames1.items():
        for j, B in frames2.items():
            for k, C in frames3.items():
                if len(A & B & C) >= min_overlap_frames:
                    # weight: total overlap across the 3 pairs (for greedy strongest-first)
                    w = len(A & B & C) #+ len(A & C) + len(B & C)
                    triple_candidates.append((w, i, j, k))

    triple_candidates.sort(reverse=True)  # strongest first

    used1, used2, used3 = set(), set(), set()
    E3 = 0
    triples = []  # optional: store which triples were formed

    for w, i, j, k in triple_candidates:
        if i in used1 or j in used2 or k in used3:
            continue
        used1.add(i); used2.add(j); used3.add(k)
        E3 += 1
        triples.append((i, j, k))

    # --- STEP 2: build pair agreements (E2) using remaining series only ---
    pair_candidates = []

    # 1-2
    for i, A in frames1.items():
        if i in used1: 
            continue
        for j, B in frames2.items():
            if j in used2:
                continue
            w = len(A & B)
            if w >= min_overlap_frames:
                pair_candidates.append((w, "12", i, j))

    # 1-3
    for i, A in frames1.items():
        if i in used1:
            continue
        for k, C in frames3.items():
            if k in used3:
                continue
            w = len(A & C)
            if w >= min_overlap_frames:
                pair_candidates.append((w, "13", i, k))

    # 2-3
    for j, B in frames2.items():
        if j in used2:
            continue
        for k, C in frames3.items():
            if k in used3:
                continue
            w = len(B & C)
            if w >= min_overlap_frames:
                pair_candidates.append((w, "23", j, k))

    pair_candidates.sort(reverse=True)  # strongest overlaps first

    E2 = 0
    pairs = []  # optional

    for w, ptype, a, b in pair_candidates:
        if ptype == "12":
            i, j = a, b
            if i in used1 or j in used2:
                continue
            used1.add(i); used2.add(j)
            E2 += 1
            pairs.append((ptype, i, j))

        elif ptype == "13":
            i, k = a, b
            if i in used1 or k in used3:
                continue
            used1.add(i); used3.add(k)
            E2 += 1
            pairs.append((ptype, i, k))

        else:  # "23"
            j, k = a, b
            if j in used2 or k in used3:
                continue
            used2.add(j); used3.add(k)
            E2 += 1
            pairs.append((ptype, j, k))

    # --- STEP 3: score = (3*E3 + 2*E2) / total_series ---
    agreement_score = (3 * E3 + 2 * E2) / total_series if total_series > 0 else np.nan

    return pd.Series({
        "behavior": behavior,
        "annotator_1": a1,
        "annotator_2": a2,
        "annotator_3": a3,
        "E2": E2,
        "E3": E3,
        "series_1_count": N1,
        "series_2_count": N2,
        "series_3_count": N3,
        "agreement_score": agreement_score,
        # Optional debug outputs:
        # "triples": triples,
        # "pairs": pairs,
    })
