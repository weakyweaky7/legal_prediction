import argparse
import json
import re
from pathlib import Path

import pandas as pd

from config import ROOT_DIR, MODELS_DIR


# ============================================================
# Paths
# ============================================================

LEGAL_NORMS_PATH = ROOT_DIR / "legal_dict" / "legal_norms.csv"
DEFAULT_LIME_PATH = MODELS_DIR / "lime_explanations" / "latest_lime_explanation.json"

LEGAL_OUTPUT_DIR = MODELS_DIR / "legal_layer"
LEGAL_OUTPUT_PATH = LEGAL_OUTPUT_DIR / "latest_legal_explanation.json"


# ============================================================
# Filtering settings
# ============================================================

STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then",
    "in", "on", "at", "by", "for", "from", "to", "of", "with",
    "as", "is", "are", "was", "were", "be", "been", "being",
    "this", "that", "these", "those", "it", "its", "itself",
    "there", "here", "any", "all", "some", "such",
    "no", "not", "so", "very", "also", "therefore", "particularly",
    "when", "where", "which", "who", "whom", "whose",
    "into", "over", "under", "above", "below", "before", "after",
    "had", "has", "have", "having", "do", "does", "did",
    "can", "could", "shall", "should", "would", "may", "might",
}

MASK_ARTIFACTS = {
    "outcome", "word", "outcomeword", "outcome_word",
    "mask", "masked", "unk", "cls", "sep", "pad"
}


# ============================================================
# Text helpers
# ============================================================

def normalize_word(word):
    word = str(word).lower()
    word = word.replace("##", "")
    word = re.sub(r"[^a-z0-9_ -]", "", word)
    word = word.strip()
    word = re.sub(r"\s+", " ", word)
    return word


def normalize_phrase(text):
    text = str(text).lower()
    text = text.replace("##", "")
    text = re.sub(r"[^a-z0-9_ -]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def is_useful_word(word):
    word = normalize_word(word)

    if not word:
        return False

    if len(word) < 3:
        return False

    if word in STOPWORDS:
        return False

    if word in MASK_ARTIFACTS:
        return False

    if word.replace(".", "").isdigit():
        return False

    if not re.search(r"[a-zA-Z]", word):
        return False

    return True


def filter_lime_items(lime_items):
    """
    Takes LIME words and removes:
    - stopwords
    - mask artifacts like outcome/word
    - very short tokens
    - numeric tokens

    Also merges repeated words.
    """
    merged = {}

    for item in lime_items:
        raw_word = item.get("word", "")
        weight = float(item.get("weight", 0.0))

        word = normalize_word(raw_word)

        if not is_useful_word(word):
            continue

        if word not in merged:
            merged[word] = {
                "word": word,
                "weight": weight,
                "absolute_weight": abs(weight)
            }
        else:
            merged[word]["weight"] += weight
            merged[word]["absolute_weight"] = abs(merged[word]["weight"])

    filtered = list(merged.values())
    filtered = sorted(filtered, key=lambda x: abs(x["weight"]), reverse=True)

    return filtered


def split_keywords(keyword_string):
    keywords = []

    for part in str(keyword_string).split(";"):
        keyword = normalize_phrase(part)

        if keyword:
            keywords.append(keyword)

    return keywords


def extract_keyword_tokens(keyword):
    tokens = re.findall(r"[a-zA-Z0-9_]+", normalize_phrase(keyword))
    tokens = [normalize_word(token) for token in tokens]
    tokens = [token for token in tokens if is_useful_word(token)]
    return tokens


# ============================================================
# Legal pattern matching
# ============================================================

def score_pattern(pattern_row, important_words, chunk_text):
    """
    Scores one legal pattern using:
    1. Phrase matches inside the influential chunk.
    2. Keyword overlap with filtered LIME words.
    """
    chunk_text_norm = normalize_phrase(chunk_text)

    important_map = {
        item["word"]: float(item["weight"])
        for item in important_words
    }

    important_set = set(important_map.keys())

    keywords = split_keywords(pattern_row["keywords"])

    matched_keywords = []
    matched_lime_words = []
    score = 0.0

    for keyword in keywords:
        keyword_tokens = extract_keyword_tokens(keyword)

        # Phrase match in the selected chunk
        if " " in keyword and keyword in chunk_text_norm:
            score += 2.5
            matched_keywords.append(keyword)

        # Single-word keyword directly appears in LIME important words
        if len(keyword_tokens) == 1:
            token = keyword_tokens[0]

            if token in important_set:
                weight = abs(important_map[token])
                score += 1.0 + min(weight * 5, 2.0)
                matched_keywords.append(keyword)
                matched_lime_words.append(token)

        # Multi-word keyword partially supported by LIME words
        elif len(keyword_tokens) > 1:
            overlap = [token for token in keyword_tokens if token in important_set]

            # Require at least half of useful tokens to overlap
                        # Require stronger overlap for multi-word legal patterns.
            # This prevents generic words like "high" and "court"
            # from incorrectly matching phrases such as "high court not justified".
            required = max(2, (len(keyword_tokens) * 3 + 3) // 4)

            if len(overlap) >= required:
                overlap_weight = sum(abs(important_map[token]) for token in overlap)
                score += 1.0 + min(overlap_weight * 5, 2.0)
                matched_keywords.append(keyword)
                matched_lime_words.extend(overlap)

    matched_keywords = sorted(set(matched_keywords))
    matched_lime_words = sorted(set(matched_lime_words))

    return {
        "pattern_id": pattern_row["pattern_id"],
        "title": pattern_row["title"],
        "outcome_type": pattern_row["outcome_type"],
        "description": pattern_row["description"],
        "score": float(score),
        "matched_keywords": matched_keywords,
        "matched_lime_words": matched_lime_words,
    }


def match_legal_patterns(
    legal_norms,
    prediction_result,
    filtered_supportive_words,
    include_all_patterns=False,
    top_k=5
):
    predicted_outcome = prediction_result["prediction"]
    chunk_text = prediction_result["most_influential_chunk_text"]

    results = []

    for _, row in legal_norms.iterrows():
        outcome_type = str(row["outcome_type"])

        # By default, keep only patterns aligned with the predicted class
        # plus neutral procedural patterns.
        if not include_all_patterns:
            if outcome_type not in {predicted_outcome, "Neutral"}:
                continue

        scored = score_pattern(
            pattern_row=row,
            important_words=filtered_supportive_words,
            chunk_text=chunk_text
        )

        if scored["score"] > 0:
            results.append(scored)

    results = sorted(results, key=lambda x: x["score"], reverse=True)

    return results[:top_k]


# ============================================================
# Loading and saving
# ============================================================

def load_lime_explanation(path):
    with open(path, "r") as f:
        return json.load(f)


def load_legal_norms(path):
    if not path.exists():
        raise FileNotFoundError(f"legal_norms.csv not found: {path}")

    df = pd.read_csv(path)

    required_columns = [
        "pattern_id",
        "title",
        "outcome_type",
        "keywords",
        "description",
    ]

    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column in legal_norms.csv: {col}")

    return df


def save_result(result, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(result, f, indent=4)

    return output_path


# ============================================================
# Printing
# ============================================================

def print_legal_explanation(result):
    prediction = result["prediction_result"]
    lime = result["filtered_lime"]
    patterns = result["related_legal_patterns"]

    print("\n================ LEGAL LAYER EXPLANATION ================")
    print(f"Prediction: {prediction['prediction']}")
    print(f"Predicted label: {prediction['predicted_label']}")
    print(f"Probability Accepted: {prediction['probability_accepted']:.4f}")
    print(f"Confidence: {prediction['confidence']:.4f}")
    print(f"Threshold: {prediction['threshold']:.2f}")

    print("\nMost influential chunk:")
    print(f"Chunk index: {prediction['most_influential_chunk_index']}")
    print(f"Chunk probability: {prediction['most_influential_chunk_probability']:.4f}")
    print("Text preview:")
    print(prediction["most_influential_chunk_text"][:1200])

    print("\nFiltered LIME words supporting the prediction:")
    if lime["supportive_words"]:
        for item in lime["supportive_words"]:
            print(f"- {item['word']}: {item['weight']:.4f}")
    else:
        print("- None")

    print("\nFiltered LIME words against the prediction:")
    if lime["opposing_words"]:
        for item in lime["opposing_words"]:
            print(f"- {item['word']}: {item['weight']:.4f}")
    else:
        print("- None")

    print("\nRelated legal patterns:")
    if patterns:
        for i, pattern in enumerate(patterns, start=1):
            print(f"\n{i}. {pattern['title']}")
            print(f"   Outcome type: {pattern['outcome_type']}")
            print(f"   Score: {pattern['score']:.4f}")
            print(f"   Matched keywords: {', '.join(pattern['matched_keywords'])}")
            print(f"   Matched LIME words: {', '.join(pattern['matched_lime_words'])}")
            print(f"   Description: {pattern['description']}")
    else:
        print("- No related legal patterns found.")

    print("==========================================================")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Legal layer over filtered LIME explanation."
    )

    parser.add_argument(
        "--lime_file",
        type=str,
        default=str(DEFAULT_LIME_PATH),
        help="Path to LIME explanation JSON."
    )

    parser.add_argument(
        "--legal_norms",
        type=str,
        default=str(LEGAL_NORMS_PATH),
        help="Path to legal_norms.csv."
    )

    parser.add_argument(
        "--output",
        type=str,
        default=str(LEGAL_OUTPUT_PATH),
        help="Path to save legal layer JSON."
    )

    parser.add_argument(
        "--top_k_patterns",
        type=int,
        default=5,
        help="Number of legal patterns to show."
    )

    parser.add_argument(
        "--include_all_patterns",
        action="store_true",
        help="If set, include patterns from all outcome types, not only predicted outcome + Neutral."
    )

    args = parser.parse_args()

    lime_path = Path(args.lime_file)
    legal_norms_path = Path(args.legal_norms)
    output_path = Path(args.output)

    if not lime_path.exists():
        raise FileNotFoundError(
            f"LIME explanation not found: {lime_path}\n"
            "Run explain_lime.py first."
        )

    lime_data = load_lime_explanation(lime_path)
    legal_norms = load_legal_norms(legal_norms_path)

    prediction_result = lime_data["prediction_result"]
    lime_result = lime_data["lime_result"]

    raw_supportive = lime_result.get("supportive_words", [])
    raw_opposing = lime_result.get("opposing_words", [])

    filtered_supportive = filter_lime_items(raw_supportive)
    filtered_opposing = filter_lime_items(raw_opposing)

    related_patterns = match_legal_patterns(
        legal_norms=legal_norms,
        prediction_result=prediction_result,
        filtered_supportive_words=filtered_supportive,
        include_all_patterns=args.include_all_patterns,
        top_k=args.top_k_patterns
    )

    final_result = {
        "prediction_result": prediction_result,
        "true_label": lime_data.get("true_label"),
        "filtered_lime": {
            "supportive_words": filtered_supportive,
            "opposing_words": filtered_opposing,
            "removed_noise_examples": [
                "the",
                "in",
                "and",
                "for",
                "outcome",
                "word"
            ]
        },
        "related_legal_patterns": related_patterns,
        "source_lime_file": str(lime_path),
        "legal_norms_file": str(legal_norms_path),
    }

    save_result(final_result, output_path)

    print_legal_explanation(final_result)

    print("\nLegal layer explanation saved:")
    print(output_path)


if __name__ == "__main__":
    main()