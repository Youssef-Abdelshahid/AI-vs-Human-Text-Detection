from __future__ import annotations

import html
import logging
import math
import pickle
import re
from pathlib import Path
from typing import Any

import joblib
import nltk
from flask import Flask, jsonify, render_template, request
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
MODELS_DIR = PROJECT_ROOT / "models"
MIN_CHUNK_SENTENCES = 2
TARGET_CHUNK_SENTENCES = 4
MAX_CHUNK_SENTENCES = 6
MAX_CHUNK_CHARS = 900
MIN_LOCAL_SECTION_CHARS = 150
LONG_LOCAL_SENTENCE_CHARS = 220
MAX_LOCAL_SECTION_CHARS = 650
TARGET_LOCAL_SECTION_SENTENCES = 3

MODEL_CONFIG = {
    "en_combined": {
        "name": "English combined detector",
        "language": "en",
        "source": "combined",
        "essay_model_key": "en_essay",
        "sentence_model_key": "en_sentence",
    },
    "en_sentence": {
        "name": "English sentence model",
        "path": MODELS_DIR / "sentences_passive_aggressive_pipeline.pkl",
        "language": "en",
        "split_mode": "chunk",
        "source": "local",
    },
    "en_essay": {
        "name": "English essay model",
        "path": MODELS_DIR / "essay_passive_aggressive_pipeline.pkl",
        "language": "en",
        "split_mode": "document",
        "source": "local",
    },
}

LABELS = {
    0: "Human",
    1: "AI-generated",
}

app = Flask(__name__, template_folder=str(BASE_DIR))
_model_cache: dict[str, Any] = {}
_model_errors: dict[str, str] = {}


def ensure_nltk_resource(resource_path: str, package_name: str) -> None:
    try:
        nltk.data.find(resource_path)
    except LookupError:
        try:
            nltk.download(package_name, quiet=True)
        except Exception:
            logger.exception("Could not download NLTK resource %s", package_name)


for resource_path, package_name in (
    ("tokenizers/punkt", "punkt"),
    ("tokenizers/punkt_tab", "punkt_tab"),
    ("corpora/stopwords", "stopwords"),
    ("corpora/wordnet", "wordnet"),
):
    ensure_nltk_resource(resource_path, package_name)


def load_stopwords(language: str) -> set[str]:
    try:
        return set(stopwords.words(language))
    except LookupError:
        logger.exception("Missing NLTK stopwords for %s", language)
        return set()


ENGLISH_STOPWORDS = load_stopwords("english")
ENGLISH_LEMMATIZER = WordNetLemmatizer()


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/model-status/<model_key>")
def model_status(model_key: str):
    if model_key not in MODEL_CONFIG:
        return jsonify({"ok": False, "error": "Unknown model."}), 404

    try:
        source = MODEL_CONFIG[model_key].get("source")
        if source == "combined":
            get_model(MODEL_CONFIG[model_key]["essay_model_key"])
            get_model(MODEL_CONFIG[model_key]["sentence_model_key"])
        else:
            get_model(model_key)
    except RuntimeError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 503
    except Exception:
        logger.exception("Model status check failed for %s", model_key)
        return jsonify({"ok": False, "error": "Model status check failed. Check the backend logs."}), 500

    return jsonify({"ok": True, "model": MODEL_CONFIG[model_key]["name"]})


def clean_english_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def preprocess_english(text: str) -> str:
    cleaned = clean_english_text(text)
    try:
        tokens = word_tokenize(cleaned)
    except LookupError:
        logger.exception("NLTK word tokenizer unavailable; using whitespace tokenization")
        tokens = cleaned.split()
    filtered = [word for word in tokens if word.lower() not in ENGLISH_STOPWORDS]
    lemmatized = [ENGLISH_LEMMATIZER.lemmatize(word) for word in filtered]
    return " ".join(lemmatized)


def split_with_regex(text: str) -> list[str]:
    pattern = r"[^.!?]+[.!?]*"
    segments = [match.group(0).strip() for match in re.finditer(pattern, text)]
    return [segment for segment in segments if segment]


def split_segments(text: str) -> list[str]:
    try:
        segments = [segment.strip() for segment in nltk.sent_tokenize(text)]
        segments = [segment for segment in segments if segment]
    except Exception:
        logger.exception("Primary sentence segmentation failed; using fallback")
        segments = split_with_regex(text)

    if not segments and text.strip():
        return [text.strip()]
    return segments


def split_paragraphs(text: str) -> list[str]:
    paragraphs = [paragraph.strip() for paragraph in re.split(r"\n\s*\n+", text)]
    return [paragraph for paragraph in paragraphs if paragraph]


def join_chunk_sentences(sentences: list[str]) -> str:
    return " ".join(sentence.strip() for sentence in sentences if sentence.strip()).strip()


def split_long_sentence(sentence: str) -> list[str]:
    sentence = sentence.strip()
    if len(sentence) <= MAX_CHUNK_CHARS:
        return [sentence] if sentence else []

    words = sentence.split()
    chunks = []
    current_words = []
    current_length = 0
    for word in words:
        next_length = current_length + len(word) + (1 if current_words else 0)
        if current_words and next_length > MAX_CHUNK_CHARS:
            chunks.append(" ".join(current_words))
            current_words = [word]
            current_length = len(word)
        else:
            current_words.append(word)
            current_length = next_length

    if current_words:
        chunks.append(" ".join(current_words))
    return chunks


def chunk_sentence_list(sentences: list[str]) -> list[str]:
    chunks = []
    current_sentences = []
    current_length = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        if len(sentence) > MAX_CHUNK_CHARS:
            if current_sentences:
                chunks.append(join_chunk_sentences(current_sentences))
                current_sentences = []
                current_length = 0
            chunks.extend(split_long_sentence(sentence))
            continue

        next_length = current_length + len(sentence) + (1 if current_sentences else 0)
        should_flush = (
            current_sentences
            and (
                len(current_sentences) >= MAX_CHUNK_SENTENCES
                or next_length > MAX_CHUNK_CHARS
                or len(current_sentences) >= TARGET_CHUNK_SENTENCES
            )
        )
        if should_flush:
            chunks.append(join_chunk_sentences(current_sentences))
            current_sentences = []
            current_length = 0

        current_sentences.append(sentence)
        current_length += len(sentence) + (1 if current_length else 0)

    if current_sentences:
        tail = join_chunk_sentences(current_sentences)
        if chunks and len(current_sentences) < MIN_CHUNK_SENTENCES and len(chunks[-1]) + len(tail) + 1 <= MAX_CHUNK_CHARS:
            chunks[-1] = f"{chunks[-1]} {tail}".strip()
        else:
            chunks.append(tail)

    return [chunk for chunk in chunks if chunk]


def merge_short_chunks(chunks: list[str]) -> list[str]:
    merged = []
    buffer = ""

    for chunk in chunks:
        if not buffer:
            buffer = chunk
            continue

        buffer_sentence_count = len(split_segments(buffer))
        chunk_sentence_count = len(split_segments(chunk))
        can_merge = len(buffer) + len(chunk) + 2 <= MAX_CHUNK_CHARS
        if can_merge and (
            buffer_sentence_count < MIN_CHUNK_SENTENCES
            or chunk_sentence_count < MIN_CHUNK_SENTENCES
        ):
            buffer = f"{buffer}\n\n{chunk}"
        else:
            merged.append(buffer)
            buffer = chunk

    if buffer:
        merged.append(buffer)
    return merged


def chunk_text(text: str) -> list[str]:
    try:
        text = text.strip()
        if not text:
            return []

        all_sentences = split_segments(text)
        if len(text) <= MAX_CHUNK_CHARS and len(all_sentences) <= MAX_CHUNK_SENTENCES:
            return [text]

        chunks = []
        for paragraph in split_paragraphs(text):
            sentences = split_segments(paragraph)
            if not sentences:
                chunks.append(paragraph)
            elif len(sentences) <= MAX_CHUNK_SENTENCES and len(paragraph) <= MAX_CHUNK_CHARS:
                chunks.append(paragraph)
            else:
                chunks.extend(chunk_sentence_list(sentences))

        chunks = merge_short_chunks(chunks)
        return chunks or [text]
    except Exception:
        logger.exception("Text chunking failed; using whole input as one chunk")
        return [text.strip()] if text.strip() else []


def get_model(model_key: str) -> Any:
    if model_key in _model_cache:
        return _model_cache[model_key]
    if model_key in _model_errors:
        raise RuntimeError(_model_errors[model_key])

    config = MODEL_CONFIG[model_key]
    model_path = config["path"]
    if not model_path.exists():
        message = f"{config['name']} file is missing: {model_path.name}"
        _model_errors[model_key] = message
        raise RuntimeError(message)

    logger.info("Loading %s from %s", config["name"], model_path)
    try:
        model = joblib.load(model_path)
    except Exception as exc:
        logger.info("joblib could not load %s; trying pickle fallback", model_path)
        try:
            with model_path.open("rb") as file:
                model = pickle.load(file)
        except Exception as pickle_exc:
            logger.exception("Failed loading model from %s", model_path)
            message = f"Could not load {config['name']} from {model_path.name}."
            _model_errors[model_key] = message
            raise RuntimeError(message) from pickle_exc

    if not hasattr(model, "predict"):
        model_type = f"{type(model).__module__}.{type(model).__name__}"
        logger.error("%s loaded as %s, which has no predict() method", model_path.name, model_type)
        message = (
            f"{model_path.name} was loaded, but it is not a trained prediction pipeline "
            f"(loaded type: {model_type}). Keep dataset pickles in data/ and trained pipelines in models/."
        )
        _model_errors[model_key] = message
        raise RuntimeError(message)

    _model_cache[model_key] = model
    logger.info("Loaded %s from %s", config["name"], model_path)
    return model


def class_index(classes: Any, target: int) -> int | None:
    if classes is None:
        return None
    for index, value in enumerate(list(classes)):
        try:
            if int(value) == target:
                return index
        except (TypeError, ValueError):
            if value == target:
                return index
    return None


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1 / (1 + z)
    z = math.exp(value)
    return z / (1 + z)


def prediction_confidences(model: Any, processed_segments: list[str]) -> tuple[list[float | None], bool]:
    if hasattr(model, "predict_proba"):
        try:
            probabilities = model.predict_proba(processed_segments)
            ai_index = class_index(getattr(model, "classes_", None), 1)
            if ai_index is None:
                ai_index = 1 if len(probabilities[0]) > 1 else 0
            return [float(row[ai_index]) for row in probabilities], True
        except Exception:
            logger.exception("predict_proba failed; falling back to labels/decision scores")

    if hasattr(model, "decision_function"):
        try:
            scores = model.decision_function(processed_segments)
            if hasattr(scores, "tolist"):
                scores = scores.tolist()

            confidences = []
            for score in scores:
                if isinstance(score, list):
                    ai_index = class_index(getattr(model, "classes_", None), 1)
                    ai_score = score[ai_index if ai_index is not None else -1]
                else:
                    ai_score = score
                    classes = list(getattr(model, "classes_", []))
                    if len(classes) == 2:
                        try:
                            if int(classes[1]) != 1:
                                ai_score = -ai_score
                        except (TypeError, ValueError):
                            pass
                confidences.append(sigmoid(float(ai_score)))
            return confidences, False
        except Exception:
            logger.exception("decision_function failed; confidence unavailable")

    return [None for _ in processed_segments], False


def overall_label(ai_percentage: float) -> str:
    if ai_percentage <= 35:
        return "Mostly Human"
    if ai_percentage <= 65:
        return "Mixed / Possibly AI-assisted"
    return "Mostly AI-generated"


def local_section_chunks(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []

    paragraphs = split_paragraphs(text) or [text]
    sections = []

    for paragraph in paragraphs:
        sentences = split_segments(paragraph)
        if not sentences:
            sections.append(paragraph)
            continue
        if len(sentences) == 1:
            sections.append(sentences[0])
            continue

        current_sentences = []
        current_length = 0
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            sentence_length = len(sentence)
            if not current_sentences and sentence_length >= LONG_LOCAL_SENTENCE_CHARS:
                sections.append(sentence)
                continue

            next_length = current_length + sentence_length + (1 if current_sentences else 0)
            should_flush = (
                current_sentences
                and (
                    next_length > MAX_LOCAL_SECTION_CHARS
                    or (
                        len(current_sentences) >= TARGET_LOCAL_SECTION_SENTENCES
                        and current_length >= MIN_LOCAL_SECTION_CHARS
                    )
                )
            )
            if should_flush:
                sections.append(join_chunk_sentences(current_sentences))
                current_sentences = []
                current_length = 0

            current_sentences.append(sentence)
            current_length += sentence_length + (1 if current_length else 0)

        if current_sentences:
            tail = join_chunk_sentences(current_sentences)
            if sections and len(tail) < MIN_LOCAL_SECTION_CHARS and len(sections[-1]) + len(tail) + 1 <= MAX_LOCAL_SECTION_CHARS:
                sections[-1] = f"{sections[-1]} {tail}".strip()
            else:
                sections.append(tail)

    return normalize_local_sections(sections)


def normalize_local_sections(sections: list[str]) -> list[str]:
    sections = [section.strip() for section in sections if section.strip()]
    if len(sections) <= 1:
        return sections

    merged = []
    buffer = ""
    for section in sections:
        if not buffer:
            buffer = section
            continue

        can_merge = len(buffer) + len(section) + 2 <= MAX_LOCAL_SECTION_CHARS
        if can_merge and (len(buffer) < MIN_LOCAL_SECTION_CHARS or len(section) < MIN_LOCAL_SECTION_CHARS):
            buffer = f"{buffer}\n\n{section}"
        else:
            merged.append(buffer)
            buffer = section

    if buffer:
        if merged and len(buffer) < MIN_LOCAL_SECTION_CHARS and len(merged[-1]) + len(buffer) + 2 <= MAX_LOCAL_SECTION_CHARS:
            merged[-1] = f"{merged[-1]}\n\n{buffer}"
        else:
            merged.append(buffer)
    return merged


def model_predictions_and_scores(model: Any, inputs: list[str]) -> tuple[list[int], list[float], bool]:
    predictions = model.predict(inputs)
    if hasattr(predictions, "tolist"):
        predictions = predictions.tolist()
    predictions = [int(prediction) for prediction in predictions]

    ai_confidences, confidence_is_probability = prediction_confidences(model, inputs)
    ai_scores = []
    for prediction, confidence in zip(predictions, ai_confidences):
        if confidence is None:
            ai_scores.append(1.0 if prediction == 1 else 0.0)
        else:
            ai_scores.append(float(confidence))
    return predictions, ai_scores, confidence_is_probability


def choose_highlighted_sections(sections: list[dict[str, Any]], essay_ai_percentage: float) -> set[int]:
    if not sections:
        return set()
    if len(sections) == 1:
        return {0} if essay_ai_percentage >= 65 else set()
    if essay_ai_percentage < 5:
        best_index, best_section = max(enumerate(sections), key=lambda item: item[1]["ai_score"])
        if best_section["ai_score"] >= 0.9 and best_section["length"] <= 250:
            return {best_index}
        return set()

    total_length = sum(section["length"] for section in sections)
    target_length = total_length * (essay_ai_percentage / 100)
    tolerance = max(total_length * 0.10, 120)
    selected: set[int] = set()
    selected_length = 0

    for index, section in sorted(enumerate(sections), key=lambda item: item[1]["ai_score"], reverse=True):
        section_length = section["length"]
        if essay_ai_percentage < 20 and section_length > target_length + tolerance and selected:
            continue

        if selected and selected_length >= max(1, target_length - tolerance):
            break

        if (
            essay_ai_percentage < 20
            and not selected
            and section_length > target_length + tolerance
            and len(sections) > 1
        ):
            continue

        selected.add(index)
        selected_length += section_length
        if selected_length >= target_length:
            break

    return selected


def predict_english_combined(text: str) -> dict[str, Any]:
    essay_model = get_model("en_essay")

    processed_document = preprocess_english(text)
    safe_document = processed_document if processed_document else " "
    try:
        essay_predictions, essay_scores, essay_uses_probability = model_predictions_and_scores(essay_model, [safe_document])
    except Exception as exc:
        logger.exception("Essay model prediction failed")
        raise RuntimeError("Essay model prediction failed. The overall score is unavailable.") from exc

    essay_ai_probability = essay_scores[0]
    essay_ai_percentage = round(essay_ai_probability * 100, 2)

    local_error = ""
    section_results = []
    highlighted_text = html.escape(text)

    try:
        sentence_model = get_model("en_sentence")
        sections = local_section_chunks(text)
        if sections:
            processed_sections = [preprocess_english(section) for section in sections]
            safe_sections = [processed if processed else " " for processed in processed_sections]
            local_predictions, local_scores, _ = model_predictions_and_scores(sentence_model, safe_sections)

            ranked_sections = [
                {
                    "text": section,
                    "prediction": prediction,
                    "ai_score": score,
                    "length": len(section),
                }
                for section, prediction, score in zip(sections, local_predictions, local_scores)
            ]
            selected_indexes = choose_highlighted_sections(ranked_sections, essay_ai_percentage)
            highlighted_parts = []

            for index, section in enumerate(ranked_sections):
                selected = index in selected_indexes
                escaped_section = html.escape(section["text"])
                if selected:
                    highlighted_parts.append(f'<mark class="ai-highlight">{escaped_section}</mark>')
                else:
                    highlighted_parts.append(f"<span>{escaped_section}</span>")

                section_results.append(
                    {
                        "index": index + 1,
                        "text": section["text"],
                        "prediction": section["prediction"],
                        "label": LABELS.get(section["prediction"], str(section["prediction"])),
                        "ai_confidence": round(section["ai_score"] * 100, 2),
                        "selected": selected,
                    }
                )

            highlighted_text = "\n\n".join(highlighted_parts)
    except Exception:
        logger.exception("Local sentence-model highlighting failed")
        local_error = "Local highlighting is unavailable, but the essay-model overall score is still shown."

    note = (
        "Overall AI percentage is based on the essay/full-text model. "
        "Highlighted sections are the most suspicious local sections according to the sentence model, "
        "limited by the essay model's overall score."
    )
    if local_error:
        note = f"{note} {local_error}"

    return {
        "ai_percentage": essay_ai_percentage,
        "human_percentage": round(100 - essay_ai_percentage, 2),
        "overall_label": overall_label(essay_ai_percentage),
        "language": "en",
        "model_key": "en_combined",
        "model_file": MODEL_CONFIG["en_essay"]["path"].name,
        "used_probability_average": essay_uses_probability,
        "segments": section_results,
        "highlighted_text": highlighted_text,
        "analysis_note": note,
        "breakdown_label": "Local Section Breakdown",
    }


def predict_segments(model_key: str, text: str) -> dict[str, Any]:
    config = MODEL_CONFIG[model_key]
    model = get_model(model_key)
    segments = [text] if config["split_mode"] == "document" else chunk_text(text)
    if not segments:
        raise ValueError("No valid text segments were found.")

    processed_segments = [preprocess_english(segment) for segment in segments]
    safe_segments = [processed if processed else " " for processed in processed_segments]

    logger.info("Running English prediction on %s chunk(s)", len(safe_segments))
    try:
        predictions = model.predict(safe_segments)
    except Exception as exc:
        logger.exception("Prediction failed for %s on %s chunk(s)", model_key, len(safe_segments))
        raise RuntimeError("Prediction failed for one or more text sections. Check the backend logs for details.") from exc
    if hasattr(predictions, "tolist"):
        predictions = predictions.tolist()
    predictions = [int(prediction) for prediction in predictions]

    ai_confidences, confidence_is_probability = prediction_confidences(model, safe_segments)
    confidence_available = any(confidence is not None for confidence in ai_confidences)
    if confidence_available:
        ai_percentage = round(
            sum(confidence or 0 for confidence in ai_confidences) / len(ai_confidences) * 100,
            2,
        )
    else:
        ai_percentage = round((sum(1 for prediction in predictions if prediction == 1) / len(predictions)) * 100, 2)

    segment_results = []
    highlighted_parts = []
    for index, (segment, prediction, confidence) in enumerate(
        zip(segments, predictions, ai_confidences),
        start=1,
    ):
        label = LABELS.get(prediction, str(prediction))
        escaped_segment = html.escape(segment)
        if config["split_mode"] != "document" and prediction == 1:
            highlighted_parts.append(f'<mark class="ai-highlight">{escaped_segment}</mark>')
        else:
            highlighted_parts.append(f'<span>{escaped_segment}</span>')

        segment_results.append(
            {
                "index": index,
                "text": segment,
                "prediction": prediction,
                "label": label,
                "ai_confidence": round(confidence * 100, 2) if confidence is not None else None,
            }
        )

    analysis_note = ""
    if config["split_mode"] == "document":
        if confidence_available and not confidence_is_probability:
            analysis_note = (
                "Essay model analyzes the full text as one document. AI/Human percentages are approximate "
                "because they are derived from the model decision score."
            )
        else:
            analysis_note = "Essay model analyzes the full text as one document, so highlighting is not sentence-level."

    return {
        "ai_percentage": ai_percentage,
        "human_percentage": round(100 - ai_percentage, 2) if confidence_available else None,
        "overall_label": overall_label(ai_percentage),
        "language": "en",
        "model_key": model_key,
        "model_file": config["path"].name,
        "used_probability_average": confidence_is_probability,
        "segments": segment_results,
        "highlighted_text": "\n\n".join(highlighted_parts),
        "analysis_note": analysis_note,
        "breakdown_label": "Document Analysis" if config["split_mode"] == "document" else "Chunk Breakdown",
    }


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    model_key = data.get("model_key") or "en_combined"
    if model_key == "en":
        model_key = "en_combined"

    logger.info("Received prediction request: model=%s, characters=%s", model_key, len(text))

    if model_key not in MODEL_CONFIG:
        return jsonify({"error": "Please choose a valid detector."}), 400
    if not text:
        return jsonify({"error": "Please paste some text to analyze."}), 400

    try:
        source = MODEL_CONFIG[model_key].get("source")
        if source == "combined":
            return jsonify(predict_english_combined(text))
        return jsonify(predict_segments(model_key, text))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 503
    except Exception:
        logger.exception("Prediction failed")
        return jsonify({"error": "Prediction failed. Please check the backend logs for details."}), 500


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
