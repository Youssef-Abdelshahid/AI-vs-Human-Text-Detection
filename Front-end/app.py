from __future__ import annotations

import html
import json
import logging
import math
import pickle
import re
from pathlib import Path
from typing import Any

import joblib
import nltk
from flask import Flask, jsonify, render_template, request, send_from_directory
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

try:
    import torch
    import torch.nn as nn
except Exception:
    torch = None
    nn = None

try:
    from transformers import ElectraConfig, ElectraModel, ElectraTokenizer, RobertaConfig, RobertaModel, RobertaTokenizer
except Exception:
    ElectraConfig = ElectraModel = ElectraTokenizer = RobertaConfig = RobertaModel = RobertaTokenizer = None


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
MODELS_DIR = PROJECT_ROOT / "models"
CACHE_DIR = PROJECT_ROOT / "cache"
TRANSFORMERS_CACHE_DIR = CACHE_DIR / "transformers"
MIN_CHUNK_SENTENCES = 2
TARGET_CHUNK_SENTENCES = 4
MAX_CHUNK_SENTENCES = 6
MAX_CHUNK_CHARS = 900
MIN_LOCAL_SECTION_CHARS = 150
LONG_LOCAL_SENTENCE_CHARS = 220
MAX_LOCAL_SECTION_CHARS = 650
TARGET_LOCAL_SECTION_SENTENCES = 3
TRANSFORMER_MAX_LEN = 256


if nn is not None:
    class DNNModel(nn.Module):
        def __init__(self, vocab_size: int, embed_dim: int, num_classes: int):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embed_dim)
            self.fc1 = nn.Linear(embed_dim, 128)
            self.dropout = nn.Dropout(0.5)
            self.fc2 = nn.Linear(128, 64)
            self.output = nn.Linear(64, num_classes)

        def forward(self, x):
            x = self.embedding(x)
            x = torch.mean(x, dim=1)
            x = torch.relu(self.fc1(x))
            x = self.dropout(x)
            x = torch.relu(self.fc2(x))
            return self.output(x)


    class BiLSTMModel(nn.Module):
        def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, output_dim: int):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embed_dim)
            self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
            self.output = nn.Linear(2 * hidden_dim, output_dim)

        def forward(self, x):
            x = self.embedding(x)
            _, (h_n, _) = self.lstm(x)
            x = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)
            return self.output(x)


    class CNNModel(nn.Module):
        def __init__(self, vocab_size: int, embed_dim: int, num_classes: int):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embed_dim)
            self.conv1d = nn.Conv1d(in_channels=embed_dim, out_channels=128, kernel_size=5)
            self.global_max_pool = nn.AdaptiveMaxPool1d(1)
            self.fc1 = nn.Linear(128, 64)
            self.dropout = nn.Dropout(0.5)
            self.fc2 = nn.Linear(64, num_classes)

        def forward(self, x):
            x = self.embedding(x)
            x = x.permute(0, 2, 1)
            x = torch.relu(self.conv1d(x))
            x = self.global_max_pool(x).squeeze(-1)
            x = torch.relu(self.fc1(x))
            x = self.dropout(x)
            return self.fc2(x)


    class GRUModel(nn.Module):
        def __init__(self, vocab_size: int, embed_dim: int, gru_hidden_dim: int, num_classes: int):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embed_dim)
            self.gru = nn.GRU(embed_dim, gru_hidden_dim, batch_first=True)
            self.fc1 = nn.Linear(gru_hidden_dim, 128)
            self.dropout = nn.Dropout(0.5)
            self.fc2 = nn.Linear(128, 64)
            self.output = nn.Linear(64, num_classes)

        def forward(self, x):
            x = self.embedding(x)
            _, hn = self.gru(x)
            x = hn[-1]
            x = torch.relu(self.fc1(x))
            x = self.dropout(x)
            x = torch.relu(self.fc2(x))
            return self.output(x)


class TorchPipeline:
    classes_ = [0, 1]

    def __init__(self, model: Any, tokenizer: Any, max_len: int, device: Any):
        self.model = model
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.device = device

    def _texts_to_tensor(self, texts: list[str]):
        rows = []
        word_index = getattr(self.tokenizer, "word_index", {})
        oov_token = getattr(self.tokenizer, "oov_token", None)
        oov_index = word_index.get(oov_token, 1) if oov_token else 1
        num_words = getattr(self.tokenizer, "num_words", None)

        for text in texts:
            tokens = str(text).split()
            sequence = []
            for token in tokens:
                index = word_index.get(token, oov_index)
                if num_words and index >= num_words:
                    index = oov_index
                sequence.append(index)
            sequence = sequence[: self.max_len]
            sequence.extend([0] * (self.max_len - len(sequence)))
            rows.append(sequence)

        return torch.tensor(rows, dtype=torch.long, device=self.device)

    def predict_proba(self, texts: list[str]):
        with torch.no_grad():
            inputs = self._texts_to_tensor(texts)
            logits = self.model(inputs)
            return torch.softmax(logits, dim=1).cpu().numpy()

    def predict(self, texts: list[str]):
        probabilities = self.predict_proba(texts)
        return probabilities.argmax(axis=1)


class TransformerPipeline:
    classes_ = [0, 1]

    def __init__(self, encoder: Any, classifier: Any, tokenizer: Any, device: Any):
        self.encoder = encoder
        self.classifier = classifier
        self.tokenizer = tokenizer
        self.device = device

    def predict_proba(self, texts: list[str]):
        encoded = self.tokenizer(
            [str(text) for text in texts],
            truncation=True,
            padding="max_length",
            max_length=TRANSFORMER_MAX_LEN,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        with torch.no_grad():
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            cls_token = outputs.last_hidden_state[:, 0, :]
            logits = self.classifier(cls_token)
            ai_prob = torch.sigmoid(logits).squeeze(-1)
            probabilities = torch.stack((1 - ai_prob, ai_prob), dim=1)
            return probabilities.cpu().numpy()

    def predict(self, texts: list[str]):
        probabilities = self.predict_proba(texts)
        return probabilities.argmax(axis=1)

MODEL_CONFIG = {
}

LABELS = {
    0: "Human",
    1: "AI-generated",
}

app = Flask(__name__, template_folder=str(BASE_DIR))
_model_cache: dict[str, Any] = {}
_model_errors: dict[str, str] = {}


def title_from_slug(slug: str) -> str:
    overrides = {
        "bilstm": "BiLSTM",
        "cnn": "CNN",
        "dnn": "DNN",
        "electra": "ELECTRA",
        "gru": "GRU",
        "roberta": "RoBERTa",
        "xgboost": "XGBoost",
        "naive_bayes": "Naive Bayes",
        "logistic_regression": "Logistic Regression",
        "passive_aggressive": "Passive Aggressive",
        "random_forest": "Random Forest",
    }
    return overrides.get(slug, slug.replace("_", " ").title())


def slug_from_model_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def load_metric_scores(category: str) -> dict[str, float]:
    metrics_path = MODELS_DIR / category / f"{category}_ml_metrics.json"
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except Exception:
        logger.exception("Could not read model metrics from %s", metrics_path)
        return {}

    scores = {}
    for item in metrics:
        model_name = item.get("model")
        if not model_name:
            continue
        scores[slug_from_model_name(model_name)] = float(item.get("f1") or item.get("accuracy") or 0)
    return scores


def load_dl_metric_scores(category: str) -> dict[str, float]:
    metrics_path = MODELS_DIR / category / f"{category}_dl_metrics.json"
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except Exception:
        logger.exception("Could not read DL model metrics from %s", metrics_path)
        return {}

    scores = {}
    for slug, item in metrics.items():
        accuracy = item.get("accuracy") or []
        scores[slug_from_model_name(slug)] = float(max(accuracy) if accuracy else 0)
    return scores


def likely_supports_predict_proba(slug: str) -> bool:
    return slug != "passive_aggressive"


def discover_model_registry() -> dict[str, dict[str, Any]]:
    registry: dict[str, dict[str, Any]] = {}
    for category in ("essay", "sentence"):
        model_dir = MODELS_DIR / category / "ml"
        metric_scores = load_metric_scores(category)
        if not model_dir.exists():
            logger.warning("Model directory is missing: %s", model_dir)
            continue

        for model_path in sorted(model_dir.glob(f"{category}_*.pkl")):
            slug = model_path.stem.removeprefix(f"{category}_")
            model_id = f"{category}_{slug}"
            model_name = title_from_slug(slug)
            registry[model_id] = {
                "id": model_id,
                "name": f"{category.title()} {model_name}",
                "path": model_path,
                "language": "en",
                "category": category,
                "source": "local",
                "model_family": "sklearn_pipeline",
                "supports_predict_proba": likely_supports_predict_proba(slug),
                "probability_fallback": "decision_function sigmoid, then hard prediction",
                "metric_score": metric_scores.get(slug),
                "split_mode": "document" if category == "essay" else "chunk",
            }

        dl_dir = MODELS_DIR / category / "dl"
        dl_config_path = dl_dir / f"{category}_dl_config.json"
        dl_metric_scores = load_dl_metric_scores(category)
        try:
            dl_config = json.loads(dl_config_path.read_text(encoding="utf-8"))
        except Exception:
            logger.exception("Could not read DL model config from %s", dl_config_path)
            dl_config = {}

        for slug, model_info in (dl_config.get("models") or {}).items():
            model_path = dl_dir / model_info.get("file", "")
            tokenizer_path = dl_dir / f"{category}_tokenizer.pkl"
            if not model_path.exists() or not tokenizer_path.exists():
                continue
            model_id = f"{category}_dl_{slug}"
            registry[model_id] = {
                "id": model_id,
                "name": f"{category.title()} {title_from_slug(slug)}",
                "path": model_path,
                "tokenizer_path": tokenizer_path,
                "language": "en",
                "category": category,
                "source": "local",
                "model_family": "torch_dl",
                "dl_architecture": slug,
                "dl_config": dl_config,
                "dl_model_config": model_info,
                "supports_predict_proba": True,
                "probability_fallback": "softmax probability",
                "metric_score": dl_metric_scores.get(slug),
                "split_mode": "document" if category == "essay" else "chunk",
            }

        transformer_dir = MODELS_DIR / category / "transformers"
        for slug in ("roberta", "electra"):
            model_family_dir = transformer_dir / slug
            checkpoints = sorted(model_family_dir.glob("best_*.pt"))
            if not checkpoints:
                continue
            checkpoint = checkpoints[-1]
            model_id = f"{category}_transformer_{slug}"
            registry[model_id] = {
                "id": model_id,
                "name": f"{category.title()} {title_from_slug(slug)}",
                "path": checkpoint,
                "language": "en",
                "category": category,
                "source": "local",
                "model_family": "transformer",
                "transformer_architecture": slug,
                "supports_predict_proba": True,
                "probability_fallback": "sigmoid probability",
                "metric_score": None,
                "split_mode": "document" if category == "essay" else "chunk",
            }
    return registry


def models_for_category(category: str) -> list[dict[str, Any]]:
    models = [config for config in MODEL_CONFIG.values() if config.get("category") == category]
    return sorted(
        models,
        key=lambda config: (
            config.get("metric_score") is None,
            -(config.get("metric_score") or 0),
            config["name"],
        ),
    )


def default_model_id(category: str) -> str | None:
    models = models_for_category(category)
    if not models:
        return None
    return models[0]["id"]


def public_model(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": config["id"],
        "name": config["name"],
        "supports_predict_proba": config["supports_predict_proba"],
        "probability_fallback": config["probability_fallback"],
    }


MODEL_CONFIG.update(discover_model_registry())


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


@app.route("/styles/<path:filename>")
def styles_asset(filename: str):
    return send_from_directory(BASE_DIR / "styles", filename)


@app.route("/scripts/<path:filename>")
def scripts_asset(filename: str):
    return send_from_directory(BASE_DIR / "scripts", filename)


@app.route("/model-status/<model_key>")
def model_status(model_key: str):
    if model_key not in MODEL_CONFIG:
        return jsonify({"ok": False, "error": "Unknown model."}), 404

    try:
        get_model(model_key)
    except RuntimeError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 503
    except Exception:
        logger.exception("Model status check failed for %s", model_key)
        return jsonify({"ok": False, "error": "Model status check failed. Check the backend logs."}), 500

    return jsonify({"ok": True, "model": MODEL_CONFIG[model_key]["name"]})


@app.route("/models")
def available_models():
    essay_models = [public_model(model) for model in models_for_category("essay")]
    sentence_models = [public_model(model) for model in models_for_category("sentence")]
    return jsonify(
        {
            "essay_models": essay_models,
            "sentence_models": sentence_models,
            "defaults": {
                "essay_model_id": default_model_id("essay"),
                "sentence_model_id": default_model_id("sentence"),
            },
        }
    )


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


def torch_device():
    if torch is None:
        raise RuntimeError("PyTorch is not installed, so this model type cannot run.")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_torch_state_dict(model_path: Path) -> Any:
    try:
        return torch.load(model_path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(model_path, map_location="cpu")


def build_dl_model(config: dict[str, Any]) -> Any:
    if torch is None or nn is None:
        raise RuntimeError("PyTorch is not installed, so deep learning models cannot run.")

    model_info = config["dl_model_config"]
    dl_config = config["dl_config"]
    architecture = config["dl_architecture"]
    vocab_size = int(dl_config.get("vocab_size", 0))
    embed_dim = int(model_info.get("embed_dim", 128))
    num_classes = int(model_info.get("num_classes", 2))

    if architecture == "cnn":
        return CNNModel(vocab_size, embed_dim, num_classes)
    if architecture == "bilstm":
        return BiLSTMModel(vocab_size, embed_dim, int(model_info.get("hidden_dim", 64)), num_classes)
    if architecture == "gru":
        return GRUModel(vocab_size, embed_dim, int(model_info.get("gru_hidden_dim", 128)), num_classes)
    if architecture == "dnn":
        return DNNModel(vocab_size, embed_dim, num_classes)
    raise RuntimeError(f"Unsupported DL architecture: {architecture}")


def load_torch_dl_pipeline(config: dict[str, Any]) -> TorchPipeline:
    device = torch_device()
    model = build_dl_model(config)
    state_dict = load_torch_state_dict(config["path"])
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    tokenizer = joblib.load(config["tokenizer_path"])
    max_len = int(config["dl_config"].get("max_len", 500))
    return TorchPipeline(model, tokenizer, max_len, device)


def load_transformer_pipeline(config: dict[str, Any]) -> TransformerPipeline:
    if torch is None or nn is None:
        raise RuntimeError("PyTorch is not installed, so transformer models cannot run.")

    architecture = config["transformer_architecture"]
    device = torch_device()
    cache_dir = str(TRANSFORMERS_CACHE_DIR)
    if architecture == "roberta":
        if RobertaConfig is None or RobertaModel is None or RobertaTokenizer is None:
            raise RuntimeError("Transformers RoBERTa support is not installed.")
        base_name = "roberta-base"
        hf_config = RobertaConfig.from_pretrained(base_name, cache_dir=cache_dir, local_files_only=True)
        tokenizer = RobertaTokenizer.from_pretrained(base_name, cache_dir=cache_dir, local_files_only=True)
        encoder = RobertaModel(hf_config)
    elif architecture == "electra":
        if ElectraConfig is None or ElectraModel is None or ElectraTokenizer is None:
            raise RuntimeError("Transformers ELECTRA support is not installed.")
        base_name = "google/electra-base-discriminator"
        hf_config = ElectraConfig.from_pretrained(base_name, cache_dir=cache_dir, local_files_only=True)
        tokenizer = ElectraTokenizer.from_pretrained(base_name, cache_dir=cache_dir, local_files_only=True)
        encoder = ElectraModel(hf_config)
    else:
        raise RuntimeError(f"Unsupported transformer architecture: {architecture}")

    checkpoint = load_torch_state_dict(config["path"])
    encoder.load_state_dict(checkpoint["model"])
    classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(hf_config.hidden_size, 1))
    classifier.load_state_dict(checkpoint["clf"])
    encoder.to(device)
    classifier.to(device)
    encoder.eval()
    classifier.eval()
    return TransformerPipeline(encoder, classifier, tokenizer, device)


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
        model_family = config.get("model_family")
        if model_family == "torch_dl":
            model = load_torch_dl_pipeline(config)
        elif model_family == "transformer":
            model = load_transformer_pipeline(config)
        else:
            try:
                model = joblib.load(model_path)
            except Exception:
                logger.info("joblib could not load %s; trying pickle fallback", model_path)
                with model_path.open("rb") as file:
                    model = pickle.load(file)
    except Exception as exc:
        logger.exception("Failed loading model from %s", model_path)
        message = f"Could not load {config['name']} from {model_path.name}."
        _model_errors[model_key] = message
        raise RuntimeError(message) from exc

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

    sections = []
    for paragraph in split_paragraphs(text) or [text]:
        sentences = split_segments(paragraph)
        sections.extend(sentences or [paragraph])

    return [section.strip() for section in sections if section.strip()]


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


def choose_highlighted_sections(
    sections: list[dict[str, Any]],
    essay_ai_percentage: float,
) -> set[int]:
    if not sections:
        return set()
    if essay_ai_percentage >= 99.995:
        return set(range(len(sections)))
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


def render_highlighted_sections(original_text: str, sections: list[dict[str, Any]], selected_indexes: set[int]) -> str:
    if not sections:
        return html.escape(original_text)

    parts = []
    cursor = 0
    for index, section in enumerate(sections):
        section_text = section["text"]
        start = original_text.find(section_text, cursor)
        if start < 0:
            logger.warning("Could not locate highlighted section in original text; using section-order fallback")
            fallback_parts = []
            for fallback_index, fallback_section in enumerate(sections):
                escaped = html.escape(fallback_section["text"])
                if fallback_index in selected_indexes:
                    fallback_parts.append(f'<mark class="ai-highlight">{escaped}</mark>')
                else:
                    fallback_parts.append(f"<span>{escaped}</span>")
            return " ".join(fallback_parts)

        parts.append(html.escape(original_text[cursor:start]))
        escaped_section = html.escape(section_text)
        if index in selected_indexes:
            parts.append(f'<mark class="ai-highlight">{escaped_section}</mark>')
        else:
            parts.append(f"<span>{escaped_section}</span>")
        cursor = start + len(section_text)

    parts.append(html.escape(original_text[cursor:]))
    return "".join(parts)


def predict_english_combined(
    text: str,
    essay_model_id: str,
    sentence_model_id: str,
) -> dict[str, Any]:
    essay_config = MODEL_CONFIG[essay_model_id]
    sentence_config = MODEL_CONFIG[sentence_model_id]
    essay_model = get_model(essay_model_id)

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
        sentence_model = get_model(sentence_model_id)
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
            selected_indexes = choose_highlighted_sections(
                ranked_sections,
                essay_ai_percentage,
            )

            for index, section in enumerate(ranked_sections):
                selected = index in selected_indexes
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

            highlighted_text = render_highlighted_sections(text, ranked_sections, selected_indexes)
    except Exception:
        logger.exception("Local sentence-model highlighting failed")
        local_error = "Local highlighting is unavailable, but the essay-model overall score is still shown."

    note = (
        "The essay model provides the overall score. "
        "The sentence model only ranks local sections for highlighting."
    )
    if local_error:
        note = f"{note} {local_error}"

    return {
        "ai_percentage": essay_ai_percentage,
        "human_percentage": round(100 - essay_ai_percentage, 2),
        "overall_label": overall_label(essay_ai_percentage),
        "language": "en",
        "essay_model_id": essay_model_id,
        "sentence_model_id": sentence_model_id,
        "essay_model_name": essay_config["name"],
        "sentence_model_name": sentence_config["name"],
        "model_file": essay_config["path"].name,
        "sentence_model_file": sentence_config["path"].name,
        "used_probability_average": essay_uses_probability,
        "segments": section_results,
        "highlighted_text": highlighted_text,
        "analysis_note": note,
        "breakdown_label": f"Section Breakdown ({sentence_config['name']})",
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
    essay_model_id = data.get("essay_model_id") or default_model_id("essay")
    sentence_model_id = data.get("sentence_model_id") or default_model_id("sentence")

    logger.info(
        "Received prediction request: essay_model=%s, sentence_model=%s, characters=%s",
        essay_model_id,
        sentence_model_id,
        len(text),
    )

    if not essay_model_id or essay_model_id not in MODEL_CONFIG or MODEL_CONFIG[essay_model_id].get("category") != "essay":
        return jsonify({"error": "Please choose a valid essay model."}), 400
    if not sentence_model_id or sentence_model_id not in MODEL_CONFIG or MODEL_CONFIG[sentence_model_id].get("category") != "sentence":
        return jsonify({"error": "Please choose a valid sentence model."}), 400
    if not text:
        return jsonify({"error": "Please paste some text to analyze."}), 400

    try:
        return jsonify(
            predict_english_combined(
                text,
                essay_model_id,
                sentence_model_id,
            )
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 503
    except Exception:
        logger.exception("Prediction failed")
        return jsonify({"error": "Prediction failed. Please check the backend logs for details."}), 500


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
