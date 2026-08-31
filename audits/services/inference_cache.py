import hashlib
import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent.parent

CACHE_DIR = BASE_DIR / "data" / "cache"
CACHE_FILE = CACHE_DIR / "inference_cache.json"


def make_cache_key(
    model_name: str,
    prompt: str,
) -> str:

    raw = f"{model_name}\n---\n{prompt}"

    return hashlib.sha256(
        raw.encode("utf-8")
    ).hexdigest()


def load_cache() -> dict:

    if not CACHE_FILE.exists():
        return {}

    with open(
        CACHE_FILE,
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def save_cache(cache: dict) -> None:

    CACHE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        CACHE_FILE,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            cache,
            f,
            indent=2,
            ensure_ascii=False,
        )


def store_cached_result(
    model_name: str,
    prompt: str,
    result: dict,
) -> None:

    cache = load_cache()

    key = make_cache_key(
        model_name,
        prompt,
    )

    cache[key] = {
        "model": model_name,
        "prompt": prompt,
        "result": result,
    }

    save_cache(cache)


def cached_model_run(
    model,
    prompt: str,
) -> tuple[dict, bool]:

    cache = load_cache()

    model_key = getattr(
        model,
        "cache_id",
        model.model_name,
    )

    key = make_cache_key(
        model_key,
        prompt,
    )

    if key in cache:
        return cache[key]["result"], True

    result = model.run(prompt)

    cache[key] = {
        "model": model.model_name,
        "cache_id": model_key,
        "prompt": prompt,
        "result": result,
    }

    save_cache(cache)

    return result, False
