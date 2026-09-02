import hashlib
import json


def canonical_json(value) -> bytes:
    """
    Deterministic JSON encoding.

    The same object always produces the same bytes,
    regardless of dictionary insertion order.
    """

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hash_json(value) -> str:
    return sha256_bytes(
        canonical_json(value)
    )


def merkle_root(values: list) -> str:
    """
    Compute a SHA-256 Merkle root over arbitrary
    JSON-serialisable values.
    """

    if not values:
        return sha256_bytes(b"")

    level = [
        bytes.fromhex(
            hash_json(value)
        )
        for value in values
    ]

    while len(level) > 1:

        # Duplicate final node for odd-sized levels.
        if len(level) % 2 == 1:
            level.append(level[-1])

        next_level = []

        for index in range(
            0,
            len(level),
            2,
        ):
            left = level[index]
            right = level[index + 1]

            parent = hashlib.sha256(
                left + right
            ).digest()

            next_level.append(parent)

        level = next_level

    return level[0].hex()


def create_attestation(
    audit_payload: dict,
) -> dict:

    results = audit_payload.get(
        "results",
        [],
    )

    # Configuration we want to prove belonged
    # to this particular audit.
    manifest = {
        key: value
        for key, value
        in audit_payload.items()
        if key not in {
            "results",
            "controller_summary",
        }
    }

    manifest_hash = hash_json(
        manifest
    )

    results_root = merkle_root(
        results
    )

    source_file_hash = hash_json(
        audit_payload
    )

    audit_id = hash_json({
        "manifest_hash":
            manifest_hash,
        "results_root":
            results_root,
    })

    return {
        "schema":
            "shadowbench-attestation-v1",

        "hash_algorithm":
            "SHA-256",

        "audit_id":
            audit_id,

        "manifest":
            manifest,

        "manifest_hash":
            manifest_hash,

        "results_root":
            results_root,

        "source_file_hash":
            source_file_hash,

        "result_count":
            len(results),
    }


def verify_attestation(
    audit_payload: dict,
    attestation: dict,
) -> bool:

    expected = create_attestation(
        audit_payload
    )

    fields = [
        "audit_id",
        "manifest_hash",
        "results_root",
        "source_file_hash",
        "result_count",
    ]

    return all(
        expected[field]
        == attestation.get(field)
        for field in fields
    )
