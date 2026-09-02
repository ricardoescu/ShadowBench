import argparse
import json
from pathlib import Path

from audits.services.attestation import (
    create_attestation,
    verify_attestation,
)


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Create a cryptographic "
            "ShadowBench audit attestation."
        )
    )

    parser.add_argument(
        "audit_file",
        type=Path,
    )

    args = parser.parse_args()

    audit_file = args.audit_file

    with open(
        audit_file,
        "r",
        encoding="utf-8",
    ) as f:
        payload = json.load(f)

    attestation = create_attestation(
        payload
    )

    output_file = (
        audit_file.parent
        / (
            audit_file.stem
            + ".attestation.json"
        )
    )

    with open(
        output_file,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            attestation,
            f,
            indent=2,
            ensure_ascii=False,
        )

    verified = verify_attestation(
        payload,
        attestation,
    )

    print()
    print("=" * 60)
    print(
        "SHADOWBENCH CRYPTOGRAPHIC ATTESTATION"
    )
    print("=" * 60)
    print()

    print(
        f"Results committed: "
        f"{attestation['result_count']}"
    )

    print()

    print("Manifest hash:")
    print(
        attestation[
            "manifest_hash"
        ]
    )

    print()

    print("Merkle results root:")
    print(
        attestation[
            "results_root"
        ]
    )

    print()

    print("Audit ID:")
    print(
        attestation[
            "audit_id"
        ]
    )

    print()

    print(
        "Verification: "
        + (
            "VALID"
            if verified
            else "INVALID"
        )
    )

    print()

    print("Attestation saved to:")
    print(output_file)


if __name__ == "__main__":
    main()
