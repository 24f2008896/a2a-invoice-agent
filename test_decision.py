
from ai_client import decide_batch

sample_packages = [
    {
        "packageId": "TEST-PACKAGE-001",
        "documents": [
            {
                "documentType": "invoice",
                "text": (
                    "Invoice INV-1001 issued by Example Supplies. "
                    "Total amount INR 500.00. "
                    "Payment reconciliation is complete and the "
                    "invoice is within the authorized payment limit. "
                    "[INV-REF-001] [REC-REF-001] [AUTH-REF-001]"
                ),
            }
        ],
    }
]

try:
    decisions = decide_batch(sample_packages)

    for decision in decisions:
        print("\nValidated decision:")
        print(decision)

except Exception as exc:
    print(f"Decision test failed: {type(exc).__name__}: {exc}")
