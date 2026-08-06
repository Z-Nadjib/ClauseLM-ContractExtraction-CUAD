"""Output schema shared by every extractor.

The 41 CUAD clause categories are mapped to snake_case JSON keys. Each key holds a
list of verbatim text spans copied from the contract. This module is the single
source of truth for field names, descriptions, JSON Schema generation and output
validation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Field:
    key: str
    cuad_name: str
    group: str
    description: str


FIELDS: tuple[Field, ...] = (
    Field("document_name", "Document Name", "general", "Title of the contract."),
    Field("parties", "Parties", "general", "Names of the parties who signed the contract."),
    Field("agreement_date", "Agreement Date", "general", "Date the contract was signed or dated."),
    Field("effective_date", "Effective Date", "general", "Date the contract becomes effective."),
    Field("expiration_date", "Expiration Date", "term", "Date or event on which the initial term expires."),
    Field(
        "renewal_term", "Renewal Term", "term", "Renewal term after the initial term, including automatic extensions."
    ),
    Field(
        "notice_period_to_terminate_renewal",
        "Notice Period To Terminate Renewal",
        "term",
        "Notice period required to terminate renewal.",
    ),
    Field("governing_law", "Governing Law", "general", "Jurisdiction whose law governs the contract."),
    Field(
        "most_favored_nation",
        "Most Favored Nation",
        "commercial",
        "Better terms offered to a third party must also be offered to the counterparty.",
    ),
    Field("non_compete", "Non-Compete", "restrictive_covenants", "Restriction on competing with the counterparty."),
    Field("exclusivity", "Exclusivity", "restrictive_covenants", "Exclusive dealing or exclusive supply commitment."),
    Field(
        "no_solicit_of_customers",
        "No-Solicit Of Customers",
        "restrictive_covenants",
        "Restriction on soliciting customers of the counterparty.",
    ),
    Field(
        "competitive_restriction_exception",
        "Competitive Restriction Exception",
        "restrictive_covenants",
        "Exceptions or carve-outs to non-compete, exclusivity or no-solicit clauses.",
    ),
    Field(
        "no_solicit_of_employees",
        "No-Solicit Of Employees",
        "restrictive_covenants",
        "Restriction on soliciting or hiring employees of the counterparty.",
    ),
    Field(
        "non_disparagement",
        "Non-Disparagement",
        "restrictive_covenants",
        "Requirement not to disparage the counterparty.",
    ),
    Field(
        "termination_for_convenience",
        "Termination For Convenience",
        "term",
        "Right to terminate without cause.",
    ),
    Field(
        "rofr_rofo_rofn",
        "Rofr/Rofo/Rofn",
        "control",
        "Right of first refusal, first offer or first negotiation.",
    ),
    Field(
        "change_of_control",
        "Change Of Control",
        "control",
        "Rights triggered by a change of control, merger or asset sale.",
    ),
    Field(
        "anti_assignment",
        "Anti-Assignment",
        "control",
        "Consent or notice required to assign the contract.",
    ),
    Field(
        "revenue_profit_sharing",
        "Revenue/Profit Sharing",
        "commercial",
        "Obligation to share revenue or profit, including royalties.",
    ),
    Field(
        "price_restrictions",
        "Price Restrictions",
        "commercial",
        "Restriction on raising or lowering prices.",
    ),
    Field(
        "minimum_commitment",
        "Minimum Commitment",
        "commercial",
        "Minimum order size, purchase amount or fee commitment.",
    ),
    Field(
        "volume_restriction",
        "Volume Restriction",
        "commercial",
        "Fee increase or consent requirement above a usage threshold.",
    ),
    Field(
        "ip_ownership_assignment",
        "Ip Ownership Assignment",
        "intellectual_property",
        "Assignment of intellectual property ownership to the counterparty.",
    ),
    Field(
        "joint_ip_ownership",
        "Joint Ip Ownership",
        "intellectual_property",
        "Joint or shared ownership of intellectual property.",
    ),
    Field("license_grant", "License Grant", "intellectual_property", "Grant of a license by one party to another."),
    Field(
        "non_transferable_license",
        "Non-Transferable License",
        "intellectual_property",
        "Limit on transferring a granted license.",
    ),
    Field(
        "affiliate_license_licensor",
        "Affiliate License-Licensor",
        "intellectual_property",
        "License grant covering intellectual property of the licensor's affiliates.",
    ),
    Field(
        "affiliate_license_licensee",
        "Affiliate License-Licensee",
        "intellectual_property",
        "License grant extended to the licensee's affiliates.",
    ),
    Field(
        "unlimited_all_you_can_eat_license",
        "Unlimited/All-You-Can-Eat-License",
        "intellectual_property",
        "License granting unlimited usage.",
    ),
    Field(
        "irrevocable_or_perpetual_license",
        "Irrevocable Or Perpetual License",
        "intellectual_property",
        "License that is irrevocable or perpetual.",
    ),
    Field(
        "source_code_escrow",
        "Source Code Escrow",
        "liability",
        "Obligation to deposit source code in escrow.",
    ),
    Field(
        "post_termination_services",
        "Post-Termination Services",
        "term",
        "Obligations that continue after termination or expiration.",
    ),
    Field("audit_rights", "Audit Rights", "liability", "Right to audit books, records or physical locations."),
    Field("uncapped_liability", "Uncapped Liability", "liability", "Liability that is not capped."),
    Field("cap_on_liability", "Cap On Liability", "liability", "Cap on liability or on the time to bring claims."),
    Field("liquidated_damages", "Liquidated Damages", "liability", "Liquidated damages or termination fees."),
    Field("warranty_duration", "Warranty Duration", "liability", "Duration of a warranty."),
    Field("insurance", "Insurance", "liability", "Requirement to maintain insurance."),
    Field(
        "covenant_not_to_sue",
        "Covenant Not To Sue",
        "liability",
        "Restriction on contesting or suing over the counterparty's rights.",
    ),
    Field(
        "third_party_beneficiary",
        "Third Party Beneficiary",
        "liability",
        "Third party beneficiary able to enforce the contract.",
    ),
)

FIELD_KEYS: tuple[str, ...] = tuple(f.key for f in FIELDS)
CUAD_NAME_TO_KEY: dict[str, str] = {f.cuad_name: f.key for f in FIELDS}
KEY_TO_FIELD: dict[str, Field] = {f.key: f for f in FIELDS}

Extraction = dict[str, list[str]]

JSON_SEPARATORS = (", ", ": ")


def empty_extraction() -> Extraction:
    return {key: [] for key in FIELD_KEYS}


def json_schema() -> dict[str, Any]:
    """JSON Schema of a chunk-level extraction.

    Every field is optional, which lets the models emit compact objects containing only the
    fields they found.
    """
    properties = {key: {"type": "array", "items": {"type": "string"}} for key in FIELD_KEYS}
    return {"type": "object", "properties": properties, "additionalProperties": False}


def to_compact_json(extraction: Extraction) -> str:
    """Serialize non-empty fields in schema order, the target format used for fine-tuning."""
    compact = {key: extraction[key] for key in FIELD_KEYS if extraction.get(key)}
    return json.dumps(compact, ensure_ascii=False, separators=JSON_SEPARATORS)


def validate(obj: Any) -> bool:
    """Return True when ``obj`` conforms to the extraction schema."""
    if not isinstance(obj, dict):
        return False
    for key, value in obj.items():
        if key not in KEY_TO_FIELD:
            return False
        if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
            return False
    return True


def parse_output(raw: str) -> tuple[Extraction, bool]:
    """Parse model output into a full extraction.

    Returns the extraction and a flag telling whether the raw output was valid JSON that
    conforms to the schema. Invalid outputs yield an empty extraction.
    """
    try:
        obj = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return empty_extraction(), False
    if not validate(obj):
        return empty_extraction(), False
    result = empty_extraction()
    for key, value in obj.items():
        result[key] = [v for v in value if v.strip()]
    return result, True
