import pandas as pd

from mediroad.stage4_2.contracts import FIELD_STATUS_COLUMNS
from mediroad.stage4_2.field_validation import audit_field_validation


def _row(venue_id="V1", value="YES", date="2026-08-20"):
    row = {
        "venue_id": venue_id,
        "venue_name": venue_id,
        "venue_type": "senior_center",
        "cluster_id": "C1",
        "admin_code": "A1",
        "sigungu": "S1",
        "verification_date": date,
    }
    row.update({c: value for c in FIELD_STATUS_COLUMNS})
    return row


def test_all_yes_and_date_is_verified():
    audit = audit_field_validation(pd.DataFrame([_row()]))
    assert audit.summary["verified"] == 1
    assert audit.normalized.loc[0, "field_operationally_feasible"]


def test_unknown_is_not_verified():
    row = _row()
    row["parking"] = "UNKNOWN"
    audit = audit_field_validation(pd.DataFrame([row]))
    assert audit.summary["verified"] == 0
    assert audit.summary["unknown_rows"] == 1


def test_no_is_complete_but_not_feasible():
    row = _row()
    row["parking"] = "NO"
    audit = audit_field_validation(pd.DataFrame([row]))
    assert audit.normalized.loc[0, "field_validation_complete"]
    assert not audit.normalized.loc[0, "field_verified"]


def test_all_unknown_dates_fail_closed_without_dtype_error():
    rows = [_row("V1", value="UNKNOWN", date="UNKNOWN"), _row("V2", value="UNKNOWN", date="")]
    audit = audit_field_validation(pd.DataFrame(rows))

    assert audit.summary["verified"] == 0
    assert audit.summary["complete"] == 0
    assert audit.normalized["verification_date"].eq("UNKNOWN").all()
