from __future__ import annotations

FIELD_STATUS_COLUMNS = [
    "large_vehicle_access",
    "parking",
    "electricity",
    "toilet",
    "indoor_waiting",
    "heating_cooling",
    "barrier_free",
    "medical_equipment_space",
    "actual_facility_exists",
    "reservation_possible",
    "contact_verified",
]
FIELD_DATE_COLUMN = "verification_date"
FIELD_REQUIRED_INPUT_COLUMNS = FIELD_STATUS_COLUMNS + [FIELD_DATE_COLUMN]
FIELD_DERIVED_COLUMNS = [
    "field_validation_complete",
    "field_verified",
    "field_operationally_feasible",
]
FIELD_KEY_COLUMNS = ["venue_id", "venue_name", "venue_type", "cluster_id", "admin_code", "sigungu"]

EVIDENCE_COLUMNS = [
    "venue_id",
    "verification_source",
    "source_reference",
    "contact_person_or_office",
    "contact_channel",
    "verifier",
    "verification_notes",
    "evidence_file_or_url",
]

TEAM_BASE_REQUIRED_COLUMNS = [
    "team_id",
    "base_id",
    "base_name",
    "latitude",
    "longitude",
    "confirmed",
    "source_reference",
    "effective_date",
]

TEAM_CAPABILITY_REQUIRED_COLUMNS = ["team_id", "bundle_id", "capable", "confirmed", "source_reference"]
VEHICLE_REQUIRED_COLUMNS = [
    "vehicle_id",
    "team_id",
    "base_id",
    "vehicle_type",
    "confirmed",
    "max_daily_drive_minutes",
    "max_daily_work_minutes",
    "source_reference",
    "effective_date",
]
CALENDAR_REQUIRED_COLUMNS = ["resource_type", "resource_id", "date", "available", "source_reference"]
TRAVEL_REQUIRED_COLUMNS = [
    "team_id",
    "venue_id",
    "travel_minutes",
    "routing_method",
    "confirmed",
    "source_reference",
    "effective_date",
]

RELEASE_DECISIONS = {
    "PASS_STAGE4_2_COMPUTATIONAL_HARDENING",
    "PASS_STAGE4_2_READY_FOR_FIELD_VALIDATION",
    "PASS_STAGE4_2_READY_FOR_OPERATIONAL_FINAL",
    "PASS_STAGE4_OPERATIONAL_FINAL",
    "FAIL_STAGE4_2",
}
