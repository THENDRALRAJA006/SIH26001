"""Initial schema — all 22 LAND-JEPA tables

Revision ID: 0001
Revises:
Create Date: 2026-09-04

Creates:
  users, roles, user_roles, locations, zones, terrain,
  rainfall, weather, soil_moisture,
  landslide_events, insar_observations,
  roads, villages, infrastructure,
  citizen_reports, field_reports,
  model_runs, risk_predictions, risk_factors,
  priority_scores, alerts, alert_deliveries,
  sync_queue, audit_logs
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from geoalchemy2 import Geometry
from sqlalchemy.dialects import postgresql

revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── Extensions (idempotent) ─────────────────────────────────────
    op.execute("CREATE EXTENSION IF NOT EXISTS postgis")
    op.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"')

    # ── roles ────────────────────────────────────────────────────────
    op.create_table(
        "roles",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("name", sa.String(50), nullable=False),
        sa.Column("description", sa.String(255), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name"),
    )

    # ── users ────────────────────────────────────────────────────────
    op.create_table(
        "users",
        sa.Column("id", postgresql.UUID(as_uuid=True), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("email", sa.String(255), nullable=False),
        sa.Column("phone", sa.String(20), nullable=True),
        sa.Column("full_name", sa.String(255), nullable=False),
        sa.Column("password_hash", sa.String(255), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("is_verified", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("preferred_lang", sa.String(10), nullable=False, server_default="en"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("email"),
    )
    op.create_index("idx_users_email", "users", ["email"])

    # ── user_roles ───────────────────────────────────────────────────
    op.create_table(
        "user_roles",
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("role_id", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["role_id"], ["roles.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("user_id", "role_id"),
    )

    # ── zones ────────────────────────────────────────────────────────
    op.create_table(
        "zones",
        sa.Column("id", postgresql.UUID(as_uuid=True), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("code", sa.String(50), nullable=False),
        sa.Column("district", sa.String(100), nullable=True),
        sa.Column("state", sa.String(100), nullable=False, server_default="Assam"),
        sa.Column("geom", Geometry(geometry_type="MULTIPOLYGON", srid=4326), nullable=False),
        sa.Column("area_km2", sa.Float(), nullable=True),
        sa.Column("population_est", sa.Integer(), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("code"),
    )
    op.create_index("idx_zones_geom", "zones", ["geom"], postgresql_using="gist")
    op.create_index("idx_zones_code", "zones", ["code"])

    # ── terrain ──────────────────────────────────────────────────────
    op.create_table(
        "terrain",
        sa.Column("id", postgresql.UUID(as_uuid=True), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("zone_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("elevation_m", sa.Float(), nullable=True),
        sa.Column("slope_deg", sa.Float(), nullable=True),
        sa.Column("aspect_deg", sa.Float(), nullable=True),
        sa.Column("curvature", sa.Float(), nullable=True),
        sa.Column("tpi", sa.Float(), nullable=True),
        sa.Column("twi", sa.Float(), nullable=True),
        sa.Column("lithology_class", sa.String(100), nullable=True),
        sa.Column("land_cover", sa.String(100), nullable=True),
        sa.Column("data_source", sa.String(100), nullable=False),
        sa.Column("data_date", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["zone_id"], ["zones.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("zone_id"),
    )

    # ── rainfall ─────────────────────────────────────────────────────
    op.create_table(
        "rainfall",
        sa.Column("id", postgresql.UUID(as_uuid=True), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("zone_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("observed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("precipitation_mm", sa.Float(), nullable=False),
        sa.Column("data_source", sa.String(100), nullable=False),
        sa.Column("is_demo", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("quality_flag", sa.String(20), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["zone_id"], ["zones.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_rainfall_zone_time", "rainfall", ["zone_id", sa.text("observed_at DESC")])
    op.create_index("idx_rainfall_time", "rainfall", [sa.text("observed_at DESC")])

    # ── weather ──────────────────────────────────────────────────────
    op.create_table(
        "weather",
        sa.Column("id", postgresql.UUID(as_uuid=True), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("zone_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("observed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("temperature_c", sa.Float(), nullable=True),
        sa.Column("humidity_pct", sa.Float(), nullable=True),
        sa.Column("wind_speed_ms", sa.Float(), nullable=True),
        sa.Column("wind_dir_deg", sa.Float(), nullable=True),
        sa.Column("pressure_hpa", sa.Float(), nullable=True),
        sa.Column("data_source", sa.String(100), nullable=False),
        sa.Column("is_demo", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("quality_flag", sa.String(20), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["zone_id"], ["zones.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_weather_zone_time", "weather", ["zone_id", sa.text("observed_at DESC")])

    # ── soil_moisture ────────────────────────────────────────────────
    op.create_table(
        "soil_moisture",
        sa.Column("id", postgresql.UUID(as_uuid=True), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("zone_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("observed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("sm_volumetric", sa.Float(), nullable=True),
        sa.Column("sm_anomaly", sa.Float(), nullable=True),
        sa.Column("depth_cm", sa.Float(), nullable=True),
        sa.Column("data_source", sa.String(100), nullable=False),
        sa.Column("is_demo", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("quality_flag", sa.String(20), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["zone_id"], ["zones.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_sm_zone_time", "soil_moisture", ["zone_id", sa.text("observed_at DESC")])

    # ── landslide_events ─────────────────────────────────────────────
    op.create_table(
        "landslide_events",
        sa.Column("id", postgresql.UUID(as_uuid=True), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("zone_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("location", Geometry(geometry_type="POINT", srid=4326), nullable=True),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("date_precision", sa.String(20), nullable=False, server_default="unknown"),
        sa.Column("event_type", sa.String(100), nullable=True),
        sa.Column("magnitude", sa.String(50), nullable=True),
        sa.Column("casualties", sa.Integer(), nullable=True),
        sa.Column("source", sa.String(100), nullable=False),
        sa.Column("is_demo", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["zone_id"], ["zones.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_ls_events_zone", "landslide_events", ["zone_id"])
    op.create_index("idx_ls_events_location", "landslide_events", ["location"], postgresql_using="gist")
    op.create_index("idx_ls_events_time", "landslide_events", [sa.text("occurred_at DESC")])

    # ── insar_observations ───────────────────────────────────────────
    op.create_table(
        "insar_observations",
        sa.Column("id", postgresql.UUID(as_uuid=True), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("zone_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("acquired_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("deformation_mm", sa.Float(), nullable=True),
        sa.Column("deformation_std_mm", sa.Float(), nullable=True),
        sa.Column("coherence", sa.Float(), nullable=True),
        sa.Column("pass_direction", sa.String(20), nullable=True),
        sa.Column("track_number", sa.Integer(), nullable=True),
        sa.Column("data_source", sa.String(100), nullable=False),
        sa.Column("is_demo", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["zone_id"], ["zones.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_insar_zone_time", "insar_observations", ["zone_id", sa.text("acquired_at DESC")])

    # ── roads ────────────────────────────────────────────────────────
    op.create_table(
        "roads",
        sa.Column("id", postgresql.UUID(as_uuid=True), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("name", sa.String(255), nullable=True),
        sa.Column("road_class", sa.String(20), nullable=True),
        sa.Column("criticality", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("geom", Geometry(geometry_type="LINESTRING", srid=4326), nullable=False),
        sa.Column("zone_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("data_source", sa.String(100), nullable=False, server_default="OSM"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.CheckConstraint("criticality BETWEEN 1 AND 5", name="ck_roads_criticality"),
        sa.ForeignKeyConstraint(["zone_id"], ["zones.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_roads_geom", "roads", ["geom"], postgresql_using="gist")

    # ── villages ─────────────────────────────────────────────────────
    op.create_table(
        "villages",
        sa.Column("id", postgresql.UUID(as_uuid=True), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("population_est", sa.Integer(), nullable=True),
        sa.Column("zone_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("location", Geometry(geometry_type="POINT", srid=4326), nullable=True),
        sa.Column("district", sa.String(100), nullable=True),
        sa.Column("state", sa.String(100), nullable=False, server_default="Assam"),
        sa.Column("data_source", sa.String(100), nullable=False, server_default="Census2011"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["zone_id"], ["zones.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_villages_geom", "villages", ["location"], postgresql_using="gist")
    op.create_index("idx_villages_zone", "villages", ["zone_id"])

    # ── infrastructure ───────────────────────────────────────────────
    op.create_table(
        "infrastructure",
        sa.Column("id", postgresql.UUID(as_uuid=True), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("name", sa.String(255), nullable=True),
        sa.Column("infra_type", sa.String(50), nullable=False),
        sa.Column("criticality", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("zone_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("location", Geometry(geometry_type="POINT", srid=4326), nullable=True),
        sa.Column("data_source", sa.String(100), nullable=False, server_default="OSM"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.CheckConstraint("criticality BETWEEN 1 AND 5", name="ck_infra_criticality"),
        sa.ForeignKeyConstraint(["zone_id"], ["zones.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_infra_geom", "infrastructure", ["location"], postgresql_using="gist")

    # ── citizen_reports ──────────────────────────────────────────────
    op.create_table(
        "citizen_reports",
        sa.Column("id", postgresql.UUID(as_uuid=True), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("reporter_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("zone_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("location", Geometry(geometry_type="POINT", srid=4326), nullable=True),
        sa.Column("reported_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("category", sa.String(50), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("photo_urls", postgresql.ARRAY(sa.String()), nullable=True),
        sa.Column("video_url", sa.String(500), nullable=True),
        sa.Column("review_status", sa.String(20), nullable=False, server_default="pending"),
        sa.Column("reviewed_by", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("reviewed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("review_notes", sa.Text(), nullable=True),
        sa.Column("is_demo", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["reporter_id"], ["users.id"]),
        sa.ForeignKeyConstraint(["reviewed_by"], ["users.id"]),
        sa.ForeignKeyConstraint(["zone_id"], ["zones.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_citizen_reports_zone", "citizen_reports", ["zone_id"])
    op.create_index("idx_citizen_reports_geom", "citizen_reports", ["location"], postgresql_using="gist")
    op.create_index("idx_citizen_reports_status", "citizen_reports", ["review_status"])

    # ── field_reports ────────────────────────────────────────────────
    op.create_table(
        "field_reports",
        sa.Column("id", postgresql.UUID(as_uuid=True), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("officer_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("zone_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("location", Geometry(geometry_type="POINT", srid=4326), nullable=True),
        sa.Column("reported_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("observation", sa.Text(), nullable=False),
        sa.Column("category", sa.String(50), nullable=True),
        sa.Column("severity", sa.String(20), nullable=True),
        sa.Column("photo_urls", postgresql.ARRAY(sa.String()), nullable=True),
        sa.Column("is_demo", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["officer_id"], ["users.id"]),
        sa.ForeignKeyConstraint(["zone_id"], ["zones.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_field_reports_zone", "field_reports", ["zone_id"])

    # ── model_runs ───────────────────────────────────────────────────
    op.create_table(
        "model_runs",
        sa.Column("id", postgresql.UUID(as_uuid=True), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("run_type", sa.String(50), nullable=False),
        sa.Column("run_status", sa.String(20), nullable=False, server_default="running"),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("config_snapshot", postgresql.JSONB(), nullable=True),
        sa.Column("metrics", postgresql.JSONB(), nullable=True),
        sa.Column("model_version", sa.String(100), nullable=True),
        sa.Column("checkpoint_path", sa.String(500), nullable=True),
        sa.Column("git_commit", sa.String(40), nullable=True),
        sa.Column("seed", sa.Integer(), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_model_runs_type", "model_runs", ["run_type", sa.text("started_at DESC")])

    # ── risk_predictions ─────────────────────────────────────────────
    op.create_table(
        "risk_predictions",
        sa.Column("id", postgresql.UUID(as_uuid=True), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("zone_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("model_run_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("predicted_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("valid_for", sa.DateTime(timezone=True), nullable=False),
        sa.Column("horizon", sa.Integer(), nullable=False),
        sa.Column("risk_score", sa.Float(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("risk_level", sa.String(10), nullable=False),
        sa.Column("model_version", sa.String(100), nullable=False),
        sa.Column("is_demo", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.CheckConstraint("risk_score BETWEEN 0 AND 1", name="ck_risk_score"),
        sa.CheckConstraint("confidence BETWEEN 0 AND 1", name="ck_confidence"),
        sa.ForeignKeyConstraint(["zone_id"], ["zones.id"]),
        sa.ForeignKeyConstraint(["model_run_id"], ["model_runs.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_risk_zone_time", "risk_predictions", ["zone_id", sa.text("predicted_at DESC")])
    op.create_index("idx_risk_horizon", "risk_predictions", ["horizon", sa.text("predicted_at DESC")])

    # ── risk_factors ─────────────────────────────────────────────────
    op.create_table(
        "risk_factors",
        sa.Column("id", postgresql.UUID(as_uuid=True), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("prediction_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("factor_name", sa.String(100), nullable=False),
        sa.Column("shap_value", sa.Float(), nullable=True),
        sa.Column("direction", sa.String(20), nullable=True),
        sa.Column("rank", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["prediction_id"], ["risk_predictions.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_risk_factors_pred", "risk_factors", ["prediction_id"])

    # ── priority_scores ──────────────────────────────────────────────
    op.create_table(
        "priority_scores",
        sa.Column("id", postgresql.UUID(as_uuid=True), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("zone_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("prediction_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("scored_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("priority_level", sa.Integer(), nullable=False),
        sa.Column("risk_score", sa.Float(), nullable=True),
        sa.Column("population_exposure", sa.Float(), nullable=True),
        sa.Column("road_criticality_score", sa.Float(), nullable=True),
        sa.Column("infra_criticality_score", sa.Float(), nullable=True),
        sa.Column("accessibility_score", sa.Float(), nullable=True),
        sa.Column("composite_score", sa.Float(), nullable=False),
        sa.Column("explanation", sa.Text(), nullable=True),
        sa.Column("is_demo", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.CheckConstraint("priority_level BETWEEN 1 AND 3", name="ck_priority_level"),
        sa.ForeignKeyConstraint(["zone_id"], ["zones.id"]),
        sa.ForeignKeyConstraint(["prediction_id"], ["risk_predictions.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_priority_zone", "priority_scores", ["zone_id", sa.text("scored_at DESC")])

    # ── alerts ───────────────────────────────────────────────────────
    op.create_table(
        "alerts",
        sa.Column("id", postgresql.UUID(as_uuid=True), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("zone_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("prediction_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("alert_type", sa.String(50), nullable=False),
        sa.Column("severity", sa.String(20), nullable=False),
        sa.Column("title_en", sa.String(500), nullable=False),
        sa.Column("body_en", sa.Text(), nullable=False),
        sa.Column("title_hi", sa.String(500), nullable=True),
        sa.Column("body_hi", sa.Text(), nullable=True),
        sa.Column("triggered_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("is_demo", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("demo_note", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["zone_id"], ["zones.id"]),
        sa.ForeignKeyConstraint(["prediction_id"], ["risk_predictions.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_alerts_zone", "alerts", ["zone_id", sa.text("triggered_at DESC")])
    op.create_index("idx_alerts_demo", "alerts", ["is_demo"])

    # ── alert_deliveries ─────────────────────────────────────────────
    op.create_table(
        "alert_deliveries",
        sa.Column("id", postgresql.UUID(as_uuid=True), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("alert_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("channel", sa.String(20), nullable=False),
        sa.Column("recipient_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("recipient_ref", sa.String(255), nullable=True),
        sa.Column("status", sa.String(20), nullable=False, server_default="pending"),
        sa.Column("sent_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("is_simulated", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["alert_id"], ["alerts.id"]),
        sa.ForeignKeyConstraint(["recipient_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_deliveries_alert", "alert_deliveries", ["alert_id"])
    op.create_index("idx_deliveries_status", "alert_deliveries", ["status"])

    # ── sync_queue ───────────────────────────────────────────────────
    op.create_table(
        "sync_queue",
        sa.Column("id", postgresql.UUID(as_uuid=True), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("entity_type", sa.String(50), nullable=False),
        sa.Column("entity_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("device_id", sa.String(255), nullable=True),
        sa.Column("payload_json", postgresql.JSONB(), nullable=False),
        sa.Column("sync_status", sa.String(20), nullable=False, server_default="pending"),
        sa.Column("attempts", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("last_attempted", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_sync_status", "sync_queue", ["sync_status", sa.text("created_at")])
    op.create_index("idx_sync_entity", "sync_queue", ["entity_type", "entity_id"])

    # ── audit_logs ───────────────────────────────────────────────────
    op.create_table(
        "audit_logs",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("actor_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("action", sa.String(50), nullable=False),
        sa.Column("resource", sa.String(100), nullable=False),
        sa.Column("resource_id", sa.String(255), nullable=True),
        sa.Column("old_data", postgresql.JSONB(), nullable=True),
        sa.Column("new_data", postgresql.JSONB(), nullable=True),
        sa.Column("ip_address", sa.String(45), nullable=True),
        sa.Column("user_agent", sa.Text(), nullable=True),
        sa.Column("occurred_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["actor_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_audit_actor", "audit_logs", ["actor_id", sa.text("occurred_at DESC")])
    op.create_index("idx_audit_resource", "audit_logs", ["resource", sa.text("occurred_at DESC")])

    # ── Seed default roles ────────────────────────────────────────────
    op.execute("""
        INSERT INTO roles (name, description) VALUES
            ('admin',        'Full system access'),
            ('authority',    'Disaster management authority'),
            ('field_officer','Field officer with assigned zones'),
            ('citizen',      'General public user')
        ON CONFLICT (name) DO NOTHING
    """)


def downgrade() -> None:
    # Drop in reverse dependency order
    for table in [
        "audit_logs", "sync_queue", "alert_deliveries", "alerts",
        "priority_scores", "risk_factors", "risk_predictions", "model_runs",
        "field_reports", "citizen_reports", "infrastructure", "villages", "roads",
        "insar_observations", "landslide_events",
        "soil_moisture", "weather", "rainfall", "terrain",
        "zones", "user_roles", "users", "roles",
    ]:
        op.drop_table(table)
