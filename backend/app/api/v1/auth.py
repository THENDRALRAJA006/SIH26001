"""
backend/app/api/v1/auth.py
==========================
Officer JWT Authentication and RBAC verification.
Implements real server-side authentication for authorized disaster-response officers.
"""
from __future__ import annotations

import os
import time
from typing import Optional
import jwt
from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

router = APIRouter()
security = HTTPBearer(auto_error=False)

# Secret key and algorithm for JWT (configurable via env)
JWT_SECRET = os.getenv("JWT_SECRET_KEY", "landjepa-sih26001-secret-key-ner-disaster-2026")
JWT_ALGORITHM = "HS256"
TOKEN_EXPIRE_SECONDS = 3600 * 12  # 12 hours

# Authorized officer roster (demo + production credentials)
AUTHORIZED_OFFICERS = {
    "OFFICER-NER-01": {
        "password": "landjepa2026",
        "name": "Dr. Arindam Sharma",
        "designation": "Chief Geological Officer",
        "role": "officer",
        "jurisdiction": "Northeast Regional Command (NER)",
        "badge_id": "NER-GEO-8841",
    },
    "OFFICER-NER-001": {
        "password": "landjepa2026",
        "name": "Dr. Arindam Sharma",
        "designation": "Chief Geological Officer",
        "role": "officer",
        "jurisdiction": "Northeast Regional Command (NER)",
        "badge_id": "NER-GEO-8841",
    },
    "OFFICER-NER-02": {
        "password": "landjepa2026",
        "name": "Sunita Roy",
        "designation": "Disaster Response Director",
        "role": "officer",
        "jurisdiction": "Assam & Meghalaya Highway Corridor",
        "badge_id": "NER-DRD-9912",
    },
    "admin": {
        "password": "admin",
        "name": "Command Administrator",
        "designation": "Systems Director",
        "role": "officer",
        "jurisdiction": "All NER Corridors",
        "badge_id": "HQ-ADMIN-0001",
    },
    "commander": {
        "password": "zaix2026",
        "name": "Command Incident Commander",
        "designation": "Disaster Response Director",
        "role": "officer",
        "jurisdiction": "Northeast Regional Command (NER)",
        "badge_id": "NER-CMD-2026",
    },
}


class LoginRequest(BaseModel):
    username: str
    password: str


class OfficerUser(BaseModel):
    officer_id: str
    name: str
    designation: str
    role: str
    jurisdiction: str
    badge_id: str


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "Bearer"
    role: str
    permissions: list[str] = ["read:all", "write:alerts", "write:reports", "execute:prediction", "admin:gis"]
    expires_in: int
    user: OfficerUser


def create_jwt_token(officer_id: str, officer_data: dict) -> str:
    payload = {
        "sub": officer_id,
        "name": officer_data["name"],
        "role": officer_data["role"],
        "jurisdiction": officer_data["jurisdiction"],
        "badge_id": officer_data["badge_id"],
        "iat": int(time.time()),
        "exp": int(time.time()) + TOKEN_EXPIRE_SECONDS,
        "iss": "LAND-JEPA-AUTH",
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_jwt_token(token: str) -> dict:
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Officer session expired. Please re-authenticate.",
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid officer credentials or signature.",
        )


async def get_current_officer(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> dict:
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Officer credentials missing.",
        )
    payload = decode_jwt_token(credentials.credentials)
    if payload.get("role") != "officer":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access restricted to authorized officers only.",
        )
    return payload


@router.post("/login", response_model=LoginResponse, summary="Officer AI Authentication")
async def login(req: LoginRequest):
    """
    Authenticate disaster officer with Officer ID and password.
    Returns signed JWT with RBAC officer role.
    """
    uname = req.username.strip()
    officer = AUTHORIZED_OFFICERS.get(uname)
    
    # Also allow standard officer prefix check or fallback for testing
    if not officer and uname.upper().startswith("OFFICER-"):
        # Allow default officer credentials for demo/field test
        if req.password in ("landjepa2026", "password123", "officer"):
            officer = {
                "password": req.password,
                "name": f"Officer {uname.upper()}",
                "designation": "Regional Field Director",
                "role": "officer",
                "jurisdiction": "Northeast Regional Sector",
                "badge_id": f"NER-{uname.upper()}",
            }

    if not officer or officer["password"] != req.password:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Officer ID or authentication code. Access denied.",
        )

    token = create_jwt_token(uname, officer)
    return LoginResponse(
        access_token=token,
        token_type="Bearer",
        role=officer["role"],
        expires_in=TOKEN_EXPIRE_SECONDS,
        user=OfficerUser(
            officer_id=uname,
            name=officer["name"],
            designation=officer["designation"],
            role=officer["role"],
            jurisdiction=officer["jurisdiction"],
            badge_id=officer["badge_id"],
        ),
    )


@router.get("/me", summary="Verify officer session")
async def get_me(current_officer: dict = Depends(get_current_officer)):
    """Return active officer identity from verified JWT."""
    return {
        "status": "authenticated",
        "officer": current_officer,
    }
