# LAND-JEPA — Security Specification

## Overview

Security is implemented in layers: authentication, authorization, input validation,
file handling, data protection, audit logging, and operational security.

---

## Authentication

### Method
- JWT (JSON Web Tokens) using `python-jose` with RS256 or HS256 (configurable)
- Access token: short-lived (default 30 minutes, configurable via `ACCESS_TOKEN_EXPIRE_MINUTES`)
- Refresh token: longer-lived (default 7 days, configurable)
- Tokens stored in: HttpOnly cookies (web) or SecureStore (mobile)

### Endpoints
- `POST /api/v1/auth/login` — rate limited, brute-force protected
- `POST /api/v1/auth/refresh` — requires valid refresh token
- `POST /api/v1/auth/logout` — invalidates refresh token

### Password Requirements
- Minimum 8 characters
- Hashed with bcrypt (cost factor configurable, default 12)
- Plaintext passwords never stored or logged

---

## Authorization (RBAC)

| Role          | Permissions                                                         |
|---------------|---------------------------------------------------------------------|
| admin         | All operations including user management and system config         |
| authority     | Read all risk/alert data, manage alerts, review reports, view priorities |
| field_officer | Submit field reports, view assigned zones, view risk data          |
| citizen       | Submit citizen reports, view public risk data                      |
| (unauthenticated) | Read-only public endpoints only                                 |

All protected endpoints check role membership via a FastAPI dependency.
Role checks are explicit, not implicit — each endpoint declares required roles.

---

## Input Validation

### API Inputs
- All request bodies validated via Pydantic schemas
- All path and query parameters validated via FastAPI type annotations
- SQL injection prevention: ORM-only (SQLAlchemy), no raw SQL with user input
- Coordinate validation: latitude ∈ [-90, 90], longitude ∈ [-180, 180]
- Timestamp validation: ISO 8601 format required
- String field limits enforced (max lengths)

### File Uploads
| Field         | Allowed types              | Max size  |
|---------------|----------------------------|-----------|
| Photos        | image/jpeg, image/png, image/webp | 10 MB each |
| Video         | video/mp4                  | 50 MB     |

- File type validated by MIME type inspection (not just extension)
- Files stored outside the web root
- Filenames sanitized (UUID-based storage names)
- Total concurrent uploads rate-limited

---

## Secrets Management

- All secrets via environment variables (no hardcoded values anywhere)
- `.env` file used locally (never committed)
- `.env.example` provided with placeholder values only
- Docker Compose uses environment variable injection
- Production: secrets manager (AWS Secrets Manager / GCP Secret Manager / HashiCorp Vault)

**Never commit**:
- Database passwords
- JWT secret keys
- API keys for external services
- Private SSL certificates

---

## CORS Configuration

```python
allowed_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
# Production: explicitly list allowed domains, no wildcard
```

---

## Rate Limiting

| Endpoint / Group     | Limit                   | Implementation |
|----------------------|-------------------------|----------------|
| POST /auth/login     | 10 req/min per IP       | slowapi         |
| POST /reports        | 20 req/min per user     | slowapi         |
| POST /model/predict  | 60 req/min per user     | slowapi         |
| GET public endpoints | 120 req/min per IP      | slowapi         |

---

## Audit Logging

Every write operation is logged in the `audit_logs` table:
- Actor (user ID)
- Action (CREATE, UPDATE, DELETE, LOGIN, LOGOUT, REVIEW)
- Resource and resource ID
- Old and new data (JSONB, sensitive fields redacted)
- IP address
- Timestamp

Audit logs are append-only (no delete or update operations on audit_logs).

---

## Data Protection

- Geolocation data: captured only for active report submission
- Photos: stored with UUID filenames, not original names
- Reports: accessible only to authorized roles (not publicly exposed by default)
- Citizen PII (email, phone): never exposed in public API responses

---

## HTTPS / TLS

- All production traffic over HTTPS (TLS 1.2 minimum, 1.3 preferred)
- Development: HTTP acceptable on localhost only
- HSTS header in production

---

## Security Headers (Backend)

```
X-Content-Type-Options: nosniff
X-Frame-Options: DENY
Content-Security-Policy: (configured per deployment)
Referrer-Policy: no-referrer
```

---

## Demo Mode Security Notes

- Demo alerts must NEVER be sent to real phone numbers or email addresses
- Demo data must NEVER appear in production API responses without `is_demo=true` flag
- Demo mode is disabled by default in production (`DEMO_MODE=false` in production .env)
- Demo toggle in admin UI is audit-logged

---

## Known Limitations

1. Mobile token storage uses Expo SecureStore (platform-protected, best available on mobile).
2. Server-side token revocation requires a token denylist (Redis recommended for production).
3. File uploads to local disk in development; S3/GCS required for production HA.
4. Rate limiting is in-process (slowapi); Redis-backed rate limiting required for multi-instance production.

---

## Security Checklist

- [ ] All secrets in environment variables
- [ ] `.env` never committed
- [ ] Passwords bcrypt-hashed
- [ ] JWT expiry configured
- [ ] File type validation on uploads
- [ ] File size limits enforced
- [ ] SQL injection: ORM-only
- [ ] CORS configured
- [ ] Rate limiting on sensitive endpoints
- [ ] Audit logging enabled
- [ ] Demo mode isolated
- [ ] No PII in logs
- [ ] HTTPS in production
