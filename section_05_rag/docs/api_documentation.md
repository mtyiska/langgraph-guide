# API Documentation — Project Atlas

## Base URL

Production: https://api.atlas.company.com/v1
Staging: https://api-staging.atlas.company.com/v1
Local: http://localhost:8000/v1

## Authentication

All endpoints require a Bearer token in the Authorization header.
Tokens are issued by the authentication service at /v1/auth/token.
Token lifetime: 8 hours. Refresh tokens valid for 30 days.

## Core Endpoints

GET /v1/reports — list all available reports with metadata.
POST /v1/reports/generate — generate a report. Body: {report_type, date_range, filters}.
GET /v1/reports/{id} — retrieve a generated report by ID.
GET /v1/datasets — list available datasets.
GET /v1/datasets/{id}/schema — retrieve the schema for a dataset.
POST /v1/data/ingest — trigger manual data ingestion. Admin only.
GET /v1/health — health check. Returns {status, version, db_connected}.

## Rate Limits

Standard users: 100 requests per minute.
Admin users: 500 requests per minute.
Report generation: 10 concurrent jobs per user.

## Error Codes

400: Invalid request parameters. 401: Authentication required. 403: Insufficient permissions.
404: Resource not found. 429: Rate limit exceeded. 500: Internal server error.