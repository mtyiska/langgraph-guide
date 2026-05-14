# System Architecture Overview — Project Atlas

## High Level Architecture

Project Atlas uses a three-tier architecture: data layer, API layer, presentation layer.
All tiers deploy to AWS using ECS Fargate containers managed through Terraform.

## Data Layer

Primary datastore: PostgreSQL 16 on AWS RDS. Multi-AZ for high availability.
Schema: 31 tables, 8 junction tables. All migrations managed with Alembic.
Caching: Redis 7.2 on AWS ElastiCache for API response caching.
Object storage: AWS S3 for generated report files and raw data lake.

## API Layer

Framework: FastAPI with Python 3.11. Async throughout using asyncio and SQLAlchemy async.
Authentication: JWT tokens issued by internal auth service. Tokens expire after 8 hours.
API versioning: URL-based versioning (/v1/). Breaking changes require new version.
Documentation: Auto-generated OpenAPI docs via FastAPI at /docs endpoint.

## Presentation Layer

Framework: React 18 with TypeScript. Component library built in-house using Storybook.
State management: React Query for server state, Zustand for client state.
Build tool: Vite 5. Bundle size target: under 200KB gzipped for initial load.

## Infrastructure

CI/CD: GitHub Actions for build and test. ArgoCD for deployment to ECS.
Monitoring: Datadog for metrics, logs, and APM. PagerDuty for alerting.
Secrets management: AWS Secrets Manager. No secrets in environment variables or code.