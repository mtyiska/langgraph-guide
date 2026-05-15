# Project Atlas — Technical Decisions

## ADR-001: Database Selection
Decision: PostgreSQL 15
Rationale: Team familiarity, JSONB support for flexible schemas, strong tooling.
Alternatives considered: MySQL 8, MongoDB.
Status: Implemented.

## ADR-002: API Framework
Decision: FastAPI (Python 3.11)
Rationale: Async support, automatic OpenAPI docs, Pydantic integration.
Alternatives considered: Django REST Framework, Flask.
Status: Implemented.

## ADR-003: Frontend Framework
Decision: React 18 with TypeScript
Rationale: Component reusability, type safety, team experience.
Alternatives considered: Vue 3, SvelteKit.
Status: Implemented.

## ADR-004: Deployment
Decision: Docker Compose for dev, Kubernetes for production.
Infrastructure provider: AWS (us-west-2 region).
Status: Dev complete, prod in progress.

## ADR-005: Authentication
Decision: OAuth 2.0 with Auth0.
Status: Planned for Sprint 15.
