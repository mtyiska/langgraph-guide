# Architecture Decision Records — Project Atlas

## ADR-001 Database Selection

Date: January 15 2024. Decision: PostgreSQL 16.
Rationale: Superior analytical query performance, native JSON support, team expertise.
Rejected alternatives: MySQL (weaker analytical features), MongoDB (consistency concerns).

## ADR-002 API Framework

Date: January 22 2024. Decision: FastAPI with Python 3.11.
Rationale: Native async support, automatic OpenAPI docs, 40% faster than Flask in benchmarks.
Rejected alternatives: Django REST Framework (too much overhead), Flask (manual everything).

## ADR-003 Frontend Framework

Date: February 1 2024. Decision: React 18 with TypeScript.
Rationale: Largest talent pool, strong ecosystem, TypeScript eliminates class of runtime errors.
Rejected alternatives: Vue 3 (smaller ecosystem), Angular (too opinionated for this team).

## ADR-004 Deployment Platform

Date: February 10 2024. Decision: AWS ECS with Fargate.
Rationale: No cluster management overhead, scales to zero, integrates with existing AWS setup.
Rejected alternatives: Self-managed Kubernetes (too much ops burden), Heroku (cost at scale).

## ADR-005 Caching Layer

Date: March 5 2024. Decision: Redis 7.2 for API response caching.
Rationale: Sub-millisecond latency, native data structures fit caching patterns, team familiarity.
Cache TTL set to 15 minutes for dashboard data, 1 hour for static reference data.