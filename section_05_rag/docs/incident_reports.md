# Incident Reports — Project Atlas

## INC-001 — Database Connection Pool Exhaustion

Date: March 18 2024. Severity: P2. Duration: 47 minutes.
Root cause: Connection pool limit set to 10 (default) was insufficient under load testing.
Resolution: Increased pool limit to 50. Added connection pool monitoring to Datadog.
Action items: Set pool limits explicitly in all environments. Add alert at 80% pool utilisation.

## INC-002 — SAP Integration Timeout

Date: April 2 2024. Severity: P3. Duration: 2 hours.
Root cause: SAP test environment scheduled maintenance was not communicated to Atlas team.
Resolution: SAP team added Atlas contacts to maintenance notification list.
Action items: Add external dependency maintenance calendar to project monitoring.

## INC-003 — Frontend Build Failure in CI

Date: May 15 2024. Severity: P4. Duration: 3 hours.
Root cause: Node version mismatch between local dev (Node 18) and CI pipeline (Node 16).
Resolution: Pinned Node version to 20 LTS in CI configuration and in onboarding docs.
Action items: Use .nvmrc file to enforce Node version in all environments.

## INC-004 — Salesforce API Deprecation

Date: August 1 2024. Severity: P2. Ongoing.
Root cause: Salesforce deprecated the API version used by Atlas connector with 90-day notice.
Resolution in progress: Migrating to Salesforce REST API v58. Target: October 1 2024.
Estimated impact: 2-week delay to Salesforce integration delivery.