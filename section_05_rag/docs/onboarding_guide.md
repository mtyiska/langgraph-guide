# Developer Onboarding Guide — Project Atlas

## Prerequisites

Install: Python 3.11, Node 20, Docker Desktop, AWS CLI v2, PostgreSQL 16 client.
Request access: AWS console (ask Dave Okafor), Confluence (ask Sarah Chen), Linear (ask Sarah Chen).

## Local Setup

Clone the repository from github.com/company/project-atlas.
Run `make setup` to install Python dependencies and Node packages.
Run `make db-setup` to create local PostgreSQL database with seed data.
Run `make dev` to start the API on port 8000 and frontend on port 3000.
API docs available at http://localhost:8000/docs (FastAPI automatic OpenAPI).

## First Week Checklist

Day 1: Environment setup and repository orientation with your team lead.
Day 2: Read all Architecture Decision Records in technical_decisions.md.
Day 3: Complete the API tutorial in /docs/api_tutorial.md.
Day 4: Submit your first PR — fix a bug from the good-first-issue label in Linear.
Day 5: Shadow a production deployment with Dave Okafor.

## Key Contacts

Technical questions: Jordan Kim (Slack @jordan.kim).
Process questions: Sarah Chen (Slack @sarah.chen).
Infrastructure issues: Dave Okafor (Slack @dave.okafor).