# knowledge_base.py
import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "knowledge_base.db")

SEED_DATA = [
    ("team", "Sarah Chen is the Project Manager for Project Orion. Email: s.chen@company.com"),
    ("team", "Dave Okafor is the Lead Engineer. He completed the CI/CD pipeline on April 3, 2024."),
    ("team", "Priya Nair is the Senior Designer. She is currently out sick."),
    ("team", "Marcus Webb is the QA Lead. He is writing the test plan."),
    ("team", "Jordan Kim is a Backend Engineer joining on May 1, 2024. Specializes in payment integrations."),
    ("team", "Alex Torres is a Frontend Engineer joining on May 1, 2024. React specialist."),
    ("project", "Project Orion launch target is Q3 2024, specifically September 30, 2024."),
    ("project", "Project Orion Phase 1 budget is $180,000 total."),
    ("project", "Project Orion uses Python backend, React frontend, and PostgreSQL database."),
    ("project", "The internal demo for Project Orion is scheduled for May 3, 2024."),
    ("project", "The stakeholder review is scheduled for April 12, 2024."),
    ("project", "Project Orion backend API is 40% complete as of April 8 — 8 of 20 endpoints done."),
    ("project", "The frontend component library is 25% complete as of April 8, 2024."),
    ("project", "The database schema has 14 tables and 3 junction tables."),
    ("budget", "Total approved budget for Project Orion is $180,000."),
    ("budget", "Amount spent to date as of April 8, 2024 is $42,000 (23% of budget)."),
    ("budget", "Remaining budget is $138,000 as of April 8, 2024."),
    ("budget", "Projected spend by Q2 end is $95,000."),
    ("budget", "Personnel costs total $121,500 for Q1 2024."),
    ("budget", "Infrastructure and tools budget is $17,000. Includes AWS, SaaS tools, PaymentCo API."),
    ("budget", "Contingency reserve is $15,000."),
    ("budget", "PaymentCo API licensing costs $12,000."),
    ("decision", "Weekly standups are every Monday at 9am PST."),
    ("decision", "PaymentCo API sandbox access was requested March 20, 2024 and is still pending."),
    ("decision", "Project status as of April 8 is YELLOW — PaymentCo blocker and Priya out sick."),
    ("decision", "CI/CD pipeline was completed 12 days ahead of schedule."),
]


def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS knowledge (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            category TEXT NOT NULL,
            content TEXT NOT NULL UNIQUE
        )
    """)
    # INSERT OR IGNORE makes this idempotent — safe to run multiple times
    cursor.executemany(
        "INSERT OR IGNORE INTO knowledge (category, content) VALUES (?, ?)",
        SEED_DATA
    )
    conn.commit()
    conn.close()
    print(f"Knowledge base ready: {len(SEED_DATA)} entries seeded")


if __name__ == "__main__":
    init_db()