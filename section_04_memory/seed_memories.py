import sys
sys.path.append("..")

from memory_store import init_db, save_memory, create_profile, update_profile

TEST_USER = "user_test_001"

SEED_MEMORIES = [
    ("User's name is Alex.", "preference", 0.95),
    ("User prefers concise bullet-point responses.", "preference", 0.9),
    ("User is working on a project called Atlas.", "fact", 0.85),
    ("User is in the Pacific timezone.", "preference", 0.7),
    ("User dislikes long introductory paragraphs.", "preference", 0.8),
    ("User asked about Q1 budget on March 15 and reviewed budget_summary.txt.", "episode", 0.4),
    ("User's team includes a designer named Priya.", "fact", 0.6),
    ("User prefers Python over JavaScript for backend work.", "preference", 0.75),
    ("User mentioned the Atlas project launch is in September.", "fact", 0.7),
    ("User asked for a calculation of remaining budget on March 16.", "episode", 0.35),
]


def seed():
    init_db()
    create_profile(TEST_USER)
    update_profile(TEST_USER, "name", "Alex")
    update_profile(TEST_USER, "style", "concise")
    update_profile(TEST_USER, "timezone", "Pacific")

    for content, category, importance in SEED_MEMORIES:
        save_memory(TEST_USER, "seed_session", content, category, importance)

    print(f"Seeded {len(SEED_MEMORIES)} memories for user '{TEST_USER}'")


if __name__ == "__main__":
    seed()