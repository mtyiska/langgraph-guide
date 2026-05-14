import uuid


class SessionManager:
    def get_or_create_session(self, session_id: str | None = None) -> str:
        if session_id:
            return session_id
        return f"session-{str(uuid.uuid4())[:8]}"

    def session_to_thread_id(self, session_id: str) -> str:
        return f"thread-{session_id}"