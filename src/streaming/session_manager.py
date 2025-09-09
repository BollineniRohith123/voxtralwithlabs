from typing import Dict, Any
import asyncio

class SessionManager:
    def __init__(self, config):
        self.config = config
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    async def create_session(self) -> str:
        # Simple unique id
        session_id = str(len(self.active_sessions) + 1)
        async with self._lock:
            self.active_sessions[session_id] = {"config": {}}
        return session_id

    async def delete_session(self, session_id: str) -> None:
        async with self._lock:
            self.active_sessions.pop(session_id, None)

    async def update_session_config(self, session_id: str, update: Dict[str, Any]) -> None:
        async with self._lock:
            if session_id in self.active_sessions:
                self.active_sessions[session_id]["config"].update(update)
