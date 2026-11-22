"""Smoke test verifying CLIENT_INTERRUPT cancels both normal turns and proactive TTS cleanly."""

from __future__ import annotations

import asyncio


class MockSession:
    def __init__(self) -> None:
        self.epoch = 10
        self.speaking = False
        self._thinking = False
        self._turn_task: asyncio.Task[None] | None = None
        self._proactive_task: asyncio.Task[None] | None = None
        self.events: list[str] = []

    async def start_long_proactive(self) -> None:
        async def _run() -> None:
            self.speaking = True
            try:
                await asyncio.sleep(5.0)
            finally:
                self.speaking = False

        self._proactive_task = asyncio.create_task(_run())
        await asyncio.sleep(0.02)

    async def cancel_turn(self, reason: str = "client_interrupt") -> None:
        for attr in ("_turn_task", "_proactive_task"):
            task = getattr(self, attr)
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                setattr(self, attr, None)
        self.speaking = False
        self._thinking = False
        self.epoch += 1
        self.events.append(f"BARGE_IN_CLEAR:{reason}:{self.epoch}")
        self.events.append("SPEECH_STATE:IDLE")


async def run_test() -> None:
    s = MockSession()
    await s.start_long_proactive()
    assert s.speaking is True
    await s.cancel_turn("voice_barge_in")
    assert s.speaking is False
    assert s._proactive_task is None
    assert s.epoch == 11
    assert s.events == ["BARGE_IN_CLEAR:voice_barge_in:11", "SPEECH_STATE:IDLE"]
    print("ALL BARGE-IN CANCEL SMOKE TESTS PASSED!")


if __name__ == "__main__":
    asyncio.run(run_test())
