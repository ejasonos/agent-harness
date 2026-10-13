import json

from agent.context import ContextManager
from agent.loop import AgentLoop
from agent.state import AgentState
from utils.task_journal import (
    TaskJournal,
    record_task_event,
    use_task_journal,
)


def test_task_journal_records_events_and_removes_temporary_file():
    with TaskJournal("answer two questions") as journal:
        journal_path = journal.path
        assert journal_path.exists()

        with use_task_journal(journal):
            record_task_event(
                "model_response",
                source="planner",
                content="First answer",
            )
            record_task_event(
                "tool_result",
                tool_name="read_file",
                success=True,
                result="Useful evidence",
            )

        entries = [
            json.loads(line)
            for line in journal.read_text().splitlines()
        ]
        assert [entry["event"] for entry in entries] == [
            "task",
            "model_response",
            "tool_result",
        ]
        assert entries[1]["content"] == "First answer"
        assert entries[2]["result"] == "Useful evidence"

    assert not journal_path.exists()


def test_failed_tool_retrieves_new_journal_entries_into_context():
    loop = AgentLoop.__new__(AgentLoop)
    loop.context = ContextManager()
    state = AgentState(task="answer the remaining questions")

    with TaskJournal(state.task) as journal:
        with use_task_journal(journal):
            record_task_event(
                "model_response",
                source="planner",
                content="Earlier answer remains valid",
            )
            loop._retrieve_journal_after_failure(
                state,
                "run_tests",
            )

        retrieved_message = state.messages[-1].content
        assert "run_tests" in retrieved_message
        assert "Earlier answer remains valid" in retrieved_message

        journal_path = journal.path

    assert not journal_path.exists()