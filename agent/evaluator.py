from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from agent.state import AgentState, ToolExecution
from model.messages import Message
from model.ollama_client import OllamaClient


class NextAction(str, Enum):
    """
    Controlled actions the agent loop can take after evaluation.

    The evaluator selects the next phase.
    The AgentLoop is responsible for executing that phase.
    """

    INVESTIGATE = "investigate"
    SYNTHESIZE = "synthesize"
    MODIFY = "modify"
    VERIFY = "verify"
    FINISH = "finish"


@dataclass
class Evaluation:
    """
    Result of evaluating the agent's current progress.

    The evaluator distinguishes between:

    - evidence_sufficient:
        Whether the agent has enough evidence to answer or act
        on the user's request.

    - task_complete:
        Whether there is nothing meaningful remaining to do.

    - goal_complete:
        Whether the user's actual objective has been accomplished.

    - next_action:
        The controlled phase the agent should enter next.
    """

    evidence_sufficient: bool
    task_complete: bool
    goal_complete: bool
    reason: str
    next_action: NextAction


class Evaluator:
    """
    Determines whether the agent has enough evidence to proceed,
    whether the user's goal has been completed, and what execution
    phase should happen next.

    The evaluator does not execute tools itself.

    It produces a controlled decision that the AgentLoop can enforce.
    """

    def __init__(
        self,
        model: OllamaClient,
    ) -> None:
        self.model = model

    def evaluate(
        self,
        state: AgentState,
        tool_execution: ToolExecution | None = None,
    ) -> Evaluation:
        """
        Evaluate the current state of the agent.
        """

        prompt = self._build_prompt(
            state=state,
            tool_execution=tool_execution,
        )

        response = self.model.chat(
            messages=[
                Message(
                    role="system",
                    content=(
                        "You are the evidence and completion evaluator "
                        "for an autonomous software engineering agent.\n\n"
                        "Your responsibility is to determine:\n"
                        "1. whether enough evidence has been gathered,\n"
                        "2. whether the user's goal is complete, and\n"
                        "3. which controlled execution phase should happen next.\n\n"
                        "Do not assume that successful tool execution "
                        "means the user's goal is complete.\n\n"
                        "Do not request additional investigation merely "
                        "because more information could theoretically "
                        "be obtained."
                    ),
                ),
                Message(
                    role="user",
                    content=prompt,
                ),
            ],
        )

        return self._parse_response(
            response.content,
        )

    def _build_prompt(
        self,
        state: AgentState,
        tool_execution: ToolExecution | None,
    ) -> str:
        """
        Build the evaluation prompt from the current agent state.
        """

        history = self._format_messages(
            state.messages,
        )

        executions = self._format_tool_executions(
            state.tool_executions,
        )

        latest_execution = "None"

        if tool_execution is not None:
            latest_execution = (
                f"Tool: {tool_execution.tool_name}\n"
                f"Reason: {tool_execution.reason_for_tool}\n"
                f"Arguments: {tool_execution.arguments}\n"
                f"Success: {tool_execution.success}\n"
                f"Result: {tool_execution.result}\n"
                f"Error: {tool_execution.error}"
            )

        return f"""
Evaluate the progress of an autonomous coding agent.

ORIGINAL USER TASK:
{state.task}

CONVERSATION HISTORY:
{history}

PREVIOUS TOOL EXECUTIONS:
{executions}

LATEST TOOL EXECUTION:
{latest_execution}

Your evaluation must determine the current execution phase.

EVIDENCE SUFFICIENCY
--------------------

Evidence is sufficient when:

- The relevant target or targets have been identified.
- The available tool results contain the information needed
  to address the user's actual request.
- Another investigation call would not materially improve
  correctness.
- The agent is not gathering information merely because
  additional information is theoretically available.

Evidence is insufficient when:

- Important information required by the user's request
  is still unknown.
- The wrong target was inspected.
- A required component has not been inspected.
- The current evidence is ambiguous, contradictory, or incomplete.
- Further investigation is materially necessary.

NEXT ACTION
-----------

You MUST choose exactly ONE of these values:

INVESTIGATE
    More evidence or inspection is materially necessary.

SYNTHESIZE
    Enough evidence exists to formulate the answer or result,
    but the final response has not yet been produced.

MODIFY
    The task requires changing code, files, configuration,
    or another project artifact, and the agent has enough
    evidence to make that change.

VERIFY
    A modification or action has occurred and verification is
    required before the goal can be considered complete.

FINISH
    The user's actual objective has been accomplished and
    no meaningful work remains.

IMPORTANT EXAMPLES
------------------

Example 1:

User task:
"What does ToolRegistry do?"

If the agent has located and inspected the implementation
of ToolRegistry and the implementation contains enough
information to explain its responsibilities:

EVIDENCE_SUFFICIENT: true
GOAL_COMPLETE: false
NEXT_ACTION: SYNTHESIZE

The agent should NOT perform another redundant inspection.

Example 2:

User task:
"Fix the bug in registry.py."

If the relevant code has been inspected and the cause
has been identified:

EVIDENCE_SUFFICIENT: true
GOAL_COMPLETE: false
NEXT_ACTION: MODIFY

Example 3:

After the modification has been made:

EVIDENCE_SUFFICIENT: true
GOAL_COMPLETE: false
NEXT_ACTION: VERIFY

Example 4:

After verification confirms the requested change works:

EVIDENCE_SUFFICIENT: true
GOAL_COMPLETE: true
NEXT_ACTION: FINISH

RULES
-----

- Do not invent work that has not happened.
- Do not select INVESTIGATE when the required evidence already exists.
- Do not select MODIFY for an informational question.
- Do not select VERIFY before a meaningful modification or action
  that requires verification.
- Do not select FINISH merely because a tool succeeded.
- If the user's objective has not been accomplished, do not select
  FINISH.
- NEXT_ACTION must contain exactly one allowed value.

Return ONLY:

EVIDENCE_SUFFICIENT: true/false
TASK_COMPLETE: true/false
GOAL_COMPLETE: true/false
REASON: <short explanation>
NEXT_ACTION: INVESTIGATE|SYNTHESIZE|MODIFY|VERIFY|FINISH
""".strip()

    def _format_messages(
        self,
        messages: list[Message],
    ) -> str:
        """
        Format conversation history for the evaluator.
        """

        if not messages:
            return "No conversation history."

        formatted: list[str] = []

        for message in messages:
            content = message.content or ""

            formatted.append(
                f"[{message.role}] {content}"
            )

        return "\n".join(formatted)

    def _format_tool_executions(
        self,
        executions: list[ToolExecution],
    ) -> str:
        """
        Format previous tool executions.
        """

        if not executions:
            return "No previous tool executions."

        formatted: list[str] = []

        for index, execution in enumerate(
            executions,
            start=1,
        ):
            formatted.append(
                (
                    f"{index}. "
                    f"Tool: {execution.tool_name}\n"
                    f"   Reason: {execution.reason_for_tool}\n"
                    f"   Arguments: {execution.arguments}\n"
                    f"   Success: {execution.success}\n"
                    f"   Result: {execution.result}\n"
                    f"   Error: {execution.error}"
                )
            )

        return "\n".join(formatted)

    def _parse_response(
        self,
        content: str,
    ) -> Evaluation:
        """
        Parse the evaluator model's response.

        Parsing is defensive because local models can occasionally
        produce additional text or inconsistent formatting.
        """

        text = content.strip()

        evidence_sufficient = self._parse_boolean(
            text,
            "EVIDENCE_SUFFICIENT",
        )

        task_complete = self._parse_boolean(
            text,
            "TASK_COMPLETE",
        )

        goal_complete = self._parse_boolean(
            text,
            "GOAL_COMPLETE",
        )

        reason = self._parse_field(
            text,
            "REASON",
            default="No reason provided.",
        )

        next_action = self._parse_next_action(
            text,
        )

        return Evaluation(
            evidence_sufficient=evidence_sufficient,
            task_complete=task_complete,
            goal_complete=goal_complete,
            reason=reason,
            next_action=next_action,
        )

    def _parse_next_action(
        self,
        text: str,
    ) -> NextAction:
        """
        Parse and normalize the controlled NEXT_ACTION value.

        Unknown values fall back to INVESTIGATE because continuing
        investigation is safer than allowing an unrecognized model
        response to trigger mutation or premature completion.
        """

        value = self._parse_field(
            text,
            "NEXT_ACTION",
            default="INVESTIGATE",
        )

        normalized = value.strip().upper()

        try:
            return NextAction(normalized.lower())
        except ValueError:
            return NextAction.INVESTIGATE

    def _parse_boolean(
        self,
        text: str,
        field_name: str,
    ) -> bool:
        """
        Parse a boolean field from the evaluator response.
        """

        value = self._parse_field(
            text,
            field_name,
            default="false",
        )

        return value.strip().lower() == "true"

    def _parse_field(
        self,
        text: str,
        field_name: str,
        default: str = "",
    ) -> str:
        """
        Extract a single field from the evaluator response.
        """

        prefix = f"{field_name}:"

        for line in text.splitlines():
            stripped = line.strip()

            if stripped.upper().startswith(
                prefix.upper()
            ):
                return stripped[
                    len(prefix):
                ].strip()

        return default