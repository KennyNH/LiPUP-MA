"""Generic stage-level ReAct agent with strict tagged output."""

from __future__ import annotations

import ast
import base64
import html
import json
import mimetypes
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .models import PhaseTrace, StageRunResult


class StageAgentRuntimeError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        stage_name: str,
        trace: List[PhaseTrace],
        full_trace: List[Dict[str, Any]],
        raw_last_response: str,
        counted_steps_used: int,
        total_attempts: int,
    ):
        super().__init__(message)
        self.stage_name = stage_name
        self.trace = list(trace)
        self.full_trace = list(full_trace)
        self.raw_last_response = raw_last_response
        self.counted_steps_used = int(counted_steps_used)
        self.total_attempts = int(total_attempts)


class StageReActAgent:
    MAX_SAME_STEP_RETRIES = 5

    def __init__(
        self,
        stage_name: str,
        client: Any,
        toolkit: Any,
        system_prompt: str,
        user_prompt: str,
        allowed_tools: List[str],
        max_steps: int = 6,
        initial_image_path: Optional[Path] = None,
        result_validator: Optional[Callable[[Dict[str, Any]], Optional[str]]] = None,
        thought_validator: Optional[Callable[[str], Optional[str]]] = None,
        min_tool_calls_before_result: int = 0,
        verbose: bool = False,
    ):
        self.stage_name = stage_name
        self.client = client
        self.toolkit = toolkit
        self.allowed_tools = set(allowed_tools)
        self.max_steps = max(1, int(max_steps))
        self.result_validator = result_validator
        self.thought_validator = thought_validator
        self.min_tool_calls_before_result = max(0, int(min_tool_calls_before_result))
        self.verbose = verbose
        self.system_message: Dict[str, Any] = {"role": "system", "content": system_prompt}
        if initial_image_path and initial_image_path.exists():
            data_url = self._image_to_data_url(initial_image_path)
            if data_url:
                self.initial_user_message = {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            else:
                self.initial_user_message = {"role": "user", "content": user_prompt}
        else:
            self.initial_user_message = {"role": "user", "content": user_prompt}
        self.messages: List[Dict[str, Any]] = [self.system_message, self.initial_user_message]

    def run(self) -> StageRunResult:
        trace: List[PhaseTrace] = []
        full_trace: List[Dict[str, Any]] = []
        last_raw = ""
        self._record_full_message(full_trace, step=0, attempt=0, message=self.system_message, note="system")
        self._record_full_message(full_trace, step=0, attempt=0, message=self.initial_user_message, note="stage_input")
        counted_steps_used = 0
        total_attempts = 0

        while counted_steps_used < self.max_steps:
            step = counted_steps_used + 1
            attempt = 0
            if step == self.max_steps:
                self._append_user_message(
                    full_trace,
                    step=step,
                    attempt=0,
                    text=(
                        "<response>{\"warning\":\"last_counted_step\","
                        "\"message\":\"This is the last counted step for this stage. "
                        "Finish any remaining necessary tool work now and do not defer it to another counted step.\"}</response>"
                    ),
                    note="last_step_warning",
                )

            while True:
                attempt += 1
                total_attempts += 1
                self._log(step, "input", self._last_user_message_text())
                resp = self.client.chat(self.messages)
                text = self.client.extract_text(resp).strip()
                last_raw = text
                assistant_message = {"role": "assistant", "content": text}
                self.messages.append(assistant_message)
                self._record_full_message(full_trace, step=step, attempt=attempt, message=assistant_message, note="assistant")
                self._log(step, "assistant", text)

                _, _, think, tool_obj, result_obj, err = self._parse_tagged_output(text)

                if err:
                    if attempt >= self.MAX_SAME_STEP_RETRIES:
                        raise StageAgentRuntimeError(
                            f"[{self.stage_name}] exceeded {self.MAX_SAME_STEP_RETRIES} same-step retries at step {step} "
                            f"due to output protocol errors. Last parser detail: {err}. Last raw response: {last_raw}",
                            stage_name=self.stage_name,
                            trace=trace,
                            full_trace=full_trace,
                            raw_last_response=last_raw,
                            counted_steps_used=counted_steps_used,
                            total_attempts=total_attempts,
                        )
                    self._append_user_message(
                        full_trace,
                        step=step,
                        attempt=attempt,
                        text=self._retry_response_text(
                            error="output_protocol_error",
                            reason=err,
                            repair_instruction=self._format_repair_instruction(err),
                        ),
                        note="format_retry",
                    )
                    continue
                thought_error = self.thought_validator(think or "") if self.thought_validator else None
                if thought_error:
                    if attempt >= self.MAX_SAME_STEP_RETRIES:
                        raise StageAgentRuntimeError(
                            f"[{self.stage_name}] exceeded {self.MAX_SAME_STEP_RETRIES} same-step retries at step {step} "
                            f"due to thought validation errors. Last validation detail: {thought_error}. "
                            f"Last raw response: {last_raw}",
                            stage_name=self.stage_name,
                            trace=trace,
                            full_trace=full_trace,
                            raw_last_response=last_raw,
                            counted_steps_used=counted_steps_used,
                            total_attempts=total_attempts,
                        )
                    self._append_user_message(
                        full_trace,
                        step=step,
                        attempt=attempt,
                        text=self._retry_response_text(
                            error="thought_validation_error",
                            reason=thought_error,
                            repair_instruction=(
                                "Rewrite the response with `<think>` first. The first line inside `<think>` must be "
                                "`skill_experience_mode: follow` or `skill_experience_mode: explore`, then output "
                                "exactly one `<tool>` or `<result>` block."
                            ),
                        ),
                        note="thought_retry",
                    )
                    continue

                if result_obj is not None:
                    tool_call_count = sum(1 for item in trace if item.kind == "tool_call")
                    if tool_call_count < self.min_tool_calls_before_result:
                        if attempt >= self.MAX_SAME_STEP_RETRIES:
                            raise StageAgentRuntimeError(
                                f"[{self.stage_name}] exceeded {self.MAX_SAME_STEP_RETRIES} same-step retries at step {step} "
                                f"because result was emitted before the required {self.min_tool_calls_before_result} successful tool call(s).",
                                stage_name=self.stage_name,
                                trace=trace,
                                full_trace=full_trace,
                                raw_last_response=last_raw,
                                counted_steps_used=counted_steps_used,
                                total_attempts=total_attempts,
                            )
                        self._append_user_message(
                            full_trace,
                            step=step,
                            attempt=attempt,
                            text=self._retry_response_text(
                                error="premature_result",
                                reason=f"Result emitted before the required {self.min_tool_calls_before_result} successful tool call(s).",
                                repair_instruction=(
                                    f"Call at least {self.min_tool_calls_before_result} successful evidence tool(s) before emitting `<result>` for this stage. "
                                    "Your next response must be `<think>...</think><tool>{...}</tool>` only."
                                ),
                            ),
                            note="result_retry",
                        )
                        continue
                    validation_error = self.result_validator(result_obj) if self.result_validator else None
                    if validation_error:
                        if attempt >= self.MAX_SAME_STEP_RETRIES:
                            raise StageAgentRuntimeError(
                                f"[{self.stage_name}] exceeded {self.MAX_SAME_STEP_RETRIES} same-step retries at step {step} "
                                f"due to result validation errors. Last validation detail: {validation_error}. "
                                f"Last raw response: {last_raw}",
                                stage_name=self.stage_name,
                                trace=trace,
                                full_trace=full_trace,
                                raw_last_response=last_raw,
                                counted_steps_used=counted_steps_used,
                                total_attempts=total_attempts,
                            )
                        self._append_user_message(
                            full_trace,
                            step=step,
                            attempt=attempt,
                            text=self._retry_response_text(
                                error="result_validation_error",
                                reason=validation_error,
                                repair_instruction=self._result_repair_instruction(validation_error),
                            ),
                            note="result_retry",
                        )
                        continue
                    trace.append(PhaseTrace(step=step, kind="assistant", payload={"text": text, "attempt": attempt}))
                    trace.append(PhaseTrace(step=step, kind="result", payload={"result": result_obj}))
                    counted_steps_used = step
                    return StageRunResult(
                        stage_name=self.stage_name,
                        structured_result=result_obj,
                        trace=trace,
                        full_trace=full_trace,
                        raw_last_response=last_raw,
                        counted_steps_used=counted_steps_used,
                        total_attempts=total_attempts,
                    )

                tool_name = str(tool_obj.get("tool", "")) if isinstance(tool_obj, dict) else ""
                args = tool_obj.get("args", {}) if isinstance(tool_obj, dict) else {}
                if not isinstance(args, dict):
                    args = {}
                dynamic_tool_names = set(self.toolkit.dynamic_tool_names()) if hasattr(self.toolkit, "dynamic_tool_names") else set()
                if tool_name not in self.allowed_tools and tool_name not in dynamic_tool_names:
                    tool_result = {
                        "error": f"tool_not_allowed_in_{self.stage_name}",
                        "tool": tool_name,
                        "allowed_tools": sorted(self.allowed_tools | dynamic_tool_names),
                    }
                else:
                    tool_result = self.toolkit.execute(tool_name, args)
                self._log(step, "tool_call", f"{tool_name} {self._short_json(args)}")
                self._log(step, "tool_result", self._short_json(tool_result))

                if self._is_tool_error(tool_result):
                    if attempt >= self.MAX_SAME_STEP_RETRIES:
                        raise StageAgentRuntimeError(
                            f"[{self.stage_name}] exceeded {self.MAX_SAME_STEP_RETRIES} same-step retries at step {step} "
                            f"due to tool errors. Tool: {tool_name}. Last tool result: {self._short_json(tool_result)}",
                            stage_name=self.stage_name,
                            trace=trace,
                            full_trace=full_trace,
                            raw_last_response=last_raw,
                            counted_steps_used=counted_steps_used,
                            total_attempts=total_attempts,
                        )
                    self._append_tool_result_message(
                        full_trace=full_trace,
                        step=step,
                        attempt=attempt,
                        tool_name=tool_name,
                        tool_result=tool_result,
                        retry_same_step=True,
                    )
                    continue

                trace.append(PhaseTrace(step=step, kind="assistant", payload={"text": text, "attempt": attempt}))
                trace.append(
                    PhaseTrace(
                        step=step,
                        kind="tool_call",
                        payload={"tool": tool_name, "args": args, "result": tool_result},
                    )
                )
                self._append_tool_result_message(
                    full_trace=full_trace,
                    step=step,
                    attempt=attempt,
                    tool_name=tool_name,
                    tool_result=tool_result,
                    retry_same_step=False,
                )
                counted_steps_used += 1
                break

        self._append_user_message(
            full_trace,
            step=self.max_steps,
            attempt=0,
            text=(
                "<response>{\"instruction\":\"finalize_required\","
                "\"message\":\"Counted step limit reached. Output one final <result> block now. "
                "Do not call another tool. If your previous attempt is malformed, fix it immediately; "
                "this retry does not consume a new counted step.\"}</response>"
            ),
            note="forced_finalize",
        )
        finalize_attempt = 0
        while True:
            finalize_attempt += 1
            total_attempts += 1
            self._log(self.max_steps, "input", self._last_user_message_text())
            resp = self.client.chat(self.messages)
            text = self.client.extract_text(resp).strip()
            last_raw = text
            assistant_message = {"role": "assistant", "content": text}
            self.messages.append(assistant_message)
            self._record_full_message(
                full_trace,
                step=self.max_steps,
                attempt=finalize_attempt,
                message=assistant_message,
                note="forced_final_assistant",
            )
            self._log(self.max_steps, "assistant", text)
            _, _, think, tool_obj, result_obj, err = self._parse_tagged_output(text)

            if err:
                if finalize_attempt >= self.MAX_SAME_STEP_RETRIES:
                    raise StageAgentRuntimeError(
                        f"[{self.stage_name}] exceeded {self.MAX_SAME_STEP_RETRIES} forced-finalize retries "
                        f"due to output protocol errors. Last parser detail: {err}. Last raw response: {last_raw}",
                        stage_name=self.stage_name,
                        trace=trace,
                        full_trace=full_trace,
                        raw_last_response=last_raw,
                        counted_steps_used=counted_steps_used,
                        total_attempts=total_attempts,
                    )
                self._append_user_message(
                    full_trace,
                    step=self.max_steps,
                    attempt=finalize_attempt,
                    text=self._retry_response_text(
                        error="output_protocol_error",
                        reason=f"Finalization requires one valid <result> block. Parser detail: {err}",
                        repair_instruction=(
                            "Output `<think>...</think><result>{...}</result>` only. "
                            "Do not include `<tool>` during forced finalization."
                        ),
                    ),
                    note="forced_finalize_retry",
                )
                continue
            thought_error = self.thought_validator(think or "") if self.thought_validator else None
            if thought_error:
                if finalize_attempt >= self.MAX_SAME_STEP_RETRIES:
                    raise StageAgentRuntimeError(
                        f"[{self.stage_name}] exceeded {self.MAX_SAME_STEP_RETRIES} forced-finalize retries "
                        f"due to thought validation errors. Last validation detail: {thought_error}. "
                        f"Last raw response: {last_raw}",
                        stage_name=self.stage_name,
                        trace=trace,
                        full_trace=full_trace,
                        raw_last_response=last_raw,
                        counted_steps_used=counted_steps_used,
                        total_attempts=total_attempts,
                    )
                self._append_user_message(
                    full_trace,
                    step=self.max_steps,
                    attempt=finalize_attempt,
                    text=self._retry_response_text(
                        error="thought_validation_error",
                        reason=thought_error,
                        repair_instruction=(
                            "Fix `<think>` first line to `skill_experience_mode: follow` or "
                            "`skill_experience_mode: explore`, then output one final `<result>` block."
                        ),
                    ),
                    note="forced_finalize_retry",
                )
                continue
            if tool_obj is not None:
                if finalize_attempt >= self.MAX_SAME_STEP_RETRIES:
                    raise StageAgentRuntimeError(
                        f"[{self.stage_name}] exceeded {self.MAX_SAME_STEP_RETRIES} forced-finalize retries "
                        "because the model kept emitting <tool> instead of <result>.",
                        stage_name=self.stage_name,
                        trace=trace,
                        full_trace=full_trace,
                        raw_last_response=last_raw,
                        counted_steps_used=counted_steps_used,
                        total_attempts=total_attempts,
                    )
                self._append_user_message(
                    full_trace,
                    step=self.max_steps,
                    attempt=finalize_attempt,
                    text=self._retry_response_text(
                        error="finalize_requires_result",
                        reason="The counted step limit has been reached, so no more tool calls are allowed.",
                        repair_instruction="Output one final `<think>...</think><result>{...}</result>` block now.",
                    ),
                    note="forced_finalize_retry",
                )
                continue
            if isinstance(result_obj, dict):
                tool_call_count = sum(1 for item in trace if item.kind == "tool_call")
                if tool_call_count < self.min_tool_calls_before_result:
                    if finalize_attempt >= self.MAX_SAME_STEP_RETRIES:
                        raise StageAgentRuntimeError(
                            f"[{self.stage_name}] exceeded {self.MAX_SAME_STEP_RETRIES} forced-finalize retries "
                            f"because result was emitted before the required {self.min_tool_calls_before_result} successful tool call(s).",
                            stage_name=self.stage_name,
                            trace=trace,
                            full_trace=full_trace,
                            raw_last_response=last_raw,
                            counted_steps_used=counted_steps_used,
                            total_attempts=total_attempts,
                        )
                    self._append_user_message(
                        full_trace,
                        step=self.max_steps,
                        attempt=finalize_attempt,
                        text=self._retry_response_text(
                            error="premature_result",
                            reason=f"Final result still needs at least {self.min_tool_calls_before_result} successful evidence tool(s) in this stage.",
                            repair_instruction=(
                                "Because forced finalization disallows new tools, return the best valid evidence-grounded result "
                                "that satisfies the schema. Do not invent unsupported evidence; if no tool evidence was collected, "
                                "state the limitation explicitly while preserving any already materialized plan."
                            ),
                        ),
                        note="forced_finalize_retry",
                    )
                    continue
                validation_error = self.result_validator(result_obj) if self.result_validator else None
                if validation_error:
                    if finalize_attempt >= self.MAX_SAME_STEP_RETRIES:
                        raise StageAgentRuntimeError(
                            f"[{self.stage_name}] exceeded {self.MAX_SAME_STEP_RETRIES} forced-finalize retries "
                            f"due to result validation errors. Last validation detail: {validation_error}. "
                            f"Last raw response: {last_raw}",
                            stage_name=self.stage_name,
                            trace=trace,
                            full_trace=full_trace,
                            raw_last_response=last_raw,
                            counted_steps_used=counted_steps_used,
                            total_attempts=total_attempts,
                        )
                    self._append_user_message(
                        full_trace,
                        step=self.max_steps,
                        attempt=finalize_attempt,
                        text=self._retry_response_text(
                            error="result_validation_error",
                            reason=validation_error,
                            repair_instruction=self._result_repair_instruction(validation_error),
                        ),
                        note="forced_finalize_retry",
                    )
                    continue
                trace.append(
                    PhaseTrace(
                        step=self.max_steps,
                        kind="assistant",
                        payload={"text": text, "forced": True, "attempt": finalize_attempt},
                    )
                )
                trace.append(
                    PhaseTrace(
                        step=self.max_steps,
                        kind="result",
                        payload={"result": result_obj, "forced": True},
                    )
                )
                return StageRunResult(
                    stage_name=self.stage_name,
                    structured_result=result_obj,
                    trace=trace,
                    full_trace=full_trace,
                    raw_last_response=last_raw,
                    counted_steps_used=counted_steps_used,
                    total_attempts=total_attempts,
                )

    def _append_tool_result_message(
        self,
        full_trace: List[Dict[str, Any]],
        step: int,
        attempt: int,
        tool_name: str,
        tool_result: Dict[str, Any],
        retry_same_step: bool,
    ) -> None:
        payload = {"tool": tool_name, "result": self._strip_internal_fields(tool_result)}
        if retry_same_step:
            payload["retry_same_step"] = True
            payload["error_reason"] = self._tool_error_reason(tool_result)
            payload["repair_instruction"] = self._tool_repair_instruction(tool_name, tool_result)
            payload["message"] = (
                "Tool call failed for the reason above. Fix that specific issue immediately; "
                "this retry stays in the same counted step."
            )
        text = f"<response>{json.dumps(payload, ensure_ascii=False)}</response>"
        image_path = self._extract_image_path(tool_result)
        if image_path:
            data_url = self._image_to_data_url(image_path)
            if data_url:
                message = {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
                self.messages.append(message)
                self._record_full_message(full_trace, step=step, attempt=attempt, message=message, note="tool_response")
                return
        self._append_user_message(full_trace, step=step, attempt=attempt, text=text, note="tool_response")

    def _append_tool_response(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})

    @staticmethod
    def _is_tool_error(tool_result: Any) -> bool:
        return isinstance(tool_result, dict) and (
            tool_result.get("error") is not None or tool_result.get("success") is False
        )

    @staticmethod
    def _retry_response_text(error: str, reason: str, repair_instruction: str) -> str:
        payload = {
            "error": error,
            "reason": str(reason or ""),
            "repair_instruction": str(repair_instruction or ""),
            "retry_same_step": True,
            "message": "Retry because of the specific reason above; do not mechanically repeat the previous output.",
        }
        return f"<response>{json.dumps(payload, ensure_ascii=False)}</response>"

    @staticmethod
    def _format_repair_instruction(reason: str) -> str:
        text = str(reason or "").lower()
        if "unclosed <result>" in text:
            return (
                "Your previous response started a `<result>` block but did not close it. "
                "Return the same valid JSON inside exactly `<think>...</think><result>{...}</result>`; "
                "the required final characters are `</result>`."
            )
        if "unclosed <tool>" in text:
            return (
                "Your previous response started a `<tool>` block but did not close it. "
                "Return exactly `<think>...</think><tool>{...}</tool>`; "
                "the required final characters are `</tool>`."
            )
        if "must include exactly one" in text and "<result>" in text and "</result>" not in text:
            return (
                "Use exactly one closed action block after `<think>`. If you choose result, "
                "the response must end with `</result>`."
            )
        if "found" in text and "action blocks" in text:
            return (
                "Your next response must contain exactly two tags in this order: "
                "`<think>...</think>` and either one `<tool>{...}</tool>` OR one `<result>{...}</result>`. "
                "Do not include both `<tool>` and `<result>` in the same response."
            )
        if "before </think>" in text:
            return (
                "Move all reasoning into `<think>...</think>`, then place exactly one action block after it. "
                "Never put `<tool>` before `<think>`."
            )
        if "json" in text:
            return "Fix the JSON inside the action tag. Use double quotes and valid JSON only."
        if "missing <think>" in text:
            return "Start the next response with `<think>...</think>`, then add exactly one action block."
        return "Follow the strict protocol exactly: `<think>...</think>` followed by one `<tool>` or one `<result>` block."

    @staticmethod
    def _result_repair_instruction(validation_error: str) -> str:
        text = str(validation_error or "")
        low = text.lower()
        if "execution.plan" in text and "apply_temp_plan_changes" in text:
            return (
                "Do not emit `<result>` yet. Your next response must be only "
                "`<think>...</think><tool>{\"tool\":\"apply_temp_plan_changes\",\"args\":{\"plan\":{...}}}</tool>` "
                "using the intended cumulative plan with full numeric AOI IDs. Wait for the tool response before finalizing."
            )
        if "cannot request finish" in text:
            return (
                "Keep the exact Stage-3 JSON schema, but repair the invalid status fields: keep unresolved focuses "
                "as `carry_over` or `blocked`, set `execution.action` to `commit`, set `reflection.next_step` "
                "to `continue`, and do not use the literal value `finish` anywhere unless every global focus is done."
            )
        if "coverage mismatch" in low or "cover every" in low:
            return "Repair the checklist and decision/evidence arrays so they exactly match the required focus and AOI IDs shown in the error."
        if "missing" in low:
            return "Add the missing required fields/items named in the error; do not change unrelated valid content."
        return "Revise only the invalid fields described in `reason`, then retry with one valid `<result>` block."

    @staticmethod
    def _tool_error_reason(tool_result: Dict[str, Any]) -> str:
        if not isinstance(tool_result, dict):
            return "Unknown tool failure."
        if tool_result.get("message"):
            return str(tool_result.get("message"))
        if tool_result.get("error"):
            return str(tool_result.get("error"))
        if tool_result.get("success") is False:
            return "Tool returned success=false."
        return "Tool call failed."

    @staticmethod
    def _tool_repair_instruction(tool_name: str, tool_result: Dict[str, Any]) -> str:
        err = str((tool_result or {}).get("error") or "").lower()
        msg = str((tool_result or {}).get("message") or "").lower()
        if "coordinate_space_mismatch" in err or "bbox_rel" in msg:
            return (
                "Retry with absolute map/positions coordinates `x1,y1,x2,y2`; do not pass Stage-1 `bbox_rel`. "
                "Use `bbox_abs` from stage requirements when available, or use an AOI-list tool instead."
            )
        if "unknown_aoi" in err or "unknown_aoi" in msg:
            return (
                "Repair the AOI ID: use the full numeric AOI ID from stage requirements or prior tool output. "
                "If you wrote a text label like `AOI_97`, extract digits and resolve to the full numeric ID before retrying."
            )
        if "missing required fields" in msg or "invalid_tool_args" in err:
            return "Retry the same tool with the exact required argument names from `expected_schema`."
        return f"Retry `{tool_name}` only after fixing the concrete error above, or choose a more suitable allowed tool."

    def _append_user_message(
        self,
        full_trace: List[Dict[str, Any]],
        step: int,
        attempt: int,
        text: str,
        note: str,
    ) -> None:
        message = {"role": "user", "content": text}
        self.messages.append(message)
        self._record_full_message(full_trace, step=step, attempt=attempt, message=message, note=note)

    def _record_full_message(
        self,
        full_trace: List[Dict[str, Any]],
        step: int,
        attempt: int,
        message: Dict[str, Any],
        note: str,
    ) -> None:
        full_trace.append(
            {
                "step": int(step),
                "attempt": int(attempt),
                "role": str(message.get("role") or ""),
                "content": self._normalize_message_content(message.get("content")),
                "note": note,
            }
        )

    @staticmethod
    def _normalize_message_content(content: Any) -> Any:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            items: List[Dict[str, Any]] = []
            for item in content:
                if not isinstance(item, dict):
                    items.append({"type": "unknown", "value": str(item)})
                    continue
                normalized = dict(item)
                if normalized.get("type") == "image_url" and isinstance(normalized.get("image_url"), dict):
                    normalized["image_url"] = {"url": "[image_url omitted]"}
                items.append(normalized)
            return items
        if isinstance(content, dict):
            return dict(content)
        return str(content)

    def _last_user_message_text(self) -> str:
        for msg in reversed(self.messages):
            if msg.get("role") != "user":
                continue
            content = msg.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts = []
                for item in content:
                    if not isinstance(item, dict):
                        continue
                    if item.get("type") == "text":
                        parts.append(str(item.get("text") or ""))
                    elif item.get("type") == "image_url":
                        parts.append("[image_url]")
                return " ".join(parts)
        return ""

    @staticmethod
    def _parse_tagged_output(
        text: str,
    ) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[Dict[str, Any]], Optional[Dict[str, Any]], Optional[str]]:
        experience_blocks = StageReActAgent._extract_tag_blocks(text, "experience")
        skill_blocks = StageReActAgent._extract_tag_blocks(text, "skill")
        experience = experience_blocks[0] if experience_blocks else None
        skill = skill_blocks[0] if skill_blocks else None

        think_matches = list(re.finditer(r"<\s*think\s*>([\s\S]*?)</\s*think\s*>", text, flags=re.IGNORECASE))
        think = think_matches[0].group(1).strip() if think_matches else None
        if len(think_matches) > 1:
            return experience, skill, think, None, None, "multiple <think> blocks found; output exactly one"
        action_matches = list(
            re.finditer(r"<\s*(tool|result)\s*>([\s\S]*?)</\s*\1\s*>", text, flags=re.IGNORECASE)
        )
        if think is None:
            return experience, skill, None, None, None, "missing <think>"
        if not action_matches:
            recovered = StageReActAgent._recover_unclosed_action_block(text, think_matches[0].end())
            if recovered is not None:
                action_kind, action_payload, action_start = recovered
                if action_start < think_matches[0].end():
                    return experience, skill, think, None, None, "action block appears before </think>; valid order is <think> then one action block"
                obj = StageReActAgent._parse_json_like(action_payload or "")
                if not isinstance(obj, dict):
                    return (
                        experience,
                        skill,
                        think,
                        None,
                        None,
                        f"unclosed <{action_kind}> block found, but JSON payload is invalid; close with </{action_kind}> and fix JSON",
                    )
                if action_kind == "tool":
                    if "tool" not in obj and "name" in obj:
                        obj["tool"] = obj.get("name")
                    if "args" not in obj:
                        for alias in ("parameters", "arguments", "input", "kwargs"):
                            if alias in obj:
                                obj["args"] = obj.get(alias)
                                break
                    if "tool" not in obj:
                        return experience, skill, think, None, None, "missing tool field"
                    if "args" not in obj:
                        obj["args"] = {}
                    return experience, skill, think, obj, None, None
                return experience, skill, think, None, obj, None
            return experience, skill, think, None, None, "must include exactly one of <tool> or <result>"
        if len(action_matches) > 1:
            kinds = [m.group(1).lower() for m in action_matches]
            return (
                experience,
                skill,
                think,
                None,
                None,
                f"found {len(action_matches)} action blocks {kinds}; output exactly one <tool> or one <result>, never both",
            )
        if action_matches[0].start() < think_matches[0].end():
            return experience, skill, think, None, None, "action block appears before </think>; valid order is <think> then one action block"

        action_kind = (action_matches[0].group(1) or "").lower()
        action_payload = action_matches[0].group(2).strip()
        if action_kind == "tool":
            tool_text = action_payload
            obj = StageReActAgent._parse_json_like(tool_text)
            if not isinstance(obj, dict):
                return experience, skill, think, None, None, "invalid <tool> JSON"
            if "tool" not in obj and "name" in obj:
                obj["tool"] = obj.get("name")
            if "args" not in obj:
                for alias in ("parameters", "arguments", "input", "kwargs"):
                    if alias in obj:
                        obj["args"] = obj.get(alias)
                        break
            if "tool" not in obj:
                return experience, skill, think, None, None, "missing tool field"
            if "args" not in obj:
                obj["args"] = {}
            return experience, skill, think, obj, None, None
        obj = StageReActAgent._parse_json_like(action_payload or "")
        if not isinstance(obj, dict):
            return experience, skill, think, None, None, "invalid <result> JSON"
        return experience, skill, think, None, obj, None

    @staticmethod
    def _extract_tag_content(text: str, tag: str) -> Optional[str]:
        match = re.search(rf"<\s*{tag}\s*>([\s\S]*?)</\s*{tag}\s*>", text, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return None

    @staticmethod
    def _extract_tag_blocks(text: str, tag: str) -> List[str]:
        return [
            m.strip()
            for m in re.findall(rf"<\s*{tag}\s*>([\s\S]*?)</\s*{tag}\s*>", text, flags=re.IGNORECASE)
        ]

    @staticmethod
    def _parse_json_like(payload: str) -> Any:
        cleaned = payload.strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        cleaned = html.unescape(cleaned)
        first = cleaned.find("{")
        last = cleaned.rfind("}")
        if first >= 0 and last >= first:
            cleaned = cleaned[first : last + 1]
        try:
            return json.loads(cleaned)
        except Exception:
            pass
        try:
            import json_repair  # type: ignore
            repaired = json_repair.repair_json(cleaned)
            if isinstance(repaired, str):
                return json.loads(repaired)
            if isinstance(repaired, dict):
                return repaired
        except Exception:
            pass
        try:
            return ast.literal_eval(cleaned)
        except Exception:
            return None

    @staticmethod
    def _recover_unclosed_action_block(text: str, after_index: int) -> Optional[Tuple[str, str, int]]:
        """Recover a single trailing action block when only the closing tag is missing."""
        tail = text[max(0, int(after_index)) :]
        starts = list(re.finditer(r"<\s*(tool|result)\s*>", tail, flags=re.IGNORECASE))
        if len(starts) != 1:
            return None
        start = starts[0]
        kind = (start.group(1) or "").lower()
        if re.search(rf"</\s*{kind}\s*>", tail[start.end() :], flags=re.IGNORECASE):
            return None
        other = "result" if kind == "tool" else "tool"
        if re.search(rf"<\s*{other}\s*>|</\s*{other}\s*>", tail[start.end() :], flags=re.IGNORECASE):
            return None
        payload = tail[start.end() :].strip()
        if not payload:
            return None
        return kind, payload, max(0, int(after_index)) + start.start()

    @staticmethod
    def _extract_image_path(tool_result: Dict[str, Any]) -> Optional[Path]:
        path_str = tool_result.get("output_path") or tool_result.get("_image_path")
        if not path_str and isinstance(tool_result.get("render_map"), dict):
            path_str = (tool_result.get("render_map") or {}).get("output_path")
        if isinstance(path_str, str):
            path = Path(path_str)
            if path.exists():
                return path
        return None

    @staticmethod
    def _strip_internal_fields(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {
                k: StageReActAgent._strip_internal_fields(v)
                for k, v in obj.items()
                if k not in {"_image_path", "output_path"}
            }
        if isinstance(obj, list):
            return [StageReActAgent._strip_internal_fields(x) for x in obj]
        return obj

    @staticmethod
    def _image_to_data_url(path: Path) -> Optional[str]:
        try:
            raw = path.read_bytes()
            mime = mimetypes.guess_type(str(path))[0] or "image/png"
            return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"
        except Exception:
            return None

    def _log(self, step: int, tag: str, message: str) -> None:
        if self.verbose:
            print(f"[StageTrace][{self.stage_name}][{step:02d}][{tag}] {message}\n")

    @staticmethod
    def _short_json(obj: Any, limit: int = 700) -> str:
        try:
            text = json.dumps(obj, ensure_ascii=False)
        except Exception:
            text = str(obj)
        return text if len(text) <= limit else text[:limit] + "..."
