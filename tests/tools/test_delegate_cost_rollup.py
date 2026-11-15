"""Delegated child costs roll up without constructing live clients."""

import json
import types
import unittest
from unittest.mock import MagicMock, patch

from tests.tools.test_delegate import _make_mock_parent
from tools.delegate_tool import delegate_task


class TestSubagentCostRollup(unittest.TestCase):
    """Port of Kilo-Org/kilocode#9448 — parent's session_estimated_cost_usd
    must include subagent spend, not just the parent's own API calls."""

    def _make_parent_with_cost_counters(self, depth=0, starting_cost=0.0):
        parent = _make_mock_parent(depth=depth)
        # The fields AIAgent exposes and the footer reads from.  Set real
        # floats/strings so the rollup can add to them rather than tripping
        # on MagicMock auto-attrs.
        parent.session_estimated_cost_usd = starting_cost
        parent.session_cost_status = "unknown"
        parent.session_cost_source = "none"
        return parent

    def test_single_child_cost_folded_into_parent(self):
        parent = self._make_parent_with_cost_counters(starting_cost=0.10)

        with patch("run_agent.AIAgent") as MockAgent:
            mock_child = MagicMock()
            mock_child.model = "claude-sonnet-4-6"
            mock_child.session_prompt_tokens = 1000
            mock_child.session_completion_tokens = 200
            mock_child.session_estimated_cost_usd = 0.42
            mock_child.run_conversation.return_value = {
                "final_response": "done",
                "completed": True,
                "interrupted": False,
                "api_calls": 2,
                "messages": [],
            }
            MockAgent.return_value = mock_child

            result = json.loads(delegate_task(goal="do stuff", parent_agent=parent))

        # Parent footer must reflect parent_cost + child_cost.
        self.assertAlmostEqual(parent.session_estimated_cost_usd, 0.52, places=6)
        # Rollup must strip the internal field before serialising to the model.
        self.assertNotIn("_child_cost_usd", result["results"][0])
        self.assertNotIn("_child_role", result["results"][0])

    def test_batch_children_costs_sum_into_parent(self):
        parent = self._make_parent_with_cost_counters(starting_cost=0.00)

        # _run_single_child is downstream of construction/tool discovery. This
        # aggregation test needs distinct child identities, not live AIAgents.
        children = [
            types.SimpleNamespace(session_id=f"cost-child-{i}", tool_progress_callback=None)
            for i in range(3)
        ]
        with (
            patch(
                "tools.delegate_tool._build_child_preserving_parent_tools",
                side_effect=children,
            ) as mock_build,
            patch("tools.delegate_tool._run_single_child") as mock_run,
        ):
            entries = [
                {
                    "task_index": 0,
                    "status": "completed",
                    "summary": "A",
                    "api_calls": 2,
                    "duration_seconds": 1.0,
                    "_child_role": "leaf",
                    "_child_cost_usd": 0.15,
                },
                {
                    "task_index": 1,
                    "status": "completed",
                    "summary": "B",
                    "api_calls": 2,
                    "duration_seconds": 1.0,
                    "_child_role": "leaf",
                    "_child_cost_usd": 0.27,
                },
                {
                    "task_index": 2,
                    "status": "failed",
                    "summary": "",
                    "error": "boom",
                    "api_calls": 0,
                    "duration_seconds": 0.1,
                    "_child_role": "leaf",
                    "_child_cost_usd": 0.03,
                },
            ]

            def run_child(*, task_index, goal, child, parent_agent, **ownership):
                self.assertIs(child, children[task_index])
                self.assertIs(parent_agent, parent)
                self.assertEqual(goal, f"Investigate module {'ABC'[task_index]}")
                self.assertEqual(ownership, {
                    "owner_session_id": None,
                    "owner_transport": None,
                    "owner_session_record": None,
                })
                return dict(entries[task_index])

            mock_run.side_effect = run_child
            result = json.loads(
                delegate_task(
                    tasks=[
                        {"goal": "Investigate module A"},
                        {"goal": "Investigate module B"},
                        {"goal": "Investigate module C"},
                    ],
                    parent_agent=parent,
                )
            )

        self.assertEqual(mock_build.call_count, 3)
        self.assertEqual(mock_run.call_count, 3)
        for i, call in enumerate(mock_build.call_args_list):
            self.assertEqual(call.kwargs["task_index"], i)
            self.assertIs(call.kwargs["parent_agent"], parent)
        self.assertEqual([entry["task_index"] for entry in result["results"]], [0, 1, 2])
        self.assertEqual([entry["status"] for entry in result["results"]],
                         ["completed", "completed", "failed"])
        # 0.15 + 0.27 + 0.03 even though one child failed — the API calls it
        # made before failing still cost money.
        self.assertAlmostEqual(parent.session_estimated_cost_usd, 0.45, places=6)
        # cost_source promoted from "none" since the parent had no direct spend.
        self.assertEqual(parent.session_cost_source, "subagent")
        self.assertEqual(parent.session_cost_status, "estimated")
        # All internal fields stripped from results.
        for entry in result["results"]:
            self.assertNotIn("_child_cost_usd", entry)
            self.assertNotIn("_child_role", entry)


if __name__ == "__main__":
    unittest.main()
