from __future__ import annotations

import unittest

import yolorag.core.agent as agent
from yolorag.core.agent import DEEP_AGENT_SYSTEM_PROMPT


class DeepAgentBehaviorTests(unittest.TestCase):
    def test_deep_agent_prompt_includes_issue_triage_layer(self) -> None:
        for issue_type in (
            "bug report",
            "usage question",
            "docs/UX gap",
            "feature request",
            "environment/install issue",
            "model/export/deployment issue",
            "platform/backend issue",
            "resolved/follow-up/thanks",
        ):
            self.assertIn(issue_type, DEEP_AGENT_SYSTEM_PROMPT)

    def test_deep_agent_prompt_requires_evidence_and_public_reply_discipline(self) -> None:
        self.assertIn("Exact versions, dates, timestamps, paths", DEEP_AGENT_SYSTEM_PROMPT)
        self.assertIn("Distinguish root-cause fixes from downstream workarounds", DEEP_AGENT_SYSTEM_PROMPT)
        self.assertIn("If docs and code disagree", DEEP_AGENT_SYSTEM_PROMPT)
        self.assertIn("Keep the final answer external-facing", DEEP_AGENT_SYSTEM_PROMPT)
        self.assertIn("80-140", DEEP_AGENT_SYSTEM_PROMPT)

    def test_deep_agent_uses_prompt_protocol_not_hardcoded_text_filtering(self) -> None:
        self.assertFalse(hasattr(agent, "_clean_final_answer"))
        self.assertFalse(hasattr(agent, "_INTERNAL_PROCESS_PREFIX_RE"))
