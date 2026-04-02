from __future__ import annotations

import unittest

from free_gpu.data import load_providers
from free_gpu.models import LocalCapabilityProfile, WorkloadRequest
from free_gpu.planner import build_plan, infer_model_size, rank_providers


class PlannerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.providers = load_providers()

    def test_infer_model_size(self) -> None:
        self.assertEqual(infer_model_size("llama-3.1-8b"), 8.0)
        self.assertEqual(infer_model_size("qwen2.5-coder-7b"), 7.0)
        self.assertIsNone(infer_model_size("nanogpt"))

    def test_inference_prefers_local_with_enough_vram(self) -> None:
        request = WorkloadRequest(workload="inference", model="qwen2.5-coder-7b")
        local = LocalCapabilityProfile(source="manual", llmfit_available=False, ram_gb=32, vram_gb=12)
        plan = build_plan(request, local, self.providers)
        self.assertEqual(plan.local_verdict, "good-local")
        self.assertEqual(plan.workflow_steps[0].recommended_environment, "local")

    def test_finetune_uses_remote_when_vram_is_low(self) -> None:
        request = WorkloadRequest(workload="finetune-lora", model="llama-3.1-8b")
        local = LocalCapabilityProfile(source="manual", llmfit_available=False, ram_gb=16, vram_gb=8)
        plan = build_plan(request, local, self.providers)
        finetune_step = [step for step in plan.workflow_steps if step.stage == "finetune"][0]
        self.assertEqual(plan.local_verdict, "cloud-assisted")
        self.assertEqual(finetune_step.recommended_environment, "remote")

    def test_agent_loop_ranking_prefers_api_capable_providers(self) -> None:
        request = WorkloadRequest(workload="agent-loop", requires_api=True, budget="free")
        ranked = rank_providers(self.providers, request, stage="agent-loop", limit=5)
        self.assertTrue(ranked)
        self.assertTrue(any("api" in provider.reason.lower() for provider in ranked[:3]))


if __name__ == "__main__":
    unittest.main()
