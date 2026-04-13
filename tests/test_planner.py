from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from free_gpu.cli import build_parser
from free_gpu.data import load_providers
from free_gpu.http_app import create_http_app
from free_gpu.models import LocalCapabilityProfile, WorkloadRequest
from free_gpu.planner import assess_compute_need, build_plan, infer_model_size, rank_providers


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

    def test_compute_need_uses_grant_scale_for_heavy_training(self) -> None:
        request = WorkloadRequest(
            workload="scratch-train",
            budget="grant",
            task_hours=24,
            min_vram_gb=40,
        )
        compute_need = assess_compute_need(request)
        self.assertEqual(compute_need.lane, "grant-scale")

    def test_workflow_steps_include_compute_need(self) -> None:
        request = WorkloadRequest(
            workload="finetune-lora",
            model="llama-3.1-8b",
            budget="under-25",
            task_hours=6,
            min_vram_gb=16,
        )
        local = LocalCapabilityProfile(source="manual", llmfit_available=False, ram_gb=32, vram_gb=12)
        plan = build_plan(request, local, self.providers)
        self.assertEqual(plan.compute_need.lane, "session")
        self.assertTrue(plan.workflow_steps)
        self.assertTrue(all(step.compute_need is not None for step in plan.workflow_steps))

    def test_cli_budget_choices_remove_under_10(self) -> None:
        parser = build_parser()
        subparsers = next(action for action in parser._actions if getattr(action, "choices", None))
        plan_parser = subparsers.choices["plan"]
        budget_action = next(action for action in plan_parser._actions if action.dest == "budget")
        self.assertNotIn("under-10", budget_action.choices)
        self.assertIn("under-25", budget_action.choices)
        self.assertIn("grant", budget_action.choices)

    def test_http_app_exposes_mcp_metadata(self) -> None:
        with TestClient(create_http_app(), base_url="https://free-gpu.vercel.app") as client:
            response = client.get("/")
            self.assertEqual(response.status_code, 200)
            payload = response.json()
            self.assertEqual(payload["name"], "free-gpu")
            self.assertEqual(payload["mcp_path"], "/mcp")

    def test_http_app_exposes_mcp_endpoint(self) -> None:
        with TestClient(create_http_app(), base_url="https://free-gpu.vercel.app") as client:
            response = client.get("/mcp")
            self.assertEqual(response.status_code, 406)


if __name__ == "__main__":
    unittest.main()
