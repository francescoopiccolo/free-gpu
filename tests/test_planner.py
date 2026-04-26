from __future__ import annotations

import asyncio
import subprocess
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from free_gpu.cli import build_parser
from free_gpu.data import load_providers
from free_gpu.http_app import create_http_app
from free_gpu.llmfit_adapter import _run_llmfit_system, load_local_profile
from free_gpu.mcp_server import _build_request, create_mcp
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

    def test_lightning_ai_free_tier_lists_high_memory_gpus(self) -> None:
        lightning = next(provider for provider in self.providers if provider.service == "Lightning AI Studio")

        self.assertIn("H200", lightning.compute)
        self.assertIn("141 GB", lightning.vram)
        self.assertIn("80 GPU hours/month", lightning.max_hours)
        self.assertIn("H200 3h", lightning.max_hours)
        self.assertEqual(lightning.credit_card_required, "No")

    def test_audited_provider_rows_keep_current_credit_details(self) -> None:
        providers = {provider.service: provider for provider in self.providers}

        self.assertIn("B200", providers["Modal Starter"].compute)
        self.assertIn("$30/month", providers["Modal Starter"].max_hours)
        self.assertIn("25 cloud credit hours", providers["AMD Developer Cloud"].max_hours)
        self.assertIn("$200 credit valid for 1 year", providers["DigitalOcean GitHub Students"].max_hours)
        self.assertIn("H200", providers["DigitalOcean Free Credit (new account)"].compute)
        self.assertIn("$100 credit for 12 months", providers["Azure for Students"].max_hours)
        self.assertIn("$200 credit for 30 days", providers["Microsoft Azure Free Account"].max_hours)
        self.assertIn("8 GPUs for a full year", providers["Nebius Research Credits Program"].max_hours)
        self.assertIn("$1,000 Starter Tier credits", providers["RunPod Startup Program"].max_hours)

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

    def test_create_mcp_registers_tools_and_resources(self) -> None:
        mcp = create_mcp(host="127.0.0.1")
        tool_names = {tool.name for tool in mcp._tool_manager.list_tools()}
        resource_uris = {str(resource.uri) for resource in mcp._resource_manager.list_resources()}
        prompt_names = {prompt.name for prompt in asyncio.run(mcp.list_prompts())}

        self.assertSetEqual(
            tool_names,
            {"plan_provider_workflow", "rank_providers_for_task", "assess_task_compute"},
        )
        self.assertSetEqual(resource_uris, {"providers://snapshot", "guide://tool-selection"})
        self.assertSetEqual(prompt_names, {"choose_free_gpu_tool"})

    def test_mcp_tool_schemas_expose_enums_and_descriptions(self) -> None:
        mcp = create_mcp(host="127.0.0.1")
        tools = {tool.name: tool for tool in asyncio.run(mcp.list_tools())}

        plan_schema = tools["plan_provider_workflow"].inputSchema
        budget_schema = plan_schema["properties"]["budget"]
        deadline_schema = plan_schema["properties"]["deadline"]
        workload_schema = plan_schema["properties"]["workload"]

        self.assertIn("any, free, under-25, grant", budget_schema["description"])
        self.assertIn("flexible", deadline_schema["description"])
        self.assertIn("urgent", deadline_schema["description"])
        self.assertIn("normalized automatically", workload_schema["description"])

    def test_mcp_resource_and_prompt_offer_client_guidance(self) -> None:
        mcp = create_mcp(host="127.0.0.1")
        resource = asyncio.run(mcp.read_resource("guide://tool-selection"))
        prompt = asyncio.run(
            mcp.get_prompt("choose_free_gpu_tool", {"user_goal": "Find a short list of free APIs for an agent run"})
        )

        self.assertEqual(resource[0].mime_type, "application/json")
        self.assertIn("canonical_workloads", resource[0].content)
        self.assertIn("Do not mention internal fallback reasoning", prompt.messages[0].content.text)

    def test_plan_provider_workflow_returns_canonical_request_metadata(self) -> None:
        mcp = create_mcp(host="127.0.0.1")
        _, payload = asyncio.run(
            mcp.call_tool(
                "plan_provider_workflow",
                {
                    "workload": "agent run",
                    "budget": "cheap",
                    "deadline": "asap",
                    "task_hours": 0,
                    "parallel_jobs": 0,
                    "limit": 999,
                },
            )
        )

        self.assertEqual(payload["canonical_request"]["workload"], "agent-loop")
        self.assertEqual(payload["canonical_request"]["budget"], "under-25")
        self.assertEqual(payload["canonical_request"]["deadline"], "urgent")
        self.assertEqual(payload["canonical_request"]["task_hours"], 1.0)
        self.assertEqual(payload["canonical_request"]["parallel_jobs"], 1)
        self.assertEqual(payload["canonical_request"]["limit"], 10)
        self.assertEqual(payload["client_guidance"]["tool"], "plan_provider_workflow")
        self.assertTrue(payload["normalization"]["workload_changed"])

    def test_build_request_normalizes_common_workload_aliases(self) -> None:
        cases = {
            "fine-tune": "finetune-lora",
            "LoRA fine tune": "finetune-lora",
            "training": "scratch-train",
            "batch evaluation": "batch-eval",
            "agent run": "agent-loop",
            "inference": "inference",
        }
        for raw, expected in cases.items():
            with self.subTest(raw=raw):
                request = _build_request(
                    workload=raw,
                    model=None,
                    params_b=None,
                    budget="free",
                    task_hours=1.0,
                    min_vram_gb=None,
                    parallel_jobs=1,
                    requires_api=False,
                    prefer_local=True,
                    deadline="flexible",
                    limit=5,
                )
                self.assertEqual(request.workload, expected)

    def test_build_request_normalizes_budget_and_deadline_aliases(self) -> None:
        request = _build_request(
            workload="inference",
            model=None,
            params_b=None,
            budget="cheap",
            task_hours=0,
            min_vram_gb=None,
            parallel_jobs=0,
            requires_api=False,
            prefer_local=True,
            deadline="asap",
            limit=999,
        )
        self.assertEqual(request.budget, "under-25")
        self.assertEqual(request.deadline, "urgent")
        self.assertEqual(request.task_hours, 1.0)
        self.assertEqual(request.parallel_jobs, 1)
        self.assertEqual(request.limit, 10)

    def test_llmfit_timeout_returns_warning(self) -> None:
        with patch("free_gpu.llmfit_adapter.subprocess.run", side_effect=subprocess.TimeoutExpired(cmd=["llmfit"], timeout=15)):
            payload, warning = _run_llmfit_system("llmfit")
        self.assertIsNone(payload)
        self.assertIsNotNone(warning)
        self.assertIn("Could not read llmfit system output", warning)

    def test_missing_llmfit_keeps_provider_first_mode(self) -> None:
        with patch("free_gpu.llmfit_adapter.resolve_llmfit_executable", return_value=None):
            profile = load_local_profile()
        self.assertEqual(profile.source, "provider-first")
        self.assertFalse(profile.has_hardware_data())

        request = WorkloadRequest(workload="inference", budget="free")
        plan = build_plan(request, profile, self.providers)
        self.assertEqual(plan.local_verdict, "unknown")
        self.assertIn("provider data only", plan.summary)


if __name__ == "__main__":
    unittest.main()
