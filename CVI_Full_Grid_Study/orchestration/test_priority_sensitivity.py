"""Light focused priority-orchestration tests; never spawns numerical workers."""
import ast
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import run_priority_sensitivity as priority


class PrioritySafety(unittest.TestCase):
    def test_priority_ten_worker_override_does_not_change_original(self):
        original=SimpleNamespace(workers=7,memory_budget_gib=8.,min_free_disk_gib=12.,heavy_factor_gib=1.)
        with patch.object(priority.os,"cpu_count",return_value=10):
            amended=priority.priority_admission(original,10)
            self.assertEqual(amended.workers,10)
            self.assertEqual(original.workers,7)
            self.assertLess(10*priority.original.weight(priority.SETTING)["resident_gib"],8.)
            with self.assertRaises(RuntimeError):priority.priority_admission(original,11)
        with patch.object(priority.os,"cpu_count",return_value=8):
            with self.assertRaises(RuntimeError):priority.priority_admission(original,10)

    def test_execution_sidecar_records_override_and_pair_identities(self):
        with tempfile.TemporaryDirectory(prefix="cvi-priority-sidecar-") as directory:
            root=Path(directory);case=root/"case";case.mkdir();phase=root/"plan.json";phase.write_text("{}")
            for method in priority.original.METHODS:
                (case/f"{method}_run.json").write_text("fixture")
                (case/f"{method}_points.npz").write_text("fixture")
            plan=dict(original_plan_sha256="original",priority_source_sha256="source",original_plan_file="/original/plan.json",
                priority_source_file="/snapshot.py",execution_policy=dict(original_worker_limit=7,priority_worker_limit=10,memory_budget_gib=8.))
            before=dict(pairs={"proposed":{}},diagnostic_prefix={})
            priority.execution_sidecar(case,phase,plan,before,10,active_at_launch=10)
            sidecar=priority.original.load(case/"priority_execution.json")
            self.assertEqual(sidecar["original_worker_limit"],7)
            self.assertEqual(sidecar["priority_worker_limit"],10)
            self.assertEqual(sidecar["observed_peak_priority_workers"],10)
            prior=set(sidecar["committed_before_launch_methods"]);scheduled=set(sidecar["methods_scheduled_in_priority"])
            self.assertFalse(prior&scheduled)
            self.assertEqual(prior|scheduled,set(priority.original.METHODS))
            self.assertEqual(set(sidecar["timed_pair_sha256"]),set(priority.original.METHODS))

    def test_scope_and_original_resume_contract(self):
        self.assertEqual(len(priority.TARGETS),10)
        self.assertEqual(set(priority.TARGETS.values()),set(range(10)))
        self.assertTrue(all(k.startswith("multi_n100_m100_seed") for k in priority.TARGETS))
        results=priority.ROOT/"results_n100"
        note={"original_plan_sha256":priority.original.digest(results/"plan.json"),
              "original_wrapper_sha256":priority.original.digest(priority.ROOT/"run_n100_study.py")}
        plan,admission,wrapper=priority.assert_original_compatibility(results,note)
        self.assertEqual(plan["seconds"],1000.)
        self.assertEqual(plan["diagnostic_seconds"],300.)
        self.assertEqual(admission.workers,7)
        self.assertTrue(wrapper.name=="run_n100_study.py")
        note["original_plan_sha256"]="wrong"
        with self.assertRaises(RuntimeError):priority.assert_original_compatibility(results,note)

    def test_durable_intent_and_spawn_pid_window_reconciliation(self):
        case="multi_n100_m100_seed0"
        intent=dict(case=case,path="/test/case",state="launching",pid=None,intent_id="unique")
        with tempfile.TemporaryDirectory(prefix="cvi-priority-test-") as directory:
            base=Path(directory);state=base/"priority";results=base/"results"
            state.mkdir();results.mkdir()
            old={"old_case":{"state":"exited","pid":1,"actual_failure_preserved":True}}
            priority.original.atomic_json(results/"launch_ledger.json",old)
            own={};priority.write_ledgers(state,results,own,intent)
            self.assertEqual(priority.original.load(state/"launch_ledger.json")[case],intent)
            compatible=priority.original.load(results/"launch_ledger.json")
            self.assertEqual(compatible["old_case"],old["old_case"])
            self.assertEqual(compatible[case],intent)
            with self.assertRaises(RuntimeError):priority.reconcile_ledgers(own,compatible)
            # Crash after original PID commit but before own PID commit.
            compatible[case]=dict(intent,pid=123,state="running",process_start="recorded")
            recovered=priority.reconcile_ledgers(own,compatible)
            self.assertEqual(recovered[case]["pid"],123)

    def test_quiescence_unknown_is_not_dead(self):
        note=dict(original_queue_quiescent=True,original_controller={"pid":1},
                  original_wrapper={"pid":2},interrupted_workers=[{"pid":3}])
        with patch.object(priority.original,"process_state",return_value={"alive":True,"evidence":"unknown"}):
            with self.assertRaises(RuntimeError):priority.assert_quiescent(note,Path("/unused"))

    def test_child_and_render_inherit_original_lock(self):
        tree=ast.parse(Path(priority.__file__).read_text())
        calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
               and isinstance(n.func.value,ast.Name) and n.func.value.id=="subprocess"
               and n.func.attr in {"Popen","run"}]
        self.assertEqual(len(calls),2)
        self.assertTrue(all(any(k.arg=="pass_fds" for k in n.keywords) for n in calls))


if __name__=="__main__":unittest.main()
