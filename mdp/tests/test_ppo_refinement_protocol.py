"""Keep continuation lineage and evaluation seeds fixed across save promotions."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from nrs_rl.tasks.manager_based.nrs_rl.mdp.evaluation import refine_fixed_force_ppo as refine


class RefinementProtocolTests(unittest.TestCase):
    def test_rejects_validation_seed_reuse(self):
        for invalid in [
            {'selection_seed': 3201},
            {'heldout_seeds': [3251, 3251]},
            {'heldout_seeds': [3251, 3241]},
            {'heldout_seeds': [3251]},
            {'selection_seed': 1.5},
        ]:
            plan = refine.normalize_plan()
            plan.update(invalid)
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                refine.normalize_plan(plan)

    def test_plan_can_lengthen_rollouts_without_changing_batch_size(self):
        plan = refine.normalize_plan()
        plan['variants'][1].update(num_envs=32, rollout_steps=1024, updates=320)
        result = refine.normalize_plan(plan)
        self.assertEqual(result['variants'][0]['num_envs'] * result['variants'][0]['rollout_steps'],
                         result['variants'][1]['num_envs'] * result['variants'][1]['rollout_steps'])
        self.assertEqual(result['variants'][1]['updates'], 320)

    def test_study_pins_save_despite_registry_promotion(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            registry = base / 'logs/experiments/uniformity_saves'
            archive = registry / 'saved'
            archive.mkdir(parents=True)
            actor, state = archive / 'actor.pt', archive / 'training.pt'
            actor.write_bytes(b'protected actor')
            state.write_bytes(b'protected optimizer and critic')
            record = {'algorithm': 'PPO', 'checkpoint': str(actor),
                      'checkpoint_sha256': refine.sha(actor), 'training_state': str(state),
                      'training_state_sha256': refine.sha(state), 'save_directory': str(archive)}
            (archive / 'SAVE.json').write_text(json.dumps(record))
            pointer = registry / 'PPO_FIXED.json'
            pointer.write_text(json.dumps(record))
            deployment = base / 'deployment'
            deployment.mkdir()
            gui = deployment / 'actor.pt'
            gui.write_bytes(b'independent GUI policy')
            (deployment / 'FIXED_MODEL.json').write_text(json.dumps(
                {'checkpoint': str(gui), 'checkpoint_sha256': refine.sha(gui)}))
            study = base / 'study'
            with patch.object(refine, 'ROOT', base):
                self.assertEqual(refine.initialize(study), record)
                pointer.write_text(json.dumps({'checkpoint': 'later save'}))
                self.assertEqual(refine.preserve(study), record)
                changed = refine.normalize_plan()
                changed['variants'][0]['rollout_steps'] *= 2
                with self.assertRaisesRegex(AssertionError, 'Cannot change'):
                    refine.initialize(study, changed)
                state.write_bytes(b'changed optimizer')
                with self.assertRaises(AssertionError):
                    refine.preserve(study)


if __name__ == '__main__':
    unittest.main()
