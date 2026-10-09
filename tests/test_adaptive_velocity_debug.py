# SPDX-License-Identifier: BSD-3-Clause
"""Legacy console and actual PNG-saving tests without launching Isaac."""
# Also support the retained scripts/ entry points and Python without an editable install.
if not __package__:
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath
    _source_root = next(p / 'source/nrs_rl' for p in _BootstrapPath(__file__).resolve().parents
                        if (p / 'source/nrs_rl/nrs_rl').is_dir())
    _bootstrap_sys.path.insert(0, str(_source_root))

from nrs_rl.tasks.manager_based.nrs_rl.paths import ROOT as _REPO_ROOT
import contextlib
import importlib.util
import io
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import torch

ROOT = _REPO_ROOT / "source/nrs_rl/nrs_rl/tasks/manager_based/nrs_rl"
PREFIX = "nrs_rl.tasks.manager_based.nrs_rl"


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


debug = load_file("adaptive_velocity_debug", ROOT / "utils/adaptive_velocity_debug.py")


class EpisodeDebugPrinterTests(unittest.TestCase):
    def test_legacy_zero_based_cadence(self):
        printer = debug.EpisodeDebugPrinter(0, 50, writer=lambda _: None)
        printer.reset()
        self.assertEqual([i for i in range(151) if printer.tick()], [0, 50, 100, 150])

    def test_nonpositive_interval_prints_every_step(self):
        for interval in (0, -1):
            printer = debug.EpisodeDebugPrinter(0, interval)
            self.assertTrue(all(printer.tick() for _ in range(10)))

    def test_reset_counts_started_episodes_without_new_format(self):
        output = []
        printer = debug.EpisodeDebugPrinter(3, 50, writer=output.append)
        printer.reset()
        printer.reset()
        self.assertEqual(printer.episode, 1)
        printer.tick()
        printer.reset()
        self.assertEqual((printer.episode, printer.step), (2, 0))
        self.assertEqual(output, [])

    def test_nonuniform_arc_to_legacy_index(self):
        for distance, index in [(-1, 0), (.5, .5), (1, 1), (3, 1.5), (7.5, 2.5), (12, 3)]:
            self.assertAlmostEqual(debug.arc_to_index(distance, [0., 1., 5., 10.]), index)

    def test_format_matches_e4efeeb_golden(self):
        actual = debug.format_polishing_live(
            episode=2, step=50, env_id=0, current_index=12, last_index=100,
            target_index=13, cursor=12.5, current_pose=[1, 2, 3, .1, .2, .3],
            target_pose=[4, 5, 6, .4, .5, .6], command_pose=[7, 8, 9],
            target_force=10., normal_force=9.25, sliding_velocity=6.,
            removal_rate=55.5, cumulative_removal=123.25, fn_offset=.75,
            action=.25, index_rate=.5, path_error_xy=2.5, reward_debug="| last_total=0.100000",
        )
        expected = (
            "\n[Polishing Live] ep2 step=50 env=0 | hdf5_index=12/100 (12.5%) | target_index=13 | cursor=12.500\n"
            "  current xyz/wxyz = (1.000, 2.000, 3.000) / (0.1000, 0.2000, 0.3000)\n"
            "  target  xyz/wxyz = (4.000, 5.000, 6.000) / (0.4000, 0.5000, 0.6000)\n"
            "  command xyz      = (7.000, 8.000, 9.000)\n"
            "  force/speed      = | target_force_N=10.0000 | normal_force_N=9.2500 "
            "| sliding_velocity_mm_s=6.0000 | removal_rate_N_mm_s=55.5000 | cumulative_removal=123.2500\n"
            "  control          = | fn_offset_mm=0.7500 | action=0.2500 | index_rate=0.5000 | path_err_xy_mm=2.500\n"
            "  rewards          = | last_total=0.100000\n"
        )
        self.assertEqual(actual, expected)

    def test_tqdm_writer_with_active_progress_bar(self):
        from tqdm import tqdm
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            with tqdm(total=10, file=stderr, mininterval=0, disable=False) as bar:
                bar.update(1)
                debug.write_debug("[Polishing Live] first line\n  second line")
                bar.update(1)
        self.assertEqual(stdout.getvalue(), "[Polishing Live] first line\n  second line\n")
        self.assertIn("2/10", stderr.getvalue())


class VisualizationTests(unittest.TestCase):
    def setUp(self):
        # Stub only Isaac-dependent imports; the actual plotting and saving
        # implementation is exercised, including PNG encoding.
        self.messages = []
        modules = {
            PREFIX + ".mdp.observation": types.SimpleNamespace(_hdf5_position=None),
            PREFIX + ".assets.assets.sensors.six_axis_ft_sensor": types.SimpleNamespace(),
            PREFIX + ".utils.debug": types.SimpleNamespace(
                print_info=self.messages.append, print_exception=lambda *args: self.fail(str(args))),
            PREFIX + ".utils.adaptive_velocity_debug": debug,
        }
        patcher = mock.patch.dict(sys.modules, modules)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.vis = load_file(PREFIX + ".utils.visualization", ROOT / "utils/visualization.py")
        directory = tempfile.TemporaryDirectory(prefix="nrs_png_unit_")
        self.addCleanup(directory.cleanup)
        self.vis.RUN_LOG_DIR = Path(directory.name)
        self.vis.REWARD_LOG_DIR = self.vis.RUN_LOG_DIR / "reward_logs"
        self.env = types.SimpleNamespace(
            common_step_counter=0, step_dt=.008,
            cfg=types.SimpleNamespace(visualization=types.SimpleNamespace(
                enable_visualizer=True, save_interval_episodes=1)),
            reward_manager=types.SimpleNamespace(_episode_sums={
                "realized_removal": torch.tensor([10.]), "safety_shield": torch.tensor([-2.75])}),
        )
        self.term = types.SimpleNamespace(
            path_cursor_mm=torch.tensor([0.]), _diagnostic_arc_mm=[0., 1., 5., 10.],
            current_sliding_velocity_mm_s=torch.tensor([6.]),
            current_mrr_n_mm_s=torch.tensor([0.]), realized_removal_step=torch.tensor([0.]),
        )

    def record(self, count, *, contact=True):
        increments = []
        for i in range(count):
            step = self.env.common_step_counter
            removal = .48 if contact and i >= 2 else 0.
            if contact and i == count - 1:
                removal = 1.234  # distinctive terminal sample
            self.term.path_cursor_mm[0] = min(10., step * .1)
            self.term.current_mrr_n_mm_s[0] = removal / self.env.step_dt
            self.term.realized_removal_step[0] = removal
            pose = torch.tensor([800. + step * .04, 350. + step * .02, 100., 0., 0., 0.])
            self.vis.record_control_step(self.env, self.term, pose, 100. if i < 2 else 10.)
            self.vis.record_control_step(self.env, self.term, pose, 10.)  # must deduplicate
            increments.append(float(self.term.realized_removal_step[0]))
            self.env.common_step_counter += 1
        return sum(increments)

    def test_pngs_terminal_metrics_rewards_and_reset_isolation(self):
        from PIL import Image
        expected_removal = self.record(12)
        self.assertEqual(len(self.vis._rl_time_buffer), 12)
        self.vis.on_episode_reset(self.env, torch.tensor([1]))
        self.assertFalse((self.vis.RUN_LOG_DIR / "ep1").exists())
        self.vis.on_episode_reset(self.env, torch.tensor([0]))
        self.assertAlmostEqual(self.vis._summary_metrics["total_removal"][-1], expected_removal)
        self.assertEqual(self.vis._summary_metrics["samples"][-1], 12)
        self.assertEqual(self.vis._summary_metrics["episode_reward"][-1], 7.25)
        expected = {
            "01_preston_rate_heatmap.png", "02_adaptive_preston_rate_profile.png",
            "03_adaptive_force_velocity_signals.png", "04_contact_path_3d.png",
            "09_command_tracking.png",
            "05_preston_rate_comparison.png", "06_constant_force_velocity_signals.png",
            "07_preston_rate_profile_comparison.png",
        }
        files = list((self.vis.RUN_LOG_DIR / "ep1").glob("*.png"))
        self.assertEqual({p.name for p in files}, expected)
        for path in files + [self.vis.REWARD_LOG_DIR / "00_reward_components.png"]:
            with Image.open(path) as png:
                self.assertGreater(png.width, 100)
                png.verify()
        summary = (self.vis.RUN_LOG_DIR / "ep1/00_summary.txt").read_text()
        self.assertIn("contact_start_index: 2", summary)
        import numpy as np
        trace = np.load(self.vis.RUN_LOG_DIR / "ep1/08_control_trace.npz")
        self.assertEqual(len(trace["time_s"]), 12)
        self.assertAlmostEqual(float(trace["removal_step_n_mm"].sum()), expected_removal)
        self.assertAlmostEqual(float(trace["removal_step_n_mm"][-1]), 1.234)
        self.assertTrue((self.vis.RUN_LOG_DIR / "00_episode_summary.csv").exists())
        self.assertTrue(any("[STAMP] Ep 1 Saved." in s for s in self.messages))
        self.assertFalse(self.vis._rl_time_buffer)
        self.vis.on_episode_reset(self.env, torch.tensor([0]))
        self.assertEqual(self.vis._episode_counter, 2)
        self.assertFalse((self.vis.RUN_LOG_DIR / "ep2").exists())

    def test_save_interval_and_short_episode_cleanup(self):
        self.env.cfg.visualization.save_interval_episodes = 2
        self.record(6)
        self.vis.on_episode_reset(self.env, [0])
        self.assertFalse((self.vis.RUN_LOG_DIR / "ep1").exists())
        self.assertEqual(self.vis._episode_counter, 2)
        self.assertFalse(self.vis._rl_removal_step_buffer)
        self.record(3)
        self.vis.on_episode_reset(self.env, [0])
        self.assertEqual(self.vis._episode_counter, 3)
        self.assertFalse(self.vis._rl_time_buffer)
        self.assertFalse((self.vis.RUN_LOG_DIR / "ep2").exists())

    def test_no_contact_still_saves_basic_pngs(self):
        self.record(6, contact=False)
        self.vis.on_episode_reset(self.env, [0])
        self.assertEqual(self.vis._summary_metrics["total_removal"][-1], 0.)
        self.assertEqual(len(list((self.vis.RUN_LOG_DIR / "ep1").glob("*.png"))), 4)

    def test_adaptive_curve_is_raw_and_axis_retains_stops_and_spikes(self):
        import numpy as np
        rate = np.array([0., 5., 8., 20., 4., 0., 15., 0.])
        profiles = self.vis._velocity_comparison_profiles(
            np.arange(len(rate)) * .008, np.full_like(rate, 10.), rate / 10., rate)
        np.testing.assert_array_equal(profiles["reward_action_rate"], rate)
        self.assertAlmostEqual(profiles["reward_rate_cv"], rate.std() / rate.mean())
        self.assertGreater(profiles["constant_rate"][1], 0.)
        self.assertGreater(profiles["constant_rate"][0], 0.)
        low, high = self.vis._contact_rate_axis_limits(rate, contact_mask=rate > 0)
        self.assertLessEqual(low, 0.)
        self.assertGreaterEqual(high, 20.)

    def test_processing_cv_includes_zero_removal(self):
        import numpy as np
        rate = np.array([10., 0., 10., 0., 10., 0.])
        xyz = np.column_stack((np.arange(6), np.zeros(6), np.full(6, 100.)))
        summary = self.vis._compute_episode_summary(
            1, np.arange(6) * .008, xyz, np.full(6, 10.), rate / 10.,
            rate * .008, rate, 0, 0., np.ones(6, dtype=bool))
        self.assertEqual(summary["contact_cv_removal"], 0.)
        self.assertAlmostEqual(summary["processing_rate_cv"], 1.)
        self.assertEqual(summary["processing_zero_removal_fraction"], .5)

    def test_rate_heatmap_is_independent_of_dwell_sample_density(self):
        import numpy as np
        x = np.r_[np.zeros(1000), np.full(10, 40.)]
        y = np.zeros_like(x)
        rate = np.full_like(x, 60.)
        extent = [-10., 50., -10., 10.]
        grid, _ = self.vis._mean_rate_heatmap_display(x, y, rate, extent, 80)
        np.testing.assert_allclose(grid.compressed(), 60., atol=1e-10)
        self.assertTrue(np.ma.getmaskarray(grid).any())
        doubled, _ = self.vis._mean_rate_heatmap_display(x, y, rate * 2, extent, 80)
        # Preserve absolute scale between maps, never normalize each to [0,1].
        np.testing.assert_allclose(doubled.compressed(), 2 * grid.compressed())
        self.assertAlmostEqual(self.vis._rate_heatmap_upper(grid, doubled), 120.)


if __name__ == "__main__":
    unittest.main()
