# Also support the retained scripts/ entry points and Python without an editable install.
if not __package__:
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath
    _source_root = next(p / 'source/nrs_rl' for p in _BootstrapPath(__file__).resolve().parents
                        if (p / 'source/nrs_rl/nrs_rl').is_dir())
    _bootstrap_sys.path.insert(0, str(_source_root))

import unittest
import numpy as np
from nrs_rl.tasks.manager_based.nrs_rl.model_based.training.train_uniformity_refinement import basis_at,force_filter,affine_objective
from nrs_rl.tasks.manager_based.nrs_rl.model_based.evaluation.uniformity_save_guard import assess_save_guard


class RefinementTests(unittest.TestCase):
    def test_guard_rejects_threshold_failure_and_seed_regression(self):
        mode6={'spatial_cv':.4,'eligible':True};saved={'spatial_cv':.352,'eligible':True}
        for cv in [.36,.356,.352,.353]:
            self.assertFalse(assess_save_guard({'spatial_cv':cv,'eligible':True},saved,mode6)['passed'])
        self.assertTrue(assess_save_guard({'spatial_cv':.344,'eligible':True},saved,mode6)['passed'])
        self.assertFalse(assess_save_guard({'spatial_cv':.30,'eligible':False},saved,mode6)['passed'])

    def test_delayed_force_response_is_causal_and_preserves_dc(self):
        time=np.arange(500)*.008;command=np.where(time>=1.,20.,8.)
        response=force_filter(command,time,.16,.4)[:,0]
        np.testing.assert_allclose(response[time<1.16],8.)
        self.assertGreater(response[-1],19.9)
        self.assertLess(response[160],20.)
        basis=basis_at(np.array([-1.,0.,25.,50.,100.,101.]),100.,5)
        np.testing.assert_allclose(basis.sum(axis=1),1.)
        np.testing.assert_allclose(basis@np.arange(5),[0,0,1,2,4,4])

    def test_affine_cv_gradient_matches_finite_differences(self):
        rng=np.random.default_rng(123);n=9
        matrix=rng.uniform(.01,.1,(150,n));offset=rng.uniform(.01,.2,150)
        a=np.c_[matrix,offset];statistics=[(a.T@a/len(a),a.mean(axis=0))]
        d2=np.diff(np.eye(n),n=2,axis=0);smooth=d2.T@d2/(n-2)
        force=rng.uniform(8.,20.,n);_,analytic=affine_objective(force,statistics,smooth)
        numeric=np.zeros(n)
        for i in range(n):
            delta=np.zeros(n);delta[i]=1.e-4
            numeric[i]=(affine_objective(force+delta,statistics,smooth)[0]-affine_objective(force-delta,statistics,smooth)[0])/2.e-4
        np.testing.assert_allclose(analytic,numeric,atol=1.e-9,rtol=1.e-6)


if __name__=='__main__':unittest.main()
