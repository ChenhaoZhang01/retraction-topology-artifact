"""Unit tests for the retraction-topology analysis (stdlib unittest; no network).

Run from the project root:   python tests/test_topology.py

Two things matter for this paper and both are tested here. First, the headline must be leakage-safe:
retraction_proximity (which can encode post-publication retractions) must never be in the clean
feature set. Second, the evaluation must actually credit a feature that carries signal more than a
feature that is pure noise, otherwise a reported AUC "gain" would be meaningless.
"""
import os, sys, unittest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import topology


class TestShortId(unittest.TestCase):
    def test_strips_openalex_prefix(self):
        self.assertEqual(topology.short_id("https://openalex.org/W12345"), "W12345")

    def test_passes_through_none(self):
        self.assertIsNone(topology.short_id(None))


class TestLeakageInvariant(unittest.TestCase):
    """The clean feature set is the leakage guarantee the whole headline rests on."""
    def test_proximity_excluded_from_clean(self):
        self.assertNotIn("retraction_proximity", topology.TOPO_CLEAN)

    def test_proximity_present_only_in_full(self):
        self.assertIn("retraction_proximity", topology.TOPO_FULL)

    def test_clean_is_subset_of_full(self):
        self.assertTrue(set(topology.TOPO_CLEAN).issubset(set(topology.TOPO_FULL)))


class TestEvalModelsCreditsSignal(unittest.TestCase):
    """A reported topology 'gain' is only meaningful if the evaluator detects real signal and
    ignores noise. We feed a known-informative topology column vs a pure-noise one and require the
    informative gain to clearly exceed the noise gain."""
    def _xy(self, informative, seed=0):
        rng = np.random.default_rng(seed)
        n = 160
        y = np.array([0, 1] * (n // 2))
        years = rng.integers(2010, 2020, n)
        ncols = len(topology.ALLCOLS)
        X = rng.normal(size=(n, ncols))                      # metadata = noise
        if informative:
            sc = topology.ALLCOLS.index("self_citation_rate")
            X[:, sc] = y + rng.normal(scale=0.3, size=n)     # one topology feature carries signal
        return X, y, years

    def test_signal_gain_exceeds_noise_gain(self):
        base, full = topology.FEATS, topology.FEATS + topology.TOPO_CLEAN
        Xs, y, yr = self._xy(informative=True, seed=1)
        Xn, _, _ = self._xy(informative=False, seed=1)
        gain_signal = topology.eval_models(Xs, y, yr, base, full, "sig")["cv_gain"]
        gain_noise = topology.eval_models(Xn, y, yr, base, full, "noise")["cv_gain"]
        self.assertGreater(gain_signal, 0.02)                # real signal lifts AUC
        self.assertGreater(gain_signal, gain_noise + 0.03)   # and clearly more than noise does


if __name__ == "__main__":
    unittest.main(verbosity=2)
