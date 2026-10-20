"""Offline software checks. Synthetic data are NEVER PathMNIST results."""
from pathlib import Path
import shutil
import tempfile
import unittest
import numpy as np
import torch
from torch.utils.data import TensorDataset
from histology.core import CompactCNN, environment, fit_model, make_loader, pass_batches, save_json, score_predictions, seed_everything, write_evaluation
from histology.data import sample_training_indices

ROOT = Path(__file__).resolve().parents[1]


def synthetic_data(n, seed):
    rng = np.random.default_rng(seed)
    labels = np.arange(n) % 9
    palette = np.array([[.9,.1,.1], [.1,.9,.1], [.1,.1,.9], [.9,.9,.1], [.9,.1,.9],
                        [.1,.9,.9], [.2,.2,.2], [.5,.5,.5], [.9,.9,.9]], dtype=np.float32)
    images = np.clip(palette[labels, None, None, :] + rng.normal(0, .03, (n,28,28,3)), 0, 1).astype(np.float32)
    tensor = torch.from_numpy(images).permute(0,3,1,2).contiguous() * 2 - 1
    return TensorDataset(tensor, torch.from_numpy(labels)), (images * 255).astype(np.uint8)


class IntegrityTests(unittest.TestCase):
    def test_metrics_known_answers_and_shape_guard(self):
        truth = np.array([0, 1, 2])
        good = score_predictions(truth, np.eye(3), ['a','b','c'])
        self.assertEqual(good['accuracy'], 1)
        self.assertEqual(good['macro_f1'], 1)
        mixed = score_predictions(truth, np.eye(3)[[0, 0, 2]], ['a','b','c'])
        self.assertAlmostEqual(mixed['accuracy'], 2/3)
        self.assertAlmostEqual(mixed['macro_f1'], 5/9)
        with self.assertRaises(ValueError):
            score_predictions(truth, np.ones((3,3)), ['a','b','c'])

    def test_train_subset_reproducible_and_local_to_input(self):
        labels = np.repeat(np.arange(9), 30)
        first = sample_training_indices(labels, 90, 42)
        second = sample_training_indices(labels, 90, 42)
        self.assertTrue(np.array_equal(first, second))
        self.assertEqual(len(set(first)), 90)
        self.assertTrue((first >= 0).all() and (first < len(labels)).all())
        self.assertTrue(np.array_equal(np.bincount(labels[first]), [10]*9))
        self.assertTrue(np.array_equal(sample_training_indices(labels, None, 42), np.arange(270)))

    def test_synthetic_training_checkpoint_and_reports(self):
        torch.set_num_threads(2)
        seed_everything(42)
        train, _ = synthetic_data(180, 10)
        val, _ = synthetic_data(90, 20)
        final, final_images = synthetic_data(90, 30)
        model = CompactCNN()
        before = model.features[0].weight.detach().clone()
        names = [f'synthetic_class_{i}' for i in range(9)]
        (ROOT/'outputs').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='histology_smoke_', dir=ROOT/'outputs') as tmp:
            output = Path(tmp)
            summary = fit_model(model, make_loader(train, 30, True), make_loader(val, 30), names, output,
                                epochs=4, patience=4, metadata={'scope':'SYNTHETIC SOFTWARE SMOKE TEST - NOT HISTOLOGY'})
            self.assertFalse(torch.equal(before, model.features[0].weight.detach()))
            loss, truth, probs = pass_batches(model, make_loader(final, 30), 'cpu')
            self.assertTrue(np.isfinite(loss))
            self.assertEqual(probs.shape, (90,9))
            replica = CompactCNN()
            replica.load_state_dict(torch.load(output/'best.pt', weights_only=True)['model_state'])
            _, truth2, probs2 = pass_batches(replica, make_loader(final,30), 'cpu')
            self.assertTrue(np.array_equal(truth, truth2))
            self.assertTrue(np.array_equal(probs, probs2))
            metrics = write_evaluation(output/'synthetic_evaluation', truth, probs, names, final_images,
                                       scope='SYNTHETIC SOFTWARE SMOKE TEST - NOT HISTOLOGY')
            self.assertEqual(sum(map(sum, metrics['confusion_matrix'])), 90)
            for name in ['metrics.json','classification_report.csv','confusion_matrix.png','correct_examples.png','incorrect_examples.png','predictions.csv']:
                self.assertTrue((output/'synthetic_evaluation'/name).is_file())
            destination = ROOT/'outputs'/'synthetic_smoke'
            destination.mkdir(parents=True, exist_ok=True)
            shutil.copytree(output, destination, dirs_exist_ok=True)
            save_json(destination/'execution.json', {
                'status':'passed', 'kind':'synthetic_software_smoke_test', 'pathmnist_training_executed':False,
                'resnet18_executed':False, 'train_n':180, 'validation_n':90, 'evaluation_n':90,
                'epochs_executed':summary['epochs_completed'], 'checkpoint_reload_predictions_identical':True,
                'parameters_updated':True, 'environment':environment(),
                'limitations':['No official images downloaded','torchvision and medmnist unavailable locally',
                               'CUDA unavailable in installed CPU-only torch','Synthetic scores are not biomedical performance']})


if __name__ == '__main__':
    unittest.main(verbosity=2)
