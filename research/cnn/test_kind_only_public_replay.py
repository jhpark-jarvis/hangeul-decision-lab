"""Public export metadata and original private-config resolution boundaries."""
import unittest
from pathlib import Path
import kind_only
from kind_only_public_replay import require_export_projection, manifest_input, config_input_adapter, ROOT, CONFIG_PATH


class PublicReaderTests(unittest.TestCase):
    def test_only_authorization_text_projection_allowed(self):
        original={'authorization':'private provenance','objective':{'kindWeight':0.5,'pointWeight':0},'approval':'USER_APPROVED'}
        exported={**original,'authorization':'public provenance'}
        require_export_projection(original,exported)
        for changed in ({**exported,'objective':{'kindWeight':1,'pointWeight':0}},
                        {**exported,'extra':True},{k:v for k,v in exported.items() if k!='approval'},
                        {**exported,'authorization':False}):
            with self.assertRaises(ValueError):require_export_projection(original,changed)

    def test_only_exact_config_path_uses_explicit_original_input(self):
        private=Path('frozen-original-config.json')
        self.assertEqual(manifest_input(CONFIG_PATH,private),private)
        self.assertEqual(manifest_input('research/cnn/kind_only.py',private),ROOT/'research/cnn/kind_only.py')
        self.assertEqual(manifest_input('research/experiments/point-only-objective-v1.json',private),ROOT/'research/experiments/point-only-objective-v1.json')

    def test_registered_loader_restored_after_failure_without_model(self):
        original=kind_only.load
        with self.assertRaisesRegex(RuntimeError,'fixture'):
            with config_input_adapter('missing'):
                self.assertIsNot(kind_only.load,original)
                raise RuntimeError('fixture')
        self.assertIs(kind_only.load,original)


if __name__=='__main__':unittest.main()
