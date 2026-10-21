"""Reader isolation and failure controls; no real-data forwards or training."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import kind_only
from kind_only_replay import data_only_loader,main
from spatial_audit import compute_guard


class ReaderTests(unittest.TestCase):
    def test_legacy_loader_replaced_explicitly_and_restored_on_failure(self):
        original=kind_only.spatial_load
        with self.assertRaisesRegex(RuntimeError,'toy'):
            with compute_guard(),data_only_loader():
                self.assertIsNot(kind_only.spatial_load,original)
                self.assertEqual(kind_only.spatial_load.__name__,'readonly_data')
                raise RuntimeError('toy')
        self.assertIs(kind_only.spatial_load,original)

    def test_missing_input_failure_preserved_exclusive_output_no_ledger(self):
        root=Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory(dir=root/'.research-output') as d:
            args=['kind_only_replay.py']
            for key in ('source','original','experiment','execution','recorded'):args.extend(['--'+key,'missing'])
            args.extend(['--output',str(Path(d)/'out')])
            with patch.object(sys,'argv',args),patch('kind_only_replay.subprocess.check_output',return_value=''):
                with self.assertRaises(FileNotFoundError):main()
                before=(Path(d)/'out/failure.json').read_bytes()
                with self.assertRaises(FileExistsError):main()
                self.assertEqual((Path(d)/'out/failure.json').read_bytes(),before)
                self.assertFalse((Path(d)/'out/forward-count.jsonl').exists())


if __name__=='__main__':unittest.main()
