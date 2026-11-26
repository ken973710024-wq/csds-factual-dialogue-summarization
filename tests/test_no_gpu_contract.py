import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'src'
sys.path.insert(0,str(SRC))

spec=importlib.util.spec_from_file_location('manual_fact_audit',SRC/'manual_fact_audit.py')
audit=importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)

class TestManualAudit(unittest.TestCase):
    def test_scoring_human_annotation(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            audit.writecsv(p/'fact_bank.csv',audit.FACT_COLS,[
                {'dialogue_id':'1','fact_id':'F1','topic_id':'order','fact_text':'退款待审核',
                 'source_ids':'C7','key_stage1':'1','key_summary':'1','final_state':'1','notes':''},
                {'dialogue_id':'1','fact_id':'F2','topic_id':'order','fact_text':'需3天',
                 'source_ids':'C8','key_stage1':'1','key_summary':'1','final_state':'0','notes':''},
            ])
            claim={'dialogue_id':'1','arm_blind':'X','output_index':'0','claim_id':'X-0-1','claim_text':'退款待审核',
                   'atomic_checked':'1','supported':'1','object_correct':'1','speaker_correct':'1',
                   'state_correct':'1','is_state_claim':'1','matched_fact_ids':'F1','issue_type':'','notes':''}
            audit.writecsv(p/'claims_to_judge.csv',audit.CLAIM_COLS,[claim])
            (p/'_unblind_after_review.json').write_text(json.dumps({'1':{'qa_lora':'X'}}))
            audit.score(p,'stage1',False)
            results=json.loads((p/'human_fact_scores.json').read_text())['systems']['qa_lora']
            assert results['fact_support_precision']==1
            assert results['key_fact_recall']==0.5
            assert results['final_status_wrong']=='0/1'

    def test_untouched_annotations_cannot_score(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            audit.writecsv(p/'fact_bank.csv',audit.FACT_COLS,[])
            audit.writecsv(p/'claims_to_judge.csv',audit.CLAIM_COLS,[])
            (p/'_unblind_after_review.json').write_text('{}')
            with self.assertRaises(ValueError):audit.score(p,'stage1',False)

if __name__=='__main__':unittest.main()
