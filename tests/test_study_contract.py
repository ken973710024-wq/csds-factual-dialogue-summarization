"""CPU-only contract regression tests (requires existing project validation modules).
Run: PYTHONPATH=src python tests/test_study_contract.py
"""
import json
from qa_summary_study_common import (
    qa_input,stage2_input,select_rows,sha_text,validate_cache_record,
)
from prepare_summary_qa_training import make_records
from train_summary_qa_conditioned import prepare_tokenized
from validate_summary_qa_training import report_group

SRC={
    'DialogueID':42,'QRole':'用户',
    'Dialogue':[
       {'turn':0,'speaker':'Q','utterance':'多久可以回电？'},
       {'turn':1,'speaker':'A','utterance':'专员将在[数字]小时之内回电。'},
    ],
    # Trap: human QA labels are NEVER copied into the Stage 1 prompt.
    'QA':[{'QueSumm':'DO_NOT_LEAK_HUMAN_QA', 'AnsSummLong':'DO_NOT_LEAK'}],
    'AgentSumm':['DO_NOT_LEAK_TARGET_SUMMARY'],
}
QA=[{'category':'回电时效','question':'何时回电','answer':'专员会在[数字]小时内回电',
     'question_ids':['U0'],'evidence_ids':['C0']}]
assert 'DO_NOT_LEAK' not in qa_input(SRC)
assert '[U0]' in qa_input(SRC) and '[C0]' in qa_input(SRC)
assert 'DO_NOT_LEAK' not in stage2_input(SRC,QA)
assert 'evidence_quotes' in stage2_input(SRC,QA)
assert '专员将在[数字]小时之内回电' in stage2_input(SRC,QA)
assert '\n【QA 核对清单】\n' not in stage2_input(SRC)
assert '\n【QA 核对清单】\n' in stage2_input(SRC,QA)
assert select_rows(list(range(20)),6)==select_rows(list(range(20)),6)

P={'sample_sha':'synthetic','train_sha256':'x'}
record={'DialogueID':42,'qa_prompt_sha256':sha_text(qa_input(SRC)),'provenance':P,'qa':QA}
assert validate_cache_record(record,SRC,P)
row={'DialogueID':42,'source':SRC,'target':'客服说明专员将在[数字]小时之内回电。'}
a,b=next(iter(make_records([row],{'42':record},P)))
assert a['target']==b['target'] and a['DialogueID']==b['DialogueID']
assert '\n【QA 核对清单】\n' not in a['prompt'] and '\n【QA 核对清单】\n' in b['prompt']

class DummyTokenizer:
    def apply_chat_template(self,messages,tokenize=True,add_generation_prompt=False):
        t=[1,len(messages[0]['content']),3]
        if len(messages)>1: t+=[4,len(messages[1]['content'])]
        return t

a_tokens=prepare_tokenized(DummyTokenizer(),[a],99)
b_tokens=prepare_tokenized(DummyTokenizer(),[b],99)
assert a_tokens[0]['labels'][:3]==[-100,-100,-100]
assert a_tokens[0]['labels'][3:]==[4,len(a['target'])]
assert b_tokens[0]['labels'][3:]==a_tokens[0]['labels'][3:]

s={'rouge_l_f1':.4,'rouge_l_precision':.4,'rouge_l_recall':.4,'chars':30}
t={'rouge_l_f1':.5,'rouge_l_precision':.5,'rouge_l_recall':.5,'chars':31}
sample={'A_old_direct':{'scores':s},'C_old_qa_guided':{'scores':s},
        'G_new_matched_direct':{'scores':s},'F_new_qa_trained':{'scores':t}}
result=report_group([sample,sample],'synthetic')
assert result['F_minus_G_matched_QA_training_effect']['f1_delta']==.1
assert result['F_minus_G_matched_QA_training_effect']['right_wins']==2
print('ALL CPU study contract tests passed.')
