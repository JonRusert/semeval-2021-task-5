#########################################################################################
Code for "NLPUIOWA at Semeval-2021 Task 5: Transferring Toxic Sets to TagToxic Spans"
#########################################################################################

BLSTM.py -> BLSTM base architecture trained on the 6 different datasets, contains methods for attention, freq-ratio, and described hybrid approaches.
Note that certain parameters will need to be adjusted as training sets differ in inputs. 

Training:
python3 BLSTM_Attention.py OFFis1_offenseval-training-v1.tsv  gold_small_testset-taska.tsv train > BLSTM_Attention_train_out



testPreds/ -> contains predictions of various BLSTM/methods on the provided test set
