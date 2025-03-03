import numpy as np
from tqdm import tqdm
from train.regression import *
from tqdm import tqdm
from huggingface_hub import hf_hub_download
import click

class SEScore2:
    def __init__(self, version):
        # load in the weights of SEScore2
        exp_config.device_id = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(f"xlm-roberta-large")
        if version == 'seg':
            cur_addr = hf_hub_download(repo_id="xu1998hz/sescore2_en_pretrained", filename="sescorex_seg.ckpt")
        elif version == 'sys':
            cur_addr = hf_hub_download(repo_id="xu1998hz/sescore2_en_pretrained", filename="sescorex_sys.ckpt")
        elif version == "pretrained":
            cur_addr = hf_hub_download(repo_id="xu1998hz/sescore2_en_pretrained", filename="sescore2_en_original.ckpt")
        else:
            print("We currently support three modes: seg, sys and pretrained")
            exit(1)

        self.model = torch.load(cur_addr).to(exp_config.device_id)
        self.model.eval()

    def score(self, refs, outs, batch_size):
        scores_ls = []
        cur_data_dict = {'pivot': refs, 'mt': outs}
        cur_data_loader = preprocess_data(cur_data_dict, self.tokenizer, exp_config.max_length, batch_size, shuffle=False, sampler=False, mode='test')

        for batch in tqdm(cur_data_loader, desc='Processing batches'):
            # generate a batch of ref, mt embeddings
            score = self.model(batch, 'last_layer').squeeze(1).tolist()
            scores_ls.extend(score)
        return scores_ls

# test the results
@click.command()
@click.option('-lang')
def main(lang):
    if lang == "zh":
        file_addr = "../.mt-metrics-eval/mt-metrics-eval-v2/wmt24/human-scores/ja-zh.mqm.seg.score"
        path = "../.mt-metrics-eval/mt-metrics-eval-v2/wmt24/system-outputs/ja-zh"
    elif lang == "es":
        file_addr = "../.mt-metrics-eval/mt-metrics-eval-v2/wmt24/human-scores/en-es.mqm.seg.score"
        path = "../.mt-metrics-eval/mt-metrics-eval-v2/wmt24/system-outputs/en-es"
    elif lang == "en":
        file_addr = "../.mt-metrics-eval/mt-metrics-eval-v2/wmt22/human-scores/zh-en.mqm.seg.score"
        path = "../.mt-metrics-eval/mt-metrics-eval-v2/wmt22/system-outputs/zh-en"
    elif lang == "de":
        file_addr = "../.mt-metrics-eval/mt-metrics-eval-v2/wmt22/human-scores/en-de.mqm.seg.score"
        path = "../.mt-metrics-eval/mt-metrics-eval-v2/wmt22/system-outputs/en-de"
    elif lang == "ru":
        file_addr = "../.mt-metrics-eval/mt-metrics-eval-v2/wmt22/human-scores/en-ru.mqm.seg.score"
        path = "../.mt-metrics-eval/mt-metrics-eval-v2/wmt22/system-outputs/en-ru"
    else:
        print("Don't put langs other than en, ru, es, zh or de")
        exit(1)

    lines = open(file_addr, 'r').readlines()
    sys_ls = []
    for line in lines:
        sys_name = line.split('\t')[0]
        if sys_name not in sys_ls:
            sys_ls.append(sys_name)
    print(sys_ls)
        

    scorer = SEScore2('sys')
    save_file = open(f'rescale_data/sys/{lang}.sescorex', 'w')
    for sys_name in sys_ls:
        refs = open(f"{path}/refA.txt", 'r').readlines()
        outs = open(f"{path}/{sys_name}.txt", 'r').readlines()

        scores_ls = scorer.score(refs, outs, 32)
        save_ls = []
        for score in scores_ls:
            save_ls+=[f"{sys_name}\t{score}\n"]
        save_file.writelines(save_ls)
    save_file.close()

if __name__ == "__main__":
    main()
