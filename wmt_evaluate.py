from comet import download_model, load_from_checkpoint
import click
import torch
from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModel,
    DataCollatorWithPadding,
)
import sys
from mt_metrics_eval import data
from train.regression import *
import functools
from typing import Any
from huggingface_hub import hf_hub_download
# 1) obtain the triplet pairs (ranking difference and use it to determine margins)
# 2) use margin ranking loss for all triplets, based on rankings

human_mapping_dict = {
    "wmt23": {
        'cs-uk': ['refA'],
        'en-cs': ['refA'],
        'en-ja': ['refA'],
        'en-zh': ['refA'],
        'zh-en': ['refA'],
        'en-de': ['refA'],
        'he-en': ['refA'],
    },
    "wmt21.flores": {
        'bn-hi': ['refA'],
        'hi-bn': ['refA'],
        'xh-zu': ['refA'],
        'zu-xh': ['refA'],
    },
    "wmt22": {
        'en-zh': ['refB'],
        'zh-en': ['refA'],
        'en-de': ['refA'],
        'en-ru': ['refA'],
        'en-cs': ['refC'],
        'en-liv': ['refA'],
        'en-hr': ['refstud'],
        "sah-ru": ['refA'],
        "en-uk": ['refA'],
        "en-ja": ['refA'],
        "cs-uk": ['refA']
    },
    "wmt21.news": {
        'en-de': ['refA'],
        'en-ru': ['refA'],
        'zh-en': ['refA'],
        'en-is': ['refA'],
        'en-cs': ['refA'],
        'en-ha': ['refA'],
        'en-ja': ['refA'],
        'en-zh': ['refA'],
        'de-fr': ['refA']
    },
    "wmt20": {
        'en-de': ['refb'],
        'zh-en': ['refb'],
        'en-pl': ['ref']
    },
}
    

def preprocess_data(triplets_score_dict, tokenizer, max_length, batch_size, shuffle=True, sampler=True, mode='train'):
    if mode == 'train':
        ds = Dataset.from_dict({"pivot": triplets_score_dict['pivot'], 'mt': triplets_score_dict['mt'], 'score': triplets_score_dict['score']})
    else:
        ds = Dataset.from_dict({"pivot": triplets_score_dict['pivot'], 'mt': triplets_score_dict['mt']})

    def preprocess_function(examples):
        model_inputs = {}
        # pivot examples added into dataloader, one pivot per instance
        pivot = tokenizer(examples['pivot'], max_length=max_length, padding='max_length', truncation=True)
        model_inputs['input_ids'], model_inputs['pivot_attn_masks'] = pivot['input_ids'], pivot['attention_mask']
        # mt examples added into dataloader, one mt per instance
        mt = tokenizer(examples['mt'], max_length=max_length, padding='max_length', truncation=True)
        model_inputs['mt_input_ids'], model_inputs['mt_attn_masks'] = mt["input_ids"], mt['attention_mask']
        # store the labels in model inputs
        if mode == 'train':
            model_inputs['score'] = examples['score']
        return model_inputs

    processed_datasets = ds.map(
        preprocess_function,
        batched=True,
        num_proc=None,
        remove_columns=ds.column_names,
        desc="Running tokenizer on dataset",
    )

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding='max_length',
        max_length=max_length,
        pad_to_multiple_of=None,
        return_tensors = 'pt'
    )

    if sampler:
        data_sampler = torch.utils.data.distributed.DistributedSampler(processed_datasets, shuffle=shuffle)
        dataloader = DataLoader(processed_datasets, batch_size=batch_size, collate_fn=data_collator, sampler=data_sampler)
    else:
        dataloader = DataLoader(processed_datasets, batch_size=batch_size, collate_fn=data_collator, shuffle=shuffle)
    return dataloader

def baselines_cl_eval(mt_outs_dict, refs, emb_type, model, batch_size, tokenizer):
    with torch.no_grad():
        # load tokenizer and models, already specified addr for tokenizer
        mt_scores_dict = {}
        # generate src embeddings
        for mt_name, mt_outs in mt_outs_dict.items():   
            mt_scores_dict[mt_name] = []
            cur_data_dict = {'pivot': refs, 'mt': mt_outs}
            cur_data_loader = preprocess_data(cur_data_dict, tokenizer, exp_config.max_length, batch_size, shuffle=False, sampler=False, mode='test')
            for batch in cur_data_loader:
                # generate a batch of ref, mt embeddings
                score = model(batch, emb_type).squeeze(1).tolist()
                mt_scores_dict[mt_name].extend(score)
        return mt_scores_dict

def baselines_comet_eval(mt_outs_dict, refs, srcs, model, batch_size):
    with torch.no_grad():
        # load tokenizer and models, already specified addr for tokenizer
        mt_scores_dict = {}
        # generate src embeddings
        for mt_name, mt_outs in mt_outs_dict.items():
            mt_scores_dict[mt_name] = []
            cur_data_dict = []
            for ref, src, mt in zip(refs, srcs, mt_outs):
                cur_data_dict.append({"src": src, "mt": mt, "ref": ref})
            model_output = model.predict(cur_data_dict, batch_size=batch_size, gpus=1)
            mt_scores_dict[mt_name].extend(model_output[0])
        return mt_scores_dict

def store_corr_eval(evs, mt_scores_dict, mode, wmt, lang, gold_scores):
    # process the ground truth human ratings
    mqm_scores = evs.Scores(mode, gold_scores)
    #print(mqm_scores.keys())
    qm_no_human = set(mqm_scores) - set(evs.all_refs)
    #qm_human = qm_no_human.copy()
    #print(qm_human)
    #qm_human.update(human_mapping_dict[wmt][lang])
    save_dict, temp_ls = {}, []

    if mode == 'sys':
        # compute system-level scores (overwrite) otherwise seg scores are available already
        mt_scores_dict = {mt_name: [sum(scores)/len(scores)] for mt_name, scores in mt_scores_dict.items()}
    #mqm_bp = evs.Correlation(mqm_scores, mt_scores_dict, qm_human)
    mqm_bp_no = evs.Correlation(mqm_scores, mt_scores_dict, qm_no_human)

    if mode == 'seg':
        #save_dict['seg_system_human']=mqm_bp.Kendall()[0]
        save_dict['seg_system']=mqm_bp_no.Kendall()[0]
        corr_fn = functools.partial(mqm_bp_no.KendallWithTiesOpt,sample_rate=1.0)
        no_grouping = corr_fn()[0]
        group_by_item, _, num_items = corr_fn(average_by="item")
        print(f"Acc-t: no_grouping: {no_grouping}, group_by_item: {group_by_item}")
        corr_fn = mqm_bp_no.Pearson
        no_grouping = corr_fn()[0]
        group_by_item, _, num_items = corr_fn(average_by="item")
        print(f"Pearson: no_grouping: {no_grouping}, group_by_item: {group_by_item}")
        corr_fn = mqm_bp_no.Kendall
        no_grouping = corr_fn()[0]
        group_by_item, _, num_items = corr_fn(average_by="item")
        print(f"Kendall: no_grouping: {no_grouping}, group_by_item: {group_by_item}")
        temp_ls.append(mqm_bp_no.Kendall()[0])
    elif mode == 'sys':
        #save_dict['sys_system_human']=mqm_bp.Pearson()[0]
        save_dict['sys_system']=mqm_bp_no.Pearson()[0]
        temp_ls.append(mqm_bp_no.Pearson()[0])
    else:
        print('Please choose between seg and sys!')
        exit(1)
    return max(temp_ls), save_dict


@click.command()
@click.option('-lr', type=float, help="learning rate", default=1e-5)
@click.option('-lang_dir', type=str, help="en_de or zh_en", default="zh_en")
@click.option('-emb_type', type=str, help="choose from last_layer, avg_first_last and states_concat", default="last_layer")
@click.option('-wmt', type=str)
@click.option('-gold_scores', type=str)
@click.option('-hidden_size', type=int, help="model base's hidden size dim: 1024 or 1152")
@click.option('-model_addr', type=str, help="The addr of the model weight")
@click.option('-model_base', type=str, help="The addr of the model weight")
def main(lang_dir, emb_type, lr, hidden_size, \
            model_addr, model_base, wmt, gold_scores):
    # load in eval data
    if model_base == 'comet':
        lang = lang_dir.replace('_', '-')
        print(f"Lang: {lang}, WMT: {wmt}, Gold_scores: {gold_scores}")
        evs = data.EvalSet(wmt, lang)
        mt_outs_dict, refs, srcs = evs.sys_outputs, evs.all_refs[evs.std_ref], evs.src
        model_path = download_model("Unbabel/wmt22-comet-da")
        model = load_from_checkpoint(model_path)
        mt_scores_dict = baselines_comet_eval(mt_outs_dict, refs, srcs, model, 120)
        step_seg_cor, save_seg_dict = store_corr_eval(evs, mt_scores_dict, 'seg', wmt, lang, gold_scores)
        _, save_sys_dict = store_corr_eval(evs, mt_scores_dict, 'sys', wmt, lang, gold_scores)
        print(save_seg_dict)
        print(save_sys_dict)
    elif model_base == 'xcomet':
        lang = lang_dir.replace('_', '-')
        print(f"Lang: {lang}, WMT: {wmt}, Gold_scores: {gold_scores}")
        evs = data.EvalSet(wmt, lang)
        mt_outs_dict, refs, srcs = evs.sys_outputs, evs.all_refs[evs.std_ref], evs.src
        model_path = download_model("Unbabel/XCOMET-XXL")
        model = load_from_checkpoint(model_path)
        mt_scores_dict = baselines_comet_eval(mt_outs_dict, refs, srcs, model, 6)
        step_seg_cor, save_seg_dict = store_corr_eval(evs, mt_scores_dict, 'seg', wmt, lang, gold_scores)
        _, save_sys_dict = store_corr_eval(evs, mt_scores_dict, 'sys', wmt, lang, gold_scores)
        print(save_seg_dict)
        print(save_sys_dict)
    else:
        # parent_dir = './MT_sescore/Sescore2_main'
        # sys.path.append(parent_dir)
        lang = lang_dir.replace('_', '-')
        print(f"Lang: {lang}, WMT: {wmt}, Gold_scores: {gold_scores}")
        evs = data.EvalSet(wmt, lang)
        mt_outs_dict, refs, srcs = evs.sys_outputs, evs.all_refs[evs.std_ref], evs.src
        
        tokenizer = AutoTokenizer.from_pretrained(f"xlm-roberta-large")
        exp_config.device_id = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        cur_addr = hf_hub_download(repo_id="xu1998hz/sescore2_en_pretrained", filename="sescorex_seg.ckpt")
        model = torch.load(cur_addr).to(exp_config.device_id)

        model.eval()
        # evaluate on the seg and sys correlations
        mt_scores_dict = baselines_cl_eval(mt_outs_dict, refs, emb_type, model, 1000, tokenizer)
        step_seg_cor, save_seg_dict = store_corr_eval(evs, mt_scores_dict, 'seg', wmt, lang, gold_scores)
        _, save_sys_dict = store_corr_eval(evs, mt_scores_dict, 'sys', wmt, lang, gold_scores)
        print(save_seg_dict)
        print(save_sys_dict)

if __name__ == "__main__":
    main()
