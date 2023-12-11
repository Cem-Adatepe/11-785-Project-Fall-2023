import argparse
import os
import torch
import numpy as np
import pandas as pd
from categories import subcategories, categories
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM
from transformers import GPT2Model, GPT2Config, GPT2Tokenizer, GPT2LMHeadModel

import time

choices = ["A", "B", "C", "D"]


def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s


def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j + 1])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt


def gen_prompt(train_df, subject, k=-1):
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(
        format_subject(subject)
    )
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt


@torch.no_grad()
def eval(args, subject, model, tokenizer, dev_df, test_df):
    cors = []
    all_probs = []
    answers = choices[: test_df.shape[1] - 2]

    for i in range(test_df.shape[0]):
        # get prompt and make sure it fits
        k = args.ntrain
        prompt_end = format_example(test_df, i, include_answer=False)
        train_prompt = gen_prompt(dev_df, subject, k)
        prompt = train_prompt + prompt_end

        input_ids = tokenizer.encode(prompt, return_tensors='pt')


        while input_ids.shape[-1] > 1024:
            k -= 1
            train_prompt = gen_prompt(dev_df, subject, k)
            prompt = train_prompt + prompt_end
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids

        label = test_df.iloc[i, test_df.shape[1] - 1]



        # decoder_input_ids = tokenizer("", return_tensors="pt").input_ids#.cuda()
        # decoder_input_ids = model._shift_right(decoder_input_ids)
        logits = model(
            input_ids=input_ids, # decoder_input_ids=decoder_input_ids
        ).logits

        '''
        logitsflat = model(
            input_ids=input_ids,  # decoder_input_ids=decoder_input_ids
        ).logits.flatten()



        probs = (
            torch.nn.functional.softmax(
                torch.tensor(
                    [
                        logitsflat[tokenizer("A").input_ids[0]],
                        logitsflat[tokenizer("B").input_ids[0]],
                        logitsflat[tokenizer("C").input_ids[0]],
                        logitsflat[tokenizer("D").input_ids[0]],
                    ]
                ),
                dim=0,
            )
            .detach()
            .cpu()
            .numpy()
        )
        print(probs)
        
        pred = {0: "A", 1: "B", 2: "C", 3: "D"}[np.argmax(probs)]
        '''
        A_id = tokenizer.encode("A", add_special_tokens=False)[0]
        B_id = tokenizer.encode("B", add_special_tokens=False)[0]
        C_id = tokenizer.encode("C", add_special_tokens=False)[0]
        D_id = tokenizer.encode("D", add_special_tokens=False)[0]
        A_logit = logits[0, -1, A_id]
        B_logit = logits[0, -1, B_id]
        C_logit = logits[0, -1, C_id]
        D_logit = logits[0, -1, D_id]
        probs = torch.nn.functional.softmax(torch.tensor([A_logit, B_logit, C_logit, D_logit]), dim=0)
        A_prob = probs[0].item()
        B_prob = probs[1].item()
        C_prob = probs[2].item()
        D_prob = probs[3].item()
        preds_dict = {A_prob:'A', B_prob:'B', C_prob:'C', D_prob:'D'}
        pred = preds_dict[max(preds_dict)]

        cor = pred == label
        cors.append(cor)
        probs = np.array([A_prob, B_prob, C_prob, D_prob])
        all_probs.append(probs)

    acc = np.mean(cors)
    cors = np.array(cors)

    all_probs = np.array(all_probs)
    print("Average accuracy {:.3f} - {}".format(acc, subject))

    return cors, acc, all_probs


def main(args):

    model = GPT2LMHeadModel.from_pretrained("gpt2")
    # model.base_model.h[0].attn.c_attn.weight = torch.nn.Parameter(
    #     torch.cat((torch.zeros(768, 768), torch.zeros(768, 768), torch.eye(768)), 1))


    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    # model = AutoModelForCausalLM.from_pretrained("bigscience/bloom-3b")

    # https://github.com/huggingface/transformers/issues/27132
    # please use the slow tokenizer since fast and slow tokenizer produces different tokens
    # tokenizer = AutoTokenizer.from_pretrained(
    #     "bigscience/bloom-3b",
    # )
    model.eval()
    subjects = sorted(
        [
            f.split("_test.csv")[0]
            for f in os.listdir(os.path.join(args.data_dir, "test"))
            if "_test.csv" in f
        ]
    )

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    if not os.path.exists(os.path.join(args.save_dir, "results_gpt")):
        os.makedirs(os.path.join(args.save_dir, "results_gpt"))

    all_cors = []
    subcat_cors = {
        subcat: [] for subcat_lists in subcategories.values() for subcat in subcat_lists
    }
    cat_cors = {cat: [] for cat in categories}

    for subject in subjects:
        dev_df = pd.read_csv(
            os.path.join(args.data_dir, "dev", subject + "_dev.csv"), header=None
        )[: args.ntrain]
        test_df = pd.read_csv(
            os.path.join(args.data_dir, "test", subject + "_test.csv"), header=None
        )

        cors, acc, probs = eval(args, subject, model, tokenizer, dev_df, test_df)
        subcats = subcategories[subject]
        for subcat in subcats:
            subcat_cors[subcat].append(cors)
            for key in categories.keys():
                if subcat in categories[key]:
                    cat_cors[key].append(cors)
        all_cors.append(cors)

        test_df["gpt_correct"] = cors
        for j in range(probs.shape[1]):
            choice = choices[j]
            test_df["gpt_choice{}_probs".format(choice)] = probs[:, j]
        test_df.to_csv(
            os.path.join(
                args.save_dir, "results_gpt", "{}.csv".format(subject)
            ),
            index=None,
        )

    for subcat in subcat_cors:
        subcat_acc = np.mean(np.concatenate(subcat_cors[subcat]))
        print("Average accuracy {:.3f} - {}".format(subcat_acc, subcat))

    for cat in cat_cors:
        cat_acc = np.mean(np.concatenate(cat_cors[cat]))
        print("Average accuracy {:.3f} - {}".format(cat_acc, cat))
    weighted_acc = np.mean(np.concatenate(all_cors))
    print("Average accuracy: {:.3f}".format(weighted_acc))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", "-k", type=int, default=5)
    parser.add_argument("--ngpu", "-g", type=int, default=2)
    parser.add_argument("--data_dir", "-d", type=str, default="eval_data")
    parser.add_argument("--save_dir", "-s", type=str, default="results")
    args = parser.parse_args()
    main(args)