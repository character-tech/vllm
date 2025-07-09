# SPDX-License-Identifier: Apache-2.0
import argparse
import json
import logging
import os
import random

import pandas as pd
import requests
import torch
import urllib3
from transformers import AutoTokenizer

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning
                         )  # disable annoying security warning

# adapted from https://github.com/character-tech/character-tech/blob/main/rayman/rayman/evals/replay_chat.py 
STATIC_PATH = "gs://character-ai-us-central1/data/evals/static_replay_chat_4_10.jsonl"
NUM_TURNS_TO_CONSIDER = 20  # Assuming this constant from the original code


def format_user_chats(prompt_data): # Format the chat context messages into a readable prompt string
    last_turns = prompt_data["chat_context_messages"][-NUM_TURNS_TO_CONSIDER:]

    chat_parts = []
    for turn in last_turns:
        is_bot = turn["type"] == 2
        turn_from = "[bot]" if is_bot else "[user]"
        turn_text = turn["text"]
        chat_parts.append(f"{turn_from} {turn_text}")

    return "\n".join(chat_parts)


def extract_prompt_response_from_generation(generation): # Extract prompt and response from a single generation
    try:
        prompt_data = generation["payload"]["prompt_data"]

        prompt = format_user_chats(prompt_data)
        response_raw = json.loads(generation["payload"]["response"])
        response = response_raw[0]["resp_sentences"]

        return prompt, response
    except Exception as e:
        logging.warning(f"Failed to extract prompt/response: {e}")
        return None, None


def load_static_generations(filename=STATIC_PATH, limit=None, random_selection=False): # Load generations from static file
    df = pd.read_json(filename, lines=True)
    if limit:
        if random_selection:
            df = df.sample(n=limit, random_state=random.randint(0, 1000000))
        else:
            df = df.head(limit)

    return df.to_dict(orient="records")


def generate_prompt_response_pairs_from_replay_chat(limit=10, random_selection=False): # 
    print("Loading chat data from static source...")
    prompt_response_pairs = []

    try:
        generations = load_static_generations(
            limit=limit, random_selection=random_selection)
        print(f"Loaded {len(generations)} generations from static file")
    except Exception as e:
        print(f"Error loading static file: {e}")
        print(
            "Note: You may need access to the GCS bucket or a local copy of the data"
        )
        return prompt_response_pairs

    valid_pairs = 0
    for i, generation in enumerate(generations):
        prompt, response = extract_prompt_response_from_generation(generation)

        if prompt and response:
            valid_pairs += 1
            if len(response) > 1:
                print(f"Expected 1 response, got {len(response)}")
            prompt_response_pairs.append((prompt, response[0]))
        else:
            print(f"Skipped generation {i+1} due to extraction error")

    print(
        f"\nProcessed {valid_pairs} valid prompt-response pairs out of {len(generations)} generations"
    )
    return prompt_response_pairs

def logits_hash(logits):# hash rounded logits (for comparison)
    import hashlib
    rounded_logits = torch.round(logits * 1000) / 1000
    return hashlib.sha256(rounded_logits.to(
        torch.float32).numpy().tobytes()).hexdigest()[:10]

tokenizer = None # global tokenizer

def load_tokenizer(): # load tokenizer from checkpoint dir
    CHECKPOINT_DIR = os.getenv(
        "CHECKPOINT_DIR",
        "/data/rohin/ckpts/roar_sft_adventure_safe.05-01-22-00-25.uc33.classi_only.tp1.minimal/"
    )
    return AutoTokenizer.from_pretrained(CHECKPOINT_DIR)


def calculate_statistics(vllm_logits_file, roller_logits_file, eos_tok_id):
    vllm_logits = torch.load(vllm_logits_file).to('cpu')
    roller_logits = torch.load(roller_logits_file).to('cpu')

    global tokenizer
    if tokenizer is None:
        tokenizer = load_tokenizer()

    vllm_tok = vllm_logits.argmax()
    roller_tok = roller_logits.argmax()

    # if vllm_tok != roller_tok:
    # print(f"DIFFERENT SELECTED TOKENS: vllm={tokenizer.decode(vllm_tok)} roller={tokenizer.decode(roller_tok)}")

    def smart_clamp(logits, max_range=100, clamp_range=50):
        if logits.max() - logits.min() > max_range:
            center = logits.median()
            return torch.clamp(logits,
                               min=center - clamp_range,
                               max=center + clamp_range)
        return logits

    roller_logits = smart_clamp(roller_logits)
    vllm_logits = smart_clamp(vllm_logits)

    vllm_log_probs = vllm_logits.log_softmax(dim=-1)
    roller_log_probs = roller_logits.log_softmax(dim=-1)

    vllm_probs = torch.exp(vllm_log_probs)
    roller_probs = torch.exp(roller_log_probs)
    
    vllm_eos_prob = vllm_probs[..., eos_tok_id].mean()
    roller_eos_prob = roller_probs[..., eos_tok_id].mean()
    eos_prob_diff = vllm_eos_prob - roller_eos_prob

    kl_div = torch.nn.functional.kl_div(vllm_log_probs,
                                        roller_log_probs,
                                        reduction="batchmean",
                                        log_target=True)
    rev_kl_div = torch.nn.functional.kl_div(roller_log_probs,
                                            vllm_log_probs,
                                            reduction="batchmean",
                                            log_target=True)

    m = 0.5 * (vllm_probs + roller_probs)
    js_div = 0.5 * torch.nn.functional.kl_div(vllm_log_probs, torch.log(m),
                                             reduction="batchmean") + \
             0.5 * torch.nn.functional.kl_div(roller_log_probs, torch.log(m),
                                             reduction="batchmean")

    tv_distance = 0.5 * torch.sum(torch.abs(vllm_probs - roller_probs),
                                  dim=-1).mean()

    hellinger = torch.sqrt(0.5 * torch.sum(
        (torch.sqrt(vllm_probs) - torch.sqrt(roller_probs))**2,
        dim=-1)).mean()

    top_k = 10
    vllm_topk_mass = torch.topk(vllm_probs, top_k,
                                dim=-1)[0].sum(dim=-1).mean()
    roller_topk_mass = torch.topk(roller_probs, top_k,
                                  dim=-1)[0].sum(dim=-1).mean()
    # print(torch.topk(vllm_probs, top_k, dim=-1)[0], torch.topk(roller_probs, top_k, dim=-1)[0])
    topk_mass_diff = vllm_topk_mass - roller_topk_mass

    vllm_entropy = -torch.sum(vllm_probs * vllm_log_probs, dim=-1).mean()
    roller_entropy = -torch.sum(roller_probs * roller_log_probs, dim=-1).mean()
    entropy_diff = torch.abs(vllm_entropy - roller_entropy)

    log_rev_kl = torch.log10(
        torch.clamp(torch.nn.functional.kl_div(roller_log_probs,
                                               vllm_log_probs,
                                               reduction="batchmean",
                                               log_target=True),
                    min=1e-10))

    return {
        'kl_div': kl_div.item(),
        'rev_kl_div': rev_kl_div.item(),
        # 'js_divergence': js_div.item(),
        # 'tv_distance': tv_distance.item(),
        # 'hellinger_dist': hellinger.item(),
        # 'log10_rev_kl': log_rev_kl.item(),
        'topk_mass_diff': topk_mass_diff.item(),
        'entropy_diff': entropy_diff.item(),
        'vllm_eos_prob': vllm_eos_prob.item(),
        'roller_eos_prob': roller_eos_prob.item(),
        'eos_prob_diff': eos_prob_diff.item(),
        'vllm_logits_hash': logits_hash(vllm_logits),
        'roller_logits_hash': logits_hash(roller_logits)
    }


def print_statistics(roller_outer_dir,
                     vllm_outer_dir,
                     prompt_dir,
                     new_files,
                     idx=None):
    global tokenizer
    if tokenizer is None:
        tokenizer = load_tokenizer()

    eos_id = tokenizer.eos_token_id
    if prompt_dir not in os.listdir(vllm_outer_dir):
        print(f"Prompt dir {prompt_dir} not found in vllm")
        return
    if prompt_dir not in os.listdir(roller_outer_dir):
        print(f"Prompt dir {prompt_dir} not found in roller")
        return

    idx = 0
    print(f"Prompt dir: {prompt_dir} processing {len(new_files)} files")
    vllm_ev_message_length = 0
    roller_ev_message_length = 0
    prob_vllm_still_generating = 1
    prob_roller_still_generating = 1
    cumulative_stats = {}
    for file in new_files:
        assert file.endswith(".pt")
        if file not in os.listdir(os.path.join(vllm_outer_dir, prompt_dir)):
            print(f"File {file} not found in vllm")
            continue

        roller_logits_file = os.path.join(roller_outer_dir, prompt_dir, file)
        vllm_logits_file = os.path.join(vllm_outer_dir, prompt_dir, file)
        print(f"Statistics for {idx}: {file=}", end='')
        computed_stats = calculate_statistics(vllm_logits_file,
                                              roller_logits_file, eos_id)
        for k, v in computed_stats.items():
            if type(v) == float:
                print(f"  {k}={v:.3e}", end='')
                cumulative_stats[k] = cumulative_stats.get(k, 0) + v
            else:
                print(f"  {k}={v}", end='')

        vllm_eos_prob = computed_stats['vllm_eos_prob']
        roller_eos_prob = computed_stats['roller_eos_prob']
        vllm_ev_message_length += vllm_eos_prob * prob_vllm_still_generating * idx
        roller_ev_message_length += roller_eos_prob * prob_roller_still_generating * idx
        prob_vllm_still_generating *= 1 - vllm_eos_prob
        prob_roller_still_generating *= 1 - roller_eos_prob
        print()
        idx += 1

    print(f"CUMULATIVE_STATS for {idx} {prompt_dir}:")
    print(
        f"CUMULATIVE_STATS: VLLM EV message length: {vllm_ev_message_length}")
    print(
        f"CUMULATIVE_STATS: Roller EV message length: {roller_ev_message_length}"
    )
    print(
        f"CUMULATIVE_STATS: VLLM EV message length / Roller EV message length: {vllm_ev_message_length / roller_ev_message_length}"
    )
    print(
        f"CUMULATIVE_STATS: P(VLLM still generating): {prob_vllm_still_generating}"
    )
    print(
        f"CUMULATIVE_STATS: P(Roller still generating): {prob_roller_still_generating}"
    )
    for k, v in cumulative_stats.items():
        print(f"CUMULATIVE_STATS: {k}: {v / idx}")

def encode(input_text):
    global tokenizer
    if tokenizer is None:
        tokenizer = load_tokenizer()
    return tokenizer.encode(input_text)


def decode(input_tokens):
    global tokenizer
    if tokenizer is None:
        tokenizer = load_tokenizer()
    return tokenizer.decode(input_tokens)


def get_prompt_dir(prompt):
    prefix_tokens = encode(prompt)
    start_idx = 0 if prefix_tokens[0] != 5 else 1
    num_tokens_for_dir = min(3, len(prefix_tokens) - start_idx)
    dir_tokens = prefix_tokens[start_idx:start_idx + num_tokens_for_dir]
    dir_name = "_".join(str(token) for token in dir_tokens)
    # print(f"{prompt=} {prefix_tokens=} {dir_name=}")
    return dir_name


def execute_vllm(prompt):
    url = "http://0.0.0.0:8000/v1/completions"
    headers = {"Content-Type": "application/json", "x-access-tokens": "token"}
    data = {"prompt": f"<|beginningofdialog|>{prompt}"}

    response = requests.post(url, headers=headers, json=data)
    return response.json()


def execute_roller(prompt):
    url = "https://127.0.0.1:7395/generate"
    x_access_tokens = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJwdWJsaWNfaWQiOiJmNzU2Zjg1Yy04ZTUxLTQ3MDMtYTBjZi1hYjY2NjQ0NGNjZGEifQ.s_PmKLKNodgPiWiLQA50Fa1aUYYxouQ3Fb00AxuC33w"
    headers = {
        "Content-Type": "application/json",
        "x-access-tokens": x_access_tokens
    }
    data = {
        "contexts": [prompt],
        "result_keys": ["resp_sentences", "output_tokens", "is_final"]
    }

    response = requests.put(url, headers=headers, json=data, verify=False)
    return response.json()


def execute_models(prompt, response):
    tokenized_prompt = encode(prompt)
    tokenized_reponse = encode(response)
    detokenized_reponse = [decode(token) for token in tokenized_reponse]
    # print(f"tokenizer output: {tokenized_prompt=} {tokenized_reponse=} {detokenized_reponse=}")
    for i in range(len(tokenized_reponse) + 1):
        current_prompt = prompt + ''.join(detokenized_reponse[:i])
        # print(f"Prompt {i}: {current_prompt}")
        execute_vllm(current_prompt)
        execute_roller(current_prompt)


def get_files_and_hashes(dir):
    return [(file, os.path.getmtime(os.path.join(dir, file)))
            for file in os.listdir(dir)]


def process_prod_response_pair(prompt,
                               response,
                               print_full_inputs=False,
                               idx=None):
    roller_outer_dir = os.getenv("ROLLER_LOGITS_DIR")
    vllm_outer_dir = os.getenv("VLLM_LOGITS_DIR")

    if not print_full_inputs:
        print(f"Prompt: {prompt[:50]}... Response: {response[:50]}... {idx=}")
    else:
        print(f"Prompt\n{prompt}\n\nResponse\n{response}\n\n")

    prompt_dir = get_prompt_dir(prompt)
    before_files = get_files_and_hashes(
        os.path.join(
            roller_outer_dir,
            prompt_dir))  # hack to allow for collisions in prompt directories
    # print(f"Before files: {before_files}")
    execute_models(prompt, response)
    after_files = get_files_and_hashes(
        os.path.join(roller_outer_dir, prompt_dir))
    # print(f"After files: {after_files}")
    new_files = []
    for file, hash in after_files:
        if (file, hash) not in before_files:
            new_files.append(file)
    print_statistics(roller_outer_dir, vllm_outer_dir, prompt_dir, new_files,
                     idx)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--random_selection", action="store_true")
    parser.add_argument("--print_full_inputs", action="store_true")
    return parser.parse_args()


def compare_vllm_roller_logits(args):
    prompt_response_pairs = generate_prompt_response_pairs_from_replay_chat(args.limit)
    for idx, (prompt, response) in enumerate(prompt_response_pairs):
        process_prod_response_pair(prompt, response, args.print_full_inputs,
                                   idx)


if __name__ == "__main__":
    args = parse_args()
    compare_vllm_roller_logits(args)
