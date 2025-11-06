import os
import unsloth
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

import argparse
import json
import gc
import random
import re


from copy import deepcopy
from collections import deque
from typing import List, Dict

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)
from peft import PeftModel
from transformers import AutoTokenizer
from datetime import datetime, timedelta

import re
import pandas as pd
import spacy


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


nlp = spacy.load("en_core_web_sm")

def clean_responses(text):
    if not isinstance(text, str):
        return None

    # Remove tokens ending with ':' and any numbers (or digits) after it
    text = re.sub(r'\b\w+:\s*\d*\s*', '', text)

    # Keep only part before 'assistant' (case-insensitive)
    text = re.split(r'assistant', text, flags=re.IGNORECASE)[0]

    # Strip whitespace
    text = text.strip()

    return text if text else None



def is_valid_sentence(text, nlp, min_words=3):
    """
    Use POS tagging and dependency parsing to check if text is a valid sentence.
    Criteria:
      - Contains at least one verb
      - Contains at least one noun or pronoun
      - Reasonable length (not just fragments)
    """
    if not isinstance(text, str) or len(text.split()) < min_words:
        return False

    doc = nlp(text)

    has_verb = any(token.pos_ in ("VERB", "AUX") for token in doc)
    has_subject = any(token.dep_ in ("nsubj", "nsubjpass", "expl") for token in doc)
    has_noun_or_pron = any(token.pos_ in ("NOUN", "PROPN", "PRON") for token in doc)

    return has_verb and has_noun_or_pron and has_subject



def create_context_card_simulation(speaker, topics_list, people_list, speaking_order, meeting_start="09:00", add_times=False):
    """
    Context card for multi-agent meeting simulation with robust time tracking.
    - Strict turn order & agenda adherence
    - Short, live-meeting style utterances
    - Each utterance begins with current_time and current agenda item
    - Agents never speak for others; everyone votes individually
    - Supports abrupt agenda changes
    """

    people_str = ", ".join(people_list)
    order_str = ", ".join(speaking_order)
    idx = speaking_order.index(speaker)
    next_speaker = speaking_order[(idx + 1) % len(speaking_order)]
    prev_speaker = speaking_order[idx - 1] if idx > 0 else speaking_order[-1]

    # Build agenda string with times
    start_hour, start_minute = map(int, meeting_start.split(":"))
    current_hour, current_minute = start_hour, start_minute
    agenda_schedule = []  # To track each topic's start and end for reference
    agenda_str = ""
    for topic in topics_list:
        duration = topic.get("duration_minutes", 30)
        vote_duration = topic.get("vote_minutes", 15)
        start_time = f"{current_hour:02d}:{current_minute:02d}"

        # Discussion end
        end_minute = current_minute + duration
        end_hour = current_hour + end_minute // 60
        end_minute %= 60
        discussion_end = f"{end_hour:02d}:{end_minute:02d}"

        # Vote end
        vote_end_minute = end_minute + vote_duration
        vote_end_hour = end_hour + vote_end_minute // 60
        vote_end_minute %= 60
        vote_end = f"{vote_end_hour:02d}:{vote_end_minute:02d}"

        agenda_str += f"\n- {start_time}–{discussion_end}: {topic['title']}\n\t- "
        agenda_str += "\n\t- ".join(topic['discussion_points'])
        
        if topic.get('decision_point',None):
            agenda_str += f"\n- {discussion_end}-{vote_end}: {topic.get('decision_point','[Decision]')}\n\n"

        # Add schedule for reference in examples
        agenda_schedule.append({
            "title": topic['title'],
            "discussion_start": start_time,
            "discussion_end": discussion_end,
            "vote_start": discussion_end,
            "vote_end": vote_end,
            "decision_point": topic.get("decision_point", "[Decision]")
        })

        current_hour, current_minute = vote_end_hour, vote_end_minute

    # Build in-context examples with explicit time & agenda reference
    example_lines = []
    current_time_hour, current_time_minute = map(int, meeting_start.split(":"))

    # Example 1: General discussion
    topic1 = agenda_schedule[0]
    time_string = ''
    if add_times:
        time_string = f"[current_time={topic1['discussion_start']}, agenda_item={topic1['title']}]"

    example_lines.append("=== Example 1: General Discussion ===")
    example_lines.append(
        f"{time_string}{speaking_order[0]}: Uh, so, let's start with {topic1['title']}. I mean, there are a few things we need to check."
    )
    current_time_minute += 1
    example_lines.append(
        f"{time_string}{speaking_order[1]}: Yeah, I was thinking the same. Maybe we should review last quarter's numbers first."
    )
    current_time_minute += 1
    example_lines.append(
        f"{time_string}{speaking_order[2]}: Uh, I just want to point out some potential issues with the current plan."
    )
    current_time_minute += 1

    # Example 2: Topic change
    topic2 = agenda_schedule[1]
    if add_times:
        time_string = f"[current_time={topic1['discussion_start']}, agenda_item={topic1['title']}]"

    example_lines.append("\n=== Example 2: Topic Change ===")
    example_lines.append(
        f"{time_string}{speaking_order[0]}: Okay, actually we need to move on to {topic2['title']} since there’s new info."
    )
    current_time_minute += 1
    example_lines.append(
        f"{time_string}{speaking_order[1]}: Right, that makes sense. Let’s address it quickly."
    )
    current_time_minute += 1

    # Example 3: Vote example
    vote_topic = agenda_schedule[0]  # Voting for first topic
    if add_times:
        time_string = f"[current_time={vote_topic['discussion_start']}, agenda_item={vote_topic['decision_point']}]"

    example_lines.append("\n=== Example 3: Voting ===")
    example_lines.append(
        f"{time_string}{speaking_order[0]}: I vote yes."
    )
    current_time_minute += 1
    example_lines.append(
        f"{time_string}{speaking_order[1]}: I vote no."
    )
    current_time_minute += 1
    example_lines.append(
        f"{time_string}{speaking_order[2]}: I vote yes."
    )
    current_time_minute += 1

    example_text = "\n".join(example_lines)
    if add_times:
        time_string = '\n- Before speaking, always state the current time and the scheduled agenda item in the format: [current_time=HH:MM, agenda_item=...]'
        time_string=""
    context_card = f"""====================
PERSONA: {speaker.upper()}
====================

PARTICIPANTS: {people_str}
TURN ORDER: {order_str}

You are {speaker}. You speak AFTER {prev_speaker} and BEFORE {next_speaker}.
Only address participants in the list. Never speak for anyone else.
Stay on topic and only speak about the agenda items below.

=== AGENDA ===
The below agenda has already been approved, please follow it closely.
{agenda_str}
=== RULES ===
- Speak only as {speaker}, stay in character and never speak for others. You may only ask questions to: {people_str}.
- Follow agenda times exactly. Stop immediately when the segment ends. Always stay on topic.
- **IMPORTANT** Keep each utterance short (1–2 sentences), with natural fillers like “uh,” “so,” “I mean.”  Speak as if in a live meeting.{time_string}
- When voting each participant states their own vote: “I vote yes/no/abstain.”

=== EXAMPLES ===
{example_text}
====================
"""
    return context_card.strip()


def create_defendant_context_card(speaker, people_list, speaking_order, add_times=False):
    """
    Generates a fully detailed, case-specific context card for John F. General.
    Includes:
      - Full case background
      - Facts of the interview
      - Conviction context
      - Defense arguments
      - Strategic guidance
      - Discussion points for appellate simulation
    """

    people_str = ", ".join(people_list)
    order_str = ", ".join(speaking_order)
    idx = speaking_order.index(speaker)
    next_speaker = speaking_order[(idx + 1) % len(speaking_order)]
    prev_speaker = speaking_order[idx - 1] if idx > 0 else speaking_order[-1]
    
    time_string = ''
    if add_times:
        time_string = '\n- Before speaking, always state the current time and the scheduled agenda item in the format: [current_time=HH:MM, agenda_item=...]'
        time_string=""
    # Full case context persona
    persona_info = f"""
You are {speaker}, the defendant in the federal appellate case General v. United States.

CASE BACKGROUND:
- Mr. General was convicted of alleged financial fraud, tax evasion, and related federal offenses.
- The key evidence against you includes statements made during a 1 hour 27 minute police interview at a local station-house.
- During the interview:
    • The room was locked; reentry restricted.
    • You were questioned sharply about multiple allegations, including prior uncharged conduct.
    • Saliva DNA sample was collected without formal consent.
    • You were not formally arrested, but were not free to leave.
- At trial, the prosecution relied heavily on statements made in this interview.
- You believe these statements should have been suppressed because the conditions were “custodial” under Miranda.

DEFENSE ARGUMENTS TO MAKE:
- The interview conditions constituted custody: tone, restrictions, and controlled environment exceeded Graham and Turner precedents.
- DNA collection was not voluntary; statements were coerced and not harmless.
- The appellate focus: whether admission of these statements violated your constitutional rights, and whether any error was harmless beyond a reasonable doubt.
- React to judges’ questions respectfully but clearly assert rights violations and factual context.
- Provide context for each agenda discussion point: summarize facts, emphasize perceived coercion, and outline legal implications without speculating for judges.
"""

    context_card = f"""====================
PERSONA: {speaker.upper()}
====================

PARTICIPANTS: {people_str}
TURN ORDER: {order_str}

You are {speaker}. You speak AFTER {prev_speaker} and BEFORE {next_speaker}.
Only address participants in the list. Never speak for anyone else.
Stay on topic and only speak about the agenda items below.

=== PERSONA DETAILS ===
{persona_info.strip()}

=== RULES ===
- Speak only as {speaker}, stay in character and never speak for others. You may only ask questions to: {people_str}.{time_string}
"""

    return context_card.strip()



def get_current_topic(topic_schedule, current_hour, current_minute):
    cur_time = current_hour * 60 + current_minute
    for t in topic_schedule:
        # Parse start/end times
        ds_h, ds_m = map(int, t["discussion_start"].split(":"))
        de_h, de_m = map(int, t["discussion_end"].split(":"))
        vs_h, vs_m = map(int, t["vote_start"].split(":"))
        ve_h, ve_m = map(int, t["vote_end"].split(":"))

        discussion_start_min = ds_h * 60 + ds_m
        discussion_end_min = de_h * 60 + de_m
        vote_start_min = vs_h * 60 + vs_m
        vote_end_min = ve_h * 60 + ve_m

        if discussion_start_min <= cur_time < discussion_end_min:
            return t["title"]
        elif vote_start_min <= cur_time < vote_end_min:
            return f"{t['decision_point']}"
    # Default to last topic if time exceeds schedule
    return topic_schedule[-1]["title"]

from typing import List, Dict, Tuple


def compute_topic_schedule(
    topics: List[Dict],
    meeting_start: str = "09:00"
) -> Tuple[List[Dict], Dict]:
    """
    Precompute a schedule for topics including discussion and vote times.

    Args:
        topics (List[Dict]): List of topics, each must have 'title', optional 'duration_minutes' and 'vote_minutes'.
        meeting_start (str): Meeting start time in "HH:MM" format.

    Returns:
        topic_schedule (List[Dict]): List of topics with computed times.
        first_topic_info (Dict): Info about the first topic (title and discussion_start)
    """
    topic_schedule = []
    hour, minute = map(int, meeting_start.split(":"))

    for topic in topics:
        duration = topic.get("duration_minutes", 30)
        vote_duration = topic.get("vote_minutes", 15)

        discussion_start = f"{hour:02d}:{minute:02d}"
        end_minute = minute + duration
        end_hour = hour + end_minute // 60
        end_minute %= 60
        discussion_end = f"{end_hour:02d}:{end_minute:02d}"

        vote_start = discussion_end
        vote_end_minute = end_minute + vote_duration
        vote_end_hour = end_hour + vote_end_minute // 60
        vote_end_minute %= 60
        vote_end = f"{vote_end_hour:02d}:{vote_end_minute:02d}"

        topic_schedule.append({
            "title": topic["title"],
            "discussion_start": discussion_start,
            "discussion_end": discussion_end,
            "vote_start": vote_start,
            "vote_end": vote_end,
            "decision_point": topic.get("decision_point", "[Decision]")
        })

        # Update hour/minute for next topic
        hour, minute = vote_end_hour, vote_end_minute

    return topic_schedule


def get_meeting_length_minutes(topic_schedule: list) -> int:
    """
    Compute total meeting length in minutes based on topic schedule.

    Args:
        topic_schedule (list): List of topics with 'discussion_start' and 'vote_end' times.

    Returns:
        int: Total meeting length in minutes.
    """
    if not topic_schedule:
        return 0

    fmt = "%H:%M"
    start_time = datetime.strptime(topic_schedule[0]["discussion_start"], fmt)
    end_time = datetime.strptime(topic_schedule[-1]["vote_end"], fmt)

    # Handle meetings that go past midnight
    delta = end_time - start_time
    if delta.total_seconds() < 0:
        delta += timedelta(days=1)

    return int(delta.total_seconds() // 60)


def query_base_model(base_model, tokenizer, prompt, max_new_tokens=250):
    """
    Queries base model without adapters
    """

    inputs = tokenizer(prompt, return_tensors="pt").to(base_model.device)
    prompt_len = inputs.input_ids.shape[1]

    # Disable adapters if using one
    # with base_model.disable_adapters():
    # Use deterministic output to reduce variability
    
    for i in range(3):
        temperature = 0.7
        top_k = 25
        top_p = 0.9

        out_tokens = base_model.generate(
            **inputs, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k, top_p=top_p )

        response = tokenizer.decode(out_tokens[0][prompt_len:], skip_special_tokens=True)
        
        if is_valid_sentence(response):
            return clean_responses(response)
        else:
            print('Filterd Response: ', response)

    return clean_responses(response)



def round_robin_order(agent_names, start=0):
    idx = start
    while True:
        yield agent_names[idx % len(agent_names)]
        idx += 1


# =========================
# AGENT CLASS
# =========================
class Agent:
    def __init__(
        self,
        name,
        model,
        tokenizer,
        adapter_path,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.2,  # Increased penalty
        max_new_tokens: int = 80,         # Lowered default
    ):
        self.name = name
        self.model = model
        self.tokenizer = tokenizer
        self.adapter_path = adapter_path
        self.conv_history = []

        # Default generation settings
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.max_new_tokens = max_new_tokens

    def generate_response(
        self,
        prompt: str,
        temperature: float = None,
        top_k: int = None,
        top_p: float = None,
        repetition_penalty: float = None,
        max_new_tokens: int = None,
        num_candidates: int = 1,
        agent_names=None,
    ) -> str:
        """
        Generate a single response (string).
        Allows overriding of default generation settings.
        """
        gen_kwargs = {
            "temperature": temperature if temperature is not None else self.temperature,
            "top_k": top_k if top_k is not None else self.top_k,
            "top_p": top_p if top_p is not None else self.top_p,
            "repetition_penalty": (
                repetition_penalty if repetition_penalty is not None else self.repetition_penalty
            ),
            "max_new_tokens": max_new_tokens if max_new_tokens is not None else self.max_new_tokens,
        }

        candidates = self.generate_candidates(
            prompt, num_candidates, agent_names or [], **gen_kwargs
        )

        return candidates[0]

    def generate_candidates(
        self,
        prompt: str,
        num_candidates: int = 1,
        agent_names=None,
        temperature: float = None,
        top_k: int = None,
        top_p: float = None,
        repetition_penalty: float = None,
        max_new_tokens: int = None,
        max_attempts: int = 3,
    ):
        """
        Generate multiple candidate responses.
        Uses defaults unless overridden by arguments.
        """

        #self.model.set_adapter(self.name)
        peft_model = PeftModel.from_pretrained(self.model, self.adapter_path, is_trainable=False)
        peft_model.set_adapter("default")  # activate the adapter loaded from adapter_path

        gen_kwargs = {
            "do_sample": True,
            "temperature": temperature if temperature is not None else self.temperature,
            "top_k": top_k if top_k is not None else self.top_k,
            "top_p": top_p if top_p is not None else self.top_p,
            "repetition_penalty": (
                repetition_penalty if repetition_penalty is not None else self.repetition_penalty
            ),
            "max_new_tokens": max_new_tokens if max_new_tokens is not None else self.max_new_tokens,
        }

        candidates = []
        inputs = self.tokenizer(prompt, return_tensors="pt").to(peft_model.device)
        input_ids = inputs.input_ids
        prompt_len = input_ids.shape[1]

        def is_multi_agent_reply(reply: str) -> bool:
            lower = reply.lower()
            return sum(lower.count(f"{name.lower()}:") for name in (agent_names or [])) > 1

        # Patch for GPT
        eos_tokens = [self.tokenizer.eos_token_id, self.tokenizer.encode("assistant")[0], self.tokenizer.encode("\n")[0],  self.tokenizer.encode("#")[0],  self.tokenizer.encode("<")[0]]

        for _ in range(num_candidates):
            attempt = 0
            while attempt < max_attempts:
                output = peft_model.generate(**inputs, **gen_kwargs, eos_token_id=eos_tokens, pad_token_id=self.tokenizer.eos_token_id
                                             )
                new_tokens = output[0][prompt_len:]
                reply = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

                if not is_multi_agent_reply(reply) and is_valid_sentence(reply, nlp, min_words=1):
                    
                    candidates.append(clean_responses(reply))
                    break
                else:
                    print('Filterd Response: ', reply)

                attempt += 1

            if attempt == max_attempts:
                candidates.append(reply)
        del peft_model, inputs, output
        torch.cuda.empty_cache()
        gc.collect()
        return candidates


def truncate_conversation(conv, speaker, max_tokens=1000, model_name="meta-llama/Meta-Llama-3-70B-Instruct"):
    """
    Truncate a conversation to approximately max_tokens, preserving whole messages.
    The earliest message is truncated with ellipses if needed.
    
    Args:
        conv (list): List of dicts, each with 'role' and 'content' keys.
        speaker (str): Name of the assistant speaker in the logs.
        max_tokens (int): Max number of tokens to keep.
        model_name (str): Model tokenizer to use.
        
    Returns:
        truncated_conv (list): List of dicts in the same format as conv.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Reverse iterate to accumulate tokens
    tokenized_msgs = []
    total_tokens = 0
    for msg in reversed(conv):
        tokens = tokenizer.encode(msg["content"], add_special_tokens=False)
        tokenized_msgs.append((tokens, msg))
        total_tokens += len(tokens)
        if total_tokens >= max_tokens:
            break
    
    # Reverse back to chronological order
    tokenized_msgs = list(reversed(tokenized_msgs))
    
    # Truncate first message if necessary
    if total_tokens > max_tokens:
        tokens, first_msg = tokenized_msgs[0]
        # Keep only the last part that fits
        keep_tokens = max_tokens - (total_tokens - len(tokens))
        truncated_text = tokenizer.decode(tokens[-keep_tokens:], skip_special_tokens=True)
        first_msg = first_msg.copy()
        speaker=first_msg["content"].split(":")[0]
        first_msg["content"] = f"{speaker}:..." + truncated_text
        tokenized_msgs[0] = (tokens, first_msg)
    
    # Build final conversation format
    truncated_conv = []
    for _, msg in tokenized_msgs:
        truncated_conv.append(msg)
    
    return truncated_conv


def merge_consecutive_messages(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Merge consecutive messages in a conversation that have the same role.
    
    Args:
        messages: A list of dicts, each with 'role' and 'content' keys.
        
    Returns:
        A new list of messages where consecutive messages of the same role are merged.
    """
    if not messages:
        return []
    
    merged = []
    prev_role = messages[0]['role']
    prev_content = messages[0]['content'].strip()
    
    for msg in messages[1:]:
        role = msg['role']
        content = msg['content'].strip()
        if role == prev_role:
            # Merge content with newline
            prev_content += "\n" + content
        else:
            # Push the previous message and start new one
            merged.append({'role': prev_role, 'content': prev_content})
            prev_role = role
            prev_content = content
    
    # Append the last accumulated message
    merged.append({'role': prev_role, 'content': prev_content})
    return merged

def disable_moe_dynamo(module):
    for name, child in module.named_children():
        if "moe" in name.lower():  # or use class check for MoE
            torch._dynamo.disable(child.forward)
        else:
            disable_moe_dynamo(child)


def load_base_model(base_model_path):

    base, _ = FastLanguageModel.from_pretrained(
        model_name = base_model_path, 
        #dtype = torch.float16,
        load_in_4bit = False,
        #device_map="balanced",
        device_map={"":0},
        #offload_folder="/playpen-ssd/smerrill/offload",
        #offload_state_dict=True,
        max_seq_length=4000,
    )
    FastLanguageModel.for_inference(base) 
    disable_moe_dynamo(base)
    model = torch.compile(base, mode="default")

    return base



def main():

    # PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=5,6,7 accelerate launch --num_processes=1 simulate.py --base_model meta-llama/Meta-Llama-3-8B-Instruct --config /playpen-ssd/smerrill/llm_decisions/configs/models.json  --agenda_item "Agenda Item No. 3.1: COVID Mask Policy.  Here we will debate weather we should require students to wear masks in the classrooms? We will then vote on the matter at the end." --vote_prompt "Agenda Item No. 3.1: Should we require students to wear masks in classrooms?"
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--personas_config", required=True)
    parser.add_argument("--micro_config", required=True)
    parser.add_argument("--topics_config", required=True)
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--add_times", type=str2bool, required=True, help="Whether to add times")
    parser.add_argument("--save_dir", default="results_simulation")
    parser.add_argument("--num_candidates", type=int, default=3)
    
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=20, help="Top-k sampling cutoff")
    parser.add_argument("--top_p", type=float, default=0.8, help="Top-p nucleus sampling cutoff")
    parser.add_argument("--repetition_penalty", type=float, default=1.3, help="Penalty for token repetition")
    parser.add_argument("--max_new_tokens", type=int, default=500, help="Maximum new tokens to generate")
    parser.add_argument("--num_repeats", type=int, default=15, help="number of times to repeat experiment")
    parser.add_argument("--system_prompt", type=str, default='full', help="System Prompt Option")
    parser.add_argument("--appelant", type=str, default='John F. General', help="Appelant Name to simulate")

    args = parser.parse_args()
    with open(args.config) as f:
        model_paths = json.load(f)


    if 'qwen' in args.base_model.lower():
        chat_template = "qwen3-instruct"
    elif 'llama' in args.base_model.lower():
        chat_template = "llama-3.3"
    elif 'gpt' in args.base_model.lower():
        chat_template = "gpt-oss"
    else:
        print("No chat template applied")

    tokenizer = AutoTokenizer.from_pretrained(list(model_paths.values())[0])
    tokenizer = get_chat_template(tokenizer, chat_template)
    base_model = load_base_model(args.base_model)
    base_model.resize_token_embeddings(len(tokenizer), mean_resizing=False)


    if 'tags' in args.config:
        tag_string = f' [PUBLIC_ADDRESSING]'
    else:
        tag_string = f''

    if args.dataset == 'Albermale':
        with open(args.persona_config.format('Albermale')) as f:
            PERSONAS = json.load(f)

        with open(args.topics_config.format('Albermale')) as f:
            simulation_topics = json.load(f)

        with open(args.micro_config.format('Albermale'), "r", encoding="utf-8") as f:
            micro_profiles_data = json.load(f)

        people_list = ['ellenosborne', 'davidoberg', 'grahampaige', 'jonnoalcaro', 'katrinacallsen', 'kateacuff', 'judyle']
        speaking_order = sorted(people_list)

        # Precompute topic schedule
        topic_schedule = compute_topic_schedule(simulation_topics, '9:00')
        first_topic = topic_schedule[0]  # first topic in the schedule
        first_time = first_topic["discussion_start"]  # usually "09:00"
        
        meeting_length = get_meeting_length_minutes(topic_schedule)
        print("Total Meeting Length, ", meeting_length)

        FIRST_SPEAKER = "grahampaige"
        if args.add_times:
            time_string = f"[current_time=[{first_time}] [agenda_item={first_topic['title']}] "
        else:
            time_string = ''

        FIRST_MESSAGE = f"{time_string}{FIRST_SPEAKER}:{tag_string} Good morning, everybody. I am calling this special board meeting for the Albemarle County School Board to order."


    elif args.dataset == 'DCAppeals':
        # appelants = ['John F. General'  "Parker et al."]. Specify throuhg args
        people_list = ['judgemcleese', 'judgeglickman', 'judgedeahl', args.appelant]

        with open(args.persona_config.format('DCAppeals')) as f:
            PERSONAS = json.load(f)

        with open(args.topics_config.format('DCAppeals')) as f:
            simulation_topics = json.load(f)

        with open(args.micro_config.format('DCAppeals'), "r", encoding="utf-8") as f:
            micro_profiles_data = json.load(f)

        people_list = ['judgemcleese', 'judgedeahl', 'judgeglickman', 'John F. General']
        speaking_order = ['judgemcleese', 'John F. General', 'judgedeahl', 'John F. General', 'judgeglickman', 'John F. General']


        # Precompute topic schedule
        topic_schedule = compute_topic_schedule(simulation_topics, '9:00')
        first_topic = topic_schedule[0]  # first topic in the schedule
        first_time = first_topic["discussion_start"]  # usually "09:00"
        
        meeting_length = get_meeting_length_minutes(topic_schedule)
        print("Total Meeting Length, ", meeting_length)

        if args.add_times:
            time_string = f"[current_time=[{first_time}] [agenda_item={first_topic['title']}] "
        else:
            time_string = ''

        FIRST_SPEAKER = "courtroomclerk"
        FIRST_MESSAGE = f"{time_string}{FIRST_SPEAKER}:{tag_string} All persons having business for the Honorable Chief Judge and Associate Judges are residing with the District of Columbia Court of Appeals. Draw near and give your attention. God save the United States and this Honorable Court. This Honorable Court is now in session. Please come forward."
        defendent_system_prompt = create_defendant_context_card(args.appelant, people_list, speaking_order, add_times=False)

    elif args.dataset == 'Waipa':
        with open(args.persona_config.format('Waipa')) as f:
            PERSONAS = json.load(f)

        with open(args.topics_config.format('Waipa')) as f:
            simulation_topics = json.load(f)

        with open(args.micro_config.format('Waipa'), "r", encoding="utf-8") as f:
            micro_profiles_data = json.load(f)

        people_list =["susanoregan", "jimmylchreest", "clairestpierre", "andrewbrown", "rogergordon",
                    "loubrown", "angeholt", "suemilner"]
        speaking_order = sorted(people_list)

        # Precompute topic schedule
        topic_schedule = compute_topic_schedule(simulation_topics, '9:00')
        first_topic = topic_schedule[0]  # first topic in the schedule
        first_time = first_topic["discussion_start"]  # usually "09:00"
        
        meeting_length = get_meeting_length_minutes(topic_schedule)
        print("Total Meeting Length, ", meeting_length)

        if args.add_times:
            time_string = f"[current_time=[{first_time}] [agenda_item={first_topic['title']}] "
        else:
            time_string = ''

        FIRST_SPEAKER = "jimmylchreest"
        FIRST_MESSAGE = f"{time_string}{FIRST_SPEAKER}:{tag_string} So, we are now live on YouTube. So, yeah, welcome everyone to the Audit and Risk Committee meeting for today."

    else:
        raise ValueError("Dataset not recognized")


    agents = {}
    for name, path in model_paths.items():
        agents[name] = Agent(
            name,
            base_model,
            tokenizer,
            path,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            max_new_tokens=args.max_new_tokens
        )
        
    # repeat experiment this number of times
    for repeat in range(args.num_repeats):
                
        save_dir = os.path.join(args.save_dir, str(repeat))
        if os.path.exists(save_dir):
            print(f"Skipping {repeat}, file already exists")
            print('-'*20)
            continue
        else:
            # Create the directory (including parents if needed)
            print(f"Running Repeat Num: {repeat}")
            print('-'*20)
            os.makedirs(save_dir, exist_ok=True)

        
        log = [{
            "speaker": FIRST_SPEAKER,
            "content": FIRST_MESSAGE
        }]
        
        current_hour, current_minute = 9, 0  # or meeting_start parsed as integers

        turn_order = deque(speaking_order)
        print("Starting Simulation, Example Context String:")
        print(create_context_card_simulation(speaking_order[0], simulation_topics, people_list, speaking_order))
        print('-'*20)
        print(f'{FIRST_MESSAGE}')
        for round_idx in range(meeting_length):
            print('-'*20)
            speaker = turn_order.popleft()
            current_topic = get_current_topic(topic_schedule, current_hour, current_minute)

            if args.add_times:
                context_prefix = f"[current_time={current_hour:02d}:{current_minute:02d}, agenda_item={current_topic}] {speaker}: "
            else:
                context_prefix = f"{speaker}: "
            
            
            conv = [{"role": "user" if msg['speaker'] != speaker else "assistant", "content": f"{msg['content']}"} for msg in log]
            conv = truncate_conversation(merge_consecutive_messages(conv), 'assistant', max_tokens=1000, model_name=args.base_model)

            if speaker == args.appelant:
                system_prompt = defendent_system_prompt
                conv.insert(0, {"role": "system", "content": system_prompt})

                if 'gpt' in args.base_model.lower():
                    # DISABLE REASONING
                    prompt = tokenizer.apply_chat_template(
                        conv,
                        tokenize=False,
                        add_generation_prompt=False,
                        reasoning_effort='low'
                    ) + f"""<|start|>assistant<|channel|>analysis<|message|><|end|><|start|>assistant<|channel|>final<|message|> {context_prefix}: """
                    
                else:
                    prompt = tokenizer.apply_chat_template(
                        conv,
                        tokenize=False,
                        add_generation_prompt=True
                    ) + context_prefix

                response = query_base_model(base_model, tokenizer, prompt, max_new_tokens=800)
            else:
                agent = agents[speaker]
                #create_context_card_simulation(speaker, persona_info, topics_list, people_list, scenario="full"):
                #system_prompt = create_context_card_simulation(speaker, PERSONAS, simulation_topics, micro_profiles_data, people_list, speaking_order, scenario=args.system_prompt)
                system_prompt= create_context_card_simulation(speaker, simulation_topics, people_list, speaking_order, add_times = args.add_times)
                conv.insert(0, {"role": "system", "content": system_prompt})
                #conv.append({"role": "system", "content": system_prompt})

                # Inside your main simulation loop

                if 'gpt' in args.base_model.lower():
                    # DISABLE REASONING
                    prompt = tokenizer.apply_chat_template(
                        conv,
                        tokenize=False,
                        add_generation_prompt=False,
                        reasoning_effort='low'
                    ) + f"""<|start|>assistant<|channel|>analysis<|message|><|end|><|start|>assistant<|channel|>final<|message|> {context_prefix}: """
                    
                else:
                    prompt = tokenizer.apply_chat_template(
                        conv,
                        tokenize=False,
                        add_generation_prompt=True
                    ) + context_prefix
                
                response = agent.generate_response(
                    prompt,
                    agent_names=people_list,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    repetition_penalty=args.repetition_penalty,
                    max_new_tokens=args.max_new_tokens
                )

            
            print(f"{context_prefix}{response}")
            
            log.append({"speaker": speaker, "content": f"{context_prefix}{response.strip()}"})
            turn_order.append(speaker)

            num_words = len(response.split())
            current_minute = current_minute + 1 + num_words//100
            if current_minute >= 60:
                current_hour += 1
                current_minute %= 60


        save_dir = os.path.join(args.save_dir, str(repeat))
        os.makedirs(save_dir, exist_ok=True)
        json.dump(log, open(f"{save_dir}/simulation.json", "w"), indent=2)



if __name__ == "__main__":
    main()