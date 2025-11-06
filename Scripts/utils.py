import os
from typing import List, Tuple
from torch.utils.data import DataLoader

# -------------------------
# ALBERMALE School Board
# -------------------------
ALBERMALE_TAGS = [
    # Core speech acts
    "ask_question",
    "give_update",
    "request_action",
    "defer_to_expert",
    "express_concern",
    "reassure",
    "clarify",
    "policy_proposal",
    "prioritization",
    "logistics_info",
    "hedge",
    "cite_external_source",
    "opinion",  

    # Interactional moves
    "acknowledge",  
    "agreeing",
    "disagreeing",
    "conceding",
    "persuaded",
    "follow_up_request",
    "interrupt_overlap",
    "public_addressing",

    # Procedural/meeting moves
    "procedural_move",
    "vote_call"
]

# -------------------------
# DC Appeals
# -------------------------
DCAPPEALS_TAGS = [
# Courtroom / Formalities
"court_opening",
"case_call",
"court_closing",
"court_procedural_remark",


# Attorney Advocacy
"opening_argument",
"substantive_argument",
"legal_citation",
"fact_statement",
"policy_argument",
"concession",
"rebuttal_argument",
"reserve_time",


# Judicial Interventions
"judge_question",
"judge_comment",
"judge_interruption",
"judge_procedural_directive",


# Interactional Moves
"acknowledge",
"hedge",
"agreement",
"disagreement",


# Case Management / Logistics
"time_management",
"case_transition",
"record_reference",
]



# -------------------------
# Council Meeting
# -------------------------
WAIPA_TAGS = [
    # Information exchange
    "ask_question",
    "provide_information",
    "cite_reference",
    "clarify_point",

    # Deliberation & framing
    "express_concern",
    "state_opinion",
    "policy_proposal",
    "evaluate_option",
    "prioritize_issue",

    # Influence & alignment
    "agree",
    "disagree",
    "build_on_point",
    "persuasion_attempt",
    "defer_to_authority",

    # Procedural control
    "procedural_move",
    "call_vote",
    "summarize_outcome",
    "defer_decision",

    # Relationship & legitimacy management
    "community_reference",
    "equity_reference",
    "reassure_stakeholders",
    "public_address",

    # Task management & closure
    "assign_action",
    "confirm_next_step",
    "report_back",
    "close_discussion"
]


def _escape_html(text: str) -> str:
    """Escape HTML special characters efficiently."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _get_split_points(parts_mask: List[bool]) -> List[int]:
    """Find split points where trainable/non-trainable sections change."""
    split_points = [0]
    for i in range(1, len(parts_mask)):
        if parts_mask[i] != parts_mask[i - 1]:
            split_points.append(i)
    split_points.append(len(parts_mask))
    return split_points


def _process_token_slice(
    input_ids: List[int], a: int, b: int, tokenizer, max_tokens: int
) -> Tuple[str, bool]:
    """Process a slice of tokens and return decoded text and truncation flag."""
    decode_token = input_ids[a:b]

    # Count padding tokens efficiently
    if hasattr(tokenizer, "pad_token_id") and tokenizer.pad_token_id is not None:
        # Find first padding token from the end
        pad_count = 0
        for token in reversed(decode_token):
            if token == tokenizer.pad_token_id:
                pad_count += 1
            else:
                break

        if pad_count > 0:
            decode_token = decode_token[:-pad_count]
    else:
        pad_count = 0

    is_truncated = b - a > max_tokens
    if is_truncated:
        decode_token = decode_token[:max_tokens]
    text = tokenizer.decode(decode_token, skip_special_tokens=False)

    if is_truncated:
        if pad_count:
            text += f"... padding {pad_count} tokens"
        else:
            text += "... (truncated)"

    return text, is_truncated


def debug_chat_dataloader_for_training(dataset, tokenizer, n_example=1):
    """
    Debug function to log samples from the training dataloader in an HTML format.
    Outputs to both terminal (with colors) and an HTML file with CSS styling.
    """

    def collate_fn(batch):
        """Convert list of examples into batched tensors"""
        input_ids = [ex["input_ids"] for ex in batch]
        labels = [ex["labels"] for ex in batch]

        # Convert to tensors (pad sequences if needed)
        # Here we assume input_ids and labels are already lists of token IDs
        import torch
        input_ids_tensor = torch.tensor(input_ids, dtype=torch.long)
        labels_tensor = torch.tensor(labels, dtype=torch.long)

        return {
            "input_ids": input_ids_tensor,
            "labels": labels_tensor,
        }

    # Example
    dataloader = DataLoader(
        dataset, 
        batch_size=1, 
        shuffle=True, 
        collate_fn=collate_fn
    )

    g = iter(dataloader)
    html_path = ".log/dataloader_examples.html"
    os.makedirs(os.path.dirname(html_path), exist_ok=True)

    # Create HTML file with CSS styling
    with open(html_path, "w") as html_file:
        html_file.write(
            """<!DOCTYPE html>
    <html>
    <head>
        <title>Dataloader Examples</title>
        <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        
        @media (prefers-color-scheme: light) {
            body { background-color: #ffffff; color: #333; }
            .trainable { background-color: #FFEBCD; color: #333; }
            .context { background-color: #E0FFE0; color: #333; }
            th { background-color: #f2f2f2; }
            th, td { border-color: #ddd; }
        }
        
        @media (prefers-color-scheme: dark) {
            body { background-color: #222; color: #f0f0f0; }
            .trainable { background-color: #664a20; color: #f0f0f0; }
            .context { background-color: #2a5a2a; color: #f0f0f0; }
            th { background-color: #444; color: #f0f0f0; }
            th, td { border-color: #555; }
        }
        
        .trainable, .context { padding: 2px; border-radius: 3px; }
        table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
        th, td { border: 1px solid; padding: 8px; text-align: left; }
        h2 { margin-top: 30px; }
        </style>
    </head>
    <body>
        <h1>Dataloader Examples</h1>
        <p>This file contains examples of training data with context and trainable parts.</p>
    """
        )

        max_print_tokens = 3000
        html_parts = []  # Collect HTML parts for batch writing

        for i in range(n_example):
            batch = next(g)
            input_ids = batch["input_ids"][0]
            label_ids = batch["labels"][0]
            parts_mask = label_ids >= 0  # True is trainable, False is context

            # Find split points efficiently
            split_points = _get_split_points(parts_mask)

            colored_parts = []
            html_parts.append(f"\n    <h2>Example {i+1}</h2>\n")
            html_parts.append(
                "    <table>\n        <tr><th>Text</th><th>Label</th></tr>\n"
            )

            for a, b in zip(split_points[:-1], split_points[1:]):
                text, is_truncated = _process_token_slice(
                    input_ids, a, b, tokenizer, max_print_tokens
                )
                is_trainable = parts_mask[a]
                token_count = b - a

                # Colored text for terminal
                color_code = "\033[93m" if is_trainable else "\033[92m"
                colored_text = f"{color_code}{text}\033[0m"
                colored_parts.append(colored_text)

                # HTML with CSS classes
                css_class = "trainable" if is_trainable else "context"
                emoji = "🟠" if is_trainable else "🟢"
                label_text = "TRAIN" if is_trainable else "CONTEXT"
                token_suffix = "tokens" if token_count > 1 else "token"
                label = f"{emoji} {label_text} {token_count} {token_suffix}"

                # Escape HTML and add row
                text_escaped = _escape_html(text)
                html_parts.append(
                    f"        <tr>\n"
                    f'            <td><span class="{css_class}">'
                    f"{text_escaped}</span></td>\n"
                    f"            <td>{label}</td>\n"
                    f"        </tr>\n"
                )

            html_parts.append("    </table>\n")

            # Print first example to terminal
            if i == 0:
                colored_output = "".join(colored_parts)
                terminal_msg = f"\n=== EXAMPLE #{i+1} ===\n{colored_output}\n"
                print(terminal_msg)

        # Write all HTML parts at once
        html_file.writelines(html_parts)
        html_file.write("</body>\n</html>")

    print(f"More training debug examples written to {html_path}")


def get_dataset(train_path, test_path, speaker, sys_message=1):
    """
    Load train and test datasets from JSON files and return examples for a specific speaker.

    Args:
        train_path: path to the training dataset JSON file
        test_path: path to the testing dataset JSON file
        speaker: the persona name to filter examples

    Returns:
        train_examples: list of training examples for the specified speaker
        test_examples: list of testing examples for the specified speaker
    """
    with open(train_path, "r", encoding="utf-8") as f:
        train_dataset = json.load(f)
    
    with open(test_path, "r", encoding="utf-8") as f:
        test_dataset = json.load(f)
    
    train_examples = train_dataset.get(speaker, [])
    test_examples = test_dataset.get(speaker, [])
    
    if not sys_message:
        train_examples = [x[1:] for x in train_examples]
        test_examples = [x[1:] for x in test_examples]
    return train_examples, test_examples
