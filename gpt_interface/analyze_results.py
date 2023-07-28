import os
import json
from collections import Counter
import pandas as pd
import numpy as np
from gpt_interface.metric_utils import *
import csv

def get_avg_cost_per_convo(results_dir):
    print()
    for root, dirs, files in os.walk(results_dir):
        for filename in files:
            label = filename[:filename.find('-')]
            data = json.load(open(os.path.join(root, filename)))
            tokens = [sum([r["used_tokens"] for r in result["results"]]) for result in data]
            cost = [t / 1000 * 0.002 for t in tokens]
            avg_cost = sum(cost) / len(cost)
            print(f"{label}: {avg_cost:.2f}")
    print()

def get_decision_language(results_dir):
    decision_strs = []
    for root, dirs, files in os.walk(results_dir):
        for filename in files:
            data = json.load(open(os.path.join(root, filename)))
            for result in data:
                for result_dict in result["results"]:
                    result_dict["generated"] = result_dict["generated"].lower()
                    decision_idx = result_dict["generated"].find('\ndecision:')
                    if decision_idx > -1:
                        decision = result_dict["generated"][decision_idx:]
                        # endline_idx = decision.rfind('\n')
                        # if endline_idx > -1:
                        #     decision = decision[:endline_idx]
                        decision_strs.append(decision)
                    else:
                        print('-' * 50)
                        print('No decision tag found!')
                        print(result_dict["generated"])
                        print('-'*50)
    print()
    decision_str_counts = Counter(decision_strs)
    print(json.dumps(decision_str_counts, indent=2))

def _is_labelled(generated):
    if "Decision: [x]" in generated or \
            "Decision: [✓]" in generated or \
            "Decision: [YES]".lower() in generated.lower() or \
            "Decision: YES".lower() in generated.lower() or \
            "Decision is: [YES]".lower() in generated.lower():
        return True
    return False

def get_overall_stats(results_dir, type="overall", do_print=True):
    if do_print:
        print('\n' + '#'*50)
        print('STATS')
        print('#' * 50 + '\n')
    stats_dict = {}
    for root, dirs, files in os.walk(results_dir):
        for filename in files:
            label = filename[:filename.find('-')]
            if do_print:
                print(label)
            data = json.load(open(os.path.join(root, filename)))
            overall_pred = []
            overall_true = []
            for result in data:
                if 'true_atodds' not in result:
                    # training convos
                    pred_labels = [int(_is_labelled(result_dict["generated"])) for result_dict in result["results"]]
                    true_labels = [1 if i in result["true"] else 0 for i in range(1, len(result["results"])*2, 2)]
                else:
                    # annotated convos
                    if type == "overall_p":
                        pred_labels = [int(_is_labelled(result_dict["generated"])) for result_dict in result["results"]]
                        true_labels = [1 if i in result["true"]+result["true_atodds"] else 0
                                       for i in range(1, len(result["results"]) * 2, 2)]
                    elif type == "overall_n":
                        pred_labels = [int(_is_labelled(result_dict["generated"])) for result_dict in result["results"]]
                        true_labels = [1 if i in result["true"] else 0
                                       for i in range(1, len(result["results"]) * 2, 2)]
                    elif type == "hagreed":
                        pred_labels, true_labels = [], []
                        pos = -1
                        for i in range(1, len(result["results"]) * 2, 2):
                            pos += 1
                            if i not in result["true_atodds"]:
                                # label was agreed upon by humans -- either 1 or 0
                                true_l = 1 if i in result["true"] else 0
                                true_labels.append(true_l)
                                pred_l = 1 if _is_labelled(result["results"][pos]["generated"]) else 0
                                pred_labels.append(pred_l)
                    elif type == "hatodds":
                        pred_labels, true_labels = [], []
                        pos = -1
                        for i in range(1, len(result["results"]) * 2, 2):
                            pos += 1
                            if i in result["true_atodds"]:
                                # label was NOT agreed upon by humans
                                true_l = 1
                                true_labels.append(true_l)
                                pred_l = 1 if _is_labelled(result["results"][pos]["generated"]) else 0
                                pred_labels.append(pred_l)
                    else:
                        raise Exception(f"type={type} is not implemented!")
                # _get_metrics(pred_labels, true_labels, do_print=do_print)  ## for metrics by convo
                overall_pred.extend(pred_labels)
                overall_true.extend(true_labels)
            if do_print:
                print('\t' + '-'*20)
                print()
            stats_dict[label] = get_metrics(overall_pred, overall_true, do_print=do_print)
    return stats_dict

def _extract_dialogue_str(prompt: str):
    sep = '----------'
    end_idx = prompt.rfind(sep)
    start_idx = prompt.rfind(sep, 0, end_idx)
    dialogue_str = prompt[start_idx+len(sep): end_idx].strip()
    return dialogue_str

def _extract_reasoning_str(generated: str):
    end_idx = generated.find("\nDecision:")
    start_idx = generated.find("Reasoning:")
    reasoning_str = generated[start_idx+len("Reasoning:"):end_idx].strip()
    return reasoning_str

def _extract_bot_turn(prompt: str):
    prefix = """:

Speaker 2:"""

    start_idx = prompt.find(prefix) + len(prefix)
    end_idx = prompt.find('\n\n', start_idx)
    bot_turn = prompt[start_idx:end_idx].strip()
    return bot_turn

def results_to_tsv(results_dir):
    for root, dirs, files in os.walk(results_dir):
        for filename in files:
            label = filename[:filename.find('-')]
            writer = csv.writer(open(f'evaluation/abc-eval-tsv/{label}.tsv', 'w'), delimiter='\t')
            writer.writerow(['Turn', 'H-agreed', 'H-atodds', 'GPT', 'Reasoning', 'Success'])
            data = json.load(open(os.path.join(root, filename)))
            for result in data:
                for idx, r in enumerate(result["results"]):
                    dialogue_str = _extract_dialogue_str(r["prompt"])
                    last_turn = dialogue_str.split('\n')[-1]
                    bot_str = _extract_bot_turn(r["prompt"])
                    reasoning_str = _extract_reasoning_str(r["generated"])
                    pred_label = int(_is_labelled(r["generated"]))
                    true_label = int((idx*2)+1 in result["true"])
                    true_atodds_label = int((idx*2)+1 in result["true_atodds"])
                    writer.writerow([last_turn, '', '', '', '', ''])
                    bot_idx = int(last_turn[:last_turn.find('.')])+1
                    bot_turn = f'{bot_idx}. Speaker 2: ' + bot_str
                    success = 1 if true_label == pred_label else 0
                    writer.writerow([bot_turn, true_label, true_atodds_label, pred_label, reasoning_str, success])
                writer.writerows([
                    ['', '', '', '', '', ''],
                    ['#'*250, '', '', '', '', ''],
                    ['', '', '', '', '', ''],
                ])

def training_convo_stats():
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)

    results_dir = "evaluation/incontext_examples"
    get_decision_language(results_dir)
    incontext_stats = get_overall_stats(results_dir, do_print=False)
    get_avg_cost_per_convo(results_dir)

    results_dir = "evaluation/instructions_first"
    get_decision_language(results_dir)
    instr_stats = get_overall_stats(results_dir, do_print=False)
    get_avg_cost_per_convo(results_dir)

    results_dir = "evaluation/dialogue_first"
    get_decision_language(results_dir)
    dia_stats = get_overall_stats(results_dir, do_print=False)
    get_avg_cost_per_convo(results_dir)

    results_dir = "evaluation/last_turn"
    get_decision_language(results_dir)
    lastturn_stats = get_overall_stats(results_dir, do_print=False)
    get_avg_cost_per_convo(results_dir)

    results_dir = "evaluation/development/sarah_testing/best_temp10"
    get_decision_language(results_dir)
    temp10_stats = get_overall_stats(results_dir, do_print=False)

    for label in incontext_stats:
        print(label)
        df = pd.DataFrame.from_dict(
            {
                'incontext': incontext_stats[label],
                'instr_first': instr_stats[label],
                'dia_first': dia_stats[label],
                'last_turn': lastturn_stats.get(label, [np.nan]*len(dia_stats[label])),
                'temp10': temp10_stats.get(label, [np.nan]*len(dia_stats[label]))
            },
            orient='index')
        df.columns = ['P', 'R', 'F1', '+', 'Po', 'Ro', 'F1o', 'A', 'Alpha', '#']
        print(df)
        print()

    """
    instr_first:  111  == 3 wins
    dia_first:    1111 == 4 wins
    in_context:   11   == 2 wins
    """

def _check_convo_match(pred_annots, true_annots):
    dialogue_str = _extract_dialogue_str(pred_annots['results'][-1]['prompt'])
    last_bot_str = _extract_bot_turn(pred_annots['results'][-1]['prompt'])
    tag = 'Speaker '
    pred_turns = [t[t.find(tag) + len(tag) + 3:].strip() for t in dialogue_str.split('\n')] + [last_bot_str]
    true_turns = [t.strip() for d in true_annots['turns'] for t in [d['user'], d['system']]]
    for t in zip(pred_turns, true_turns):
        assert t[0] == t[1]

from data.analysis import project_data
import gpt_interface.prompts
def average_metrics(results_dir):
    only_double = True
    human_label_project = project_data.surge_evaluation
    convo_dict = {}
    for label in list(prompts.instructions.keys()):
        convo_ls = []
        for dialogue_id, dialogue in human_label_project.dialogues.items():
            doubly_annotated = len(dialogue.turns[0].behavior_annotations[label]) > 1
            if not only_double or doubly_annotated:
                turns = [{'user': turn_pair.user_turn, 'system': turn_pair.bot_turn} for turn_pair in dialogue.turns]
                scores = [[x.score for x in turn.behavior_annotations[label][:2]] for turn in dialogue.turns]
                convo_ls.append({"id": dialogue_id, "turns": turns, "scores": scores})
        convo_dict[label] = convo_ls

    print('AVG ALPHA:')
    print()
    for root, dirs, files in os.walk(results_dir):
        for filename in files:
            label = filename[:filename.find('-')]
            true_annots_ls = convo_dict[label]
            data = json.load(open(os.path.join(root, filename)))
            all_t1 = []
            all_t2 = []
            all_p = []
            for i, pred_annots in enumerate(data):
                true_annots = true_annots_ls[i]
                _check_convo_match(pred_annots, true_annots)
                t1 = [a[0] for a in true_annots['scores']]
                all_t1 += t1
                t2 = [a[1] for a in true_annots['scores']]
                all_t2 += t2
                p = [1 if _is_labelled(r['generated']) else 0 for r in pred_annots['results']]
                all_p += p
            df = pd.DataFrame.from_dict({'pred': all_p, 'true': all_t1}, orient='index').T
            alpha1 = krippendorfs_alpha(df, ci=False, level='nominal')
            df = pd.DataFrame.from_dict({'pred': all_p, 'true': all_t2}, orient='index').T
            alpha2 = krippendorfs_alpha(df, ci=False, level='nominal')
            avg_alpha = (alpha1 + alpha2) / 2
            print(f"{label}: {avg_alpha:.2f} [{len(all_p)}]")
            metrics_t1 = np.array(get_metrics(all_p, all_t1, do_print=False))
            metrics_t2 = np.array(get_metrics(all_p, all_t2, do_print=False))
            avg_metrics = (metrics_t1 + metrics_t2) / 2
            print(' / '.join([f"{x:.2f}" for x in avg_metrics[:3]]))
            print(f"& {all_p.count(1)} & {int((all_t1.count(1) + all_t2.count(1)) / 2)}")
            print('& ' + ' / '.join([f"{x:.2f}" for x in avg_metrics[4:7]]) + f' & {avg_metrics[7]:.2f}')
    print()

def abc_eval_stats(results_dir):
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)

    get_decision_language(results_dir)

    overall_p_stats = get_overall_stats(results_dir, type='overall_p', do_print=False)
    overall_n_stats = get_overall_stats(results_dir, type='overall_n', do_print=False)
    hagreed_stats = get_overall_stats(results_dir, type='hagreed', do_print=False)
    hatodds_stats = get_overall_stats(results_dir, type='hatodds', do_print=False)

    for label in overall_p_stats:
        print(label)
        df = pd.DataFrame.from_dict(
            {
                'overall-p': overall_p_stats[label],
                'overall-n': overall_n_stats[label],
                'h-agreed': hagreed_stats[label],
                'h-atodds': hatodds_stats[label]
            },
            orient='index')
        df.columns = ['P', 'R', 'F1', '+', 'Po', 'Ro', 'F1o', 'A', 'Alpha', '#']
        print(df)
        print()


if __name__ == '__main__':
    results_dir = 'evaluation/abc-eval'

    abc_eval_stats(results_dir)
    average_metrics(results_dir)