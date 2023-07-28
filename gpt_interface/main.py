import json
import datetime
import time
import openai
import os

import load_data
import prompts
from analyze_results import _is_labelled

def segmented_classification(instruction, examples, question, question_lastturn, reasoning, convo: load_data.Convo):
    segments = convo.segments()
    results = []
    labelled_turns = []
    for content, target_idx in segments:
        generator = prompts.Classify_DiaFirst(content, question, convo.turn_string(target_idx), instruction, reasoning)
        s = time.perf_counter()
        generated, output, used_tokens = generator.generate()
        dur = time.perf_counter() - s
        results.append({
            'elapsed': f"{int(dur)}",
            'used_tokens': used_tokens,
            'prompt': generator.prompt,
            'generated': generated,
            'output': output,
            'full': generator.prompt.split('\n') + generated.split('\n'),
        })
        if _is_labelled(generated):
            labelled_turns.append(target_idx)
        elif "Decision: []" in generated:
            pass
        else:
            pass
    if isinstance(convo, load_data.ConvoDouble):
        return {"results": results, "true": convo.labelled, "true_atodds": convo.atodds_labelled, "predicted": labelled_turns}
    else:
        return {"results": results, "true": convo.labelled, "predicted": labelled_turns}

def _stats(predicted, true, total_turns):
    overlap = set(predicted).intersection(true)
    extra = set(predicted) - set(true)
    missing = set(true) - set(predicted)
    prec = len(overlap) / len(predicted) if len(predicted) > 0 else 0
    rec = len(overlap) / len(true) if len(true) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if prec > 0 or rec > 0 else 0
    scores = {
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "num_correct": len(overlap),
        "num_mistakes": len(extra) + len(missing)
    }
    return scores

def stats(predicted, convo, total_turns):
    if isinstance(convo, load_data.ConvoDouble):
        stats = _stats(predicted, convo.labelled + convo.atodds_labelled, total_turns)
    else:
        stats = _stats(predicted, convo.labelled, total_turns)
    return stats

if __name__ == '__main__':
    openai.api_key = os.getenv("OPENAI")

    convos = load_data.abc_eval(list(prompts.instructions.keys()), only_double=True)

    for label, convos in convos.items():
        if label not in prompts.instructions:
            continue
        instruction = prompts.instructions[label]
        question = prompts.questions[label]
        question_lastturn = prompts.questions_lastturn[label]
        reasoning = prompts.reasoning[label]
        examples = prompts.examples[label]
        generations = []
        logfile = f'evaluation/no_instr/{label}-{datetime.datetime.now().strftime("%m%d%y_%H%M%S")}.json'
        print(logfile)
        total_tokens = 0
        start = time.time()
        for i, convo in enumerate(convos):
            print(f'\tExample {i + 1} generating...')
            results_dict = segmented_classification(instruction, examples, question, question_lastturn, reasoning, convo)
            total_tokens += sum([r["used_tokens"] for r in results_dict["results"]])
            results_dict["scores"] = stats(results_dict["predicted"], convo, int(len(convo.turns) / 2))
            generations.append(results_dict)
            if (i+1) % 10 == 0:
                json.dump(generations, open(logfile, 'w'), indent=2)
                print(f"{total_tokens} generated => ${(total_tokens / 1000 * 0.002):.2f} [{time.time()-start:.2f}]")
        print(f'\tFinished! [{time.time()-start:.2f}]')
        json.dump(generations, open(logfile, 'w'), indent=2)
        print(f"{total_tokens} generated => ${(total_tokens / 1000 * 0.002):.2f}")



