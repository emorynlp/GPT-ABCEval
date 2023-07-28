import attrs
import os
import json

speaker_map = {
    'user': 'Speaker 1',
    'system': 'Speaker 2'
}

segmentation_strategy = {
    'commonsense contradiction': 'history',
    'self contradiction': 'history',
    'partner contradiction': 'history',
    'redundant': 'history',
    'incorrect fact': 'history',
    'empathetic': 'history',
    'lack of empathy': 'history',
    'ignore': 'history',
    'irrelevant': 'history'
}

@attrs.define
class Convo:
    label: str
    turns: list[tuple[str, str]]
    labelled: list[int]

    def content(self, i=None, include_last_turn=False):
        if i is None:
            i = len(self.turns)
        if include_last_turn:
            i = min(i+1, len(self.turns))
        cstr = '\n'.join([f"{idx+1}. {speaker_map[speaker]}: {text}" for idx, (speaker, text) in enumerate(self.turns[:i])])
        return cstr

    def segments(self, include_last_turn=False):
        strategy = segmentation_strategy[self.label]
        if strategy == 'history':
            segments = [
                (self.content(i, include_last_turn=include_last_turn), i)
                for i, target in enumerate(self.turns)
                if i % 2 == 1
            ]
        return segments

    def turn_string(self, idx):
        speaker, text = self.turns[idx]
        return f"{speaker_map[speaker]}: {text}"

@attrs.define
class ConvoDouble(Convo):
    atodds_labelled: list[int]


labels = {
    "consistency_label": ["self contradiction", "partner contradiction", "redundant"],
    "commonsense": ["commonsense contradiction"],
    "grammar": ["uninterpretable"],
    "sociality": ["antisocial"],
    "empathy": ["empathetic", "lack of empathy"],
    "knowledge": ["correct fact", "incorrect fact"],
    "personal_information": ["preference info", "life info"],
    "transitions": ["irrelevant", "ignore", "follow up", "topic switch"]
}

correct_for_label = {
    "commonsense contradiction": {"commonsense_q": "This response contradicts common knowledge."},
    "uninterpretable": {"grammar_q": "This response is uninterpretable."},
    "antisocial": {"sociality_q": "This response exhibits antisocial behavior."},
    "empathetic": {"Is this response empathetic?": "Yes, the speaker demonstrates an understanding of their partner's emotions."},
    "lack of empathy": {"Is this response empathetic?": "No, the speaker misinterprets their partner's emotions or inappropriately ignores their partner's feelings."},
    "correct fact": {
        "_disjunction": True,
        "options": [
            {
                "Does SPEAKER_X's response use, claim, or assume any FACTS (either correct or incorrect)?": "SPEAKER_X's response incorporates or assumes at least one fact.",
                "Do you know whether ALL of the facts that SPEAKER_X uses or assumes are accurate?": "Yes, I know for sure ALL facts are accurate."
            },
            {
                "Does SPEAKER_X's response use, claim, or assume any FACTS (either correct or incorrect)?": "SPEAKER_X's response incorporates or assumes at least one fact.",
                "Do you know whether ALL of the facts that SPEAKER_X uses or assumes are accurate?": "I don't know for sure whether ALL of the facts are accurate.",
                "Take 60 seconds to search ALL unknown facts on the internet. Does your search verify or falsify ALL the facts?": "ALL facts are accurate; a credible source verified the facts in my search."
            }
        ]
    },
}

multi_correct_for_label = {
    "incorrect fact": [
        {
            "_disjunction": True,
            "options": [
                {
                    "Does SPEAKER_X's response use, claim, or assume any FACTS (either correct or incorrect)?": "SPEAKER_X's response incorporates or assumes at least one fact.",
                    "Do you know whether ALL of the facts that SPEAKER_X uses or assumes are accurate?": "No, I know for sure that one of the facts is inaccurate, false, or highly implausible."
                },
                {
                    "Does SPEAKER_X's response use, claim, or assume any FACTS (either correct or incorrect)?": "SPEAKER_X's response incorporates or assumes at least one fact.",
                    "Do you know whether ALL of the facts that SPEAKER_X uses or assumes are accurate?": "I don't know for sure whether ALL of the facts are accurate.",
                    "Take 60 seconds to search ALL unknown facts on the internet. Does your search verify or falsify ALL the facts?": "One of the facts is inaccurate; a credible source falsified the fact or revealed that it is highly implausible."
                }
            ]
        },
        {
            "_disjunction": True,
            "options": [
                {
                    "Does SPEAKER_X's response use, claim, or assume any FACTS (either correct or incorrect)?": "SPEAKER_X's response incorporates or assumes at least one fact.",
                    "Do you know whether ALL of the facts that SPEAKER_X uses or assumes are accurate?": "It is misleading for SPEAKER_X to claim or assume one of the facts, because there is no way that SPEAKER_X knows whether that fact is accurate."
                },
                {
                    "Does SPEAKER_X's response use, claim, or assume any FACTS (either correct or incorrect)?": "SPEAKER_X's response incorporates or assumes at least one fact.",
                    "Do you know whether ALL of the facts that SPEAKER_X uses or assumes are accurate?": "I don't know for sure whether ALL of the facts are accurate.",
                    "Take 60 seconds to search ALL unknown facts on the internet. Does your search verify or falsify ALL the facts?": "I couldn't find enough credible evidence in my search to either verify or falsify one of the facts."
                },
                {
                    "Does SPEAKER_X's response use, claim, or assume any FACTS (either correct or incorrect)?": "SPEAKER_X's response incorporates or assumes at least one fact.",
                    "Do you know whether ALL of the facts that SPEAKER_X uses or assumes are accurate?": "No, I know for sure that one of the facts is inaccurate, false, or highly implausible."
                },
                {
                    "Does SPEAKER_X's response use, claim, or assume any FACTS (either correct or incorrect)?": "SPEAKER_X's response incorporates or assumes at least one fact.",
                    "Do you know whether ALL of the facts that SPEAKER_X uses or assumes are accurate?": "I don't know for sure whether ALL of the facts are accurate.",
                    "Take 60 seconds to search ALL unknown facts on the internet. Does your search verify or falsify ALL the facts?": "One of the facts is inaccurate; a credible source falsified the fact or revealed that it is highly implausible."
                }
            ]
        }
    ]
}

label_in_list = {
    "self contradiction": "cont_s_ctxt",
    "partner contradiction": "cont_p_ctxt",
    "redundant": "redundant"
}

label_in_dict = {
    "preference info": {"value_q": "This response shares information about the speaker's preferences/values."},
    "life info": {"experience_q": "This response shares information about the speaker's life."},
    "ignore": {"Does SPEAKER_X appropriately acknowledge SPEAKER_Y with this response?": "No, SPEAKER_X ignored SPEAKER_Y."},
    "topic switch": {"Is SPEAKER_X introducing a new topic?": "Yes, SPEAKER_X is changing the topic of the conversation."},
    "follow up": {"Is SPEAKER_X introducing a new topic?": "No, SPEAKER_X is ONLY responding to, building on, or further exploring what SPEAKER_Y said in the previous turn."},
    "irrelevant": {"Is SPEAKER_X's response appropriately relevant?": "No, the response feels abrupt, and interrupts the current discussion because it is irrelevant."}
}

def is_labelled(turn, label):
    if label in correct_for_label:
        return len(turn) == 4 and turn[2] == correct_for_label[label]
    if label in multi_correct_for_label:
        return len(turn) == 4 and turn[2] in multi_correct_for_label[label]
    if label in label_in_list:
        return len(turn) == 4 and len(turn[2]) > 0 and label_in_list[label] in list(turn[2].values())[0]
    if label in label_in_dict:
        return len(turn) == 4 and list(label_in_dict[label].items())[0] in list(turn[2].items())

def training():
    training_convos = {}
    dir = '../annotator_training'
    for training_file in sorted(os.listdir(dir)):
        if '.json' in training_file:
            for _, data in json.load(open(f"{dir}/{training_file}")).items():
                for label in labels.get(data['annotation_tasks'][0], []):
                    training_convos.setdefault(label, []).append(
                        Convo(
                            label=label,
                            turns=[t[:2] for t in data["turns"]],
                            labelled=[
                                i for i, t in enumerate(data["turns"])
                                if len(t) == 4 and is_labelled(t, label)
                            ]
                        )
                    )
    return training_convos

from data.analysis import project_data
def abc_eval(labels, only_double=True):
    human_label_project = project_data.surge_evaluation
    convo_dict = {}
    for label in labels:
        convo_ls = []
        for dialogue_id, dialogue in human_label_project.dialogues.items():
            doubly_annotated = len(dialogue.turns[0].behavior_annotations[label]) > 1
            if not only_double or doubly_annotated:
                turns = [turn for turn_pair in dialogue.turns for turn in [('user', turn_pair.user_turn), ('system', turn_pair.bot_turn)]]
                agreed_labelled_idx = [(i*2)+1 for i, turn in enumerate(dialogue.turns) if set([x.score for x in turn.behavior_annotations[label][:2]]) == {1}]
                atodds_labelled_idx = [(i*2)+1 for i, turn in enumerate(dialogue.turns) if set([x.score for x in turn.behavior_annotations[label][:2]]) == {0, 1}]
                convo = ConvoDouble(label=label, turns=turns, labelled=agreed_labelled_idx, atodds_labelled=atodds_labelled_idx)
                convo_ls.append(convo)
        convo_dict[label] = convo_ls
    return convo_dict