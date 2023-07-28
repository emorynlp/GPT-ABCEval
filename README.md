# GPT-ABCEval

Paper: **Leveraging Large Language Models for Automated Dialogue Analysis** (to appear at SIGDIAL 2023)

 This work aims to automatically identify the occurrence of various dialogue behaviors in human-bot conversations using ChatGPT-3.5. 
 
This repo contains:
 
* our best-performing ChatGPT prompts and code to use them

  *Location:* `gpt_interface/main.py`


* the outputs produced during our experiments on the test data (`outputs/`), including:  

  * the behavior classification results from human judges, GPT, and the specialized classifier (where applicable) for the bot responses of the dialogues 

    *Location:* `outputs/behavior-classification-results.json`
  
    *Format:*
    ```
    {
        dialogue_id: [
            turn_dict, 
            ...
        ],
        ...
    }
    ```
  
    where `turn_dict` looks like:
  
    ```
        {
          "user": "yes it is super fun! you should definitely try iy",
          "system": "Yeah. Have you ever tried any virtual reality games before?",
          "behaviors": {
            "self contradiction": {"human": [0, 0], "gpt": 1, "specialized": 0},
            "empathetic": {"human": [0, 1], "gpt": 0, "specialized": 1},
            "lack of empathy": {"human": [0, 0], "gpt": 0, "specialized": 0},
            "irrelevant": {"human": [0, 0], "gpt": 0, "specialized": 0},
            "ignore": {"human": [0, 0], "gpt": 1, "specialized": 0},
            "incorrect fact": {"human": [0, 0], "gpt": 0, "specialized": 1},
            "commonsense contradiction": {"human": [1, 1], "gpt": 0},
            "partner contradiction": {"human": [1, 1], "gpt": 0},
            "redundant": {"human": [0, 1], "gpt": 0}
          }
        }
      ```

  * the input prompt and output of GPT for each bot response in the test data for the behavior classification tasks

    *Location:* `outputs/gpt_outputs/`
  
    *Format:*
    ```
    {
        dialogue_id: [
            turn_dict, 
            ...
        ],
        ...
    }
    ```
  
    where `turn_dict` looks like:
  
    ```
    {
        "user": "Hi!",
        "system": "How are you doing today? :d I'm good. I had a good day. I went to different restaurants and tried a variety of cuisines. Do you cook or eat out much?",
        "gpt_prompt": "----------\nDIALOGUE\n----------\n1. Speaker 1: Hi!\n----------\n\nIf this were the next response in the dialogue, would it contradict commonsense:\n\nSpeaker 2: How are you doing today? :d I'm good. I had a good day. I went to different restaurants and tried a variety of cuisines. Do you cook or eat out much?\n\nTo identify contradictions of commonsense, judge whether a vast majority of people would agree that the response doesn't make sense because the response:\n- contradicts common knowledge\n- makes unfounded assumptions\n- is highly illogical or self-contradictory\n- asks a question where the answer is already obvious\n\nDo NOT mark responses that don't make sense because they:\n- are off-topic or irrelevant as responses\n- don't have any clear meaning (e.g. overly vague or ill-formed responses)\n\nProvide your reasoning when considering this question starting with \"Reasoning:\". Then, finish by writing your final decision as one of: \"Decision: [YES]\" or \"Decision: [NO]\". Do NOT fill in your decision with any terms other than YES or NO.\n",
        "gpt_output": "Reasoning: The response does not contradict commonsense. It is a typical response to ask someone if they cook or eat out often after discussing food. It does not make unfounded assumptions, is not highly illogical or self-contradictory, and does not ask a question where the answer is already obvious. \nDecision: [NO]",
        "tokens": 306
    },
    ```


  * the results of the error analysis which consists of lists of GPT reasoning examples that correspond to each error type as denoted by filename ("Error Analysis" paper section)

    *Location:* `outputs/gpt_error_analysis/`
  
    *Format:*
    ```
    [
        turn_dict,
        ...
    ]
    ```
  
    where `turn_dict` looks like:
  
    ```
    {
        "behavior_type": "lack_of_empathy",
        "gpt_label": 1.0,
        "turns": [
            "1. Speaker 1: Hi!",
            "2. Speaker 2: How are you doing today? :d I'm good. I had a good day. I went to different restaurants and tried a variety of cuisines. Do you cook or eat out much?",
            "3. Speaker 1: I prefer to cook.",
            "4. Speaker 2: Cooking is fun because its easy and can be rewarding too. I also like cooking different types of foods from around the world. What are your favorite foods?",
            "5. Speaker 1: I like pasta and pizza.",
            "6. Speaker 2: Pizza is great. I really like any kind of pizza. Do you have any pets?",
            "7. Speaker 1: No, I don't.",
            "8. Speaker 2: Oh that's a shame. What do you like to do in your free time? I love to swim and go to the opera too.",
            "9. Speaker 1: I like to watch anime.",
            "10. Speaker 2: I've heard a lot about it but I haven't watched it much. Do you like any particular shows?",
            "11. Speaker 1: I really like My Hero Academia and Demon Slayer.",
            "12. Speaker 2: I'll have to check it out. Are those on netflix or hulu? I mainly use those sources."
        ],
        "gpt_reasoning": "Speaker 2's response does not show any indication of understanding or acknowledging Speaker 1's feelings about their favorite anime shows. Instead, Speaker 2 shifts the conversation to a different topic about streaming services. Therefore, Speaker 1 may feel like their feelings are not being understood by Speaker 2."
    }
    ```

### Dependencies

* Python 3.9
* `requirements.txt`

### Dialogue Behaviors

The dialogue behaviors under study are sourced from the ABC-Eval work:
* [Don't Forget Your ABC's: Evaluating the State-of-the-Art in Chat-Oriented Dialogue Systems](https://arxiv.org/pdf/2212.09180.pdf)

There are 16 behaviors total in the ABC-Eval paper, but we focus on 9 of those behaviors which were found to be informative for the perceived quality of human-bot conversations:

* empathetic
* lack of empathy
* commonsense contradiction
* incorrect fact
* self contradiction
* partner contradiction
* redundant
* ignore
* irrelevant

Each dialogue behavior has a corresponding ChatGPT prompt, with instructions and details tailored to enable ChatGPT to identify the behavior when it occurs in a dialogue response (`gpt_interface/prompts.py`).

### Test Data

Dialogues labeled for behaviors by humans can be found in `data/data.json`.

Python script `explore_data.py` provides an example of interfacing with the data.



