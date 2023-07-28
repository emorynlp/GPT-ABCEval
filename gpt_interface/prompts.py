from gpt_interface.gpt_utils import Prompt

class Classify_DiaFirst(Prompt):
    """
    Identify occurrence of dialogue behaviors
    """
    template = """----------
DIALOGUE
----------
{}
----------

{}

{}

{}

{} Then, finish by writing your final decision as one of: "Decision: [YES]" or "Decision: [NO]". Do NOT fill in your decision with any terms other than YES or NO.
"""



instructions = {
    "commonsense contradiction": """To identify contradictions of commonsense, judge whether a vast majority of people would agree that the response doesn't make sense because the response:
- contradicts common knowledge
- makes unfounded assumptions
- is highly illogical or self-contradictory
- asks a question where the answer is already obvious

Do NOT mark responses that don't make sense because they:
- are off-topic or irrelevant as responses
- don't have any clear meaning (e.g. overly vague or ill-formed responses)""",

    "self contradiction": """Self contradictions occur when Speaker 2 says something that is a contradiction of what they have said previously or it is extremely implausible based on the information they have already shared.
Self contradictions may also occur within a single turn if Speaker 2 shares two contradictory things.
If Speaker 2 shares world knowledge that is factually incorrect, this is NOT enough on its own to warrant a self contradiction.
If Speaker 2 contradicts something the other speaker Speaker 1 has said, this is NOT a self-contradiction.""",

    "partner contradiction": """Partner contradictions occur when Speaker 2:
- shares an assumption about Speaker 1 that is impossible to know based on what has already been said
- shares an inference about Speaker 1 that is implausible based on what has already been said
- contradicts something Speaker 1 shared about themselves
- asks a repetitive question about Speaker 1 when the answer is already known based on what has already been said

If Speaker 2 says something that makes it seem like they have forgotten or misremembered what their partner Speaker 1 has said earlier in the dialogue, this is a partner contradiction.
If Speaker 2 shares a difference of opinion or situation in their own life as compared to Speaker 1, this is NOT a partner contradiction.""",

    "redundant": """A response is repetitive if:
- it repeats something from earlier in the dialogue
- it includes asking a question whose answer has already been shared

If any part of the response is repetitive, then it should be labelled as repetitive.
Note that sometimes repetition is useful, such as for emphasis, acknowledgement, clarification, or elaboration, and in these cases it should NOT be labelled as repetitive.""",

    "incorrect fact": """Incorrect facts occur when the response includes information that is either:
- false
- unproven
- highly controversial
- highly implausible
- clearly misleading
    
If an organization, person, place, etc. is mentioned as a part of public knowledge, but it does not exist or it is inaccurately represented, then this is an incorrect fact. 

Do NOT consider a turn as an incorrect fact if the turn could be interpreted as expressing:
- preference or value judgements
- estimates or predictions
- personal information about the speaker or their partner
- information about things in either speaker's life that are not publicly relevant""",

    "empathetic": """ A response is empathetic when Speaker 2 does ONE of the following:
- clearly demonstrates an understanding of Speaker 1's emotions
- reacts with the appropriate sentiment or emotion to Speaker 1's shared experience
- understands or appropriately reacts to Speaker 1's experience or emotions
- appropriately reassures, encourages, or supports Speaker 1""",

    "lack of empathy": """A response displays a lack of empathy when:
- it indicates a misunderstanding of how Speaker 1 feels based on what Speaker 1 just said
- the tone, emotion, or sentiment of the response is clearly inappropriate for what Speaker 1 just said
- the response has an inappropriate lack of emotion to what Speaker 1 just said

Do NOT consider its empathy relative to previous topics in the conversation if the dialogue has moved on from them.
Instead, only consider the most recent dialogue context when evaluating the empathy of a response.""",

    "ignore": """Responses that are completely off-topic, fail to address the asked question, or are otherwise completely inappropriate in the context are considered to be ignoring the other speaker.""",

    "irrelevant": """If a response fails to continue the current discussion or jumps to a new and off-topic discussion, it is considered to be irrelevant.
Responses that are irrelevant feel abrupt and interrupt the discussion, usually because they present questions or ideas that are unrelated to the previous turn.
Short reactions to or acknowledgements of the previous turn are NOT irrelevant."""

}

questions = {
    "commonsense contradiction": "If this were the next response in the dialogue, would it contradict commonsense:",
    "self contradiction": "If this were the next response in the dialogue, is it a self-contradiction by Speaker 2:",
    "partner contradiction": "If this were the next response in the dialogue, is Speaker 2 saying something about Speaker 1 that is contradicting what Speaker 1 has already shared:",
    "redundant": "Is this response repeating something that has already been said:",
    "incorrect fact": "Does this response include an incorrect fact:",
    "empathetic": 'Is this an empathetic response by Speaker 2:',
    "lack of empathy": 'If this were the next response in the dialogue, would Speaker 1 feel like their feelings are not being understood by Speaker 2:',
    "ignore": 'If this were the next response in the dialogue, does it completely ignore the immediate last turn from Speaker 1:',
    "irrelevant": 'If this were the next response in the dialogue, is it completely irrelevant to what was just said:'
}

questions_lastturn = {
    "commonsense contradiction": "Given the dialogue above, does its final response contradict commonsense?",
    "self contradiction": "Given the dialogue above, is its final response a self-contradiction by Speaker 2?",
    "partner contradiction": "Given the dialogue above, does its final response say something about Speaker 1 that contradicts what Speaker 1 has already shared?",
    "redundant": "Given the dialogue above, does its final response response repeat something that has already been said?",
    "incorrect fact": "Given the dialogue above, does its final response include an incorrect fact?",
    "empathetic": 'Given the dialogue above, is its final response an empathetic response by Speaker 2?',
    "lack of empathy": 'Given the dialogue above, does its final response make Speaker 1 feel like their feelings are not being understood by Speaker 2?',
    "ignore": 'Given the dialogue above, does its final response completely ignore the immediate final turn from Speaker 1?',
    "irrelevant": 'Given the dialogue above, is its final response completely irrelevant to what Speaker 1 just said?'
}

reasoning = {
    "commonsense contradiction": 'Provide your reasoning when considering this question starting with "Reasoning:".',
    "self contradiction": 'Provide your reasoning when considering this question starting with "Reasoning:".',
    "partner contradiction": 'Provide your reasoning when considering this question starting with "Reasoning:".',
    "redundant": 'Provide your reasoning when considering this question starting with "Reasoning:".',
    "incorrect fact": 'Provide your reasoning when considering this question starting with "Reasoning:".',
    "empathetic": 'Provide your reasoning when considering this question starting with "Reasoning:".',
    "lack of empathy": 'Provide your reasoning when considering this question starting with "Reasoning:".',
    "ignore": 'Provide your reasoning when considering this question starting with "Reasoning:".',
    "irrelevant": 'Provide your reasoning when considering this question starting with "Reasoning:".'
}

examples = {
    'commonsense contradiction': """----------
DIALOGUE
----------
1. Speaker 1: I'm sad the weekend is over.
2. Speaker 2: Me too.
3. Speaker 1: What did you do?
----------

If this were the next response in the dialogue, would it contradict commonsense:

"Speaker 2: I went surfing at the lake this weekend."

Reasoning: surfing generally requires ocean waves, making the claim implausible.

Decision: [YES]

----------
DIALOGUE
----------
1. Speaker 1: Hi, how are you?
2. Speaker 2: Frustrated. My mom and I got into a fight this weekend.
3. Speaker 1: Oh no, what did you fight about?
4. Speaker 2: My political views.
5. Speaker 1: What was so controversial?
----------

If this were the next response in the dialogue, would it contradict commonsense:

"Speaker 2: I think Kanye West has done a great job as president."

Reasoning: even though the response is factually inaccurate, it does not violate common knowledge. This is because information about Kanye West and the president does not count as common knowledge; most people do not learn who the president is through direct experience (most people learn who the president is through the news or by someone else telling them).

Decision: [NO] 

----------
DIALOGUE
----------
1. Speaker 1: It's your birthday soon. Do you want a party?
2. Speaker 2: Sure, that sounds like fun.
3. Speaker 1: What kind of cake do you want?
----------

If this were the next response in the dialogue, would it contradict commonsense:

"Speaker 2: I don't like chocolate."

Reasoning: Speaker 2 is making to make a claim about his own preferences, so the response does not violate commonsense knowledge.

Decision: [NO]

----------
DIALOGUE
----------
1. Speaker 1: My high school graduation was last week.
----------

If this were the next response in the dialogue, would it contradict commonsense:

"Speaker 2: Oh, I'm sorry to hear that."

Reasoning: Speaker 2's sympathy is a highly illogical social response to Speaker 1 sharing that he graduated, and contradicts the common knowledge that graduating high school is a positive achievement. Given that Speaker 2 does not provide additional explanation as to why the sympathy is warranted, it is a violation of commonsense knowledge.

Decision: [YES]

----------
DIALOGUE
----------
1. Speaker 2: I can understand that . it sounds like you have a lot going on in your life right now .
2. Speaker 1: No, not really. Why do you say that?
----------

If this were the next response in the dialogue, would it contradict commonsense:

"Speaker 2: i just like to think of all the things i ' ve been doing for the better part of my life ."

Reasoning: Although Speaker 2's response does not make sense as an answer to Speaker 1's question, this cannot be explained by a contradiction of common knowledge. Rather, Speaker 2 seems to be presenting an off-topic comment.

Decision: [NO]

----------
DIALOGUE
----------
1. Speaker 1: Do you like soccer?
----------

If this were the next response in the dialogue, would it contradict commonsense:

"Speaker 2: I really like playing midfield, but I don't think I've ever tried playing soccer before."

Reasoning: Speaker 2's response is illogical and self-contradictory since you must play soccer in order to like playing a specific position in soccer.

Decision: [YES]       

----------
DIALOGUE
----------
1. Speaker 1: I have a boy and a girl, they're eight and twelve.
----------

If this were the next response in the dialogue, would it contradict commonsense:

"Speaker 2: Oh okay. Do you have any kids?"

Reasoning: Speaker 2's question is at odds with the information already presented in the conversation: everyone knows that if you have a boy and a girl, you have two kids.

Decision: [YES]""",

    "self contradiction": """----------
DIALOGUE
----------
1. Speaker 1: What do you do for a living?
2. Speaker 2: I'm a software engineer.
3. Speaker 1: What kinds of things do you work on?
----------

If this were the next response in the dialogue, is it a self-contradiction by Speaker 2:

"Speaker 2: I work as a university instructor, so I give lectures and grade assignments."

Reasoning: Speaker 2 presents two mutually exclusive answers for his occupation.

Decision: [YES]

----------
DIALOGUE
----------
1. Speaker 1: What are you going to have for dinner?
2. Speaker 2: Probably a burger.
3. Speaker 1: Sounds good.
----------

If this were the next response in the dialogue, is it a self-contradiction by Speaker 2:

"Speaker 2: Actually I just remembered tonight is pasta night."

Reasoning: Self-corrections do not count as contradictions.

Decision: [NO]""",

    "partner contradiction": """----------
DIALOGUE
----------
1. Speaker 1: I have one sister and one brother.
2. Speaker 2: Oh ok, do you get along?
3. Speaker 1: Yeah.
----------

Is Speaker 2 saying something about Speaker 1 that is contradicting what Speaker 1 has already shared:

"Speaker 2: What do you like to do with your brothers?"

Reasoning: Speaker 2's question implies Speaker 1 has multiple brothers, which contradicts information Speaker 1 has previously shared.

Decision: [YES]

----------
DIALOGUE
----------
1. Speaker 2:My favorite movie is Iron Man.
2. Speaker 1: Cool, I like the Marvel movies.
----------

Is Speaker 2 saying something about Speaker 1 that is contradicting what Speaker 1 has already shared:

"Speaker 2: Do you like the Marvel movies?"

Reasoning: Speaker 2 is asking a question that Speaker 1 has already provided an answer to, so it seems as if Speaker 2 has forgotten what their partner has said, which makes this a contradiction.

Decision: [YES]

----------
DIALOGUE
----------
1. Speaker 1: I have one sister and one brother.
2. Speaker 2: That's great.
3. Speaker 1: Yeah we get along well.
----------

Is Speaker 2 saying something about Speaker 1 that is contradicting what Speaker 1 has already shared:

"Speaker 2: What do you like to do with your brother and sister?"

Reasoning: Speaker 2 does not misremember Speaker 1's siblings.

Decision: [NO]""",

    "redundant": """----------
DIALOGUE
----------
1. Speaker 2: My favorite movie is Iron Man.
2. Speaker 1: Cool, I like the Marvel movies.
----------

Is this response repeating something that has already been said:

"Speaker 2: Yeah me too, I especially like Iron Man."

Reasoning: Speaker 2's response does not make any novel contribution to the dialogue, because it covers previously stated information.

Decision: [YES]      

----------
DIALOGUE
----------
1. Speaker 2: My favorite movie is Iron Man.
2. Speaker 1: Cool, I like the Marvel movies.
----------

Is this response repeating something that has already been said:

"Speaker 2: Do you like the Marvel movies?"

Reasoning: Speaker 2 is asking a question that Speaker 1 has already provided an answer to, so it is repetitive.

Decision: [YES]

----------
DIALOGUE
----------
1. Speaker 2: Let's talk about food.
2. Speaker 1: Well, pizza is the best food.
----------

Is this response repeating something that has already been said:

"Speaker 2: Pizza is the best food."

Reasoning: Although Speaker 2's response makes the same claim as Speaker 1's, the primary function of the response is to agree with Speaker 1. Since their agreement on the best food has not previously been established, the response does not constitute a repetitive turn.

Decision: [NO]

----------
DIALOGUE
----------
1. Speaker 2: Have you ever been to Europe?
2. Speaker 1: I like to watch movies in my free time.
----------

Is this response repeating something that has already been said:

"Speaker 2: But have you ever been to Europe?"

Reasoning: Since Speaker 1 did not answer Speaker 2's question, repeating the question is not repetitive.

Decision: [NO]""",

    "incorrect fact": """----------
DIALOGUE
----------
1. Speaker 1: Do you have a bucket list spot you want to climb?
----------

Does this response include an incorrect fact:

"Speaker 2: Mount Mitchell, since it's the tallest mountain."

Reasoning: Speaker 2's response states that Mount Mitchell is the tallest mountain, which is an incorret factual claim. Mount Everest is actually the tallest mountain in the world.

Decision: [YES]
    
----------
DIALOGUE
----------
1. Speaker 1: What have you been up to?
----------

Does this response include an incorrect fact:

"Speaker 2: I just got back from France."

Reasoning: Although no specific claims are made about France, Speaker 2 is mentioning France as if it is a public entity that Speaker 1 should know, which means we need to verify Speaker 2 is not misrepresenting this entity in any way. Since France is a location, it is reasonable for Speaker 2 to claim they just returned from it, so this is NOT an incorrect fact.

Decision: [NO]

----------
DIALOGUE
----------
1. Speaker 1: What did you think about the movie?
----------

Does this response include an incorrect fact:

"Speaker 2: Robert Downey Jr. is a great actor."

Reasoning: Although overall Speaker 2 is giving an opinion, their response mentions Robert Downey Jr. as a specific public figure, which means we need to verify Speaker 2 is not misrepresenting this entity in any way. Since Robert Downey Jr. is a real actor, this response is NOT using an incorrect fact.

Decision: [NO]

----------
DIALOGUE
----------
1. Speaker 1: What did you do over the summer? Take any trips?
----------

Does this response include an incorrect fact:

"Speaker 2: I went to Lankren for a few weeks, it was a great vacation."

Reasoning: Speaker 2 mentions a specific location - Lankren - as if it's a real place that Speaker 1 might know about. Even though Lankren is a made-up place, Speaker 2's presentation of Lankren as a specific, publicly relevant location constitutes an attempt at incorporating factual knowledge. Therefore, Speaker 2's presentation of this location is an incorrect usage of factual information.

Decision: [YES]

----------
DIALOGUE
----------
1. Speaker 1: Tell me about yourself.
----------

Does this response include an incorrect fact:

"Speaker 2: I like Dave's Killer Burger, which is a special recipe that my friend Dave makes at barbeques."

Reasoning: Speaker 2 mentions a specific entity - Dave's killer burger. However, because Speaker 2 continues to clarify that Dave's killer burger is her friend's recipe, she is not presenting Dave's killer burger as publicly relevant. Thus, no factual information is being used here, only information personal to the speaker.

Decision: [NO]

----------
DIALOGUE
----------
1. Speaker 1: I really like cats.
----------

Does this response include an incorrect fact:

"Speaker 2: Did you know, cats have a wider field of vision that humans."

Reasoning: Speaker 2 incorporates specific factual knowledge about cats into her response. Since she presents this information as objective, rather than her own opinion or belief, we need to verify the correctness of this fact. Since cats do indeed have a wider field of vision on average than humans, this is NOT an incorrect fact.

Decision: [NO]

----------
DIALOGUE
----------
1. Speaker 1: What's something that scares you?
----------

Does this response include an incorrect fact:

"Speaker 2: It might sound silly to some people, but I think ghosts are real and they really scare me."

Reasoning: Speaker 2 is presenting a personal belief that ghosts are real, but is not presenting this as an objective fact, so this is NOT an incorrect fact, even though there is no scientific proof of ghosts.

Decision: [NO]

----------
DIALOGUE
----------
1. Speaker 1: What's your favorite kind of food?
----------

Does this response include an incorrect fact:

"Speaker 2: I like Thai food."

Reasoning: Although Speaker 2 mentions Thai food in her response, she presents it as a broad category of food rather than as a specific mention of a publicly relevant entity. Broad categories such as this do not count as factual information. If Speaker 2 had mentioned something more specific, such as the country Thailand or the dish Pad Thai, she would have been incorporating factual information and this would have needed to then be checked for its correctness.

Decision: [NO]

----------
DIALOGUE
----------
1. Speaker 1: What country would you visit?
----------

Does this response include an incorrect fact:

"Speaker 2: I'd go to Texas."

Reasoning: Texas is a state in the USA but is not a country, so this is incorrect.

Decision: [YES]

----------
DIALOGUE
----------
1. Speaker 1: You look like you want to say something.
----------

Does this response include an incorrect fact:

"Speaker 2: Our universe is actually a computer simulation created by a more intelligent life form."

Reasoning: Speaker 2 makes a factual claim that might be possible, but this fact is not currently known to be true or false. Therefore, it is misleading for Speaker 2 to present the fact as if she knows it is true.

Decision: [YES]

----------
DIALOGUE
----------
1. Speaker 1: What did you do over break?
----------

Does this response include an incorrect fact:

"Speaker 2: I went to Kathmandu, the capital of Nepal last month."

Reasoning: It is true that Kathmandu is the capital of Nepal.

Decision: [NO]

----------
DIALOGUE
----------
1. Speaker 1: What did you do over break?
----------

Does this response include an incorrect fact:

"Speaker 2: I went to Kiritipur, the capital of Nepal last month."

Reasoning: It is not true that Kiritipur is the capital of Nepal; instead, the capital is Kathmandu.

Decision: [YES]""",

    "empathetic": """----------
DIALOGUE
----------
1. Speaker 1: I graduated college last week, so I've been celebrating.
----------

Is this an empathetic response by Speaker 2:

"Speaker 2: I'm sorry to hear that."

Reasoning: Speaker 2's response attempts to sympathize with Speaker 1, which is inappropriate because Speaker 1 is sharing good news.

Decision: [NO]

----------
DIALOGUE
----------
1. Speaker 1: I graduated college last week, so I've been celebrating.
----------

Is this an empathetic response by Speaker 2:

"Speaker 2: Congratulations!"

Reasoning: Speaker 2's response appropriately congratulates Speaker 1, demonstrating an understanding that Speaker 1 feels accomplished.

Decision: [YES]

----------
DIALOGUE
----------
1. Speaker 1: My dog passed away last week.
----------

Is this an empathetic response by Speaker 2:

"Speaker 2: Was your dog long-haired or short-haired?"

Reasoning: Speaker 2's response has an inappropriate lack of emotion, and does not sufficiently acknowledge or sympathize with Speaker 1's bad news.

Decision: [NO]

----------
DIALOGUE
----------
1. Speaker 1: I took a walk today in the park near my house.
----------

Is this an empathetic response by Speaker 2:

"Speaker 2: Oh my gosh."

Reasoning: Speaker 1 did not state or imply an emotion, so Speaker 2's emotional reaction is inappropriate.

Decision: [NO]

----------
DIALOGUE
----------
1. Speaker 1: I took a walk today in the park near my house.
----------

Is this an empathetic response by Speaker 2:

"Speaker 2: Okay."

Reasoning: Speaker 2's response does not convey an emotional reaction.

Decision: [NO]""",

    "lack of empathy": """----------
DIALOGUE
----------
1. Speaker 1: I graduated college last week, so I've been celebrating.
----------

If this were the next response in the dialogue, would Speaker 1 feel like their feelings are not being understood by Speaker 2:

"Speaker 2: I'm sorry to hear that."

Reasoning: Speaker 2's response attempts to sympathize with Speaker 1, which is inappropriate because Speaker 1 is sharing good news.

Decision: [YES]

----------
DIALOGUE
----------
1. Speaker 1: I graduated college last week, so I've been celebrating.
----------

If this were the next response in the dialogue, would Speaker 1 feel like their feelings are not being understood by Speaker 2:

"Speaker 2: Congratulations!"

Reasoning: Speaker 2's response appropriately congratulates Speaker 1, demonstrating an understanding that Speaker 1 feels accomplished.

Decision: [NO]

----------
DIALOGUE
----------
1. Speaker 1: My dog passed away last week.
----------

If this were the next response in the dialogue, would Speaker 1 feel like their feelings are not being understood by Speaker 2:

"Speaker 2: Was your dog long-haired or short-haired?"

Reasoning: Speaker 2's response has an inappropriate lack of emotion, and does not sufficiently acknowledge or sympathize with Speaker 1's bad news.

Decision: [YES]

----------
DIALOGUE
----------
1. Speaker 1: I took a walk today in the park near my house.
----------

If this were the next response in the dialogue, would Speaker 1 feel like their feelings are not being understood by Speaker 2:

"Speaker 2: Oh my gosh."

Reasoning: Speaker 1 did not state or imply an emotion, so Speaker 2's emotional reaction is inappropriate.

Decision: [YES]

----------
DIALOGUE
----------
1. Speaker 1: I took a walk today in the park near my house.
----------

If this were the next response in the dialogue, would Speaker 1 feel like their feelings are not being understood by Speaker 2:

"Speaker 2: Okay."

Reasoning: Speaker 2's response does not convey an emotional reaction, which is acceptable in this situation since Speaker 1 did not share anything too emotionally salient.

Decision: [NO]""",

    "ignore": """----------
DIALOGUE
----------
1. Speaker 2: I like Avengers. Probably because I first saw it as a kid, so there's some nostalgia.
2. Speaker 1: Right that makes sense.
----------

If this were the next response in the dialogue, does it completely ignore the immediate last turn from Speaker 1:

"Speaker 2: So, what other movies do you like?"

Reasoning: Since Speaker 1 simply acknowledged his understanding of Speaker 2's first turn, Speaker 1 did not introduce any questions, comments, or ideas that Speaker 2 is expected to respond to, so it is impossible for Speaker 2 to ignore Speaker 1 in this situation.

Decision: [NO]
    
----------
DIALOGUE
----------
1. Speaker 2: What's your favorite movie?
2. Speaker 1: I really like Star Wars.
----------

If this were the next response in the dialogue, does it completely ignore the immediate last turn from Speaker 1:

"Speaker 2: Have you ever seen Inception?"

Reasoning: Even though Speaker 2's response is relevant to the current discussion, Speaker 2 does not acknowledge what Speaker 1 said in any way.

Decision: [YES] 

----------
DIALOGUE
----------
1. Speaker 2: What's your favorite movie?
2. Speaker 1: I really like Star Wars.
----------

If this were the next response in the dialogue, does it completely ignore the immediate last turn from Speaker 1:

"Speaker 2: Oh okay. My favorite is probably Avengers."

Reasoning: Although generic and simple, Speaker 2's acknowledgement ("oh okay") signals that she understood what Speaker 1 said.

Decision: [NO] 

----------
DIALOGUE
----------
1. Speaker 2: What's your favorite movie?
2. Speaker 1: I really like Star Wars.
----------

If this were the next response in the dialogue, does it completely ignore the immediate last turn from Speaker 1:

"Speaker 2: What did you think of the recent ones?"

Reasoning: Although Speaker 2 does not acknowledge what Speaker 1 said directly, her response implicitly signals that she understood Speaker 1 by following-up on what Speaker 1 said.

Decision: [NO]""",

    "irrelevant": """----------
DIALOGUE
----------
1. Speaker 1: I am trying to find a good pizza place near my house.
----------

If this were the next response in the dialogue, is it completely irrelevant to what was just said:

"Speaker 2: Are you having a party?"

Reasoning: Speaker 2 follows up on what Speaker 1 said by asking for more detail and explanation.

Decision: [NO]

----------
DIALOGUE
----------
1. Speaker 1: I am trying to find a good pizza place near my house.
----------

If this were the next response in the dialogue, is it completely irrelevant to what was just said:

"Speaker 2: I hope you find a good place. I actually ate pizza earlier today."

Reasoning: Speaker 2 both reacts to what Speaker 1 said ("I hope you find a good place.") and introduces a new related talking point ("I actually ate pizza earlier today").

Decision: [NO]

----------
DIALOGUE
----------
1. Speaker 1: I am trying to find a good pizza place near my house.
----------

If this were the next response in the dialogue, is it completely irrelevant to what was just said:

"Speaker 2: I'm sure there's lots of good places if you live in Chicago."

Reasoning: Speaker 2 expands on the discussion point introduced by Speaker 1.

Decision: [NO]

----------
DIALOGUE
----------
1. Speaker 1: I am trying to find a good pizza place near my house.
----------

If this were the next response in the dialogue, is it completely irrelevant to what was just said:

"Speaker 2: Have you ever been fishing?"

Reasoning: There is little to no relation between Speaker 2's question about fishing and what Speaker 1 just said. So, Speaker 2's question abruptly and inappropriately changes the course of the conversation.

Decision: [YES]

----------
DIALOGUE
----------
1. Speaker 2: So that's why it's my favorite.
2. Speaker 1: Yeah that makes sense.
----------

If this were the next response in the dialogue, is it completely irrelevant to what was just said:

"Speaker 2: So anyway, do you have any plans for the weekend?"

Reasoning: Although Speaker 2's question about the weekend is unrelated to the previous discussion, the previous discussion had reached a natural conclusion. In this context, Speaker 2's response is a natural topic transition and is not abrupt or interruptive.

Decision: [NO]

----------
DIALOGUE
----------
1. Speaker 2: What's your favorite movie?
2. Speaker 1: I really like Star Wars.
----------

If this were the next response in the dialogue, is it completely irrelevant to what was just said:

"Speaker 2: My favorite is probably Avengers."

Reasoning: Even though Speaker 2's response is slightly inappropriate because it does not acknowledge what Speaker 1 just said, the response is relevant and does not interrupt the current discussion.

Decision: [NO]"""
}