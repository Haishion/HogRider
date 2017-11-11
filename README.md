# HogRider -- Malmo Collaborative AI Challenge 2017

    Team member: 
        Xiong, Yanhai, Zhao, Mengchen and Chen, Haipeng 
        AMI Research Group, Nanyang Technological University

![Screenshot of HogRider](HogRider.png?raw=true "Screenshot of HogRider")


## How to run HogRider agent

1. Download the '.py' files in this folder and put them into the default ai_challenge/pig_chase folder. [Install the challenge platform follwing instructions on this page.]( https://github.com/Microsoft/malmo-challenge)
2. Use command "python pig_chase_eval_HogRider.py" to run HogRider V.S. ChallengeAgent. Note that it’s necessary to specify PigChaseSymbolicStateBuilder for HogRider in "pig_chase_eval_HogRider.py".
3. Results can be found in "pig_chase_HogRider.json"
4. If the program is aborted due to Minecraft error with TypeNone state transferred in, please retry step 2.

## Approach 

The approach mainly consists of two components: 
(1) agent type recognition based on a generalized Bayesian update, with a hyperbolic tangent function to "squash" the update, and 
(2) a novel Q-learning approach with 
    (a) a warm start based on rule-based human reasoning ([(Watch our vedio on YouTube for more details.)](https://youtu.be/Ho7GZa3Klcc))
    (b) a state-actoin abstraction, and 
    (c) active-epsilon-greedy exploration
    
For details of our algorithm, please refer to the supplementary .pdf file, which is accepted as proceedings in AAAI-18 conference.

   ![Overview of HogRider](HogRider_overview.png?raw=true "Overview of HogRider")
     



 

