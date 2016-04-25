# Train a Smartcab How to Drive

Reinforcement Learning Project v1 2016/04/25

## Instructor notes
FYI - Slightly modified environment.py and planner.py to notify the agent the game has finished by propagating the 
message to agent.LearningAgent.on_finished.  
   
### Files: 

Report: Smartcab.pdf
Report Source: https://docs.google.com/document/d/1Xfzw4powfsGocTZ8ANuXM4ahrjCKpCuyfB9jLfRFW8g/edit?usp=sharing
Notepad for plots: smartcab_viz.ipynb

## Install

This project requires Python 2.7 with the pygame library installed:

https://www.pygame.org/wiki/GettingStarted

## Code

Open `smartcab/agent.py` and implement `LearningAgent`. Follow `TODO`s for further instructions.

## Run

Make sure you are in the top-level project directory `smartcab/` (that contains this README). Then run:

```python smartcab/agent.py```

OR:

```python -m smartcab.agent```

## References 

https://en.wikipedia.org/wiki/Q-learning
http://artint.info/html/ArtInt_265.html
