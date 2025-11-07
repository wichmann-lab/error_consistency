## Analysing Brain-Score EC values

To reproduce this analysis, you first have to download the Brain-Score benchmark data from [the Brain-Score website](https://www.brain-score.org/vision/) (button "download CSV").

Then, we need to reverse the normalization by ceiling that Brain-Score applies: They transform ECs to scores in $[0, 1]$ by dividing a model's score for an experiment by the human-human EC on that experiment. 