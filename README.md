<h1>SEScoreX</h1>

<h3>Email: zihan_ma@ucsb.edu, wendaxu@cs.ucsb.edu</h3>

<h3>Install all dependencies:</h3>

````
pip install -r requirement/requirements.txt
````

<h3>Instructions to score sentences using SEScoreX:</h3>

SEScoreX pretrained weights can be found in google drive: https://drive.google.com/drive/u/2/folders/1TOUXEDZOsjoq_lg616iKUyWJaK9OXhNP


To run SEScoreX for text generation evaluation:

````
from sescorex import *

scorer = sescorex() # load in metric with specified language, en (English), de (German), ja ('Japanese'),  es ('Spanish'), zh ('Chinese'). We have SEScore2 that is only pretrained on synthetic data which only supports five languages (mode: pretrained) and further finetuned on all available human rating data (supports up to 100 languages, mode: seg or sys, by default we choose seg).
refs = ["SEScore is a simple but effective next generation text generation evaluation metric", "you went to hotel"]
outs = ["SEScore is a simple effective text evaluation metric for next generation", "you went to zoo"]
scores_ls = scorer.score(refs, outs, 1)
````


### Table: Model Performance Comparison

| Model   | cs-uk | en-cs | en-ja | en-zh | bn-hi | hi-bn | xh-zu* | zu-xh* | en-hr | en-uk | en-af* | en-am* | en-ha* |
|---------|-------|-------|-------|-------|-------|-------|--------|--------|-------|-------|--------|--------|--------|
| XCOMET  | 0.533 | 0.499 | 0.564 | 0.566 | 0.493 | 0.521 | **0.573** | 0.623  | 0.512 | 0.493 | **0.550** | 0.568  | 0.662  |
| COMET22 | **0.550** | **0.522** | **0.580** | **0.586** | 0.503 | **0.528** | 0.564  | 0.657  | **0.551** | **0.540** | 0.548  | 0.570  | **0.693** |
| Ours    | 0.540 | 0.514 | 0.565 | 0.575 | **0.504** | 0.521 | 0.572  | **0.658** | 0.537 | 0.524 | 0.535  | **0.570** | 0.663  |


| Model   | en-ig* | en-rw* | en-lg* | en-ny* | en-om* | en-sn* | en-ss* | en-sw* | en-tn* | en-xh* | en-yo* | en-zu* | en-gu |
|---------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|-------|
| XCOMET  | 0.502  | 0.446  | 0.579  | 0.494  | 0.653  | 0.702  | 0.548  | 0.650  | 0.479  | 0.633  | 0.541  | 0.551  | **0.694** |
| COMET22 | **0.539** | 0.456  | 0.582  | **0.535** | 0.672  | 0.807  | 0.580  | **0.679** | **0.605** | 0.692  | 0.575  | 0.589  | 0.596 |
| Ours    | 0.538  | **0.478** | **0.603** | 0.529  | **0.697** | **0.820** | **0.598** | 0.674  | 0.585  | **0.702** | **0.591** | **0.597** | 0.607 |

| Model   | en-hi | en-ml | en-mr | en-ta |
|---------|-------|-------|-------|-------|
| XCOMET  | **0.700** | **0.713** | **0.667** | **0.663** |
| COMET22 | 0.587  | 0.617  | 0.570  | 0.626  |
| Ours    | 0.580  | 0.606  | 0.528  | 0.604  |

**Note:** * indicates African languages.
