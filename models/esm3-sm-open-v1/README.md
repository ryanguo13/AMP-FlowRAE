---
language:
- en
tags:
- biology
- esm
- protein
extra_gated_heading: Agree to license and to share information
extra_gated_description: "The information you provide will be collected, stored, processed and shared in accordance with the [EvolutionaryScale Privacy Policy](https://www.evolutionaryscale.ai/legal/privacy)"
extra_gated_prompt: >-

  [Cambrian Non-Commercial License Agreement](https://www.evolutionaryscale.ai/policies/cambrian-non-commercial-license-agreement)

  **The Big Picture:**

  1. The weights of the EvolutionaryScale AI Models (ESM-3 and ESMC 600M) are available under this License Agreement for **non-commercial use** by users at **non-commercial organizations**. Note that other terms apply for API access.
  2. You **may not** use the EvolutionaryScale AI Models or any derivative works:
    a. for any **commercial activities** or monetary compensation
    b. to provide **outputs as a service**
    c. without proper attribution ("Built with ESM" and "ESM" in derivative work titles)
  3. You **can** reproduce and distribute copies with modifications for **non-commercial purposes** (e.g. fine-tuning weights or modifying code), provided you:
    a. include the license agreement
    b. mark modified files
    c. maintain all copyright and attribution notices

extra_gated_fields:
  Name: text  
  Country: country
  Role: text
  Affiliation (Lab/group/division and Institution): text
  Please describe your intended use of ESM3: text
  Is your background primarily computational or experimental: 
    type: select
    options:
      - Computational
      - Experimental
      - Mixed
  How did you learn about ESM3?: 
    type: select
    options:
      - Social media post
      - News article
      - Paper pre-print
      - Word of mouth
      - Other
  I accept the CLA: checkbox
  I agree to use this dataset for non-commercial use ONLY: checkbox
---

# Model Card for esm3-sm-open-v1

`esm3-sm-open-v1` is trained on 2.78 billion natural proteins. With synthetic data augmentation, this led to 3.15 billion protein sequences, 236 million protein structures, and 539 million proteins with function annotations, totaling 771 billion tokens.
`esm3-sm-open-v1` is a generative model capable of designing proteins conditioned on partial prompts of sequence, structure and function.

Safety is an important part of our model - data related to viruses has been removed from the training dataset, as well as some proteins belonging to organisms on the [USDA Select Agents and Toxins](https://www.selectagents.gov/sat/list.htm) list.
The function decoder has been filtered for potentially harmful keywords.

## Usage

Using `ESM3` requires [esm](https://github.com/evolutionaryscale/esm)

```
pip install esm
```

Please refer to the readme and notebooks in the [esm repository](https://github.com/evolutionaryscale/esm?tab=readme-ov-file#quickstart) for details on how to use the model.

## License

This repository is under a custom non-commercial [license](https://github.com/evolutionaryscale/esm/blob/main/LICENSE.md).
