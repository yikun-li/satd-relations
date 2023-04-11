# Replication Package for Automatically Identifying Relations Between Self-Admitted Technical Debt Across Different Sources

##### Authors: Yikun Li, Mohamed Soliman, and Paris Avgeriou


## Description of This Study

Technical debt refers to taking sub-optimal decisions that prioritize short-term benefits over long-term maintainability and evolvability in software development.
The type of technical debt, acknowledged by developers through documentation, is referred to as 
*Self-Admitted Technical Debt* or *SATD* can be found in various sources, such as source code comments, commit messages, issue tracking systems, and pull requests.
Previous research has established the existence of relations between SATD items in different sources; such relations can be useful for investigating and improving SATD management.
However, there is currently a lack of approaches for automatically detecting these SATD relations.
To address this, we proposed and evaluated approaches for automatically identifying SATD relations across different sources.
Our findings show that our approach outperforms baseline approaches by a large margin, achieving an average F1-score of 0.829 in identifying relations between SATD items. 
Moreover, we explored the characteristics of SATD relations in 103 open-source projects and describe nine major cases in which related SATD is documented in a second source, and give a quantitative overview of 26 kinds of relations.


## Structure of the Replication Package

The replication package includes both the dataset and the trained SATD relation detector.

```
├── LICENSE
├── README.md
├── SATD relation detector
│   └── satd_relation_detector.py
└── SATD_relation_dataset.csv
```


## Getting Started With SATD Relation Detector

### Requirements

- torch
- transformers


### Identifying Relation Between SATD Items

1. Download model weight at [LINK](https://zenodo.org/record/7819763)
2. Replace the file path with the real path and run the following command:

```bash
python satd_relation_detector.py --snapshot "{PATH}/satd_relation_detector.pt"
```


### Example Output

```
SATD Text A: Enable periodic rebalance as a temporary work-around for the Helix issue.
SATD Text B: TODO: Enable periodic rebalance per 10 seconds as a temporary work-around for the Helix issue.
-----
Predicted result: SATD-duplication


SATD Text A: TODO: Remove the legacy delimiter after releasing 0.6.0
SATD Text B: Same as before. We should try to enhance the same code.
-----
Predicted result: no-relation


SATD Text A: Servlet injection does not always work for servlet container. We use a hacking here to initialize static variables at Spring wiring time.
SATD Text B: Remove temporary hacking and use Official way to wire-up servlet with injection under Spring.
-----
Predicted result: SATD-repayment


SATD Text A: I added this TODO. Kept the old config names from BoundedBBPool for BC.
SATD Text B: // TODO better config names?
-----
Predicted result: SATD-duplication


SATD Text A: Since the state machine is not implemented yet, we should get the configured dummy message from Ratis
SATD Text B: Please take care of the checkStyle issues and ASF licensing issue while committing.
-----
Predicted result: no-relation


SATD Text A: // TODO: Include the generated file name in the response to the server
SATD Text B: // Add a TODO to include the generated file name in the response to server
-----
Predicted result: SATD-duplication
```


## Paper

Latest version available on [arXiv](https://arxiv.org/abs/2303.07079)

If this work helps your research and you publish a paper, we encourage you to cite the following paper in your publication:

```
@article{li2023automatically,
  title={Automatically Identifying Relations Between Self-Admitted Technical Debt Across Different Sources},
  author={Li, Yikun and Soliman, Mohamed and Avgeriou, Paris},
  journal={arXiv preprint arXiv:2303.07079},
  year={2023}
}
```

## Contact

- Please use the following email addresses if you have questions:
    - :email: <yikun.li@rug.nl>
