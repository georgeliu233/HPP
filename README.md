# Dual Predictions Integrated Planning for Autonomous Driving
[Haochen Liu](https://scholar.google.com/citations?user=iizqKUsAAAAJ&hl=en), [Zhiyu Huang](https://mczhi.github.io/),[Wenhui Huang](https://scholar.google.com/citations?user=Hpatee0AAAAJ&hl=en),[Haohan Yang](https://scholar.google.com/citations?user=KmKMahwAAAAJ&hl=en), [Xiaoyu Mo](https://scholar.google.com/citations?user=JUYVmAQAAAAJ&hl=zh-CN), [Chen Lv](https://scholar.google.com/citations?user=UKVs2CEAAAAJ&hl=en)

[AutoMan Research Lab, Nanyang Technological University](https://lvchen.wixsite.com/automan)

## Abstract 
Autonomous driving systems must have the ability to fully understand and predict their surrounding agents to make informed decisions in complex real-world scenarios. While recent advancements in learning-based systems have highlighted the integration between prediction and planning modules, they have also brought three major challenges: accurate predictions that align with perceptions, consistency between joint and agent-wise prediction patterns, and social coherence in prediction and planning. Inspired by this integrative nature, we introduce DPP, a dual-predictions integrated planning system that harnesses differentiable integration between prediction and planning. With context queries perceived from the bird's-eye view (BEV) space that are collectively encoded, DPP tackles the aforementioned challenges through three modules. First, we introduce occupancy prediction to align joint predictions with perceptions. Our proposed MS-OccFormer module aims to achieve multi-stage alignment per occupancy forecasting with consistent awareness from agent-wise motion predictions. Second, we propose a game-theoretic motion predictor termed GTFormer to model the interactive future among individual agents with their joint predictive awareness. Dual prediction patterns are concurrently integrated with Ego Planner and optimized with prediction guidance. DPP achieves state-of-the-art performance on the nuScenes dataset, demonstrating superior accuracy, safety, and consistency for end-to-end paradigms in prediction and planning. Moreover, we also report the long-term open-loop and closed-loop performance of DPP on the Waymo Open Motion Dataset and CARLA benchmark, demonstrating enhanced accuracy and compatibility over other integrated prediction and planning pipelines.

<div align=center><img src="./pics/pic-1.png" style="width:60%;"></div>

## Method Overview 
<div align=center><img src="./pics/fig_2.png" style="width:90%;"></div>

Systematic overview of the proposed Dual predictions integrated planning (***DPP***) framework. DPP is established upon query-based ADS co-design optimizations of interactive planning with dual predictions integration (IPP and IOP), informed by BEV perceptions. With encoded perception scene context, DPP delivers predictions planning co-design in three-fold. Joint occupancy prediction are iteratively refined in **MS-OccFormer**, sharing mutual consistency over marginal motion prediction in **GTFormer**, which performs interactive reasoning between marginal prediction and planning. Reasoned outcomes and ego features are served to query dual predictions-aware planning in **Ego Planner**. Eventually, optimizations are scheduled to refine planning with dual predictions guidance.

## Open-loop Planning (nuScenes)
The planner outputs a planned trajectory at each time step. All agents are replayed in the dataset. 

| <video muted controls width=380> <source src="./vids/1.mp4"  type="video/mp4"> </video> | <video muted controls width=380> <source src="./vids/2.mp4"  type="video/mp4"> </video> |

| <video muted controls width=380> <source src="./vids/3.mp4"  type="video/mp4"> </video> | <video muted controls width=380> <source src="./vids/5.mp4"  type="video/mp4"> </video> |

| <video muted controls width=380> <source src="./vids/6.mp4"  type="video/mp4"> </video> | <video muted controls width=380> <source src="./vids/7.mp4"  type="video/mp4"> </video> |

<div align=center><img src="./pics/nuscene_legend.png" style="width:80%;"></div>

## Closed-loop Planning (WOMD)
The planner outputs a planned trajectory at each time step, which is used to simulate the vehicle’s state at the next time step. The other agents are replayed from a log according to their observed states in the dataset.

| <video muted controls width=380> <source src="./vids/2ef8b857eb575693 (1).mp4"  type="video/mp4"> </video> | <video muted controls width=380> <source src="./vids/45c6cd9309b3ce87.mp4"  type="video/mp4"> </video> |

| <video muted controls width=380> <source src="./vids/5bcb4673b6c09a82.mp4"  type="video/mp4"> </video> | <video muted controls width=380> <source src="./vids/93a82dbe9425898c.mp4"  type="video/mp4"> </video> |

| <video muted controls width=380> <source src="./vids/a1bdf5c2af01557a.mp4"  type="video/mp4"> </video> | <video muted controls width=380> <source src="./vids/c45f2781605c47f0.mp4"  type="video/mp4"> </video> |

<div align=center><img src="./pics/womd_legend.png" style="width:80%;"></div>

## Acknowledgements

Official release for our previous work: 

-[DIPP](https://github.com/MCZhi/DIPP) 🔥 [![](https://img.shields.io/github/stars/MCZhi/DIPP?style=social&label=Code Stars)](https://github.com/MCZhi/DIPP)

-[GameFormer](https://github.com/MCZhi/GameFormer) 🚀 [![](https://img.shields.io/github/stars/MCZhi/GameFormer?style=social&label=Code Stars)](https://github.com/MCZhi/GameFormer)

-[STrajNet](https://github.com/georgeliu233/STrajNet) 🚀 [![](https://img.shields.io/github/stars/georgeliu233/STrajNet?style=social&label=Code Stars)](https://github.com/georgeliu233/STrajNet)

-[OPGP](https://github.com/georgeliu233/OPGP) 🚀 [![](https://img.shields.io/github/stars/georgeliu233/OPGP?style=social&label=Code Stars)](https://github.com/georgeliu233/OPGP)
