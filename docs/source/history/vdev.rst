.. _vdev:

Develop version
===============

- **FORCE L2PS**

  - When merging chips of the same day/sensor to avoid data redundancy, the merging 
    strategy for the QAI files has been revised. Before, it was just updating the images,
    i.e., merely overlaying them on top of each other. Now, the QAI files are merged feature-based
    on a custom logic such that the most restrictive QAI value is kept. This ensures that reproducibility
    is guaranteed, independent of the order in which the chips are merged. 


