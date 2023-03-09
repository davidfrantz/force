Inflate QAI bit layers
======================

Quality Assurance Information (QAI) is generated for Level 2 data (QAI product), and is included as layer in the Level 3 compositing information product (INF). The QAI layers are stored with bit-encoding (see Table 6), which makes it very useful to store much information with fairly low data volume. However, the QAI need to be parsed to extract all the useful information. The program force-qai-inflate can be used to inflate the QAI to individual masks. The 1st argument specifies the QAI dataset that should be inflated (QAI or INF products for Level 2 and Level 3, respectively). The 2nd argument is the directory where the masks should be stored. The 3rd argument gives the output format (ENVI or GTiff). The output is a multilayer image (bands = flags in Table 6) with product type QIM (Quality Inflated Masks). It is advised to not store these images in the same directory as the QAI image. It is also advised to not use this tool operationally, adjust your programs to use the QAI layer directly (saves disk space and time).

Usage
^^^^^

.. code-block:: bash
    
    force-qai-inflate [-h] [-v] [-i] input-file output-dir

  -h  = show this help
  -v  = show version
  -i  = show program's purpose

  Positional arguments:
  - 'input-file': QAI file
  - 'output-dir': Output directory for QIM files.'
  