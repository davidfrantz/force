.. _vdev:

Develop version
===============

- **FORCE L2PS**

  - Added the option to buffer cirrus clouds.
    Cirrus clouds are quite hard to detect reliably (see CMIX paper).
    Buffering cirrus clouds often results in flagging out huge areas. 
    To avoid that, we somewhen took the decision to not go for a cirrus buffer.
    This remains the standard behaviour.
    
    However, it is now possible to enable a buffer by using a buffer size > 0 for the new parameter ``CIRRUS_BUFFER``.
    Note, that both the cloud and cirrus buffer are then represented by the same QAI bit, hence not allowing computational distinction in the origin of the buffer.
    
    Thanks to Max Freudenberg for bringing this up.

  

.. -- No further changes yet.
