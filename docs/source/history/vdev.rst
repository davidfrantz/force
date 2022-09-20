.. _vdev:

Develop version
===============

- **FORCE L2PS**

  - Added the option to buffer cirrus clouds.
    Cirrus clouds are quite hard to detect reliably (see CMIX paper).
    As such, buffering cirrus clouds often results in flagging out huge areas.
    To avoid that, we somewhen took the decision to not go for a cirrus buffer.
    This remains the standard behaviour.

    However, it is now possible to enable a buffer by using a buffer size > 0 for the new parameter ``CIRRUS_BUFFER``.
    Note, that both the cirrus buffer won't be represented in the "cloud buffer" QAI bit, but will be subsumed in the cirrus cloud bit (similar to snow buffer).
    This is because the bit structure would need to be disruptively changed to accomodate for a dsitinction.
    
    Thanks to Max Freudenberg for bringing this up.

  .. -- No further changes yet.
