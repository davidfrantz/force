.. _vdev:

Develop version
===============

FORCE-dev
---------

* **FORCE L2PS**

  * Changed the behaviour of the cloud shadow flag.
    Before, the cloud shadow flag was not set if the cloud flag (any state) was set.
    Thus, when users wanted to use confident clouds only, i.e. not the buffered clouds, 
    there was a gap between the cloud and the shadow mask.
    This is mitigated now: cloud and cloud shadow flags can be set simultaneously. 
    This also enables "cloud shadow on top of cloud" scenarios.
    Thanks to Haili Hu for reporting this issue.

Master release: TBA

-- No changes yet, master is in sync with develop.
