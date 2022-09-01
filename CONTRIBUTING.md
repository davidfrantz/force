# Contributing to FORCE

FORCE is open source software and embraces open science principles.
Community contributions are very much welcomed!

This guide documents the best way to make various types of contributions to FORCE,
including what is required before submitting a code change.

Contributing to FORCE doesn't just mean writing code.
Helping other users, testing releases and bug fixes, and improving documentation are all essential and valuable contributions!


## Contributing by Helping Other Users

A great way to contribute to FORCE is to help answer user questions on the [discussion forum](https://github.com/davidfrantz/force/discussions)
or in the [issues section](https://github.com/davidfrantz/force/issues).
Taking a few minutes to help answer a question is a very valuable community service, especially to onboard new users.

If asking a question, please tell us who you are, and what your background is.
And super-important, please report back whether the answer was helpful, and make sure to mark useful responses as ``answered``.

We pledge to act and interact in ways that contribute to an open, welcoming, diverse, inclusive, and healthy community.
For more information, see our [code of conduct](https://github.com/davidfrantz/force/blob/main/CODE_OF_CONDUCT.md).

Contributors should ideally subscribe to these channels and follow them in order to keep up to date on what's happening in FORCE.
Answering questions is an excellent and visible way to help our community, and also demonstrates your expertise.


## Contributing Bug Reports

Filling a bug report is likely the simplest and most useful way to contribute to the project.
It helps us to identify issues and provide patches and therefore to make FORCE more stable and useful.

Report a bug using the "New issue" button in the [issues section](https://github.com/davidfrantz/force/issues) of this project.

Please help us answering your issue by providing as much information as possible to understand - or if possible to reproduce your problem.
Always indicate the version number, your OS, whether you are using Docker (name the image), or a local build (and if you deviated from the installation instructions). Your GDAL version may help, too.

Always add the full commandline that produced the error.
If applicable, upload the parameterfile, and other small files that may be related to the problem.

If your issue is rather related to usage or general questions, please use the [discussion forum](https://github.com/davidfrantz/force/discussions) instead.
While you are there, please take a few minutes to see if you can answer questions of fellow users.
This a very valuable community service!


## Contributing to the Documentation

To propose a change to the [documentation](https://force-eo.readthedocs.io/en/latest/),
edit the documentation files in this repository's [docs/](github.com/davidfrantz/force/tree/develop/docs) directory.
A ``reStructuredText`` preview plugin for your IDE of choice is recommended.

The documentation displays the pages in the ``develop`` branch. Please commit directly to this branch.
Then open a pull request with the proposed changes.


## Contributing Code Changes

In this software project, we aim at following the git branching model outlined [here](https://nvie.com/posts/a-successful-git-branching-model/).
The working branch is ``develop``.
We collect a meaningful number of features before periodically merging into the ``main`` branch, wherein each and every commit is tagged as a release.
Please do not directly commit to the ``main branch``.

Before committing changes, create a new feature branch based on the ``develop`` branch.
Make sure to test your changes before opening the pull request.
A code reviewer will get in touch.


### Contributing Bug Fixes

Contributing bug fixes is the best way to gain experience and credibility within the community and also to become an effective project contributor.

Especially, fixing bugs labeled as ``help wanted`` is highly appreciated!


### Contributing New Features

Before contributing a new feature, submit a new feature proposal in the [issues section](https://github.com/davidfrantz/force/issues) of the project and discuss it with the community and developers.

This is important to identify possible overlaps with other planned features and avoid misunderstandings and conflicts.


### Code styling

FORCE does not impose a strict code formatting style.
However, the following settings should be applied to harmonize the codebase as much as possible.

* Use spaces for indentation
* Tab size: 2
* declare all variables at the top of a function
* use comments
* use snake_case for variables. Do not use camelCase.
* take a couple of minutes to browse a bit through the existing code to learn about common variable names (like ny and nx for rows and columns, as well as i and j for indexing these).
* write a small usage for any function (see examples in code)
* use whitespaces and braces like this:

```
if (a > 0){
  
  for (i=0; i<n; i++) b[i] = my_function(1, c, args);

  for (j=0; j<m; j++){

    tmp = 5 + j;
    b[i][j] = my_function(1, c, args);

  }
  
}
```


## License, citations, etc

New files must include the appropriate license header boilerplate and the author name(s) and contact email(s).

Please pay special attention to citing research and properly acknowledging code that you may adapt for inclusion in FORCE.
See this [example](https://github.com/davidfrantz/force/blob/main/src/lower-level/equi7-ll.c).
If applicable, make use of the [citeme](https://github.com/davidfrantz/force/blob/main/src/cross-level/cite-cl.c) functionality, e.g. ``cite_me(_CITE_EQUI7_);``
