This is how I build projects using pytorch lighnting :)
It serves as a template to provide structure, organization, replacability and some scalability.

Specifically, I adapted this such that it fits my needs of testing multipe modelling approaches, each with multiple model implementations, which is most often all I need.

It is meant to be ran as a module ( python -m experiment.file ), using the files in the experiment/ directory as entry points.
Alternatively you can also package the whole thing which allows you to run it normally from the root dir with python experiment/file.py

