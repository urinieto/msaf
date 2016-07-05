.. _config:

Configuration
=============

The ``config`` module contains many ``attributes`` that modify MSAF's behavior.
Many of these attributes are consulted during the import of the ``msaf`` module and many are assumed to be
read-only.

*As a rule, the attributes in this module should not be modified by user code.*

MSAF's code comes with default values for these attributes, but you can
override them from your ``.msafrc`` file, and override those values in turn by
the :envvar:`MSAF_FLAGS` environment variable.

The order of precedence is:

1. an assignment to ``msaf.config.<property>``
2. an assignment in :envvar:`MSAF_FLAGS`
3. an assignment in the .msafrc file (or the file indicated in :envvar:`MSAFRC`)

You can print out the current/effective configuration at any time by printing
``msaf.config``.
For example, to see a list  of all active configuration variables, type this from the command-line::

	python -c 'import msaf; print(msaf.config)' | less
