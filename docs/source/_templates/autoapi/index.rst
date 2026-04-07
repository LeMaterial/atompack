.. Copyright 2026 Entalpic

API Reference
=============

This page contains leads to the documentation of the ``atompack`` package.

The API reference was auto-generated with ``autoapi`` [#f1]_.

You can see the source code on `Github <https://github.com/Entalpic/atompack>`_ and explore the rendered documentation here ⬇️

.. toctree::
   :titlesonly:

   {% for page in pages|selectattr("is_top_level_object") %}
   {{ page.include_path }}
   {% endfor %}

.. [#f1] Created with `sphinx-autoapi <https://github.com/readthedocs/sphinx-autoapi>`_
