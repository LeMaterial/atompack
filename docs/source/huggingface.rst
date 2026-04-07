.. Copyright 2026 Entalpic

Hugging Face
============

Atompack can reopen datasets directly from remote Hugging Face repositories and can also publish
local datasets through a small API in ``atompack.hub``.

Installation
------------

Hub support ships with the base package:

.. code-block:: bash

   uv pip install "git+https://github.com/Entalpic/atompack.git@main#subdirectory=atompack-py"

Open Remote Datasets
--------------------

A common workflow is to read Atompack shards directly from a remote dataset repository. For
example, ``LeMaterial/Atompack`` exposes layouts such as ``omat/train`` and ``omol/train``:

.. code-block:: python

   import atompack

   db = atompack.hub.open(
       repo_id="LeMaterial/Atompack",
       path_in_repo="omat/train",
   )
   print(len(db))
   print(db[0].energy)
   db.close()

   db = atompack.hub.open(
       repo_id="LeMaterial/Atompack",
       path_in_repo="omol/train",
   )
   batch = db.get_molecules([0, 1, 2])
   db.close()

``AtompackReader`` presents one flat index space whether the source is a single ``.atp`` file or a
directory of shard files. Shard ordering is lexicographic by path, and the reader is read-only.
Call ``close()`` when you are done.

Download Remote Data First
--------------------------

Download only:

.. code-block:: python

   import atompack

   local_path = atompack.hub.download(
       repo_id="LeMaterial/Atompack",
       path_in_repo="omat/train",
   )

   print(local_path)

Open the downloaded data through the same reader API:

.. code-block:: python

   import atompack

   db = atompack.hub.open_path(local_path)
   print(db[0].energy)
   db.close()

Upload A Local Dataset
----------------------

Upload a single ``.atp`` file:

.. code-block:: python

   import atompack

   atompack.hub.upload(
       "data/train.atp",
       repo_id="your-org/atompack-demo",
       path_in_repo="exports",
   )

Upload a shard directory:

.. code-block:: python

   import atompack

   atompack.hub.upload(
       "exports/omat/train",
       repo_id="your-org/atompack-demo",
       path_in_repo="omat/train",
   )

Directory uploads include ``**/*.atp`` plus ``**/manifest.json`` when present.
