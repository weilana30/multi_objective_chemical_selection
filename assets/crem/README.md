# CReM fragment database

`chembl33_sa2_f5.db` is the ChEMBL33 CReM fragment database filtered to source molecules with synthetic accessibility score <= 2, from the CReM database dataset on Zenodo (record DOI [10.5281/zenodo.16909329](https://doi.org/10.5281/zenodo.16909329)). The compressed source was `chembl33_sa2_f5.db.gz`; its publisher-listed MD5 was verified before extraction: `4a81b3d251819135b06852c2f9d63001`. The database includes legacy radius tables 1–5 and precomputed fragment properties.

The extracted database is kept outside Git because it is about 1.9 GB. Download `chembl33_sa2_f5.db.gz` from the linked Zenodo record and extract it here as `assets/crem/chembl33_sa2_f5.db` before running the CReM workflow.
