# Dataset Format

Each patient contains multimodal data including imaging features and clinical variables.

Example structure:

DATA_ROOT/

    patient001/
        lesion/
            slice_01.png
            slice_02.png
        space/
            slice_01.png
            slice_02.png

    patient002/
        lesion/
        space/

Additional files:

clinical.csv
radiomics.npy
pet.npy


## Clinical Variables

clinical.csv contains:

- patient_id
- age
- sex
- treatment type
- DFS time
- DFS event
- OS time
- OS event


## Radiomics Features

Stored in:

radiomics.npy

Each row corresponds to one patient.


## PET Features

Stored in:

pet.npy

Each row corresponds to one patient.
