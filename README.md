# SAGEGly
`SAGEGly: Using multimodal information to predict proteinglycan binding sites with the Graph Sample and Aggregate Networks Framework`
## `SAGEGly Pipeline:`
<img width="1252" height="569" alt="image" src="https://github.com/user-attachments/assets/1d345ac7-01c9-44c9-b1d5-41968c3ce59a" />

## `Performance of the SAGEGly model:`
<img width="981" height="649" alt="image" src="https://github.com/user-attachments/assets/06f7e374-7306-4e79-bd6f-47d77c5ad119" />

## `Comparison with SOTA Methods:`
<img width="1227" height="675" alt="image" src="https://github.com/user-attachments/assets/1a144152-cb2a-4e7a-a265-00a62f65b7cd" />

## Dependencies
biopython==1.83  
fair-esm===2.0.0  
matplotlib==3.7.5  
numpy==1.24.1  
pandas==2.0.3  
python==3.8.19  
scikit-learn==1.3.2  
scipy==1.10.1  
torch-cluster==1.6.3+pt24cu118  
torch-geometric==2.6.1  
torch-scatter== 2.1.2+pt24cu118  
torch-sparse== 0.6.18+pt24cu118  
torch-spline-conv==1.2.2+pt24cu118  
torchaudio==2.4.1+cu118  
torchvision==0.19.1+cu118  
freesasa==2.0.3.post7

## 1 PPI Data Preparation

`Preprocessing the embeddings of the PPI prediction task data.`

The training and test embedding data used for model training can be generated using the following notebooks:

```bash
/PPI/train_embedding.ipynb
/PPI/test_embedding.ipynb
```

The coordinates, sequences, solvent-accessible surface area, and ESM parameters of GBPs and glycoproteins are available from Zenodo. These files are required to run the embedding scripts.

## 2 PPI Data Training

`Training the PPI prediction model.`

Run the following command:

```bash
python /PPI/train.py
```

The trained model parameters are saved in the following file:

```bash
/PPI/ppi_train.pt
```

## 3 Geometric Filtering

`Filtering candidate binding sites based on geometric features.`

Run the following command:

```bash
/Geometric_filtering/geometric_filtering.ipynb
```

## 4 Glycan-Binding Data Preparation

`Preprocessing the embeddings of the glycan-binding prediction task data.`

The training and test embedding data used for model training can be generated using the following notebooks:

```bash
/Glycan_binding/sugar_embedding_train.ipynb
/Glycan_binding/sugar_embedding_test.ipynb
```

The glycan information, coordinates, sequences, solvent-accessible surface area, and ESM parameters of GBPs and glycoproteins are available from Zenodo. These files are required to run the embedding scripts.

In addition, the processed embedding data are also available from Zenodo in the following directories:

```bash
/glycan binding/train_npz
/glycan binding/test_npz
```

## 5 Glycan-Binding Data Training

`Training the glycan-binding prediction model.`

Run the following command:

```bash
python /Glycan_binding/train_glycan.py
```

The trained model parameters are available in the following file:

```bash
/Glycan_binding/glycan_binding.pt
```

## 6 Visualization

`Visualization process.`

The visualization can be performed using the following notebook:

```bash
/Visualization/Visualization.ipynb
```

The model parameters used for visualization are available in the following file:

```bash
/Glycan_binding/glycan_binding.pt
```
## Reference and cite content
SAGEGly: Using multimodal information to predict protein-glycan binding sites with the Graph Sample and Aggregate Networks Framework.
