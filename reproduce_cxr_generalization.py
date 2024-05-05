import torchxrayvision as xrv
import skimage, torch, torchvision
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score
import pickle
import os
from tqdm import tqdm


grid = [
        # ('mimic', 'mimic'),
        #  ('mimic', 'chexpert'),
        #  ('mimic', 'padchest'),
        #  ('mimic', 'nih'),
         # ('chexpert', 'mimic'),
         # ('chexpert', 'chexpert'),
         # ('chexpert', 'padchest'),
         # ('chexpert', 'nih'),
         # ('padchest', 'mimic'),
         # ('padchest', 'chexpert'),
         # ('padchest', 'padchest'),
         # ('padchest', 'nih'),
         # ('nih', 'mimic'),
         # ('nih', 'chexpert'),
         # ('nih', 'padchest'),
         # ('nih', 'nih'),
         ('all', 'mimic'),
         ('all', 'chexpert'),
         # ('all', 'padchest'),
         # ('all', 'nih')
]

# For reproducibility 
seed = 0 
pathologies = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", 
               "Lesion", "Pneumonia", "Pneumothorax", "No Finding"]

transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(),
                                            xrv.datasets.XRayResizer(224)])
gss = GroupShuffleSplit(train_size=0.8,test_size=0.2, random_state=seed)
gss_val = GroupShuffleSplit(train_size=0.875,test_size=0.125, random_state=seed)

# Function that convert the model output to standard 8 labels 
output_mapping = {
    "mimic": [0, 10, 1, 4, 14, 8, 3], 
    "chexpert": [0, 10, 1, 4, 14, 8, 3], 
    "padchest": [0, 10, 1, 4, 8, 3], 
    "nih": [0, 10, 1, 4, 8, 3], 
    "all": [0, 10, 1, 4, 14, 8, 3],
}

output_dir = "table2_reproduce/"
if not os.path.exists(output_dir): 
    os.makedirs(output_dir)

def convert_output(model_name, outputs): 
    new_outputs = outputs[:, output_mapping[model_name]]

    if model_name == "padchest" or model_name == "nih": 
        lesion = torch.max(outputs[:, 11], outputs[:, 12])
        new_outputs = torch.hstack((new_outputs[:, :4], lesion.unsqueeze(1), new_outputs[:, 4:]))

    # Append "No Findings" output as last column 
    no_finding_output = outputs[:, [i for i, x in enumerate(model.pathologies) if x != '']]
    # Max probability over all class with positive finding, then take the inverse 
    no_finding = 1. - no_finding_output.max(axis=1)[0]
    new_outputs = torch.hstack((new_outputs, no_finding.unsqueeze(1)))
    return new_outputs

# Adapted from torchxrayvision train_utils
def valid_test_epoch(name, epoch, model, model_name, device, data_loader, criterion, limit=None):
    model.to(device)
    model.eval()

    avg_loss = []
    task_outputs={}
    task_targets={}
    for task in range(data_loader.dataset[0]["lab"].shape[0]):
        task_outputs[task] = []
        task_targets[task] = []
        
    with torch.no_grad():
        t = tqdm(data_loader)
        for batch_idx, samples in enumerate(t):

            if limit and (batch_idx > limit):
                print("breaking out")
                break
            
            images = samples["img"].to(device)
            targets = samples["lab"].to(device)

            outputs = model(images)
            outputs = convert_output(model_name, outputs)
            
            loss = torch.zeros(1).to(device).double()
            for task in range(targets.shape[1]):
                task_output = outputs[:,task]
                task_target = targets[:,task]
                mask = ~torch.isnan(task_target)
                task_output = task_output[mask]
                task_target = task_target[mask]
                if len(task_target) > 0:
                    loss += criterion(task_output.double(), task_target.double())
                
                task_outputs[task].append(task_output.detach().cpu().numpy())
                task_targets[task].append(task_target.detach().cpu().numpy())

            loss = loss.sum()
            
            avg_loss.append(loss.detach().cpu().numpy())
            t.set_description(f'Epoch {epoch + 1} - {name} - Loss = {np.mean(avg_loss):4.4f}')
            
        for task in range(len(task_targets)):
            task_outputs[task] = np.concatenate(task_outputs[task])
            task_targets[task] = np.concatenate(task_targets[task])
    
        task_aucs = []
        for task in range(len(task_targets)):
            if len(np.unique(task_targets[task]))> 1:
                task_auc = roc_auc_score(task_targets[task], task_outputs[task])
                task_aucs.append(task_auc)
            else:
                task_aucs.append(np.nan)

    task_aucs = np.asarray(task_aucs)
    auc = np.mean(task_aucs[~np.isnan(task_aucs)])
    print(f'Epoch {epoch + 1} - {name} - Avg AUC = {auc:4.4f}')

    return auc, task_aucs, task_outputs, task_targets

for g in grid: 
    model_name = g[0]
    dataset_name = g[1]
    print(f"Start model {model_name} with test data {dataset_name}")
    
    if dataset_name == "mimic": 
        dataset = xrv.datasets.MIMIC_Dataset(imgpath="/data/healthy-ml/gobi1/data/MIMIC-CXR-JPG/files", 
                                    csvpath="/data/healthy-ml/gobi1/data/MIMIC-CXR-JPG/mimic-cxr-2.0.0-chexpert.csv.gz", 
                                    metacsvpath="/data/healthy-ml/gobi1/data/MIMIC-CXR-JPG/mimic-cxr-2.0.0-metadata.csv.gz", 
                                    transform=transform, views=["AP", "PA"], unique_patients=False)
        # Standardization for MIMIC-CXR 
        dataset.pathologies = ["Lesion" if x == "Lung Lesion" else x for x in dataset.pathologies]
        new_labels = pd.DataFrame(dataset.labels, columns = dataset.pathologies) 
        no_findings_list = list(new_labels.columns)
        no_findings_list.remove("Support Devices") 
        # If all negative findings for selected labels, set as no finding  
        new_labels['No Finding'] = new_labels[no_findings_list].eq(0).all(axis=1).astype(float)
        dataset.pathologies = new_labels.columns.values
        dataset.labels = new_labels.values
    elif dataset_name == "chexpert": 
        dataset = xrv.datasets.CheX_Dataset(imgpath="/data/healthy-ml/gobi1/data/CheXpert-v1.0-small",
                                   csvpath="/data/healthy-ml/gobi1/data/CheXpert-v1.0-small/train.csv",
                                   transform=transform, views=["PA", "AP"], unique_patients=False)
        # Standardization for CheXpert 
        dataset.pathologies = ["Lesion" if x == "Lung Lesion" else x for x in dataset.pathologies]
        new_labels = pd.DataFrame(dataset.labels, columns = dataset.pathologies) 
        no_findings_list = list(new_labels.columns)
        no_findings_list.remove("Support Devices") 
        new_labels['No Finding'] = new_labels[no_findings_list].eq(0).all(axis=1).astype(float)
        dataset.pathologies = new_labels.columns.values
        dataset.labels = new_labels.values
    elif dataset_name == "padchest": 
        dataset = xrv.datasets.PC_Dataset(imgpath="/data/healthy-ml/gobi1/data/PadChest/images-224", 
                               transform=transform, views=["PA", "AP"], unique_patients=False)
        # Standardization for PadChest 
        new_labels = pd.DataFrame(dataset.labels, columns = dataset.pathologies) 
        # Combine "Mass" and "Nodule" as "Lesion" class
        new_labels['Lesion'] = new_labels['Mass'] + new_labels['Nodule']
        new_labels['Lesion'][new_labels['Lesion'] > 1] = 1
        no_findings_list = list(new_labels.columns)
        no_findings_list.remove("Support Devices") 
        no_findings_list.remove("Tube")
        new_labels['No Finding'] = new_labels[no_findings_list].eq(0).all(axis=1).astype(float)
        dataset.pathologies = new_labels.columns.values
        dataset.labels = new_labels.values
    elif dataset_name == "nih":
        dataset = xrv.datasets.NIH_Dataset(imgpath="/data/healthy-ml/gobi1/data/ChestXray8/images",
                                 transform=transform, views=["PA","AP"], unique_patients=False)
        # Standardization for NIH 
        new_labels = pd.DataFrame(dataset.labels, columns = dataset.pathologies)
        # Combine "Mass" and "Nodule" as "Lesion" class
        new_labels['Lesion'] = new_labels['Mass'] + new_labels['Nodule']
        new_labels['Lesion'][new_labels['Lesion'] > 1] = 1
        # If all negative findings for original labels, set as no finding  
        new_labels['No Finding'] = new_labels.eq(0).all(axis=1).astype(float)
        dataset.pathologies = new_labels.columns.values
        dataset.labels = new_labels.values
        
    xrv.datasets.relabel_dataset(pathologies, dataset)
    train_inds, test_inds = next(gss.split(X=range(len(dataset)), groups=dataset.csv.patientid))
    train_inds, val_inds = next(gss_val.split(X=range(len(train_inds)), groups=train_inds))
    split_datasets = {}
    split_datasets["train"] = xrv.datasets.SubsetDataset(dataset, train_inds)
    split_datasets["val"] = xrv.datasets.SubsetDataset(dataset, val_inds)
    split_datasets["test"] = xrv.datasets.SubsetDataset(dataset, test_inds)

    if model_name == "mimic": 
        model = xrv.models.DenseNet(weights="densenet121-res224-mimic_ch")
    elif model_name == "chexpert": 
        model = xrv.models.DenseNet(weights="densenet121-res224-chex")
    elif model_name == "padchest": 
        model = xrv.models.DenseNet(weights="densenet121-res224-pc")
    elif model_name == "nih": 
        model = xrv.models.DenseNet(weights="densenet121-res224-nih")
    elif model_name == "all": 
        model = xrv.models.DenseNet(weights="densenet121-res224-all")

    data_loader = torch.utils.data.DataLoader(split_datasets["test"], batch_size=8, shuffle=False)
    auc, task_aucs, task_outputs, task_targets = valid_test_epoch("test", 0, model, model_name, 
                                                              "cuda", data_loader, torch.nn.BCEWithLogitsLoss(),)
    print("Total AUC: {}".format(auc))
    print(task_aucs)
    
    with open(f"{output_dir}{model_name}_{dataset_name}_auc.pkl", "wb") as f: 
        pickle.dump(auc, f) 
        
    with open(f"{output_dir}{model_name}_{dataset_name}_task_aucs.pkl", "wb") as f: 
        pickle.dump(task_aucs, f) 
    
    with open(f"{output_dir}{model_name}_{dataset_name}_task_outputs.pkl", "wb") as f: 
        pickle.dump(task_outputs, f) 
    
    with open(f"{output_dir}{model_name}_{dataset_name}_task_targets.pkl", "wb") as f: 
        pickle.dump(task_targets, f) 

    print(f"Finished for model {model_name} with test data {dataset_name}")
    