import os
import time
import threading

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import transforms, models

from datasets import load_dataset, DatasetDict

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from codecarbon import EmissionsTracker

import psutil
import os
import time
import random
import threading
import platform
import socket
import json
import warnings
import subprocess

import numpy as np
import pandas as pd
import psutil

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

from torchvision import transforms
from torchvision.models import resnet18

from datasets import load_dataset

from codecarbon import EmissionsTracker

from thop import profile

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

IMG_SIZE = 224

BATCH_SIZE = 32

EPOCHS = 5

LEARNING_RATE = 0.001

NUM_WORKERS = 2

DATASET_NAME = "hf-vision/chest-xray-pneumonia"

RESULTS_DIR = "results"

CODECARBON_DIR = os.path.join(
    RESULTS_DIR,
    "codecarbon"
)

MODELS_DIR = os.path.join(
    RESULTS_DIR,
    "models"
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)

os.makedirs(
    CODECARBON_DIR,
    exist_ok=True
)

os.makedirs(
    MODELS_DIR,
    exist_ok=True
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)

np.random.seed(SEED)

torch.manual_seed(SEED)

if torch.cuda.is_available():

    torch.cuda.manual_seed_all(SEED)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# PRINT HEADER
# ============================================================

print("=" * 80)

print("CHEST X-RAY PNEUMONIA")
print("AI MODEL ENERGY CONSUMPTION EXPERIMENT")

print("=" * 80)

print()

print("Device:", device)


# ============================================================
# HARDWARE INFORMATION
# ============================================================

def get_gpu_information():

    gpu_name = "None"

    gpu_memory_gb = 0

    cuda_version = "None"

    gpu_count = 0

    if torch.cuda.is_available():

        gpu_count = torch.cuda.device_count()

        gpu_name = torch.cuda.get_device_name(0)

        properties = (
            torch.cuda.get_device_properties(0)
        )

        gpu_memory_gb = (
            properties.total_memory
            / (1024 ** 3)
        )

        cuda_version = torch.version.cuda

    return (
        gpu_name,
        gpu_memory_gb,
        cuda_version,
        gpu_count
    )


def get_hardware_information():

    (
        gpu_name,
        gpu_memory_gb,
        cuda_version,
        gpu_count
    ) = get_gpu_information()

    memory = psutil.virtual_memory()

    info = {

        # Identification
        "computer_name":
            socket.gethostname(),

        # OS
        "operating_system":
            platform.system(),

        "os_version":
            platform.version(),

        "architecture":
            platform.machine(),

        # Python
        "python_version":
            platform.python_version(),

        # PyTorch
        "pytorch_version":
            torch.__version__,

        # CPU
        "cpu":
            platform.processor(),

        "cpu_physical_cores":
            psutil.cpu_count(
                logical=False
            ),

        "cpu_logical_cores":
            psutil.cpu_count(
                logical=True
            ),

        # RAM
        "ram_total_gb":
            round(
                memory.total
                / (1024 ** 3),
                2
            ),

        # GPU
        "gpu_available":
            torch.cuda.is_available(),

        "gpu_count":
            gpu_count,

        "gpu_name":
            gpu_name,

        "gpu_vram_gb":
            round(
                gpu_memory_gb,
                2
            ),

        "cuda_version":
            cuda_version,

        # Experiment
        "seed":
            SEED,

        "image_size":
            IMG_SIZE,

        "batch_size":
            BATCH_SIZE,

        "epochs":
            EPOCHS,

        "learning_rate":
            LEARNING_RATE,

        "num_workers":
            NUM_WORKERS,

        "dataset":
            DATASET_NAME
    }

    return info


hardware = get_hardware_information()


# ============================================================
# SAVE HARDWARE INFORMATION
# ============================================================

with open(
    os.path.join(
        RESULTS_DIR,
        "hardware_info.json"
    ),
    "w"
) as f:

    json.dump(
        hardware,
        f,
        indent=4
    )


print()
print("=" * 80)
print("HARDWARE")
print("=" * 80)

for key, value in hardware.items():

    print(
        f"{key}: {value}"
    )


# ============================================================
# LOAD DATASET
# ============================================================

print()
print("=" * 80)
print("LOADING DATASET")
print("=" * 80)

dataset = load_dataset(
    DATASET_NAME
)

print(dataset)


# ============================================================
# DATASET INFORMATION
# ============================================================

train_size = len(
    dataset["train"]
)

if "test" in dataset:

    test_split_name = "test"

elif "validation" in dataset:

    test_split_name = "validation"

else:

    raise RuntimeError(
        "The dataset does not contain "
        "a test or validation split."
    )


test_size = len(
    dataset[test_split_name]
)


print()
print("Training images:", train_size)

print(
    "Testing images:",
    test_size
)


# ============================================================
# TRANSFORMS
# ============================================================

train_transform = transforms.Compose([

    transforms.Grayscale(
        num_output_channels=3
    ),

    transforms.Resize(
        (IMG_SIZE, IMG_SIZE)
    ),

    transforms.RandomHorizontalFlip(
        p=0.5
    ),

    transforms.RandomRotation(
        5
    ),

    transforms.ToTensor(),

    transforms.Normalize(

        mean=[
            0.485,
            0.456,
            0.406
        ],

        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


test_transform = transforms.Compose([

    transforms.Grayscale(
        num_output_channels=3
    ),

    transforms.Resize(
        (IMG_SIZE, IMG_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(

        mean=[
            0.485,
            0.456,
            0.406
        ],

        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


# ============================================================
# PYTORCH DATASET
# ============================================================

class ChestXRayDataset(
    torch.utils.data.Dataset
):

    def __init__(
        self,
        hf_dataset,
        transform=None
    ):

        self.dataset = hf_dataset

        self.transform = transform

    def __len__(self):

        return len(
            self.dataset
        )

    def __getitem__(
        self,
        index
    ):

        item = self.dataset[index]

        image = item["image"]

        label = item["label"]

        if self.transform:

            image = self.transform(
                image
            )

        label = torch.tensor(
            label,
            dtype=torch.long
        )

        return image, label


# ============================================================
# DATASETS
# ============================================================

train_dataset = ChestXRayDataset(

    dataset["train"],

    transform=train_transform
)


test_dataset = ChestXRayDataset(

    dataset[test_split_name],

    transform=test_transform
)


# ============================================================
# DATALOADERS
# ============================================================

train_loader = DataLoader(

    train_dataset,

    batch_size=BATCH_SIZE,

    shuffle=True,

    num_workers=NUM_WORKERS,

    pin_memory=torch.cuda.is_available()
)


test_loader = DataLoader(

    test_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=NUM_WORKERS,

    pin_memory=torch.cuda.is_available()
)


# ============================================================
# LIGHT CNN
# ============================================================

class LightCNN(nn.Module):

    def __init__(
        self,
        num_classes=2
    ):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(
                3,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(32),

            nn.ReLU(),

            nn.MaxPool2d(2),


            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(64),

            nn.ReLU(),

            nn.MaxPool2d(2),


            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(128),

            nn.ReLU(),

            nn.MaxPool2d(2),


            nn.AdaptiveAvgPool2d(
                (1, 1)
            )
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Dropout(0.3),

            nn.Linear(
                128,
                num_classes
            )
        )

    def forward(self, x):

        x = self.features(x)

        x = self.classifier(x)

        return x


# ============================================================
# RESNET18
# ============================================================

def create_resnet18():

    model = resnet18(
        weights=None
    )

    model.fc = nn.Linear(
        model.fc.in_features,
        2
    )

    return model


# ============================================================
# PARAMETERS
# ============================================================

def count_parameters(model):

    total = sum(

        p.numel()

        for p in model.parameters()

    )

    trainable = sum(

        p.numel()

        for p in model.parameters()

        if p.requires_grad

    )

    return total, trainable


# ============================================================
# FLOPS
# ============================================================

def calculate_flops(model):

    model = model.to(device)

    dummy_input = torch.randn(

        1,
        3,
        IMG_SIZE,
        IMG_SIZE

    ).to(device)

    flops, params = profile(

        model,

        inputs=(
            dummy_input,
        ),

        verbose=False

    )

    return flops


# ============================================================
# RESOURCE MONITOR
# ============================================================

monitoring = False

monitoring_data = []


def monitor_resources(
    interval=1
):

    global monitoring

    while monitoring:

        timestamp = time.time()

        cpu_percent = (
            psutil.cpu_percent(
                interval=None
            )
        )

        ram_percent = (
            psutil.virtual_memory().percent
        )

        gpu_percent = np.nan

        gpu_memory_mb = np.nan

        gpu_temperature = np.nan

        gpu_power_w = np.nan


        if torch.cuda.is_available():

            try:

                import pynvml

                pynvml.nvmlInit()

                handle = (
                    pynvml
                    .nvmlDeviceGetHandleByIndex(0)
                )

                utilization = (
                    pynvml
                    .nvmlDeviceGetUtilizationRates(
                        handle
                    )
                )

                memory = (
                    pynvml
                    .nvmlDeviceGetMemoryInfo(
                        handle
                    )
                )

                gpu_percent = (
                    utilization.gpu
                )

                gpu_memory_mb = (

                    memory.used
                    / (1024 ** 2)

                )

                try:

                    gpu_temperature = (
                        pynvml
                        .nvmlDeviceGetTemperature(
                            handle,
                            pynvml.NVML_TEMPERATURE_GPU
                        )
                    )

                except Exception:

                    pass

                try:

                    gpu_power_w = (

                        pynvml
                        .nvmlDeviceGetPowerUsage(
                            handle
                        )
                        / 1000.0

                    )

                except Exception:

                    pass

            except Exception:

                pass


        monitoring_data.append({

            "timestamp":
                timestamp,

            "elapsed_time_seconds":
                timestamp,

            "cpu_percent":
                cpu_percent,

            "ram_percent":
                ram_percent,

            "gpu_percent":
                gpu_percent,

            "gpu_memory_mb":
                gpu_memory_mb,

            "gpu_temperature_c":
                gpu_temperature,

            "gpu_power_w":
                gpu_power_w

        })

        time.sleep(
            interval
        )


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(model):

    model.eval()

    all_predictions = []

    all_labels = []

    total_loss = 0

    total_samples = 0

    criterion = nn.CrossEntropyLoss()


    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)

            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            total_loss += (

                loss.item()
                * images.size(0)

            )

            total_samples += (
                images.size(0)
            )

            _, predictions = torch.max(

                outputs,
                1

            )

            all_predictions.extend(

                predictions
                .cpu()
                .numpy()

            )

            all_labels.extend(

                labels
                .cpu()
                .numpy()

            )


    accuracy = accuracy_score(

        all_labels,

        all_predictions

    )


    precision = precision_score(

        all_labels,

        all_predictions,

        zero_division=0

    )


    recall = recall_score(

        all_labels,

        all_predictions,

        zero_division=0

    )


    f1 = f1_score(

        all_labels,

        all_predictions,

        zero_division=0

    )


    average_loss = (

        total_loss
        / total_samples

    )


    cm = confusion_matrix(

        all_labels,

        all_predictions

    )


    return (

        average_loss,

        accuracy,

        precision,

        recall,

        f1,

        cm

    )


# ============================================================
# RESOURCE SUMMARY
# ============================================================

def summarize_resources(df):

    if len(df) == 0:

        return {

            "cpu_mean_percent":
                np.nan,

            "cpu_max_percent":
                np.nan,

            "ram_mean_percent":
                np.nan,

            "ram_max_percent":
                np.nan,

            "gpu_mean_percent":
                np.nan,

            "gpu_max_percent":
                np.nan,

            "gpu_memory_max_mb":
                np.nan,

            "gpu_temperature_max_c":
                np.nan,

            "gpu_power_mean_w":
                np.nan,

            "gpu_power_max_w":
                np.nan

        }


    return {

        "cpu_mean_percent":
            df["cpu_percent"].mean(),

        "cpu_max_percent":
            df["cpu_percent"].max(),

        "ram_mean_percent":
            df["ram_percent"].mean(),

        "ram_max_percent":
            df["ram_percent"].max(),

        "gpu_mean_percent":
            df["gpu_percent"].mean(),

        "gpu_max_percent":
            df["gpu_percent"].max(),

        "gpu_memory_max_mb":
            df["gpu_memory_mb"].max(),

        "gpu_temperature_max_c":
            df["gpu_temperature_c"].max(),

        "gpu_power_mean_w":
            df["gpu_power_w"].mean(),

        "gpu_power_max_w":
            df["gpu_power_w"].max()

    }


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model(
    model,
    model_name
):

    global monitoring

    global monitoring_data


    model = model.to(device)


    criterion = nn.CrossEntropyLoss()


    optimizer = optim.Adam(

        model.parameters(),

        lr=LEARNING_RATE

    )


    history = []


    monitoring_data = []


    print()
    print("=" * 80)
    print(
        "TRAINING:",
        model_name
    )
    print("=" * 80)


    # --------------------------------------------------------
    # Monitor
    # --------------------------------------------------------

    monitoring = True


    monitor_thread = threading.Thread(

        target=monitor_resources,

        daemon=True

    )


    monitor_thread.start()


    # --------------------------------------------------------
    # CodeCarbon
    # --------------------------------------------------------

    tracker = EmissionsTracker(

        project_name=model_name,

        output_dir=CODECARBON_DIR,

        measure_power_secs=1,

        log_level="error"

    )


    tracker.start()


    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    start_time = time.time()


    for epoch in range(EPOCHS):

        model.train()


        epoch_loss = 0

        epoch_correct = 0

        epoch_total = 0


        epoch_start = time.time()


        for images, labels in train_loader:

            images = images.to(device)

            labels = labels.to(device)


            optimizer.zero_grad()


            outputs = model(images)


            loss = criterion(

                outputs,

                labels

            )


            loss.backward()


            optimizer.step()


            epoch_loss += (

                loss.item()
                * images.size(0)

            )


            _, predicted = torch.max(

                outputs,

                1

            )


            epoch_total += (
                labels.size(0)
            )


            epoch_correct += (

                predicted == labels

            ).sum().item()


        loss_value = (

            epoch_loss
            / epoch_total

        )


        accuracy_value = (

            epoch_correct
            / epoch_total

        )


        epoch_time = (

            time.time()
            - epoch_start

        )


        history.append({

            "epoch":
                epoch + 1,

            "loss":
                loss_value,

            "accuracy":
                accuracy_value,

            "time_seconds":
                epoch_time

        })


        print(

            f"Epoch "
            f"{epoch + 1}/{EPOCHS} | "

            f"Loss: "
            f"{loss_value:.4f} | "

            f"Accuracy: "
            f"{accuracy_value:.4f} | "

            f"Time: "
            f"{epoch_time:.2f}s"

        )


    total_training_time = (

        time.time()
        - start_time

    )


    # --------------------------------------------------------
    # Stop CodeCarbon
    # --------------------------------------------------------

    emissions = tracker.stop()


    # --------------------------------------------------------
    # Stop monitor
    # --------------------------------------------------------

    monitoring = False


    monitor_thread.join(
        timeout=3
    )


    monitoring_df = pd.DataFrame(

        monitoring_data

    )


    # --------------------------------------------------------
    # Save history
    # --------------------------------------------------------

    history_df = pd.DataFrame(
        history
    )


    history_df.to_csv(

        os.path.join(

            RESULTS_DIR,

            f"{model_name}_history.csv"

        ),

        index=False

    )


    # --------------------------------------------------------
    # Save resource monitor
    # --------------------------------------------------------

    monitoring_df.to_csv(

        os.path.join(

            RESULTS_DIR,

            f"{model_name}_resources.csv"

        ),

        index=False

    )


    return (

        model,

        history_df,

        total_training_time,

        emissions,

        monitoring_df

    )


# ============================================================
# MODEL 1
# ============================================================

print()
print("=" * 80)
print("MODEL 1: LIGHT CNN")
print("=" * 80)


cnn = LightCNN()


cnn_parameters, cnn_trainable = (
    count_parameters(cnn)
)


cnn_flops = calculate_flops(
    cnn
)


print(
    "Parameters:",
    f"{cnn_parameters:,}"
)


print(
    "FLOPs:",
    f"{cnn_flops:,.0f}"
)


(
    cnn,
    cnn_history,
    cnn_training_time,
    cnn_emissions,
    cnn_monitor

) = train_model(

    cnn,

    "LightCNN"

)


(
    cnn_loss,
    cnn_accuracy,
    cnn_precision,
    cnn_recall,
    cnn_f1,
    cnn_cm

) = evaluate_model(
    cnn
)


cnn_resources = summarize_resources(
    cnn_monitor
)


# ============================================================
# SAVE CNN
# ============================================================

torch.save(

    cnn.state_dict(),

    os.path.join(

        MODELS_DIR,

        "lightcnn.pth"

    )

)


# ============================================================
# MODEL 2
# ============================================================

print()
print("=" * 80)
print("MODEL 2: RESNET-18")
print("=" * 80)


resnet = create_resnet18()


resnet_parameters, resnet_trainable = (
    count_parameters(resnet)
)


resnet_flops = calculate_flops(
    resnet
)


print(
    "Parameters:",
    f"{resnet_parameters:,}"
)


print(
    "FLOPs:",
    f"{resnet_flops:,.0f}"
)


(
    resnet,
    resnet_history,
    resnet_training_time,
    resnet_emissions,
    resnet_monitor

) = train_model(

    resnet,

    "ResNet18"

)


(
    resnet_loss,
    resnet_accuracy,
    resnet_precision,
    resnet_recall,
    resnet_f1,
    resnet_cm

) = evaluate_model(
    resnet
)


resnet_resources = summarize_resources(
    resnet_monitor
)


# ============================================================
# SAVE RESNET
# ============================================================

torch.save(

    resnet.state_dict(),

    os.path.join(

        MODELS_DIR,

        "resnet18.pth"

    )

)


# ============================================================
# GET CODECARBON FILES
# ============================================================

codecarbon_files = []

for root, dirs, files in os.walk(
    CODECARBON_DIR
):

    for filename in files:

        codecarbon_files.append(

            os.path.join(
                root,
                filename
            )

        )


# ============================================================
# FINAL RESULTS
# ============================================================

results = pd.DataFrame({

    "experiment_id": [

        f"{hardware['computer_name']}_LightCNN",

        f"{hardware['computer_name']}_ResNet18"

    ],

    "computer_name": [

        hardware["computer_name"],

        hardware["computer_name"]

    ],

    "operating_system": [

        hardware["operating_system"],

        hardware["operating_system"]

    ],

    "cpu": [

        hardware["cpu"],

        hardware["cpu"]

    ],

    "cpu_physical_cores": [

        hardware["cpu_physical_cores"],

        hardware["cpu_physical_cores"]

    ],

    "cpu_logical_cores": [

        hardware["cpu_logical_cores"],

        hardware["cpu_logical_cores"]

    ],

    "ram_total_gb": [

        hardware["ram_total_gb"],

        hardware["ram_total_gb"]

    ],

    "gpu": [

        hardware["gpu_name"],

        hardware["gpu_name"]

    ],

    "gpu_vram_gb": [

        hardware["gpu_vram_gb"],

        hardware["gpu_vram_gb"]

    ],

    "cuda_version": [

        hardware["cuda_version"],

        hardware["cuda_version"]

    ],

    "dataset": [

        DATASET_NAME,

        DATASET_NAME

    ],

    "train_images": [

        train_size,

        train_size

    ],

    "test_images": [

        test_size,

        test_size

    ],

    "image_size": [

        IMG_SIZE,

        IMG_SIZE

    ],

    "batch_size": [

        BATCH_SIZE,

        BATCH_SIZE

    ],

    "epochs": [

        EPOCHS,

        EPOCHS

    ],

    "learning_rate": [

        LEARNING_RATE,

        LEARNING_RATE

    ],

    "model": [

        "LightCNN",

        "ResNet18"

    ],

    "parameters": [

        cnn_parameters,

        resnet_parameters

    ],

    "trainable_parameters": [

        cnn_trainable,

        resnet_trainable

    ],

    "FLOPs": [

        cnn_flops,

        resnet_flops

    ],

    "training_time_seconds": [

        cnn_training_time,

        resnet_training_time

    ],

    "test_loss": [

        cnn_loss,

        resnet_loss

    ],

    "test_accuracy": [

        cnn_accuracy,

        resnet_accuracy

    ],

    "test_precision": [

        cnn_precision,

        resnet_precision

    ],

    "test_recall": [

        cnn_recall,

        resnet_recall

    ],

    "test_f1": [

        cnn_f1,

        resnet_f1

    ],

    "cpu_mean_percent": [

        cnn_resources[
            "cpu_mean_percent"
        ],

        resnet_resources[
            "cpu_mean_percent"
        ]

    ],

    "cpu_max_percent": [

        cnn_resources[
            "cpu_max_percent"
        ],

        resnet_resources[
            "cpu_max_percent"
        ]

    ],

    "ram_mean_percent": [

        cnn_resources[
            "ram_mean_percent"
        ],

        resnet_resources[
            "ram_mean_percent"
        ]

    ],

    "ram_max_percent": [

        cnn_resources[
            "ram_max_percent"
        ],

        resnet_resources[
            "ram_max_percent"
        ]

    ],

    "gpu_mean_percent": [

        cnn_resources[
            "gpu_mean_percent"
        ],

        resnet_resources[
            "gpu_mean_percent"
        ]

    ],

    "gpu_max_percent": [

        cnn_resources[
            "gpu_max_percent"
        ],

        resnet_resources[
            "gpu_max_percent"
        ]

    ],

    "gpu_memory_max_mb": [

        cnn_resources[
            "gpu_memory_max_mb"
        ],

        resnet_resources[
            "gpu_memory_max_mb"
        ]

    ],

    "gpu_temperature_max_c": [

        cnn_resources[
            "gpu_temperature_max_c"
        ],

        resnet_resources[
            "gpu_temperature_max_c"
        ]

    ],

    "gpu_power_mean_w": [

        cnn_resources[
            "gpu_power_mean_w"
        ],

        resnet_resources[
            "gpu_power_mean_w"
        ]

    ],

    "gpu_power_max_w": [

        cnn_resources[
            "gpu_power_max_w"
        ],

        resnet_resources[
            "gpu_power_max_w"
        ]

    ],

    "codecarbon_emissions_value": [

        cnn_emissions,

        resnet_emissions

    ]

})


# ============================================================
# SAVE FINAL CSV
# ============================================================

results_file = os.path.join(

    RESULTS_DIR,

    "experiment_summary.csv"

)


results.to_csv(

    results_file,

    index=False

)


# ============================================================
# SAVE CONFUSION MATRICES
# ============================================================

np.savetxt(

    os.path.join(

        RESULTS_DIR,

        "LightCNN_confusion_matrix.csv"

    ),

    cnn_cm,

    delimiter=",",

    fmt="%d"

)


np.savetxt(

    os.path.join(

        RESULTS_DIR,

        "ResNet18_confusion_matrix.csv"

    ),

    resnet_cm,

    delimiter=",",

    fmt="%d"

)


# ============================================================
# PRINT FINAL TABLE
# ============================================================

print()
print("=" * 80)
print("FINAL RESULTS")
print("=" * 80)

print()

display_columns = [

    "computer_name",

    "gpu",

    "model",

    "parameters",

    "FLOPs",

    "training_time_seconds",

    "test_accuracy",

    "test_precision",

    "test_recall",

    "test_f1",

    "cpu_mean_percent",

    "ram_mean_percent",

    "gpu_mean_percent",

    "gpu_memory_max_mb",

    "gpu_power_mean_w",

    "codecarbon_emissions_value"

]


print(

    results[
        display_columns
    ].to_string(
        index=False
    )

)


# ============================================================
# FINISH
# ============================================================

print()
print("=" * 80)

print(
    "EXPERIMENT FINISHED SUCCESSFULLY"
)

print("=" * 80)

print()

print(
    "Results:",
    results_file
)

print(

    "Hardware information:",

    os.path.join(
        RESULTS_DIR,
        "hardware_info.json"
    )

)

print()

print(
    "All files are stored in:"
)

print(
    os.path.abspath(
        RESULTS_DIR
    )
)

print()

print(
    "IMPORTANT: Keep the entire 'results' folder."
)

print(
    "It will be used later to compare computers."
)