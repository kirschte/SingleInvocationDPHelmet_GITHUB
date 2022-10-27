#!/usr/bin/env python
# coding: utf-8

# original template by Facebook: https://github.com/pytorch/opacus/blob/v0.15.0/examples/mnist.py, Apache-2.0 license
# modified by Authors of "Single SMPC Invocation DPHelmet: Differentially Private Distributed Learning on a Large Scale"

import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from tqdm import tqdm

from easydict import EasyDict as edict
from opacus import PrivacyEngine
from opacus.utils.uniform_sampler import UniformWithReplacementSampler

# os.environ["CUDA_VISIBLE_DEVICES"] = ""  # You can activate a specific GPU here

print(torch.__version__)


class LinearMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(6144, 10)

    def forward(self, x):
        # x of shape [B, 6144]
        x = self.fc(x)  # -> [B, 10]
        return x

    def name(self):
        return "LinearMLP"


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    criterion = nn.CrossEntropyLoss()
    losses = []
    correct = 0
    for _batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        pred = output.argmax(
            dim=1, keepdim=True
        )  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()

    if not args.disable_dp:
        epsilon, best_alpha = optimizer.privacy_engine.get_privacy_spent(args.delta)
        print(
            f"Train Epoch: {epoch} \t"
            f"Loss: {np.mean(losses):.6f} \t"
            f"Accuracy: {100.0 * correct / len(train_loader.dataset):.2f}% "
            f"(ε = {epsilon:.2f}, δ = {args.delta}) for α = {best_alpha}"
        )
    else:
        print(f"Train Epoch: {epoch} \t Loss: {np.mean(losses):.6f}")


def test(args, model, device, test_loader):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in tqdm(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )
    return correct / len(test_loader.dataset)


code_space = np.load("code_space.npy")
labels = np.load("labels.npy")

dataset = TensorDataset(
    torch.Tensor(code_space), torch.tensor(labels, dtype=torch.long)
)

args = edict(
    {
        "sample_rate": 1024 / 20000,  # sample rate used for batch construction
        "test_batch_size": 1024,  # input batch size for testing
        "epochs": 40,  # number of epochs to train
        "n_runs": 5,  # number of runs to average on
        "lr": 4,  # learning rate
        "sigma": 16,  # Noise multiplier --> change for different eps
        "max_per_sample_grad_norm": 0.1,  # Clip per-sample gradients to this norm
        "delta": 1e-5,  # Target delta
        "device": "cuda",  # GPU ID for this process
        "save_model": False,  # Save the trained model
        "disable_dp": False,  # Disable privacy training and just train with vanilla SGD
        "secure_rng": False,  # Enable Secure RNG to have trustworthy privacy guarantees. Comes at a performance cost
        "nb_splits": 6,  # Number of CV splits
    }
)
validator = StratifiedKFold(n_splits=args.nb_splits, shuffle=True)

device = torch.device(args.device)

kwargs = {"num_workers": 1, "pin_memory": True}

if args.secure_rng:
    try:
        import torchcsprng as prng
    except ImportError as e:
        msg = (
            "To use secure RNG, you must install the torchcsprng package! "
            "Check out the instructions here: https://github.com/pytorch/csprng#installation"
        )
        raise ImportError(msg) from e

    generator = prng.create_random_device_generator("/dev/urandom")

else:
    generator = None

run_results = []
for _ in range(args.n_runs):
    for train_ids, test_ids in validator.split(*dataset.tensors):
        # dataset prep
        train_dataset = torch.utils.data.Subset(dataset, train_ids)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            generator=generator,
            batch_sampler=UniformWithReplacementSampler(
                num_samples=len(train_dataset),
                sample_rate=args.sample_rate,
                generator=generator,
            ),
            **kwargs,
        )
        test_dataset = torch.utils.data.Subset(dataset, test_ids)
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.test_batch_size,
            shuffle=True,
            **kwargs,
        )

        # training
        model = LinearMLP().to(device)

        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0)
        if not args.disable_dp:
            privacy_engine = PrivacyEngine(
                model,
                sample_rate=args.sample_rate,
                sample_size=len(train_dataset),
                alphas=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 96)),
                noise_multiplier=args.sigma,
                max_grad_norm=args.max_per_sample_grad_norm,
                secure_rng=args.secure_rng,
                poisson=True,
            )
            privacy_engine.attach(optimizer)
        for epoch in range(1, args.epochs + 1):
            train(args, model, device, train_loader, optimizer, epoch)
        run_results.append(test(args, model, device, test_loader))

if len(run_results) > 1:
    print(
        "Accuracy averaged over {} runs: {:.2f}% ± {:.2f}%".format(
            len(run_results), np.mean(run_results) * 100, np.std(run_results) * 100
        )
    )

repro_str = (
    f"{model.name()}_{args.lr}_{args.sigma}_"
    f"{args.max_per_sample_grad_norm}_{args.sample_rate}_{args.epochs}"
)
np.save(f"run_results_{repro_str}.npy", run_results)

if args.save_model:
    torch.save(model.state_dict(), f"simclr_6144d_{repro_str}.pt")
