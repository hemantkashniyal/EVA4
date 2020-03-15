import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from tqdm import tqdm



class TrainingManager(object):
    def __init__(self, config, dataset, model, scheduler, optimizer):
        self.config = config
        self.dataset = dataset
        self.model = model
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.device = torch.device(self.config.device)

        assert self.config.epochs > 0

        self.train_losses = []
        self.train_acc = []
        selftest_losses = []
        self.test_acc = []

        self.consecutive_desired_accuracy_remaining = self.config.consecutive_desired_accuracy

    def start(self):
        for epoch in range(self.config.epochs):
            print("EPOCH:", epoch)
            self.train()
            accuracy = self.test()
            scheduler.step()
            if self.config.break_on_reaching_desired_accuracy:
                if accuracy >= self.config.desired_accuracy:
                    self.consecutive_desired_accuracy_remaining -= 1
                else:
                    self.consecutive_desired_accuracy_remaining = self.config.consecutive_desired_accuracy

                if self.consecutive_desired_accuracy_remaining == 0:
                    break            


    def train(self):
        self.model.train()
        pbar = tqdm(self.dataset.train_loader)
        correct = 0
        processed = 0
        for batch_idx, (data, target) in enumerate(pbar):
            # get samples
            data, target = data.to(self.device), target.to(self.device)

            # Init
            self.optimizer.zero_grad()
            # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes. 
            # Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly.

            # Predict
            y_pred = self.model(data)

            # Calculate loss
            loss = F.nll_loss(y_pred, target)
            self.train_losses.append(loss)

            # Backpropagation
            loss.backward()
            self.optimizer.step()

            # Update pbar-tqdm
            
            pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            processed += len(data)

            pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
            accuracy = 100*correct/processed
            self.train_acc.append(accuracy)

    def test(self):
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        self.test_losses.append(test_loss)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
        accuracy = 100. * correct / len(test_loader.dataset)
        self.test_acc.append(accuracy)
        return accuracy