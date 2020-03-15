import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from tqdm import tqdm



class TrainingManager(object):
    def __init__(self, config, dataset, model, loss_fn, scheduler, optimizer, regularizer=None):
        self.config = config
        self.dataset = dataset
        self.model = model
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.regularizer = regularizer
        self.device = torch.device(self.config.device)

        assert self.config.epochs > 0

        self.train_losses = []
        self.train_acc = []
        self.test_losses = []
        self.test_acc = []

        self.consecutive_desired_accuracy_remaining = self.config.consecutive_desired_accuracy

    def start(self):
        try:
            for epoch in range(self.config.epochs):
                print("EPOCH:", epoch)
                self.train()
                accuracy = self.test()
                self.scheduler.step()
                if self.config.break_on_reaching_desired_accuracy:
                    if accuracy >= self.config.desired_accuracy:
                        self.consecutive_desired_accuracy_remaining -= 1
                    else:
                        self.consecutive_desired_accuracy_remaining = self.config.consecutive_desired_accuracy

                    if self.consecutive_desired_accuracy_remaining == 0:
                        break   
        except Exceptin as e:
            print("Error: ", e)         


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
            loss = self.loss_fn(y_pred, target)

            if self.regularizer is not None:
                loss = self.regularize(self.model)

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
            for data, target in self.dataset.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(self.dataset.test_loader.dataset)
        self.test_losses.append(test_loss)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(self.dataset.test_loader.dataset),
            100. * correct / len(self.dataset.test_loader.dataset)))
        accuracy = 100. * correct / len(self.dataset.test_loader.dataset)
        self.test_acc.append(accuracy)
        return accuracy


    def summarize(self):
        fig, axs = plt.subplots(2,2,figsize=(15,10))
        axs[0, 0].plot(self.train_losses)
        axs[0, 0].set_title("Training Loss")
        axs[1, 0].plot(self.train_acc)
        axs[1, 0].set_title("Training Accuracy")
        axs[0, 1].plot(self.test_losses)
        axs[0, 1].set_title("Test Loss")
        axs[1, 1].plot(self.test_acc)
        axs[1, 1].set_title("Test Accuracy")