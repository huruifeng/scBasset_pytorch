import torch
import torch.nn as nn
import torch.optim as optim

class RegularizedLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, weight_decay=0.001):
        super(RegularizedLinear, self).__init__(in_features, out_features, bias)
        self.weight_decay = weight_decay

    def forward(self, input):
        output = super(RegularizedLinear, self).forward(input)
        # Add L2 regularization
        l2_reg = torch.norm(self.weight) ** 2
        output += 0.5 * self.weight_decay * l2_reg
        return output

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = RegularizedLinear(256, 128, weight_decay=0.001)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = MyModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Train the model
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Print loss for every 1000 iterations
        if (i+1) % 1000 == 0:
            print('Epoch [{}/{}], Iteration [{}/{}], Loss: {:.4f}'.format(
                epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.item()))
'''
Example1: In this example, we use the nn.CrossEntropyLoss criterion, which combines the softmax function and the negative log-likelihood loss. 
We then add an L2 regularization term to the loss function by calculating the L2 norm of the model's parameters and multiplying it by a regularization strength hyperparameter. 
Finally, we use the optim.SGD optimizer with a weight decay hyperparameter to perform stochastic gradient descent with L2 regularization.
'''


#############################################################################
import torch
import torch.nn as nn
import torch.optim as optim

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = MyModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD([
                {'params': model.fc1.parameters(), 'weight_decay': 0.001},
                {'params': model.fc2.parameters()},
                {'params': model.fc3.parameters()}], lr=0.01, momentum=0.9)

# Train the model
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        # Print loss for every 1000 iterations
        if (i+1) % 1000 == 0:
            print('Epoch [{}/{}], Iteration [{}/{}], Loss: {:.4f}'.format(
                epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.item()))

'''
Example2: In this example, we specify the weight decay hyperparameter for the first fully connected layer model.fc1 in the optimizer, but not for the other layers model.fc2 and model.fc3. 
We use the optim.SGD optimizer with the momentum hyperparameter to perform stochastic gradient descent with momentum, and only apply weight decay to the first fully connected layer. 
The other layers are updated normally.
'''

#####################################################################################
import torch
import torch.nn as nn
import torch.optim as optim

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = torch.relu(self.fc1(x))

        # Add L2 regularization to fc2 layer
        x = torch.relu(self.fc2(x))
        l2_reg = torch.norm(self.fc2.weight)
        x = x + 0.001 * l2_reg

        x = self.fc3(x)
        return x

model = MyModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Train the model
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Print loss for every 1000 iterations
        if (i+1) % 1000 == 0:
            print('Epoch [{}/{}], Iteration [{}/{}], Loss: {:.4f}'.format(
                epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.item()))
'''
Example3: In this example, we add an L2 regularization term to the fc2 layer by calculating the L2 norm of its weight tensor and multiplying it by a regularization strength hyperparameter. 
We then add this regularization term to the output of the fc2 layer before passing it to the fc3 layer. 
The other layers do not have any regularization added. 
We use the optim.SGD optimizer without any weight decay hyperparameter to perform stochastic gradient descent without regularization.
'''

################################################################################
import torch
import torch.nn as nn
import torch.optim as optim

class RegularizedLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, weight_decay=0.0):
        super(RegularizedLinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.weight_decay = weight_decay

    def forward(self, x):
        return self.linear(x)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}, weight_decay={}'.format(
            self.linear.in_features, self.linear.out_features, self.linear.bias is not None, self.weight_decay
        )

    def regularization_loss(self):
        return 0.5 * self.weight_decay * torch.norm(self.linear.weight)**2

model = nn.Sequential(
    RegularizedLinear(784, 256, weight_decay=0.001),
    nn.ReLU(),
    nn.Linear(256, 10)
)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Train the model
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Add L2 regularization to the first layer only
        if isinstance(model[0], RegularizedLinear):
            loss += model[0].regularization_loss()

        loss.backward()
        optimizer.step()

        # Print loss for every 1000 iterations
        if (i+1) % 1000 == 0:
            print('Epoch [{}/{}], Iteration [{}/{}], Loss: {:.4f}'.format(
                epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.item()))
'''
Example4: In this example, we define a custom RegularizedLinear module that wraps the nn.Linear layer and adds L2 regularization to its weights. 
We then use this module in the first layer of the model, and add the regularization loss to the total loss only if the first layer is a RegularizedLinear. 
Note that we define a regularization_loss method that calculates the L2 regularization term, and a extra_repr method that adds the weight_decay hyperparameter to the module's string representation.
'''

##########################################################################
import torch
import torch.nn as nn
import torch.optim as optim

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = MyModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Train the model
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Add L2 regularization to only the second layer
        l2_reg = torch.tensor(0., requires_grad=True)
        for name, param in model.named_parameters():
            if name == 'fc2.weight' or name == 'fc2.bias':
                l2_reg += torch.norm(param)
        loss += 0.001 * l2_reg

        loss.backward()
        optimizer.step()

        # Print loss for every 1000 iterations
        if (i+1) % 1000 == 0:
            print('Epoch [{}/{}], Iteration [{}/{}], Loss: {:.4f}'.format(
                epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.item()))
'''
In this example, we use the named_parameters method of the model to iterate through all the parameters and check if their name matches the name of the second layer. 
If it does, we calculate the L2 norm of that parameter and add it to the regularization term. 
Note that we use separate if statements for the weight and bias parameters of the second layer, because they have different names (fc2.weight and fc2.bias, respectively). 
We then multiply the regularization term by a hyperparameter and add it to the loss function. 
The rest of the code is the same as before, with the exception of the optimizer not having a weight_decay argument since we are only regularizing a single layer.
'''








