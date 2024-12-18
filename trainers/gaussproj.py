import torch
import torch.nn as nn 

from tqdm import tqdm

def kl_div(mu, logvar, true_mu, wise=False):
    loss = -0.5 * torch.sum(1+logvar - (true_mu - mu)**2 - logvar.exp(), dim=1)
    if not wise:
        loss = torch.mean(loss, dim=0)
    return loss

class GaussianProjectorTrainer:
    def __init__(self, model, optimizer, lamda, radius, gamma, proj_dim, eps, device):
        
        self.model = model
        self.optimizer = optimizer
        self.lamda = lamda
        self.radius = radius
        self.gamma = gamma
        self.proj_dim=proj_dim
        self.eps = eps
        self.device = device
        self.mean=None

    def initialize_mean(self, trainloader):
        n_samples = 0
        mean = torch.zeros(self.proj_dim).to(self.device)

        self.model.eval()
        with torch.no_grad():
            for x, _ in tqdm(trainloader, desc="Initialisation du centre"):
                x = x.to(self.device)
                mu, logvar = self.model(x)
                n_samples += mu.shape[0]
                mean += torch.sum(mu, dim=0)
        mean /= n_samples

        mean[(abs(mean) < self.eps) & (mean < 0)] = -self.eps
        mean[(abs(mean) < self.eps) & (mean > 0)] = self.eps

        self.model.train()
        self.mean = mean
        return mean
    
    def train(self, trainloader, learning_rate, lr_scheduler, total_epochs, 
                warm_up_epochs, ascent_step_size, ascent_num_steps):
        
        if self.mean is None:
            raise ValueError("Le centre n'a pas été initialisé. Appelez `initialize_center` avant d'entraîner.")
        
        self.ascent_step_size=ascent_step_size
        self.ascent_num_steps=ascent_num_steps

        for epoch in range(total_epochs): 
            self.model.train()
            lr_scheduler(epoch, total_epochs, warm_up_epochs, learning_rate, self.optimizer)
            

            epoch_adv_loss = torch.tensor([0]).type(torch.float32).to(self.device)  #AdvLoss
            epoch_normal_loss = 0  
            
            batch_idx = -1
            for data, target in tqdm(trainloader):
                batch_idx += 1
                data, target = data.to(self.device), target.to(self.device)
                # Data Processing
                data = data.to(torch.float)
                target = target.to(torch.float)
                target = torch.squeeze(target)
                self.optimizer.zero_grad()
                
                mean, logvar = self.model(data)
                normal_loss = kl_div(mean, logvar, self.mean).mean()

                epoch_normal_loss += normal_loss

                if  epoch >= warm_up_epochs:
                    data = data[target == 1]
                    # AdvLoss 
                    adv_loss = self.one_class_adv_loss(data)
                    epoch_adv_loss += adv_loss
                    loss = normal_loss + adv_loss * self.lamda
                else: 
                    # If only CE based training has to be done
                    loss = normal_loss
                
                # Backprop
                loss.backward()
                self.optimizer.step()
                    
            epoch_normal_loss = epoch_normal_loss/(batch_idx + 1)  #Average CE Loss
            epoch_adv_loss = epoch_adv_loss/(batch_idx + 1) #Average AdvLoss
            print('Epoch: {}, CE Loss: {}, AdvLoss: {}'.format(
                epoch+1, epoch_normal_loss.item(), epoch_adv_loss.item())) 
    

    def one_class_adv_loss(self, x_train_data):
        # Randomly sample points around the training data
        # We will perform SGD on these to find the adversarial points
        x_adv = torch.randn(x_train_data.shape).to(self.device).detach().requires_grad_()
        x_adv_sampled = x_adv + x_train_data

        for step in range(self.ascent_num_steps):
            with torch.enable_grad():
                
                mu, logvar = self.model(x_adv_sampled)         
                new_loss = kl_div(mu, logvar, true_mu=self.mean)
                grad = torch.autograd.grad(new_loss, [x_adv_sampled])[0]
                grad_norm = torch.norm(grad, p=2, dim = tuple(range(1, grad.dim())))
                grad_norm = grad_norm.view(-1, *[1]*(grad.dim()-1))
                grad_normalized = grad/grad_norm 
            with torch.no_grad():
                x_adv_sampled.add_(self.ascent_step_size * grad_normalized)
            if (step + 1) % 10==0:
                # Project the normal points to the set N_i(r)
                h = x_adv_sampled - x_train_data
                norm_h = torch.sqrt(torch.sum(h**2, 
                                                dim=tuple(range(1, h.dim()))))
                alpha = torch.clamp(norm_h, self.radius, 
                                    self.gamma * self.radius).to(self.device)
                # Make use of broadcast to project h
                proj = (alpha/norm_h).view(-1, *[1] * (h.dim()-1))
                h = proj * h
                x_adv_sampled = x_train_data + h  #These adv_points are now on the surface of hyper-sphere
        adv_mu, adv_logvar = self.model(x_adv_sampled)
        adv_loss = kl_div(adv_mu, adv_logvar, self.mean)
        assert not torch.isnan(adv_loss), "NAN"
        return adv_loss
    
    def get_model(self):
        return {
            "model":self.model,
            "mean":self.mean
        }