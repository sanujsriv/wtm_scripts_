from torch import Tensor
import torch
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 
from torch.nn import Parameter

torch.cuda.empty_cache()
# en1_units_x = 100
# en2_units_x = 100
# learning_rate = 0.001
# beta1 = 0.99
# beta2 = 0.999
# drop_rate = 0.0
# num_topic = 10
# emb_size = 300
# smoothen = 1e-30
# variance_x = 1.0

# sig_phi = 1.0

pi = torch.acos(torch.zeros(1)).item() * 2 

# def Log_NormDist(x,mu,sigma,d):
#   # sigma = sigma + smoothen
#   First = d/2.0 * torch.log(torch.tensor(1.0/(2.0*pi*sigma)))
#   Second = -(1.0/(2.0*sigma)) * (torch.norm(x-mu,dim=-1))
#   # Second1 = -(1.0/(2.0*sigma)) 
#   # Second2 = ((x-mu)**2).sum(-1)
#   ret = First + Second
#   return ret

#phi
def gaussian(alpha): return -0.5*alpha 
#def inverse_multi_quadric(alpha): return -0.5*torch.log(torch.ones_like(alpha) + alpha)
def inverse_quadratic(alpha): return -torch.log(torch.ones_like(alpha) + alpha)
 
class Model(nn.Module):
    def __init__(self, num_input, en1_units_x, en2_units_x, num_coordinate, num_topic, drop_rate, variance_x, bs, 
                 embedding_words, word_emb_size,smoothen,isbn_muz,isbn_x,isbn_phi,distance="gaussian"):
      
        super(Model, self).__init__()
        self.num_input, self.num_coordinate, self.num_topic, self.variance_x, self.bs \
            = num_input, num_coordinate, num_topic, variance_x, bs

        self.embedding_words = embedding_words
        self.emb_size = word_emb_size

        self.isbn_muz = isbn_muz
        self.isbn_x = isbn_x
        self.isbn_phi = isbn_phi
        self.smoothen = smoothen
 
        # encoder
        self.en1_fc     = nn.Linear(num_input, en1_units_x)    
        self.en2_fc     = nn.Linear(en1_units_x, en2_units_x)           
        self.en2_drop   = nn.Dropout(drop_rate)
        self.mean_fc    = nn.Linear(en2_units_x, num_coordinate) 
        self.logvar_fc  = nn.Linear(en2_units_x, num_coordinate)  

        self.mean_bn    = nn.BatchNorm1d(num_coordinate)                    
        self.logvar_bn  = nn.BatchNorm1d(num_coordinate)
        self.decoder_x_bn = nn.BatchNorm1d(num_coordinate)    
        self.decoder_phi_bn = nn.BatchNorm1d(num_coordinate) 
        self.decoder_bn = nn.BatchNorm1d(self.num_topic)                              
        
        self.en1_fc_e     = nn.Linear(300, en1_units_x) 
        self.en2_fc_e     = nn.Linear(en1_units_x, en2_units_x)   
        self.en2_drop_e   = nn.Dropout(drop_rate)  

        # RBF
        self.in_features = self.num_coordinate
        self.out_features = self.num_topic
        self.centres = nn.Parameter(torch.Tensor(self.out_features, self.in_features)) # K x 2
        self.mu_z = nn.Parameter(torch.Tensor(self.out_features,self.emb_size))# K x 300
        # self.mu_z =nn.Embedding(self.out_features,self.emb_size)# K x 300
        # mu_z_MultiNormal = torch.normal(mean=self.embedding_words.mean(0).unsqueeze(0).expand(self.out_features,self.emb_size),std=0.00001) # Kx300
        # self.mu_z = nn.Parameter(mu_z_MultiNormal)

        if distance=="gaussian": self.basis_func = gaussian
        if distance=="inverse_quadratic": self.basis_func = inverse_quadratic
        self.init_parameters()
               
        # prior mean and variance as constant buffers
        prior_mean   = torch.Tensor(1, num_coordinate).fill_(0)
        prior_var    = torch.Tensor(1, num_coordinate).fill_(variance_x)
        self.prior_mean = nn.Parameter(prior_mean, requires_grad=False)
        self.prior_var  = nn.Parameter(prior_var, requires_grad=False)
        self.prior_logvar = nn.Parameter(prior_var.log(), requires_grad=False)

    def init_parameters(self):
        nn.init.normal_(self.centres, 0, 0.01)
        nn.init.normal_(self.mu_z, 0, 0.01)
        
    def encode(self, input_,normalized_input_):
        #N, *_ = normalized_input_.size()
        # en1 = F.softplus(self.en1_fc(normalized_input_)) 

        N, *_ = input_.size()
        en1 = F.softplus(self.en1_fc(input_))                           # en1_fc   output
        en2 = F.softplus(self.en2_fc(en1))                              # encoder2 output
        # en3 = F.softplus(self.en3_fc(en2))
        en2 = self.en2_drop(en2)
        posterior_mean   = self.mean_bn(self.mean_fc(en2))          # posterior mean
        posterior_logvar = self.logvar_bn(self.logvar_fc(en2))          # posterior log variance
        # posterior_mean   = (self.mean_fc(en2))          # posterior mean
        # posterior_logvar = (self.logvar_fc(en2))          # posterior log variance
        posterior_var    = posterior_logvar.exp()
        
        return en2, posterior_mean, posterior_logvar, posterior_var
    
    def take_sample(self, input_, posterior_mean, posterior_var, prior_var):
        eps = input_.data.new().resize_as_(posterior_mean.data).normal_(std=1) # noise
        z = posterior_mean + posterior_var.sqrt() * eps                   # reparameterization
        return z
    
    def get_beta(self): 
      # mu_z_unit = self.mu_z/(torch.norm(self.mu_z,dim=-1).unsqueeze(1))
      if not self.isbn_muz:
         return F.softmax(torch.mm(self.mu_z,self.embedding_words.T),dim=-1)
      if self.isbn_muz:
        return F.softmax(self.decoder_bn( torch.mm(self.mu_z,self.embedding_words.T).T).T,dim=-1)

    def decode(self, z):

        ## Theta - P(z|x,phi)
        N, *_ = z.size()
        size = (N, self.out_features, self.in_features) # N,K,2

        if self.isbn_x: zx =  self.decoder_x_bn(z) # Nx2
        if not self.isbn_x: zx = z
    
        x = zx.view(N, 1, self.num_coordinate).expand(size) # Nx1x2

        if self.isbn_phi: zc = self.decoder_phi_bn(self.centres)
        if not self.isbn_phi: zc = self.centres
        
        c = zc.view(1, self.num_topic, self.num_coordinate).expand(size)
        d = (x-c).pow(2).sum(-1)
        distances = self.basis_func(d)
        zx_phi = torch.exp(distances - torch.logsumexp(distances, dim=-1, keepdim=True)) # N x K

        beta = self.get_beta()
        recon_v = torch.mm(zx_phi,beta)

        return recon_v, zx, zx_phi, d,zc
    
    def forward(self, input_,normalized_input_, compute_loss=False):  
        en2, posterior_mean, posterior_logvar, posterior_var = self.encode(input_,normalized_input_)
        z = self.take_sample(input_, posterior_mean, posterior_var, self.variance_x)
        recon_v, zx, zx_phi,d,zc= self.decode(z)
        
        if compute_loss:
            return recon_v, zx,self.loss(input_, recon_v, zx_phi, posterior_mean, posterior_logvar, posterior_var, d,zx)
        else: return z, recon_v, zx,zc, zx_phi
 
    def KLD(self, posterior_mean,posterior_logvar,posterior_var):
        N = posterior_mean.shape[0]
        prior_mean   = self.prior_mean.expand_as(posterior_mean)
        prior_var    = self.prior_var.expand_as(posterior_mean)
        prior_logvar = self.prior_logvar.expand_as(posterior_mean)

        var_division    = posterior_var  / prior_var 
        diff            = posterior_mean - prior_mean
        diff_term       = diff * diff / prior_var
        logvar_division = prior_logvar - posterior_logvar
        
        xKLD = 0.5 * ((var_division + diff_term + logvar_division).sum(-1) - self.num_coordinate) 
        return xKLD
 
    def loss(self, input_, recon_v, zx_phi, posterior_mean, posterior_logvar, posterior_var, d, zx,avg=True):
        N = posterior_mean.shape[0]

        NL = - (input_ * (recon_v+self.smoothen).log()).sum(-1) # (Word_Count_input_loss)
        # NL = - (input_ * recon_v).sum(-1) # (Word_Count_input_loss)
        NL= NL.mean(0)
        KLD = self.KLD(posterior_mean,posterior_logvar,posterior_var).mean(0)
        loss = NL+KLD
        return loss,NL,KLD
