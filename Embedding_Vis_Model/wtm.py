from torch import Tensor
import torch
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 
from torch.nn import Parameter

torch.cuda.empty_cache()

# cos_sim = nn.CosineSimilarity(dim=-1, eps=1e-8)

# bs = 256 
# en1_units_x = 100
# en2_units_x = 100
# learning_rate = 0.001
# beta1 = 0.99
# beta2 = 0.999
# drop_rate = 0.5
# smoothen = 1e-8
# variance_x = 1.0
# pi = torch.acos(torch.zeros(1)).item() * 2 

def get_unit_len_embeddings(my_tensor):
  return my_tensor/(torch.norm(my_tensor,dim=-1).unsqueeze(1))

def get_centered_embeddings(embedding_tensor):
  embedding_centered_tensor = embedding_tensor - embedding_tensor.mean(0)
  return embedding_centered_tensor

#phi
def gaussian(alpha): return -0.5*alpha 

class Model(nn.Module):
    def __init__(self, num_input, en1_units_x, en2_units_x, num_coordinate, num_topic, drop_rate, variance_x, bs, 
                 embedding_words, word_emb_size,smoothen,isbn_muz,isbn_x,isbn_phi,have_KLD,distance="gaussian"):
      
        super(Model, self).__init__()
        self.num_input, self.num_coordinate, self.num_topic, self.variance_x, self.bs \
            = num_input, num_coordinate, num_topic, variance_x, bs

        self.embedding_words = embedding_words
        self.emb_size = word_emb_size

        self.isbn_muz = isbn_muz
        self.isbn_x = isbn_x
        self.isbn_phi = isbn_phi
        self.have_KLD = have_KLD
        self.smoothen = smoothen
 
        self.embedding_words = embedding_words
        self.emb_size = word_emb_size
 
        # encoder
        self.en1_fc     = nn.Linear(num_input, en1_units_x) 
        self.en2_fc     = nn.Linear(en1_units_x, en2_units_x)
        self.mu1_fc     = nn.Linear(2, 100) 
        self.mu2_fc     = nn.Linear(100, 100)
        self.mu_fc      = nn.Linear(100, 300)
        self.mu_z = 0

        self.en2_drop   = nn.Dropout(drop_rate)
        self.mean_fc    = nn.Linear(en2_units_x, num_coordinate) 
        self.logvar_fc  = nn.Linear(en2_units_x, num_coordinate) 

        self.mean_bn    = nn.BatchNorm1d(num_coordinate)                    
        self.logvar_bn  = nn.BatchNorm1d(num_coordinate)
        self.decoder_x_bn = nn.BatchNorm1d(num_coordinate)    
        self.decoder_phi_bn = nn.BatchNorm1d(num_coordinate) 
        self.decoder_bn = nn.BatchNorm1d(self.num_topic)     
        self.mu_z_bn = nn.BatchNorm1d(self.emb_size)                          

        # RBF
        self.in_features = self.num_coordinate
        self.out_features = self.num_topic
        self.centres = nn.Parameter(torch.Tensor(self.out_features, self.in_features)) # K x 2
        #self.mu_z = nn.Parameter(torch.Tensor(self.out_features,self.emb_size))# K x 300
        
        if distance=="gaussian": self.basis_func = gaussian
        if distance=="inverse_quadratic": self.basis_func = inverse_quadratic
        self.init_parameters()
               
        # prior mean and variance as constant buffers
        prior_mean   = torch.Tensor(1, num_coordinate).fill_(0)
        prior_var    = torch.Tensor(1, num_coordinate).fill_(variance_x)
        self.prior_mean = nn.Parameter(prior_mean, requires_grad=False)
        self.prior_var  = nn.Parameter(prior_var, requires_grad=False)
        self.prior_logvar = nn.Parameter(prior_var.log(), requires_grad=False)

        self._max_norm_val = 1.0

    def init_parameters(self):
        nn.init.normal_(self.centres, 0, 0.1)
        #nn.init.normal_(self.mu_z, 0, 0.01)

    def max_norm_(self, w): #https://discuss.pytorch.org/t/how-to-correctly-implement-in-place-max-norm-constraint/96769/4
      with torch.no_grad():
        norm = w.norm(2, dim=0, keepdim=True)#.clamp(min=self._max_norm_val / 2)
        desired = torch.clamp(norm, max=self._max_norm_val)
        w *= (desired / norm)

    def encode(self, input_,normalized_input_):
        N, *_ = input_.size()
        
        en1 = F.softplus((self.en1_fc(input_)))                           # en1_fc   output
        en2 = F.softplus((self.en2_fc(en1)))                              # encoder2 output
        # en3 = F.relu(self.en3_fc(en2))
        en2 = self.en2_drop(en2)

        #self.max_norm_(self.mean_fc.weight)
        posterior_mean   = self.mean_bn(self.mean_fc(en2))          # posterior mean
        #self.max_norm_(self.logvar_fc.weight)

        posterior_logvar = self.logvar_bn(self.logvar_fc(en2))          # posterior log variance
        posterior_var    = posterior_logvar.exp()
        
        return en2, posterior_mean, posterior_logvar, posterior_var

    def take_sample(self, input_, posterior_mean, posterior_var, prior_var):
        eps = input_.data.new().resize_as_(posterior_mean.data).normal_(std=1.0) # noise
        z = posterior_mean + posterior_var.sqrt() * eps     # reparameterization
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

      mu1 = F.softplus((self.mu1_fc(zc)))                           # en1_fc   output
      mu2 = F.softplus(self.en2_drop(self.mu2_fc(mu1)))
      self.mu_z = self .mu_fc(mu2)

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

        NL = - (input_ * (recon_v+self.smoothen).log()).sum(-1)
        NL= NL.mean(0)
     
        if str(self.have_KLD) == "True": 
          KLD = self.KLD(posterior_mean,posterior_logvar,posterior_var).mean(0) # self.kld1 + self.kld2 +
        else: KLD = torch.tensor(0.0)

        loss = NL + KLD
        return loss,NL,KLD
