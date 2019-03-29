'''
Created on Mar 22, 2019

@author: jens
'''

import torch
import torch.nn as nn

class SparseCoding(nn.Module):

    def __init__(self, active=True, ema_decay=0.99, steepness_factor=12, clip_threshold=0.01, target_min_boost=0.96, target_max_boost=3.84, boost_factor=0.1, boost_update_count=50):
        super(SparseCoding, self).__init__()
        self.active = active
        self.ema_decay=ema_decay
        self.steepness_factor=steepness_factor
        self.clip_threshold=clip_threshold
        self.target_min_freq=target_min_boost
        self.target_max_freq=target_max_boost
        self.boost_factor=boost_factor
        self.boost_update_count = boost_update_count
        self.N = 0
        self.not_initialized = True

    """
    batch, dim, dim, input_dim (prim_capsules), output_dim
    
    Find max routing values across all input capsules -> batch,dim,dim,output_dim
    (max of input_dim)
    
    Sum 2nd dim (column), and then sum 1st dim (row) -> batch,output_dim
    """

    def forward(self, x):
        """
        x: batch_size, output_dim, h, (dim_x, dim_y)

        Converts the capsule mask into an appropriate shape then applies it to
        the capsule embedding.
    
        Args:
          route: tensor, output of the last capsule layer.
          num_prime_capsules: scalar, number of primary capsules.
          num_latent_capsules: scalar, number of latent capsules.
          verbose: boolean, visualises the variables in Tensorboard.
          ema_decay: scalar, the expontential decay rate for the moving average.
          steepness_factor: scalar, controls the routing mask weights.
          clip_threshold: scalar, threshold for clipping values in the mask.
    
        Returns:
          The routing mask and the routing ranks.
        """

        if self.not_initialized:
            self.register_buffer('boosting_weights', torch.ones(x[1].shape[2]))
    
        """ Calculate routing coefficients """
        # route: batch_size, input_dim, output_dim, dim_x, dim_y
        shp = x[1].shape
        capsule_routing = x[1].view(shp[0]*shp[1],shp[2],-1)   #.max(dim=1)[0]
        capsule_routing = capsule_routing.sum(dim=-1) # batch-size*input_dim, output_dim
    
    
        """ Boosting """
        capsule_routing *= self.boosting_weights
    
        """ Rank routing coefficients """
        order = capsule_routing.sort(1, descending=True)[1]
        ranks = (-order).sort(1, descending=True)[1]
    
        """ Winning frequency """
        transposed_ranks = ranks.transpose(1,0)  # output_dim, batch_size*input_dim
        win_counts = (transposed_ranks == 0).sum(dim=1) # output_dim
        freq = win_counts.float() / transposed_ranks.shape[1] # output_dim
    
        """ Moving average """
        if self.not_initialized:
            self.register_buffer('freq_ema', freq)
            self.softmax = nn.Softmax(dim=0)
            self.target_max_freq /= ranks.shape[1]
            self.target_min_freq /= ranks.shape[1]
            self.not_initialized = False
        else:
            self.freq_ema = self.ema_decay * self.freq_ema + (1 - self.ema_decay) * freq # output_dim

        if not self.active:
            return x[2]
    
        # Normalise the rankings
        normalised_ranks = ranks.float() / (ranks.shape[1] - 1) # batch_size*input_dim, output_dim
    
        routing_mask = torch.exp(-self.steepness_factor * normalised_ranks) # batch_size*input_dim, output_dim
    
        # Set values < 0.1 to zero
        routing_mask = routing_mask - ( routing_mask * (routing_mask < self.clip_threshold).float() )

        self.N += 1
        if self.N == self.boost_update_count:
            self.N = 0
            #step = self.softmax(self.freq_ema) - 1/ranks.shape[1]
            #self.boosting_weights -= step * self.boost_factor #* self.boosting_weights
            #self.boosting_weights = self.boosting_weights.clamp(min=0.01, max=1.0)
            
            #mask = self.freq_ema < 1e-8
            #mask *= 10
            #mask += 1
            #self.boosting_weights *= mask.float()
            
            self.boosting_weights += (self.freq_ema < self.target_min_freq).float() * self.boost_factor
            self.boosting_weights -= (self.freq_ema > self.target_max_freq).float() * self.boost_factor
            self.boosting_weights = self.boosting_weights.clamp(max=1.0)

        votes = routing_mask.view(shp[0], shp[1], shp[2], 1, 1, 1).expand_as(x[2]) * x[2]
        return votes, None # is None, to force bias detach
        #return routing_mask[:,:,None,None,None].expand_as(x[0]) * x[0]
