from util import *

class RestrictedBoltzmannMachine():
    '''
    For more details : A Practical Guide to Training Restricted Boltzmann Machines https://www.cs.toronto.edu/~hinton/absps/guideTR.pdf
    '''
    def __init__(self, ndim_visible, ndim_hidden, is_bottom=False, image_size=[28,28], is_top=False, n_labels=10, batch_size=10):

        """
        Args:
          ndim_visible: Number of units in visible layer.
          ndim_hidden: Number of units in hidden layer.
          is_bottom: True only if this rbm is at the bottom of the stack in a deep belief net. Used to interpret visible layer as image data with dimensions "image_size".
          image_size: Image dimension for visible layer.
          is_top: True only if this rbm is at the top of stack in deep beleif net. Used to interpret visible layer as concatenated with "n_label" unit of label data at the end. 
          n_label: Number of label categories.
          batch_size: Size of mini-batch.
        """
       
        self.ndim_visible = ndim_visible

        self.ndim_hidden = ndim_hidden

        self.is_bottom = is_bottom

        if is_bottom : self.image_size = image_size
        
        self.is_top = is_top

        if is_top : self.n_labels = n_labels

        self.batch_size = batch_size        
                
        # Momentum buffers (arrays are safer than scalars for later updates)
        self.delta_bias_v = np.zeros((self.ndim_visible,))

        self.delta_weight_vh = np.zeros((self.ndim_visible,self.ndim_hidden))

        self.delta_bias_h = np.zeros((self.ndim_hidden,))

        # params initialised ~ N(0, 0.01)
        self.bias_v = np.random.normal(loc=0.0, scale=0.01, size=(self.ndim_visible))

        self.weight_vh = np.random.normal(loc=0.0, scale=0.01, size=(self.ndim_visible,self.ndim_hidden))

        self.bias_h = np.random.normal(loc=0.0, scale=0.01, size=(self.ndim_hidden))
        
        self.delta_weight_v_to_h = 0

        self.delta_weight_h_to_v = 0        
        
        self.weight_v_to_h = None
        
        self.weight_h_to_v = None

        self.learning_rate = 0.01
        
        self.momentum = 0.7

        self.print_period = 5000
        
        self.rf = { # receptive-fields. Only applicable when visible layer is input data
            "period" : 5000, # iteration period to visualize
            "grid" : [5,5], # size of the grid
            "ids" : np.random.randint(0,self.ndim_hidden,25) # pick some random hidden units
            }
        
        return

        
    def cd1(self,visible_trainset, n_iterations=10000, n_epochs=None, shuffle=True, return_history=False):
    
        """Contrastive Divergence with k=1 full alternating Gibbs sampling

        Args:
        visible_trainset: training data for this rbm, shape is (size of training set, size of visible layer)
        n_iterations: number of iterations of learning (each iteration learns a mini-batch)

        Extra (for the lab writeup):
        - If you pass n_epochs (e.g. 10..20), we interpret "each epoch = one full swipe through the training set",
          divided into mini-batches of size self.batch_size (around 20).
        - If return_history=True, we return stability metrics per epoch:
          recon_loss (avg MSE), mean_h_prob (avg hidden ON-prob), mean_dW_norm (avg ||ΔW||).
        """

        print ("learning CD1")
        
        n_samples = visible_trainset.shape[0]

        # history for monitoring "stability/convergence"
        history = {
            "recon_loss": [],
            "mean_h_prob": [],
            "mean_dW_norm": []
        }

        # ------------------------------------------------------------
        # EPOCH MODE: full sweep through dataset each epoch
        # ------------------------------------------------------------
        if n_epochs is not None:

            n_batches = n_samples // self.batch_size
            global_it = 0

            for ep in range(n_epochs):

                # epoch accumulators
                epoch_recon = 0.0
                epoch_hprob = 0.0
                epoch_dW    = 0.0

                if shuffle:
                    perm = np.random.permutation(n_samples)
                else:
                    perm = np.arange(n_samples)

                for b in range(n_batches):

                    # [TODO TASK 4.1] run k=1 alternating Gibbs sampling : v_0 -> h_0 ->  v_1 -> h_1.
                    # you may need to use the inference functions 'get_h_given_v' and 'get_v_given_h'.
                    # note that inference methods returns both probabilities and activations (samples from probablities) and you may have to decide when to use what.

                    # [TODO TASK 4.1] update the parameters using function 'update_params'
                    
                    # visualize once in a while when visible layer is input images

                    # pick a minibatch (full sweep order)
                    idx = perm[b*self.batch_size:(b+1)*self.batch_size]
                    v0 = visible_trainset[idx, :]

                    # Gibbs chain: v0 -> h0 -> v1 -> h1
                    h0_prob, h0_act = self.get_h_given_v(v0)

                    v1_prob, v1_act = self.get_v_given_h(h0_act)

                    h1_prob, h1_act = self.get_h_given_v(v1_act)

                    self.update_params(v_0=v0, h_0=h0_prob, v_k=v1_prob, h_k=h1_prob)

                    # monitor stability metrics
                    recon_loss = np.mean((v0 - v1_prob) ** 2)
                    epoch_recon += recon_loss
                    epoch_hprob += float(np.mean(h0_prob))
                    epoch_dW    += float(np.linalg.norm(self.delta_weight_vh))

                    # RF visualization (use global iteration counter)
                    if global_it % self.rf["period"] == 0 and self.is_bottom:
                        viz_rf(weights=self.weight_vh[:, self.rf["ids"]].reshape((self.image_size[0], self.image_size[1], -1)),
                               it=global_it, grid=self.rf["grid"])

                    global_it += 1

                # epoch averages
                epoch_recon /= n_batches
                epoch_hprob /= n_batches
                epoch_dW    /= n_batches

                history["recon_loss"].append(epoch_recon)
                history["mean_h_prob"].append(epoch_hprob)
                history["mean_dW_norm"].append(epoch_dW)

                # print per epoch (recommended for "stability")
                print("epoch=%3d/%3d  avg_recon_loss=%4.4f  mean_h_prob=%4.4f  mean||dW||=%4.4f"
                      % (ep+1, n_epochs, epoch_recon, epoch_hprob, epoch_dW))

            if return_history:
                return history
            return

        # ------------------------------------------------------------
        # ITERATION MODE: original behaviour (random minibatches)
        # ------------------------------------------------------------
        for it in range(n_iterations):

            # [TODO TASK 4.1] run k=1 alternating Gibbs sampling : v_0 -> h_0 ->  v_1 -> h_1.
            # you may need to use the inference functions 'get_h_given_v' and 'get_v_given_h'.
            # note that inference methods returns both probabilities and activations (samples from probablities) and you may have to decide when to use what.

            # [TODO TASK 4.1] update the parameters using function 'update_params'
            
            # visualize once in a while when visible layer is input images

            # pick a minibatch
            batch_idx = np.random.randint(0, n_samples, self.batch_size)
            v0 = visible_trainset[batch_idx, :]

            # Gibbs chain: v0 -> h0 -> v1 -> h1
            h0_prob, h0_act = self.get_h_given_v(v0)

            v1_prob, v1_act = self.get_v_given_h(h0_act)

            h1_prob, h1_act = self.get_h_given_v(v1_act)

            self.update_params(v_0=v0, h_0=h0_prob, v_k=v1_prob, h_k=h1_prob)
            
            if it % self.rf["period"] == 0 and self.is_bottom:
                
                viz_rf(weights=self.weight_vh[:, self.rf["ids"]].reshape((self.image_size[0], self.image_size[1], -1)),
                    it=it, grid=self.rf["grid"])

            # print progress
            
            if it % self.print_period == 0 :
                recon_loss = np.mean((v0 - v1_prob) ** 2)
                print("iteration=%7d recon_loss=%4.4f" % (it, recon_loss))

        if return_history:
            return history  # empty in iteration mode unless you choose to also fill it
        return
    

    def update_params(self,v_0,h_0,v_k,h_k):

        """Update the weight and bias parameters.

        You could also add weight decay and momentum for weight updates.

        Args:
           v_0: activities or probabilities of visible layer (data to the rbm)
           h_0: activities or probabilities of hidden layer
           v_k: activities or probabilities of visible layer
           h_k: activities or probabilities of hidden layer
           all args have shape (size of mini-batch, size of respective layer)
        """

        # [TODO TASK 4.1] get the gradients from the arguments (replace the 0s below) and update the weight and bias parameters
        
        batch_size = v_0.shape[0]

        """ here i calculate the gradients for bias of visible layer, bias of hidden layer and weight between visible and hidden layer
          using the formulas derived from the contrastive divergence algorithm i.e 
          ∇bv​=B1​n=1∑B​(v0(n)​−vk(n)​),
          ∇bh​=B1​n=1∑B​(h0(n)​−hk(n)​),
          ∇W=B1​(n∑​v0(n)​(h0(n)​)T−n∑​vk(n)​(hk(n)​)T)."""

        grad_bv = np.mean(v_0 - v_k, axis=0)
        grad_bh = np.mean(h_0 - h_k, axis=0)
        grad_W  = (v_0.T @ h_0 - v_k.T @ h_k) / batch_size
        
        """ here i update the parameters using the gradients calculated above, and also add momentum and learning rate to the updates as to stabilise the learning process."""
        self.delta_bias_v    = self.momentum * self.delta_bias_v    + self.learning_rate * grad_bv
        self.delta_weight_vh = self.momentum * self.delta_weight_vh + self.learning_rate * grad_W
        self.delta_bias_h    = self.momentum * self.delta_bias_h    + self.learning_rate * grad_bh

        
        self.bias_v += self.delta_bias_v
        self.weight_vh += self.delta_weight_vh
        self.bias_h += self.delta_bias_h
        
        return

    def get_h_given_v(self,visible_minibatch):
        
        """Compute probabilities p(h|v) and activations h ~ p(h|v) 

        Uses undirected weight "weight_vh" and bias "bias_h"
        
        Args: 
           visible_minibatch: shape is (size of mini-batch, size of visible layer)
        Returns:        
           tuple ( p(h|v) , h) 
           both are shaped (size of mini-batch, size of hidden layer)
        """
        
        assert self.weight_vh is not None


        # [TODO TASK 4.1] compute probabilities and activations (samples from probabilities) of hidden layer (replace the zeros below) 
        
        """ Here I calculate the total input for each hidden unit h_j= b_j + sum_i v_i w_ij, where b_j is the bias of hidden unit j, 
        v_i is the activity of visible unit i and w_ij is the weight between visible unit i and hidden unit j. 
        Then I use the sigmoid activation function to calculate the probabilities of hidden units being ON, and then sample binary activations from these probabilities. """

        n_samples = visible_minibatch.shape[0]

        support = visible_minibatch @ self.weight_vh + self.bias_h[None, :]
        h_prob = sigmoid(support) # here we use the sigmoid activation function to calculate the probabilities of hidden units being ON
        h_act = sample_binary(h_prob) # here i set h_act=0 if h_prob < random number between 0 and 1, and h_act=1 otherwise, to sample binary activations from the probabilities.

        return h_prob, h_act


    def get_v_given_h(self,hidden_minibatch):
        
        """Compute probabilities p(v|h) and activations v ~ p(v|h)

        Uses undirected weight "weight_vh" and bias "bias_v"
        
        Args: 
           hidden_minibatch: shape is (size of mini-batch, size of hidden layer)
        Returns:        
           tuple ( p(v|h) , v) 
           both are shaped (size of mini-batch, size of visible layer)
        """
        
        assert self.weight_vh is not None

        n_samples = hidden_minibatch.shape[0]
        support = hidden_minibatch @ self.weight_vh.T + self.bias_v[None, :]


        if self.is_top:

            """
            Here visible layer has both data and labels. Compute total input for each unit (identical for both cases),
            and split into two parts, something like support[:, :-self.n_labels] and support[:, -self.n_labels:].
            Then, for both parts, use the appropriate activation function to get probabilities and a sampling method \
            to get activities. The probabilities as well as activities can then be concatenated back into a normal visible layer.
            """

            # [TODO TASK 4.1] compute probabilities and activations (samples from probabilities) of visible layer (replace the pass below). \
            # Note that this section can also be postponed until TASK 4.2, since in this task, stand-alone RBMs do not contain labels in visible layer.
            
            # split into data part and label part
            data_support = support[:, :-self.n_labels]
            lbl_support  = support[:, -self.n_labels:]

            v_data_prob = sigmoid(data_support)
            v_lbl_prob  = softmax(lbl_support)

            v_data_act = sample_binary(v_data_prob)
            v_lbl_act  = sample_categorical(v_lbl_prob)

            v_prob = np.concatenate([v_data_prob, v_lbl_prob], axis=1)
            v_act  = np.concatenate([v_data_act,  v_lbl_act],  axis=1)

            return v_prob, v_act
            
        else:
                        
            # [TODO TASK 4.1] compute probabilities and activations (samples from probabilities) of visible layer (replace the pass and zeros below)         

            v_prob = sigmoid(support)
            v_act  = sample_binary(v_prob) 

            return v_prob, v_act


    
    """ rbm as a belief layer : the functions below do not have to be changed until running a deep belief net """

    

    def untwine_weights(self):
        
        self.weight_v_to_h = np.copy( self.weight_vh )
        self.weight_h_to_v = np.copy( np.transpose(self.weight_vh) )
        self.weight_vh = None

    def get_h_given_v_dir(self,visible_minibatch):

        """Compute probabilities p(h|v) and activations h ~ p(h|v)

        Uses directed weight "weight_v_to_h" and bias "bias_h"
        
        Args: 
           visible_minibatch: shape is (size of mini-batch, size of visible layer)
        Returns:        
           tuple ( p(h|v) , h) 
           both are shaped (size of mini-batch, size of hidden layer)
        """
        
        assert self.weight_v_to_h is not None

        n_samples = visible_minibatch.shape[0]

        # [TODO TASK 4.2] perform same computation as the function 'get_h_given_v' but with directed connections (replace the zeros below) 
        
        support = visible_minibatch @ self.weight_v_to_h + self.bias_h[None, :]
        h_prob = sigmoid(support)
        h_act = sample_binary(h_prob)

        return h_prob, h_act


    def get_v_given_h_dir(self,hidden_minibatch):


        """Compute probabilities p(v|h) and activations v ~ p(v|h)

        Uses directed weight "weight_h_to_v" and bias "bias_v"
        
        Args: 
           hidden_minibatch: shape is (size of mini-batch, size of hidden layer)
        Returns:        
           tuple ( p(v|h) , v) 
           both are shaped (size of mini-batch, size of visible layer)
        """
        
        assert self.weight_h_to_v is not None
        
        n_samples = hidden_minibatch.shape[0]
        
        if self.is_top:

            """
            Here visible layer has both data and labels. Compute total input for each unit (identical for both cases),
            and split into two parts, something like support[:, :-self.n_labels] and support[:, -self.n_labels:].
            Then, for both parts, use the appropriate activation function to get probabilities and a sampling method \
            to get activities. The probabilities as well as activities can then be concatenated back into a normal visible layer.
            """
            
            # [TODO TASK 4.2] Note that even though this function performs same computation as 'get_v_given_h' but with directed connections,
            # this case should never be executed : when the RBM is a part of a DBN and is at the top, it will have not have directed connections.
            # Appropriate code here is to raise an error (replace pass below)
            
            raise RuntimeError("get_v_given_h_dir() should not be called for the top RBM in a DBN (top RBM stays undirected).")
            
        else:
                        
            # [TODO TASK 4.2] performs same computaton as the function 'get_v_given_h' but with directed connections (replace the pass and zeros below)             

            support = hidden_minibatch @ self.weight_h_to_v + self.bias_v[None, :]
            v_prob = sigmoid(support)
            v_act = sample_binary(v_prob)

            return v_prob, v_act
            
    def update_generate_params(self,inps,trgs,preds):
        
        """Update generative weight "weight_h_to_v" and bias "bias_v"
        
        Args:
           inps: activities or probabilities of input unit
           trgs: activities or probabilities of output unit (target)
           preds: activities or probabilities of output unit (prediction)
           all args have shape (size of mini-batch, size of respective layer)
        """

        # [TODO TASK 4.3] find the gradients from the arguments (replace the 0s below) and update the weight and bias parameters.
        
        self.delta_weight_h_to_v += 0
        self.delta_bias_v += 0
        
        self.weight_h_to_v += self.delta_weight_h_to_v
        self.bias_v += self.delta_bias_v 
        
        return
    
    def update_recognize_params(self,inps,trgs,preds):
        
        """Update recognition weight "weight_v_to_h" and bias "bias_h"
        
        Args:
           inps: activities or probabilities of input unit
           trgs: activities or probabilities of output unit (target)
           preds: activities or probabilities of output unit (prediction)
           all args have shape (size of mini-batch, size of respective layer)
        """

        # [TODO TASK 4.3] find the gradients from the arguments (replace the 0s below) and update the weight and bias parameters.

        self.delta_weight_v_to_h += 0
        self.delta_bias_h += 0

        self.weight_v_to_h += self.delta_weight_v_to_h
        self.bias_h += self.delta_bias_h
        
        return
