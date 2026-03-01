from util import *
from rbm import RestrictedBoltzmannMachine

class DeepBeliefNet():    

    ''' 
    For more details : Hinton, Osindero, Teh (2006). A fast learning algorithm for deep belief nets. https://www.cs.toronto.edu/~hinton/absps/fastnc.pdf

    network          : [top] <---> [pen] ---> [hid] ---> [vis] 
                               `-> [lbl] 
    lbl : label
    top : top
    pen : penultimate
    hid : hidden
    vis : visible
    '''
    
    def __init__(self, sizes, image_size, n_labels, batch_size):

        """
        Args:
          sizes: Dictionary of layer names and dimensions
          image_size: Image dimension of data
          n_labels: Number of label categories
          batch_size: Size of mini-batch
        """

        self.rbm_stack = {
            
            'vis--hid' : RestrictedBoltzmannMachine(ndim_visible=sizes["vis"], ndim_hidden=sizes["hid"],
                                                    is_bottom=True, image_size=image_size, batch_size=batch_size),
            
            'hid--pen' : RestrictedBoltzmannMachine(ndim_visible=sizes["hid"], ndim_hidden=sizes["pen"], batch_size=batch_size),
            
            'pen+lbl--top' : RestrictedBoltzmannMachine(ndim_visible=sizes["pen"]+sizes["lbl"], ndim_hidden=sizes["top"],
                                                        is_top=True, n_labels=n_labels, batch_size=batch_size)
        }
        
        self.sizes = sizes

        self.image_size = image_size

        self.batch_size = batch_size
        
        self.n_gibbs_recog = 15
        
        self.n_gibbs_gener = 200
        
        self.n_gibbs_wakesleep = 5

        self.print_period = 2000
        
        return

    # -------------------- helpers (added) --------------------

    def _compute_hidden_probs_undirected(self, rbm, visible_dataset, batch_size=200):
        """
        Compute p(h|v) for an entire dataset using the RBM's undirected weights.
        This is used during greedy pretraining BEFORE untwining.
        """
        N = visible_dataset.shape[0]
        H = rbm.ndim_hidden
        out = np.zeros((N, H), dtype=np.float32)

        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            v = visible_dataset[start:end, :]
            h_prob, h_act = rbm.get_h_given_v(v)
            out[start:end, :] = h_prob.astype(np.float32)

        return out

    def _avg_recon_loss(self, rbm, visible_dataset, batch_size=200):
        """
        Average one-step reconstruction MSE over a dataset (for reporting).
        Works for both normal RBMs and top RBM (in top visible space).
        """
        N = visible_dataset.shape[0]
        total = 0.0
        count = 0

        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            v0 = visible_dataset[start:end, :]

            h_prob, h_act = rbm.get_h_given_v(v0)
            v1_prob, v1_act = rbm.get_v_given_h(h_act)

            total += float(np.mean((v0 - v1_prob) ** 2))
            count += 1

        return total / max(count, 1)

    # -------------------- TASK 4.2 --------------------

    def recognize(self,true_img,true_lbl, return_convergence=False):

        """Recognize/Classify the data into label categories and calculate the accuracy

        Args:
          true_imgs: visible data shaped (number of samples, size of visible layer)
          true_lbl: true labels shaped (number of samples, size of label layer). Used only for calculating accuracy, not driving the net
        """
        
        n_samples = true_img.shape[0]
        
        # We will do inference in mini-batches to avoid memory issues.
        predicted_lbl = np.zeros(true_lbl.shape, dtype=np.float32)

        # convergence diagnostics for "label units"
        mean_maxprob = np.zeros(self.n_gibbs_recog, dtype=np.float64)
        mean_acc_step = np.zeros(self.n_gibbs_recog, dtype=np.float64)

        n_batches = int(np.ceil(n_samples / self.batch_size))

        for b in range(n_batches):
            start = b * self.batch_size
            end = min(start + self.batch_size, n_samples)

            vis = true_img[start:end, :]  # visible layer gets the image data
            
            # initialise label units with 0.1 (uniform, sums to 1)
            lbl = np.ones((end-start, self.sizes["lbl"]), dtype=np.float32) * 0.1
        
            # [TODO TASK 4.2] fix the image data in the visible layer and drive the network bottom to top. In the top RBM, run alternating Gibbs sampling \
            # and read out the labels (replace pass below and 'predicted_lbl' to your predicted labels).
            # NOTE : inferring entire train/test set may require too much compute memory (depends on your system). In that case, divide into mini-batches.

            # ---- bottom-up feedforward using directed weights ----
            hid_prob, hid_act = self.rbm_stack["vis--hid"].get_h_given_v_dir(vis)
            pen_prob, pen_act = self.rbm_stack["hid--pen"].get_h_given_v_dir(hid_prob)

            # Clamp pen representation: we keep pen fixed during top-level Gibbs
            pen = pen_prob.astype(np.float32)

            # Top visible is [pen | lbl]
            v_top = np.concatenate([pen, lbl], axis=1)

            lbl_prob = lbl

            for t in range(self.n_gibbs_recog):
                # ---- Gibbs sampling in the top RBM (undirected) ----
                h_top_prob, h_top_act = self.rbm_stack["pen+lbl--top"].get_h_given_v(v_top)
                v_top_prob, v_top_act = self.rbm_stack["pen+lbl--top"].get_v_given_h(h_top_act)

                # Clamp the pen part back to feedforward pen representation
                v_top_prob[:, :self.sizes["pen"]] = pen

                # Read out label probabilities (softmax already applied inside get_v_given_h for top RBM)
                lbl_prob = v_top_prob[:, -self.sizes["lbl"]:]
                
                # For next iteration, we keep pen clamped and feed current lbl probabilities
                v_top = np.concatenate([pen, lbl_prob], axis=1)

                # ---- convergence measurements (mean max prob, accuracy per step) ----
                mean_maxprob[t] += float(np.mean(np.max(lbl_prob, axis=1)))

                pred_step = np.argmax(lbl_prob, axis=1)
                true_step = np.argmax(true_lbl[start:end, :], axis=1)
                mean_acc_step[t] += float(np.mean(pred_step == true_step))

            predicted_lbl[start:end, :] = lbl_prob

        # average convergence curves over mini-batches
        mean_maxprob /= max(n_batches, 1)
        mean_acc_step /= max(n_batches, 1)

        acc = 100. * np.mean(np.argmax(predicted_lbl,axis=1)==np.argmax(true_lbl,axis=1))
        print ("accuracy = %.2f%%"%acc)

        if return_convergence:
            return predicted_lbl, mean_maxprob, mean_acc_step
        
        return

    def generate(self,true_lbl,name):
        
        """Generate data from labels

        Args:
          true_lbl: true labels shaped (number of samples, size of label layer)
          name: string used for saving a video of generated visible activations
        """
        
        n_sample = true_lbl.shape[0]
        
        records = []
        # Higher render resolution for mp4 output
        fig,ax = plt.subplots(1,1,figsize=(6,6))
        fig.set_dpi(250)
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        ax.set_xticks([]); ax.set_yticks([])

        lbl = true_lbl.astype(np.float32)

        # [TODO TASK 4.2] fix the label in the label layer and run alternating Gibbs sampling in the top RBM. From the top RBM, drive the network \ 
        # top to the bottom visible layer (replace 'vis' from random to your generated visible layer).
        
        # initialise pen randomly
        pen = sample_binary(np.random.rand(n_sample, self.sizes["pen"]).astype(np.float32))
        v_top = np.concatenate([pen, lbl], axis=1)

        for _ in range(self.n_gibbs_gener):

            # Gibbs in top RBM
            h_top_prob, h_top_act = self.rbm_stack["pen+lbl--top"].get_h_given_v(v_top)
            v_top_prob, v_top_act = self.rbm_stack["pen+lbl--top"].get_v_given_h(h_top_act)

            # Clamp labels (fixed during generation)
            v_top_prob[:, -self.sizes["lbl"]:] = lbl
            v_top_act[:, -self.sizes["lbl"]:] = lbl

            # Update pen from top visible sample/probabilities
            pen = v_top_act[:, :self.sizes["pen"]]

            # Drive downward through directed generative weights:
            hid_prob, hid_act = self.rbm_stack["hid--pen"].get_v_given_h_dir(pen)
            vis_prob, vis_act = self.rbm_stack["vis--hid"].get_v_given_h_dir(hid_act)

            vis = vis_prob  # show probabilities

            records.append([
                ax.imshow(
                    vis.reshape(self.image_size),
                    cmap="bwr",
                    vmin=0,
                    vmax=1,
                    animated=True,
                    interpolation="nearest",
                )
            ])

            v_top = np.concatenate([pen, lbl], axis=1)
            
        anim = stitch_video(fig,records)
        anim.save(
            "%s.generate%d.mp4"%(name,np.argmax(true_lbl)),
            writer="ffmpeg",
            dpi=250,
            bitrate=5000,
            extra_args=[
                "-vcodec", "libx264",
                "-pix_fmt", "yuv420p",
                "-crf", "18",
                "-preset", "medium",
            ],
        )
        plt.close(fig)
            
        return

    def train_greedylayerwise(self, vis_trainset, lbl_trainset, n_iterations, n_epochs=None):

        """
        Greedy layer-wise training by stacking RBMs. This method first tries to load previous saved parameters of the entire RBM stack. 
        If not found, learns layer-by-layer (which needs to be completed) .
        Notice that once you stack more layers on top of a RBM, the weights are permanently untwined.

        Args:
          vis_trainset: visible data shaped (size of training set, size of visible layer)
          lbl_trainset: label data shaped (size of training set, size of label layer)
          n_iterations: number of iterations of learning (each iteration learns a mini-batch)

        Extra:
          n_epochs: if provided, trains each RBM in epoch mode (recommended for reporting).
        """

        try :

            self.loadfromfile_rbm(loc="trained_rbm",name="vis--hid")
            self.rbm_stack["vis--hid"].untwine_weights()            
            
            self.loadfromfile_rbm(loc="trained_rbm",name="hid--pen")
            self.rbm_stack["hid--pen"].untwine_weights()
            
            self.loadfromfile_rbm(loc="trained_rbm",name="pen+lbl--top")        

        except IOError :

            # [TODO TASK 4.2] use CD-1 to train all RBMs greedily
        
            print ("training vis--hid")
            """ 
            CD-1 training for vis--hid 
            """
            if n_epochs is not None:
                self.rbm_stack["vis--hid"].cd1(vis_trainset, n_epochs=n_epochs, shuffle=True, return_history=False)
            else:
                self.rbm_stack["vis--hid"].cd1(vis_trainset, n_iterations=n_iterations)

            # report recon loss (pixel space)
            loss_vis_hid = self._avg_recon_loss(self.rbm_stack["vis--hid"], vis_trainset)
            print("vis--hid avg recon loss (train) = %.6f" % loss_vis_hid)

            # save BEFORE untwining
            self.savetofile_rbm(loc="trained_rbm",name="vis--hid")

            # compute representation for next RBM (use probabilities for stability)
            hid_train = self._compute_hidden_probs_undirected(self.rbm_stack["vis--hid"], vis_trainset, batch_size=200)

            print ("training hid--pen")
            """ 
            CD-1 training for hid--pen 
            """
            # now untwine vis--hid (becomes directed for DBN)
            self.rbm_stack["vis--hid"].untwine_weights()

            if n_epochs is not None:
                self.rbm_stack["hid--pen"].cd1(hid_train, n_epochs=n_epochs, shuffle=True, return_history=False)
            else:
                self.rbm_stack["hid--pen"].cd1(hid_train, n_iterations=n_iterations)

            # report recon loss (hid space)
            loss_hid_pen = self._avg_recon_loss(self.rbm_stack["hid--pen"], hid_train)
            print("hid--pen avg recon loss (train, hid-space) = %.6f" % loss_hid_pen)

            # save BEFORE untwining
            self.savetofile_rbm(loc="trained_rbm",name="hid--pen")

            # compute pen representation for top RBM training
            pen_train = self._compute_hidden_probs_undirected(self.rbm_stack["hid--pen"], hid_train, batch_size=200)

            print ("training pen+lbl--top")
            """ 
            CD-1 training for pen+lbl--top 
            """
            # untwine hid--pen (becomes directed for DBN)
            self.rbm_stack["hid--pen"].untwine_weights()

            # clamp labels by concatenating true label units with pen representations
            top_vis_train = np.concatenate([pen_train, lbl_trainset.astype(np.float32)], axis=1)

            if n_epochs is not None:
                self.rbm_stack["pen+lbl--top"].cd1(top_vis_train, n_epochs=n_epochs, shuffle=True, return_history=False)
            else:
                self.rbm_stack["pen+lbl--top"].cd1(top_vis_train, n_iterations=n_iterations)

            loss_top = self._avg_recon_loss(self.rbm_stack["pen+lbl--top"], top_vis_train)
            print("pen+lbl--top avg recon loss (train, pen+lbl-space) = %.6f" % loss_top)

            # save top RBM
            self.savetofile_rbm(loc="trained_rbm",name="pen+lbl--top")            

        return    

    def train_wakesleep_finetune(self, vis_trainset, lbl_trainset, n_iterations):

        """
        Wake-sleep method for learning all the parameters of network. 
        First tries to load previous saved parameters of the entire network.

        Args:
          vis_trainset: visible data shaped (size of training set, size of visible layer)
          lbl_trainset: label data shaped (size of training set, size of label layer)
          n_iterations: number of iterations of learning (each iteration learns a mini-batch)
        """
        
        print ("\ntraining wake-sleep..")

        try :
            
            self.loadfromfile_dbn(loc="trained_dbn",name="vis--hid")
            self.loadfromfile_dbn(loc="trained_dbn",name="hid--pen")
            self.loadfromfile_rbm(loc="trained_dbn",name="pen+lbl--top")
            
        except IOError :            

            self.n_samples = vis_trainset.shape[0]

            for it in range(n_iterations):            
                                                
                # [TODO TASK 4.3] wake-phase : drive the network bottom to top using fixing the visible and label data.

                # [TODO TASK 4.3] alternating Gibbs sampling in the top RBM for k='n_gibbs_wakesleep' steps, also store neccessary information for learning this RBM.

                # [TODO TASK 4.3] sleep phase : from the activities in the top RBM, drive the network top to bottom.

                # [TODO TASK 4.3] compute predictions : compute generative predictions from wake-phase activations, and recognize predictions from sleep-phase activations.
                # Note that these predictions will not alter the network activations, we use them only to learn the directed connections.
                
                # [TODO TASK 4.3] update generative parameters : here you will only use 'update_generate_params' method from rbm class.

                # [TODO TASK 4.3] update parameters of top rbm : here you will only use 'update_params' method from rbm class.

                # [TODO TASK 4.3] update generative parameters : here you will only use 'update_recognize_params' method from rbm class.

                if it % self.print_period == 0 : print ("iteration=%7d"%it)
                        
            self.savetofile_dbn(loc="trained_dbn",name="vis--hid")
            self.savetofile_dbn(loc="trained_dbn",name="hid--pen")
            self.savetofile_rbm(loc="trained_dbn",name="pen+lbl--top")            

        return

    
    def loadfromfile_rbm(self,loc,name):
        
        self.rbm_stack[name].weight_vh = np.load("%s/rbm.%s.weight_vh.npy"%(loc,name))
        self.rbm_stack[name].bias_v    = np.load("%s/rbm.%s.bias_v.npy"%(loc,name))
        self.rbm_stack[name].bias_h    = np.load("%s/rbm.%s.bias_h.npy"%(loc,name))
        print ("loaded rbm[%s] from %s"%(name,loc))
        return
        
    def savetofile_rbm(self,loc,name):
        
        np.save("%s/rbm.%s.weight_vh"%(loc,name), self.rbm_stack[name].weight_vh)
        np.save("%s/rbm.%s.bias_v"%(loc,name),    self.rbm_stack[name].bias_v)
        np.save("%s/rbm.%s.bias_h"%(loc,name),    self.rbm_stack[name].bias_h)
        return
    
    def loadfromfile_dbn(self,loc,name):
        
        self.rbm_stack[name].weight_v_to_h = np.load("%s/dbn.%s.weight_v_to_h.npy"%(loc,name))
        self.rbm_stack[name].weight_h_to_v = np.load("%s/dbn.%s.weight_h_to_v.npy"%(loc,name))
        self.rbm_stack[name].bias_v        = np.load("%s/dbn.%s.bias_v.npy"%(loc,name))
        self.rbm_stack[name].bias_h        = np.load("%s/dbn.%s.bias_h.npy"%(loc,name))
        print ("loaded rbm[%s] from %s"%(name,loc))
        return
        
    def savetofile_dbn(self,loc,name):
        
        np.save("%s/dbn.%s.weight_v_to_h"%(loc,name), self.rbm_stack[name].weight_v_to_h)
        np.save("%s/dbn.%s.weight_h_to_v"%(loc,name), self.rbm_stack[name].weight_h_to_v)
        np.save("%s/dbn.%s.bias_v"%(loc,name),        self.rbm_stack[name].bias_v)
        np.save("%s/dbn.%s.bias_h"%(loc,name),        self.rbm_stack[name].bias_h)
        return
