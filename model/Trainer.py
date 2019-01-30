import torch
from model.MuseGAN import MuseGAN, Discrimitor
import torch.autograd as autograd
from model.libs.utils import *
import time
import copy

class MuseGANTrainer:
    def __init__(self, trainer_type, device, z_inter_dim, z_intra_dim, track_dim, lmbda, print_batch=True):
        """ Object to hold data iterators, train the model """

        self.__dict__.update(locals())

        self.museGan = MuseGAN(trainer_type, track_dim, z_inter_dim, z_intra_dim).to(device)
        self.discriminator = Discrimitor(trainer_type).to(device)

        g_parameters_n = self.count_parameters(self.museGan)
        d_parameters_n = self.count_parameters(self.discriminator)

        print('# of parameters in G (generator)                 |', g_parameters_n)
        print('# of parameters in D (discriminator)             |', d_parameters_n)

    def count_parameters(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def generate_inter_intra(self, batch_size):
        z_inter = torch.normal(torch.zeros(batch_size, self.z_inter_dim), 0.1).to(self.device)
        z_intra = torch.normal(torch.zeros(batch_size, self.z_intra_dim, self.track_dim), 0.1).to(self.device)

        return z_inter, z_intra

    def generate_inter_intra_control(self, batch_size):

        control_arr = [15, 40, 43, 10, 19, 9]
        z_inter_arr = []
        z_intra_arr = []
        for item in control_arr:
            z_inter = torch.normal(torch.zeros(1, self.z_inter_dim), 0.1).to(self.device)
            z_intra = torch.normal(torch.zeros(1, self.z_intra_dim, self.track_dim), 0.1).to(self.device)

            z_inter = z_inter.repeat((item, 1))
            z_intra = z_intra.repeat((item, 1, 1))

            z_inter_arr.append(z_inter)
            z_intra_arr.append(z_intra)

        z_inter = torch.cat(z_inter_arr)
        z_intra = torch.cat(z_intra_arr)

        return z_inter, z_intra


    def train(self, train_iter, num_epochs, model_path='', lr = 2e-4):

        optimizerG = torch.optim.Adam(self.museGan.parameters(), lr=lr, betas=(0.5, 0.9))
        optimizerD = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(0.5, 0.9))

        counter = 0
        for epoch in range(num_epochs):

            print('{:-^120}'.format(' Epoch {} Start '.format(epoch)))
            num_batch = len(train_iter)
            for batch_idx, batch in enumerate(train_iter):

                batch = (batch[0].to(self.device), batch[1].to(self.device))

                batch_start_time = time.time()

                num_iters_D = 100 if counter < 25 or counter % 500 == 0 else 5

                for j in range(num_iters_D):

                    optimizerD.zero_grad()
                    d_loss, gp = self.train_D(batch)

                    d_loss.backward()
                    optimizerD.step()
                    # Clip weights of discriminator
                    # for p in self.discriminator.parameters():
                    #     p.data.clamp_(-1, 1)

                optimizerG.zero_grad()
                g_loss = self.train_G(batch)

                g_loss.backward()
                optimizerG.step()


                if self.print_batch:
                    print('---{}--- epoch: {:2d} | batch: {:4d}/{:4d} | time: {:6.2f} ' \
                          .format('test', epoch,
                                  batch_idx, num_batch, time.time() - batch_start_time))
                    print('GP: %6.2f, D loss: %6.2f, G loss: %6.2f' % (gp, -d_loss.item(), g_loss.item()))

                if counter%500==0:
                    self.run_sampler(batch, str(counter))

                counter += 1
            self.save_model('type%d_checkpoint_%d.ckpt' % (self.trainer_type, epoch))

        print('{:=^120}'.format(' Training End '))

    def gradient_penalty(self, real, fake, layer_conditions):
        batch_size = real.shape[0]

        alpha = torch.rand(batch_size, 1, 1, 1)
        alpha = alpha.expand_as(real).to(self.device)

        interpolates = (real + alpha * (fake - real)).to(self.device)

        interpolates = autograd.Variable(interpolates, requires_grad=True)

        disc_interpolates = self.discriminator(interpolates, layer_conditions)

        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=torch.ones(disc_interpolates.size()).to(self.device),
                                  create_graph=True, retain_graph=True)[0]

        gradients = gradients.view(batch_size, -1)

        slopes = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-8)

        gradient_penalty = ((slopes - 1) ** 2).mean() * self.lmbda

        return gradient_penalty

    def train_D(self, batch):
        for param in self.discriminator.parameters():
            param.requires_grad = True

        self.museGan.train()

        images, conditions = batch
        batch_size = images.shape[0]
        z_tuple = self.generate_inter_intra(batch_size)

        G_output, layer_conditions = self.museGan(z_tuple, conditions)
        G_output = G_output.detach()
        layer_conditions = [x.detach() for x in layer_conditions]

        DX_score = self.discriminator(images, layer_conditions)
        DG_score = self.discriminator(G_output, layer_conditions)

        gp = self.gradient_penalty(images, G_output, layer_conditions)

        d_loss = torch.mean(DG_score) - torch.mean(DX_score)
        d_loss += gp


        return d_loss, gp

    def train_G(self, batch):
        # to avoid extra computation
        for param in self.discriminator.parameters():
            param.requires_grad = False

        self.museGan.train()

        images, conditions = batch
        batch_size = images.shape[0]
        z_tuple = self.generate_inter_intra(batch_size)


        G_output, layer_conditions = self.museGan(z_tuple, conditions)

        DG_score = self.discriminator(G_output, layer_conditions)

        g_loss = -torch.mean(DG_score)

        return g_loss

    def run_sampler(self, batch, prefix='sample'):
        self.museGan.eval()
        images, conditions = batch
        batch_size = images.shape[0]
        z_tuple = self.generate_inter_intra(batch_size)

        G_output, layer_conditions = self.museGan(z_tuple, conditions)
        G_output = G_output.detach().permute(0, 2, 3, 1).cpu()

        G_output_binary = copy.deepcopy(G_output)
        G_output_binary[G_output_binary > 0] = 1
        G_output_binary[G_output_binary <= 0] = -1

        images = images.permute(0, 2, 3, 1).cpu()

        gen_dir = 'test'
        if not os.path.exists(gen_dir):
            os.makedirs(gen_dir)

        save_midis(G_output_binary, file_path=os.path.join(gen_dir, prefix + '.mid'))

        sample_shape = get_sample_shape(batch_size)
        save_bars(G_output, size=sample_shape, file_path=gen_dir, name=prefix, type_=0)
        save_bars(G_output_binary, size=sample_shape, file_path=gen_dir, name=prefix + '_binary', type_=0)
        save_bars(images, size=sample_shape, file_path=gen_dir, name=prefix + '_origin', type_=0)

    def eval_sampler(self, conditions, name="gen_song"):

        song_len = len(conditions)
        z_tuple = self.generate_inter_intra(song_len)

        outputs, _ = self.museGan(z_tuple, conditions)

        outputs = outputs.detach().permute(0, 2, 3, 1).cpu().numpy()
        outputs[outputs > 0] = 1
        outputs[outputs <= 0] = -1

        gen_dir = 'gen/type0'
        if self.trainer_type == 1:
            gen_dir = 'gen/type1'

        if not os.path.exists(gen_dir):
            os.makedirs(gen_dir)


        save_midis(outputs, file_path=os.path.join(gen_dir, name + '.mid'))

        outputs = ((outputs + 1).astype(np.float) / 2.0).astype(np.bool)
        np.save(os.path.join(gen_dir, name + '.npy'), outputs)

    def save_model(self, path):
        path = 'checkpoints/'+path
        torch.save(self.museGan.state_dict(), path)

    def load_model(self, path):
        self.museGan.load_state_dict(torch.load(path))
        self.museGan.eval()

