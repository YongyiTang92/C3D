import numpy as np
import torch
from C3D_models import R3D_34, R21D_34
import torch.nn as nn
from torch.autograd import Variable


class C3D_basic(object):
    def __init__(self, FLAGS, data_type='rgb'):
        self.FLAGS = FLAGS
        if FLAGS.model == 'R3D_34':
            self.model = R3D_34()
        elif FLAGS.model == 'R21D_34':
            self.model = R21D_34()
        else:
            raise('Error model type: ', FLAGS.model)

        self.data_type = data_type
        self.max_gradient_norm = FLAGS.max_gradient_norm
        self.loss = nn.CrossEntropyLoss()
        self.learning_rate = FLAGS.lr
        self.set_optimizer(self.learning_rate)
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            self.loss = self.loss.cuda()

    def set_optimizer(self, lr):
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=lr, momentum=self.FLAGS.momentum, weight_decay=self.FLAGS.weight_decay)

    def train_step(self, image_tensor, label_tensor, forward_only):
        """
        Inputs:
            image_tensor: (batch_size, in_channel, frames_per_clip, 112, 112) # one TenCroped sample
            label_tensor: (batch_size, 101)
            forward_only: True for train, False for evaluate
        Output:
            Loss:
        """
        self.model.zero_grad()
        if forward_only:
            self.model.eval()
        else:
            self.model.train()

        _, labels_index = torch.max(label_tensor, 1)
        labels_index = labels_index.long()
        img_var, label_var = self.to_variable(image_tensor), self.to_variable(labels_index)
        predict_score = self.model(img_var)
        loss = self.loss(predict_score, label_var[:])
        _, predict_labels = torch.max(predict_score, 1)
        correct = predict_labels.eq(label_var)

        if not forward_only:
            loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm(self.model.parameters(), self.max_gradient_norm)
            self.optimizer.step()
        return self.to_data(loss), self.to_data(correct)

    def test_step(self, image_tensor, label_tensor):
        """
        Inputs:
            image_tensor: (frames, 10, in_channel, frames_per_clip, 112, 112) # one TenCroped sample
            label_tensor: (frames, 10, 101)
        Output:
            Loss:
        """
        self.model.zero_grad()
        self.model.eval()
        _, labels_index = torch.max(label_tensor[0:1, 0, :], 1)  # The label for a video must the same
        labels_index = labels_index.long()
        img_var, label_var = self.to_variable(image_tensor), self.to_variable(labels_index)
        img_var = img_var.view(-1, img_var.size(2), img_var.size(3), img_var.size(4), img_var.size(5))
        predict_score = self.model(img_var)  # May OOM
        if self.FLAGS.pooling == 'max':
            predict_score_avg, _ = torch.max(predict_score, 0, keepdim=True)
        else:
            predict_score_avg = torch.mean(predict_score, 0, keepdim=True)
        loss = self.loss(predict_score_avg, label_var[:])
        _, predict_labels = torch.max(predict_score_avg, 1)
        correct = predict_labels.eq(label_var)
        return self.to_data(loss), self.to_data(correct)

    def to_variable(self, x, requires_grad=None):
        """Convert tensor to variable."""
        if torch.cuda.is_available():
            x = x.cuda()
        if requires_grad:
            return Variable(x, requires_grad=requires_grad)
        else:
            return Variable(x)

    def to_data(self, x):
        """Convert variable to tensor."""
        if torch.cuda.is_available():
            x = x.cpu()
        return x.data
