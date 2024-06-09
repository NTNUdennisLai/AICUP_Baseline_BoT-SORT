import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
# from torch.backends import cudnn
import torch.nn as nn

from fast_reid.fastreid.config import get_cfg
from fast_reid.fastreid.modeling.meta_arch import build_model
from fast_reid.fastreid.utils.checkpoint import Checkpointer
from fast_reid.fastreid.engine import DefaultTrainer, default_argument_parser, default_setup, launch

# cudnn.benchmark = True


def setup_cfg(config_file, opts):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(config_file)
    cfg.merge_from_list(opts)
    cfg.MODEL.BACKBONE.PRETRAIN = False

    cfg.freeze()

    return cfg


def postprocess(features):
    # Normalize feature to compute cosine distance
    features = F.normalize(features)
    features = features.cpu().data.numpy()
    return features


def preprocess(image, input_size):
    if len(image.shape) == 3:
        padded_img = np.ones((input_size[1], input_size[0], 3), dtype=np.uint8) * 114
    else:
        padded_img = np.ones(input_size) * 114
    img = np.array(image)
    r = min(input_size[1] / img.shape[0], input_size[0] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    )
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    return padded_img, r


class FastReIDInterface:
    def __init__(self, config_file, weights_path, device, batch_size=8):
        super(FastReIDInterface, self).__init__()
        if device != 'cpu':
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        self.batch_size = batch_size

        self.cfg = setup_cfg(config_file, ['MODEL.WEIGHTS', weights_path])

        self.model = build_model(self.cfg)
        self.model.eval()

        Checkpointer(self.model).load(weights_path)

        if self.device != 'cpu':
            self.model = self.model.eval().to(device='cuda').half()
        else:
            self.model = self.model.eval()

        self.pH, self.pW = self.cfg.INPUT.SIZE_TEST

    def inference(self, image, detections):

        if detections is None or np.size(detections) == 0:
            return []

        H, W, _ = np.shape(image)

        batch_patches = []
        patches = []
        patch_flows = []
        for d in range(np.size(detections, 0)):
            tlbr = detections[d, :4].astype(np.int_)

            # print('tlbr', tlbr)
            # # Load the flow data
            # flow_data = np.load('/home/F0970996972/pytorch/yolo_v8_file/aicup_data/AICUP_Baseline_BoT-SORT/fast_reid/optflow/cluster_centers_0902_150000_151900_cam_0.npy')

            # # Define the bounding box [x_min, y_min, x_max, y_max]
            # bbox = tlbr

            # # Extract the region of interest from the flow data
            # # Note: bbox is [x_min, y_min, x_max, y_max]
            # roi_flow_data = flow_data[bbox[1]:bbox[3], bbox[0]:bbox[2]]

            # # Compute the mean across all pixels and flow vectors in this region
            # # axis=(0, 1) to average over both pixel dimensions, retaining the 3 flow centers and their 2 components
            # average_flow_vectors = np.mean(roi_flow_data, axis=(0, 1))

            # # print("Average Flow Vectors within the specified bounding box:", average_flow_vectors)
            # flattened_vector = average_flow_vectors.flatten()
            # # print('flattened_vector', flattened_vector)
            # flattened_tensor = torch.tensor(flattened_vector, dtype=torch.float32)
            # # print('flattened_tensor', flattened_tensor)
            # # flattened_vector = torch.tensor(flattened_vector)
            # linear_layer = nn.Linear(in_features=6, out_features=2048)
            # output_vector = linear_layer(flattened_tensor)
            # # print('output_vector', output_vector)
            # patch_flows.append(flattened_tensor)

            tlbr[0] = max(0, tlbr[0])
            tlbr[1] = max(0, tlbr[1])
            tlbr[2] = min(W - 1, tlbr[2])
            tlbr[3] = min(H - 1, tlbr[3])
            patch = image[tlbr[1]:tlbr[3], tlbr[0]:tlbr[2], :]
            # the model expects RGB inputs
            patch = patch[:, :, ::-1]

            # Apply pre-processing to image.
            patch = cv2.resize(patch, tuple(self.cfg.INPUT.SIZE_TEST[::-1]), interpolation=cv2.INTER_LINEAR)
            # patch, scale = preprocess(patch, self.cfg.INPUT.SIZE_TEST[::-1])

            # plt.figure()
            # plt.imshow(patch)
            # plt.show()

            # Make shape with a new batch dimension which is adapted for network input
            patch = torch.as_tensor(patch.astype("float32").transpose(2, 0, 1))
            patch = patch.to(device=self.device).half()

            patches.append(patch)

            if (d + 1) % self.batch_size == 0:
                patches = torch.stack(patches, dim=0)
                batch_patches.append(patches)
                patches = []

        if len(patches):
            patches = torch.stack(patches, dim=0)
            batch_patches.append(patches)

        # features = np.zeros((0, 2054))
        features = np.zeros((0, 2048))
        # features = np.zeros((0, 768))

        for idx, patches in enumerate(batch_patches):

            # Run model
            patches_ = torch.clone(patches)
            # print('patches shape', patches.shape)
            # print('idx', idx)
            pred = self.model(patches)
            pred[torch.isinf(pred)] = 1.0

            feat = postprocess(pred)
            # print(feat.shape)
            # print(len(feat))

            # feat_tensor = torch.tensor(feat, dtype=torch.float32)
            # print('patch len', len(patch_flows))
            # # opt_feature = torch.tensor(patch_flows, dtype=torch.float32)
            # opt_feature = torch.tensor([item.cpu().detach().numpy() for item in patch_flows])
            # if opt_feature.dim() == 1:
            #     opt_feature = opt_feature.unsqueeze(0)  # Adds a new dimension at index 0
            # if feat_tensor.dim() == 1:
            #     feat_tensor = feat_tensor.unsqueeze(0)  # Adds a new dimension at index 0

            # combined_feat = torch.cat((opt_feature, feat_tensor), dim=1)
            # combined_feat_np = combined_feat.numpy()

            nans = np.isnan(np.sum(feat, axis=1))
            if np.isnan(feat).any():
                for n in range(np.size(nans)):
                    if nans[n]:
                        # patch_np = patches[n, ...].squeeze().transpose(1, 2, 0).cpu().numpy()
                        patch_np = patches_[n, ...]
                        patch_np_ = torch.unsqueeze(patch_np, 0)
                        pred_ = self.model(patch_np_)

                        patch_np = torch.squeeze(patch_np).cpu()
                        patch_np = torch.permute(patch_np, (1, 2, 0)).int()
                        patch_np = patch_np.numpy()

                        plt.figure()
                        plt.imshow(patch_np)
                        plt.show()

            # features = np.vstack((features, combined_feat_np))
            features = np.vstack((features, feat))
            # print(features)
        return features

