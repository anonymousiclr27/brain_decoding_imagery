from pathlib import Path

import clip
import numpy as np
import scipy as sp
import torch
import torchvision.models as tvmodels
import torchvision.transforms as transforms
import torchvision.transforms as T
from PIL import Image
from scipy.stats import binom
from skimage.color import rgb2gray
from skimage.metrics import structural_similarity as ssim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# Network cache: without it, every call rebuilds and re-downloads six models and
# registers the hooks again. The key includes the device. The hook is registered
# once: `fn` writes into the global feat_list, which is cleared on every call.
_NET_CACHE = {}

def _compute_image_generation_features(images, emb_batch_size=32, device="cuda"):
    class batch_generator_external_images(Dataset):
        def __init__(self, images: list, net_name="clip"):
            self.images = images
            self.net_name = net_name

            if self.net_name == "clip":
                self.normalize = transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                )
            else:
                self.normalize = transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                )

        def __getitem__(self, idx):
            img = self.images[idx]
            img = T.functional.resize(img, (224, 224))
            img = T.functional.to_tensor(img).float()
            img = self.normalize(img)
            return img

        def __len__(self):
            return len(self.images)

    global feat_list
    feat_list = []

    def fn(module, inputs, outputs):
        feat_list.append(outputs.cpu().numpy())

    # solo le reti che ti servono
    net_list = [
        ("inceptionv3", "avgpool"),
        ("clip", "final"),
        ("alexnet", 2),
        ("alexnet", 5),
        ("efficientnet", "avgpool"),
        ("swav", "avgpool"),
    ]

    result = dict()

    for net_name, layer in net_list:
        feat_list = []
        dataset = batch_generator_external_images(images=images, net_name=net_name)
        loader = DataLoader(dataset, emb_batch_size, shuffle=False)

        chiave = (net_name, layer, str(device))
        if chiave in _NET_CACHE:
            net = _NET_CACHE[chiave]
        else:
            if net_name == "inceptionv3":
                net = tvmodels.inception_v3(pretrained=True)
                if layer == "avgpool":
                    net.avgpool.register_forward_hook(fn)

            elif net_name == "alexnet":
                net = tvmodels.alexnet(pretrained=True)
                if layer == 2:
                    net.features[4].register_forward_hook(fn)
                elif layer == 5:
                    net.features[11].register_forward_hook(fn)

            elif net_name == "clip":
                model, _ = clip.load("ViT-L/14", device=device)
                net = model.visual
                net = net.to(torch.float32)
                if layer == "final":
                    net.register_forward_hook(fn)

            elif net_name == "efficientnet":
                net = tvmodels.efficientnet_b1(weights=True)
                net.avgpool.register_forward_hook(fn)

            elif net_name == "swav":
                net = torch.hub.load("facebookresearch/swav:main", "resnet50")
                net.avgpool.register_forward_hook(fn)

            net = net.to(device)
            net.eval()
            _NET_CACHE[chiave] = net

        with torch.no_grad():
            for i, x in tqdm(enumerate(loader), total=len(loader)):
                x = x.to(device)
                _ = net(x)

        if net_name == "clip":
            feat_list = np.concatenate(feat_list)
        else:
            feat_list = np.concatenate(feat_list)

        result[f"{net_name}-{layer}"] = feat_list

    return result


def _pairwise_corr_all(ground_truth, predictions):
    r = np.corrcoef(ground_truth, predictions)
    r = r[: len(ground_truth), len(ground_truth) :]
    congruents = np.diag(r)

    success = r < congruents
    success_cnt = np.sum(success, 0)

    perf = np.mean(success_cnt) / (len(ground_truth) - 1)
    p = 1 - binom.cdf(
        perf * len(ground_truth) * (len(ground_truth) - 1),
        len(ground_truth) * (len(ground_truth) - 1),
        0.5,
    )

    return perf, p


def compute_image_generation_metrics(
    preds,
    trues,
    imsize_for_pixel_level_metrics=425,
    emb_batch_size=32,
    device="cuda",
):
    result = dict()

    assert len(preds) == len(trues)
    n = len(preds)

    # estraggo feature
    all_images = trues + preds
    feats = _compute_image_generation_features(
        all_images, emb_batch_size=emb_batch_size, device=device
    )

    gt_feats = {k: feats[k][:n] for k in feats}
    eval_feats = {k: feats[k][n:] for k in feats}

    distance_fn = sp.spatial.distance.correlation
    for metric_name in gt_feats.keys():
        gt_feat = gt_feats[metric_name].reshape((n, -1))
        eval_feat = eval_feats[metric_name].reshape((n, -1))

        net_name, _ = metric_name.split("-")

        if net_name in ["efficientnet", "swav"]:
            distances = np.array(
                [distance_fn(gt_feat[i], eval_feat[i]) for i in range(n)]
            )
            result[metric_name] = distances.mean()
        else:
            result[metric_name] = _pairwise_corr_all(gt_feat, eval_feat)[0]

    # metriche pixel-level
    ssim_list = []
    pixcorr_list = []
    for i in range(n):
        gen_image = preds[i].resize(
            (imsize_for_pixel_level_metrics, imsize_for_pixel_level_metrics)
        )
        gt_image = trues[i].resize(
            (imsize_for_pixel_level_metrics, imsize_for_pixel_level_metrics)
        )

        gen_image = np.array(gen_image) / 255.0
        gt_image = np.array(gt_image) / 255.0

        pixcorr_res = np.corrcoef(gt_image.reshape(-1), gen_image.reshape(-1))[0, 1]
        pixcorr_list.append(pixcorr_res)

        ssim_res = ssim(
            rgb2gray(gen_image),
            rgb2gray(gt_image),
            gaussian_weights=True,
            sigma=1.5,
            use_sample_covariance=False,
            data_range=1.0,
        )
        ssim_list.append(ssim_res)

    result["pixcorr"] = np.mean(pixcorr_list)
    result["ssim"] = np.mean(ssim_list)

    return result
