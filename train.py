

import os
import re
import glob
import math
import random
import shutil
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import argparse
from tqdm import tqdm

from utils.data_processing import DataAugmentation
from utils.ds_reader import Dataset_Loader, get_dataset_series_names
from model.models import TransformerEncoderModel, WeakDecoder, Discriminator
from utils.config import read_config
from model.loss import (
    ReconstructionLoss, DiscriminatorLoss, AdversarialLoss,
    ReconWeightedInfoNCELoss
)

warnings.filterwarnings("ignore", category=UserWarning, message="KMeans is known to have a memory leak")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_config_snapshot(config_path: str, config_dict: dict, output_dir: str):
    """
    把“原始配置文件”和“解析后的配置快照”写到同一存档目录，保证可复现。
    - 原始文件：原样拷贝（如果路径存在）
    - 快照：config_used.yaml（用 safe_dump 写出解析后的 dict）
    """
    os.makedirs(output_dir, exist_ok=True)


    try:
        if config_path and os.path.isfile(config_path):
            dst = os.path.join(output_dir, os.path.basename(config_path))
            shutil.copy2(config_path, dst)
            print(f"[Config] raw config copied to: {dst}")
        else:
            print(f"[Config] raw config path not found, skip copy: {config_path}")
    except Exception as e:
        print(f"[Config] raw config copy failed: {e}")


    try:
        snap_path = os.path.join(output_dir, "config_used.yaml")
        with open(snap_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(config_dict, f, allow_unicode=True, sort_keys=False)
        print(f"[Config] parsed config snapshot saved to: {snap_path}")
    except Exception as e:
        print(f"[Config] config snapshot dump failed: {e}")


class DBPM:
    """
    Dynamic Bad Pair Mining (DBPM)
    """
    def __init__(self, num_samples, max_epochs, beta_np=1.5, beta_fp=1.5, warmup_epochs=5):
        self.num_samples = num_samples
        self.max_epochs = max_epochs
        self.beta_np = beta_np
        self.beta_fp = beta_fp
        self.warmup_epochs = warmup_epochs
        self.memory = np.zeros((num_samples, max_epochs), dtype=np.float32)

    def update_memory(self, sample_losses, epoch, batch_start_idx=None, indices=None):
        sample_losses = np.asarray(sample_losses, dtype=np.float32)

        if indices is not None:
            idx = np.asarray(indices, dtype=np.int64)
            mask = (idx >= 0) & (idx < self.num_samples)
            if np.any(mask):
                self.memory[idx[mask], epoch] = sample_losses[mask]
            return

        if batch_start_idx is None:
            raise ValueError("Either indices or batch_start_idx must be provided.")

        batch_size = len(sample_losses)
        end_idx = min(int(batch_start_idx) + batch_size, self.num_samples)
        if end_idx > batch_start_idx:
            self.memory[int(batch_start_idx):end_idx, epoch] = sample_losses[:(end_idx - int(batch_start_idx))]

    def compute_historical_stats(self, epoch):
        if epoch <= 0:
            return None, None, None
        mean_losses = np.mean(self.memory[:, :epoch], axis=1)
        global_mean = float(np.mean(mean_losses))
        global_std = float(np.std(mean_losses))
        return mean_losses, global_mean, global_std

    def compute_weights(self, current_losses, epoch):
        current_losses = np.asarray(current_losses, dtype=np.float64)
        if epoch < self.warmup_epochs:
            return np.ones(len(current_losses), dtype=np.float64)

        _, global_mean, global_std = self.compute_historical_stats(epoch)
        if global_mean is None or global_std is None or global_std == 0:
            return np.ones(len(current_losses), dtype=np.float64)

        z = (current_losses - global_mean) / (global_std + 1e-12)
        weights = (1.0 / (global_std * math.sqrt(2 * math.pi))) * np.exp(-0.5 * (z ** 2))
        return weights


def train(config_path='./config.yaml'):
    set_seed(42)
    config = read_config(config_path)

    exp_config = config.get('experiment', {})
    exp_name = exp_config.get('name', 'default_exp')
    output_dir = os.path.join('./experiments', f"{exp_name}")
    os.makedirs(output_dir, exist_ok=True)

    print(f"--- Experiment Setup ---\nExperiment Name: {exp_name}\nAll outputs will be saved to: {output_dir}\n----------------------")


    save_config_snapshot(config_path, config, output_dir)

    device = torch.device(config['training']['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    num_workers = int(config.get('training', {}).get('num_workers', 0))
    pin_memory = bool(config.get('training', {}).get('pin_memory', True))
    persistent_workers = bool(config.get('training', {}).get('persistent_workers', num_workers > 0))

    ts_name_list = get_dataset_series_names(
        config['data']['dataset_name'],
        config['data'].get('data_path')
    )
    ts_name_filter = config.get('data', {}).get('ts_name_filter', None)
    if ts_name_filter:
        wanted = {str(x) for x in ts_name_filter}
        ts_name_list = [name for name in ts_name_list if name in wanted]
        print(f"[Data] ts_name_filter enabled: {len(ts_name_list)} selected")
    print(f"[Data] using {len(ts_name_list)} series from utils.ds_reader for dataset={config['data']['dataset_name']}")
    all_series_names = get_dataset_series_names(
        config['data']['dataset_name'],
        config['data'].get('data_path')
    )
    series_to_idx = {name: idx for idx, name in enumerate(all_series_names)}
    for loop_idx, current_ts_name in enumerate(ts_name_list):
        ts_idx = series_to_idx[current_ts_name]
        print(f"\n===== Starting Training for {current_ts_name} (Dataset {loop_idx+1}/{len(ts_name_list)}) =====")

        dataset_instance = Dataset_Loader(
            dataset=config['data']['dataset_name'],
            data_path=config['data']['data_path'],
            ts_num=ts_idx,
            window_size=config['data']['window_size'],
            step_size=config['data'].get('step_size', 1)
        )

        shuffle_train = bool(config['training'].get('shuffle', True))
        try:
            train_loader = dataset_instance.train_loader_generation(
                batch_size=config['training']['batch_size'],
                shuffle=shuffle_train,
                return_indices=True,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers
            )
            indices_supported = True
        except TypeError:
            indices_supported = False
            if shuffle_train:
                print("[Warn] train_loader_generation 不支持 return_indices；已强制 shuffle=False 以避免 DBPM 记忆错位。")
                shuffle_train = False
            train_loader = dataset_instance.train_loader_generation(
                batch_size=config['training']['batch_size'],
                shuffle=shuffle_train
            )

        dbpm = DBPM(
            num_samples=len(train_loader.dataset),
            max_epochs=int(config['training']['num_epochs']),
            **config.get('dbpm', {})
        )

        data_aug = DataAugmentation(config)

        synthetic_negatives_enabled = True
        in_batch_negatives_enabled = True


        input_dim = int(config['model']['encoder']['input_dim'])
        projection_dim = int(config['model']['encoder']['proj_dim'])
        hidden_dim = int(config['model']['encoder']['hidden_dim'])
        num_layers = int(config['model']['encoder']['num_layers'])
        nhead = int(config['model']['encoder']['nhead'])
        dim_feedforward = int(config['model']['encoder']['dim_feedforward'])
        dropout = float(config['model']['encoder']['dropout'])

        projection_layer = nn.Linear(input_dim, projection_dim).to(device)
        encoder = TransformerEncoderModel(projection_dim, hidden_dim, nhead, num_layers, dim_feedforward, dropout).to(device)
        decoder = WeakDecoder(hidden_dim=hidden_dim, output_dim=input_dim).to(device)

        discriminator = Discriminator(
            input_dim=hidden_dim,
            hidden_dim=int(config['model']['discriminator']['hidden_dim']),
            output_dim=int(config['model']['discriminator']['output_dim'])
        ).to(device)


        lr = float(config['training']['learning_rate'])
        wd = float(config['training']['weight_decay'])

        optimizer_E_D = optim.AdamW(
            list(projection_layer.parameters()) + list(encoder.parameters()) + list(decoder.parameters()),
            lr=lr, weight_decay=wd
        )
        optimizer_D = optim.AdamW(
            discriminator.parameters(),
            lr=lr, weight_decay=wd
        )

        scheduler_E_D = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_E_D, T_max=int(config['training']['num_epochs']))
        scheduler_D = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_D, T_max=int(config['training']['num_epochs']))

        recon_loss_fn = ReconstructionLoss().to(device)
        disc_loss_fn = DiscriminatorLoss().to(device)
        adv_loss_fn = AdversarialLoss().to(device)

        print("Using the release main objective: reconstruction + reconstruction-guided contrastive + adversarial loss.")
        contrastive_loss_fn = ReconWeightedInfoNCELoss(
            temperature=float(config['contrastive']['temperature']),
            margin=float(config['contrastive']['recon_margin'])
        ).to(device)

        use_adv = True


        checkpoint_pattern = os.path.join(output_dir, f"{current_ts_name}_epoch_*.pth")
        all_ckpts = glob.glob(checkpoint_pattern)
        start_epoch = 0

        if len(all_ckpts) > 0:
            epochs_found = []
            for p in all_ckpts:
                m = re.search(rf"{re.escape(current_ts_name)}_epoch_(\d+)\.pth", os.path.basename(p))
                if m:
                    epochs_found.append(int(m.group(1)))
            if len(epochs_found) > 0:
                latest_epoch = max(epochs_found)
                ckpt_path = os.path.join(output_dir, f"{current_ts_name}_epoch_{latest_epoch}.pth")
                print(f"Found existing checkpoint: {ckpt_path}. Resuming from epoch {latest_epoch}.")

                checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

                if 'projection_layer_state_dict' in checkpoint:
                    projection_layer.load_state_dict(checkpoint['projection_layer_state_dict'])
                else:
                    print("Warning: 'projection_layer_state_dict' not found in checkpoint. It will be randomly initialized.")

                encoder.load_state_dict(checkpoint['encoder_state_dict'])
                decoder.load_state_dict(checkpoint['decoder_state_dict'])
                discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
                optimizer_E_D.load_state_dict(checkpoint['optimizer_E_D_state_dict'])
                optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])

                if 'dbpm_memory' in checkpoint:
                    old_memory = checkpoint['dbpm_memory']
                    old_samples, old_epochs = old_memory.shape
                    num_epochs = int(config['training']['num_epochs'])
                    train_dataset_size = len(train_loader.dataset)

                    if old_samples != train_dataset_size or old_epochs < num_epochs:
                        print(f"Adjusting DBPM memory size from ({old_samples}, {old_epochs}) to ({train_dataset_size}, {num_epochs})")
                        new_memory = np.zeros((train_dataset_size, num_epochs), dtype=np.float32)
                        copy_samples = min(old_samples, train_dataset_size)
                        copy_epochs = min(old_epochs, num_epochs)
                        new_memory[:copy_samples, :copy_epochs] = old_memory[:copy_samples, :copy_epochs]
                        dbpm.memory = new_memory
                    else:
                        dbpm.memory = old_memory.astype(np.float32)

                start_epoch = latest_epoch


        num_epochs_total = int(config['training']['num_epochs'])
        mask_rates = config.get('acae_mask_rates', [0.05, 0.15, 0.3])

        for epoch in range(start_epoch, num_epochs_total):
            actual_epoch = epoch + 1
            projection_layer.train()
            encoder.train()
            decoder.train()

            discriminator.train()

            total_losses = {k: 0.0 for k in ['recon', 'disc', 'adv', 'con']}

            lambda_rec = float(config['training'].get('rec', 0.0))
            lambda_con = float(config['training'].get('con', 0.0))
            lambda_adv = float(config['training'].get('adv', 0.0))

            print(f"Epoch {actual_epoch} | Lambdas -> Rec: {lambda_rec:.2f}, Con: {lambda_con:.4f}, Adv: {lambda_adv:.2f}")

            for batch_idx, batch_pack in enumerate(tqdm(train_loader, desc=f"Epoch {actual_epoch}")):
                if isinstance(batch_pack, (list, tuple)) and len(batch_pack) == 3:
                    batch, _, indices = batch_pack
                    indices_np = indices.detach().cpu().numpy()
                else:
                    batch, _ = batch_pack
                    indices_np = None


                batch_cpu = batch.permute(0, 2, 1).contiguous()
                batch_for_model = batch_cpu.to(device, non_blocking=True)

                projected_anchor = projection_layer(batch_for_model)


                projected_positive = projected_anchor.clone()
                B, W, D = projected_positive.shape
                mr = torch.tensor(mask_rates, device=device, dtype=torch.float32)
                p = mr[torch.randint(0, mr.numel(), (B,), device=device)]
                mask = (torch.rand((B, W), device=device) < p.unsqueeze(1)).unsqueeze(-1)
                projected_positive = projected_positive.masked_fill(mask, 0.0)

                disc_loss = torch.tensor(0.0, device=device)
                projected_anomalous = None
                anomalous_samples = None

                if synthetic_negatives_enabled:

                    _, anomalous_samples_np = data_aug(batch_cpu.numpy())
                    if anomalous_samples_np is not None and len(anomalous_samples_np) > 0:
                        anomalous_np = np.asarray(anomalous_samples_np, dtype=np.float32)
                        anomalous_samples = torch.from_numpy(anomalous_np).to(device, non_blocking=True)
                        projected_anomalous = projection_layer(anomalous_samples)


                if use_adv and lambda_adv > 0 and (projected_anomalous is not None):
                    optimizer_D.zero_grad()
                    with torch.no_grad():
                        real_emb = torch.mean(encoder(projected_anchor), dim=1)
                        fake_emb = torch.mean(encoder(projected_anomalous), dim=1)
                    disc_output_real = discriminator(real_emb)
                    disc_output_fake = discriminator(fake_emb)
                    loss_real = disc_loss_fn(disc_output_real, torch.ones_like(disc_output_real))
                    loss_fake = disc_loss_fn(disc_output_fake, torch.zeros_like(disc_output_fake))
                    disc_loss = (loss_real + loss_fake) / 2.0
                    disc_loss.backward()
                    optimizer_D.step()


                optimizer_E_D.zero_grad()

                encoded_anchor = encoder(projected_anchor)
                encoded_positive = encoder(projected_positive)
                encoded_negative = encoder(projected_anomalous) if (synthetic_negatives_enabled and projected_anomalous is not None) else None

                reconstructed_anchor = decoder(encoded_anchor)
                recon_loss = recon_loss_fn(reconstructed_anchor, batch_for_model).mean()

                anchor_emb = nn.functional.normalize(torch.mean(encoded_anchor, dim=1), p=2, dim=1)
                positive_emb = nn.functional.normalize(torch.mean(encoded_positive, dim=1), p=2, dim=1)

                all_negative_embs_list = []
                all_negative_recon_errors_list = []


                if in_batch_negatives_enabled:
                    all_negative_embs_list.append(anchor_emb)
                    with torch.no_grad():
                        recon_error_in_batch = torch.mean(recon_loss_fn(reconstructed_anchor, batch_for_model), dim=(1, 2))
                    all_negative_recon_errors_list.append(recon_error_in_batch)


                if synthetic_negatives_enabled and (encoded_negative is not None) and (anomalous_samples is not None):
                    negative_emb = nn.functional.normalize(torch.mean(encoded_negative, dim=1), p=2, dim=1)
                    all_negative_embs_list.append(negative_emb)
                    with torch.no_grad():
                        recon_error_external = torch.mean(recon_loss_fn(decoder(encoded_negative), anomalous_samples), dim=(1, 2))
                    all_negative_recon_errors_list.append(recon_error_external)


                if len(all_negative_embs_list) > 0:
                    final_negative_embs = torch.cat(all_negative_embs_list, dim=0)
                    final_negative_recon_errors = torch.cat(all_negative_recon_errors_list, dim=0)

                    contrastive_losses_per_sample = contrastive_loss_fn(
                        anchor_emb, positive_emb,
                        final_negative_embs,
                        final_negative_recon_errors.detach()
                    )
                else:
                    contrastive_losses_per_sample = torch.zeros((anchor_emb.shape[0],), device=device)


                cl_np = contrastive_losses_per_sample.detach().cpu().numpy()
                if indices_np is not None:
                    dbpm.update_memory(cl_np, epoch, indices=indices_np)
                else:
                    dbpm.update_memory(cl_np, epoch, batch_start_idx=batch_idx * train_loader.batch_size)

                if epoch >= dbpm.warmup_epochs:
                    w = dbpm.compute_weights(cl_np, epoch)
                    weights = torch.from_numpy(w).float().to(device)
                    contrastive_loss = (contrastive_losses_per_sample * weights).mean()
                else:
                    contrastive_loss = contrastive_losses_per_sample.mean()

                adv_loss = torch.tensor(0.0, device=device)
                if use_adv and lambda_adv > 0:
                    disc_output_adv = discriminator(torch.mean(encoded_anchor, dim=1))
                    adv_loss = adv_loss_fn(disc_output_adv, torch.ones_like(disc_output_adv))

                total_loss = lambda_rec * recon_loss + lambda_con * contrastive_loss + lambda_adv * adv_loss
                total_loss.backward()
                optimizer_E_D.step()

                total_losses['recon'] += float(recon_loss.item())
                total_losses['con'] += float(contrastive_loss.item())
                total_losses['adv'] += float(adv_loss.item())
                total_losses['disc'] += float(disc_loss.item())

            scheduler_E_D.step()
            if use_adv:
                scheduler_D.step()

            print(
                f"  Avg Losses -> Recon: {total_losses['recon']/len(train_loader):.4f}, "
                f"Contrastive: {total_losses['con']/len(train_loader):.4f}, "
                f"Discriminator: {total_losses['disc']/len(train_loader):.4f}, "
                f"Adversarial: {total_losses['adv']/len(train_loader):.4f}"
            )


            save_path = os.path.join(output_dir, f"{current_ts_name}_epoch_{actual_epoch}.pth")
            torch.save({
                'projection_layer_state_dict': projection_layer.state_dict(),
                'encoder_state_dict': encoder.state_dict(),
                'decoder_state_dict': decoder.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'optimizer_E_D_state_dict': optimizer_E_D.state_dict(),
                'optimizer_D_state_dict': optimizer_D.state_dict(),
                'dbpm_memory': dbpm.memory
            }, save_path)
            print(f"Model saved to {save_path}")


        final_path = os.path.join(output_dir, f"{current_ts_name}_final.pth")
        torch.save({
            'projection_layer_state_dict': projection_layer.state_dict(),
            'encoder_state_dict': encoder.state_dict(),
            'decoder_state_dict': decoder.state_dict(),
            'discriminator_state_dict': discriminator.state_dict(),
            'optimizer_E_D_state_dict': optimizer_E_D.state_dict(),
            'optimizer_D_state_dict': optimizer_D.state_dict(),
            'dbpm_memory': dbpm.memory
        }, final_path)
        print(f"[Done] {current_ts_name} final checkpoint saved to {final_path}")


    save_config_snapshot(config_path, config, output_dir)
    print(f"\nAll training finished. Artifacts saved in: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./config.yaml")
    args = parser.parse_args()
    train(config_path=args.config)
