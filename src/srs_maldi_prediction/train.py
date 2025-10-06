"""Training script for MALDI prediction.

Usage: python train.py
"""
import os
import argparse
import torch
from torch.utils.data import DataLoader, random_split
from model import HyperPix2pixGenerator, HyperPix2pixDiscriminator
from datasets import PairedNPYDataset, CustomTransform
from losses import SAMLoss, CharbonnierLoss, compute_channel_correlation
import numpy as np
from torch.optim import Adam
from config import DEFAULTS, MODEL_DIR, DATA_DIR
import torch.nn as nn
from tqdm import tqdm


def train_entrypoint(data_image_files, data_target_files, checkpoint_dir=MODEL_DIR, device=None):
    """Convenience wrapper that sets up loaders, optimizers and calls the full `train(...)` implementation.

    Default scheduling and learning-rate settings follow the project's common defaults: StepLR(step_size=50, gamma=0.5),
    lr_G=1e-4, lr_D=4e-5, pretrain_epochs=DEFAULTS['pretrain_epochs'], num_epochs=DEFAULTS['num_epochs'], lambda_L2=1000.
    """
    device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    transform = CustomTransform()
    dataset = PairedNPYDataset(data_image_files, data_target_files, transform=transform)

    # Default model shape: generator expects 4-channel SRS input -> 100-channel MALDI output
    generator = HyperPix2pixGenerator(input_nc=4, output_nc=100, ngf=64, num_downs=4, use_dropout=True).to(device)
    # Discriminator concatenates SRS and MALDI: 4 + 100 = 104
    discriminator = HyperPix2pixDiscriminator(input_nc=104, ndf=64, n_layers=3).to(device)

    # split using a fixed seed for reproducibility
    rng = torch.Generator().manual_seed(42)
    trainset, valset = random_split(dataset, [int(len(dataset) * 0.85), len(dataset) - int(len(dataset) * 0.85)], generator=rng)
    train_loader = DataLoader(trainset, batch_size=DEFAULTS['batch_size'], shuffle=True, num_workers=4)
    val_loader = DataLoader(valset, batch_size=DEFAULTS['batch_size'], shuffle=False, num_workers=4)

    # losses and optimizers (default settings)
    criterion_GAN = torch.nn.BCEWithLogitsLoss()
    criterion_L2 = torch.nn.MSELoss()
    criterion_spec = SAMLoss()
    optimizer_G = Adam(generator.parameters(), lr=DEFAULTS.get('lr_G', 1e-4), betas=(0.5, 0.999))
    optimizer_D = Adam(discriminator.parameters(), lr=DEFAULTS.get('lr_D', 4e-5), betas=(0.5, 0.999))

    scheduler_G = torch.optim.lr_scheduler.StepLR(optimizer_G, step_size=50, gamma=0.5)
    scheduler_D = torch.optim.lr_scheduler.StepLR(optimizer_D, step_size=50, gamma=0.5)

    # default settings (with user's preference for lambda_L2)
    num_epochs = DEFAULTS.get('num_epochs', 200)
    pretrain_epochs = DEFAULTS.get('pretrain_epochs', 50)
    lambda_L2 = 500
    lambda_cosine = 1

    os.makedirs(checkpoint_dir, exist_ok=True)

    # call the full train loop
    train(generator=generator,
          discriminator=discriminator,
          train_loader=train_loader,
          valid_loader=val_loader,
          criterion_GAN=criterion_GAN,
          criterion_L2=criterion_L2,
          criterion_spec=criterion_spec,
          optimizer_G=optimizer_G,
          optimizer_D=optimizer_D,
          scheduler_G=scheduler_G,
          scheduler_D=scheduler_D,
          num_epochs=num_epochs,
          pretrain_epochs=pretrain_epochs,
          lambda_L2=lambda_L2,
          lambda_cosine=lambda_cosine,
          device=device,
          save_prefix='srs_hyperpix2pix')
# Note: `train_entrypoint` is a small convenience wrapper for quick programmatic runs.
# The fuller `train(...)` function and a complete CLI/example are defined below.


def train(generator, discriminator, train_loader, valid_loader, criterion_GAN, criterion_L2, criterion_spec,
          optimizer_G, optimizer_D, scheduler_G, scheduler_D, num_epochs, pretrain_epochs, lambda_L2, lambda_cosine,
          device, save_prefix='model'):

    train_losses_G = []
    train_losses_D = []
    val_losses_G = []
    val_losses_D = []
    val_corr = []
    best_val_loss = float('inf')
    best_val_corr = -float('inf')

    # Pretrain generator
    print("Starting generator pretraining...")
    for epoch in range(pretrain_epochs):
        generator.train()
        epoch_loss_G_L2 = 0.0
        for i, (inputs, targets) in enumerate(tqdm(train_loader)):
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer_G.zero_grad()
            fake_image = generator(inputs)
            loss_spec = criterion_spec(fake_image, targets) * lambda_cosine
            loss_G_L2 = criterion_L2(fake_image, targets) * lambda_L2
            loss_G_L2 = loss_G_L2 + loss_spec
            loss_G_L2.backward()
            optimizer_G.step()
            epoch_loss_G_L2 += loss_G_L2.item()
        epoch_loss_G_L2 /= max(1, len(train_loader))
        print(f"Pretrain Epoch [{epoch+1}/{pretrain_epochs}], Average Loss_G_L2: {epoch_loss_G_L2:.4f}")
    if pretrain_epochs > 0:
        torch.save(generator.state_dict(), os.path.join(MODEL_DIR, f'{save_prefix}_generator_pretrain.pth'))

    print("Generator pretraining completed.")
    print("Starting GAN training...")

    for epoch in range(num_epochs):
        generator.train()
        discriminator.train()
        epoch_loss_G = 0.0
        epoch_loss_D = 0.0
        for i, (inputs, targets) in enumerate(tqdm(train_loader)):
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Update D
            optimizer_D.zero_grad()
            real_input = torch.cat((inputs, targets), 1)
            output_real = discriminator(real_input)
            label_real = torch.full_like(output_real, 0.9, device=device)
            loss_D_real = criterion_GAN(output_real, label_real)
            fake_image = generator(inputs)
            fake_input = torch.cat((inputs, fake_image.detach()), 1)
            output_fake = discriminator(fake_input)
            label_fake = torch.full_like(output_real, 0.1, device=device)
            loss_D_fake = criterion_GAN(output_fake, label_fake)
            loss_D = (loss_D_real + loss_D_fake) * 0.5
            loss_D.backward()
            optimizer_D.step()

            # Update G
            optimizer_G.zero_grad()
            fake_input = torch.cat((inputs, fake_image), 1)
            output_fake = discriminator(fake_input)
            label_real = torch.full_like(output_real, 0.9, device=device)
            loss_G_GAN = criterion_GAN(output_fake, label_real)
            loss_cosine = criterion_spec(fake_image, targets) * lambda_cosine
            loss_G_L2 = criterion_L2(fake_image, targets) * lambda_L2
            loss_G = loss_G_GAN + loss_G_L2 + loss_cosine
            loss_G.backward()
            optimizer_G.step()

            epoch_loss_G += loss_G.item()
            epoch_loss_D += loss_D.item()

        scheduler_G.step()
        scheduler_D.step()
        avg_loss_G = epoch_loss_G / max(1, len(train_loader))
        avg_loss_D = epoch_loss_D / max(1, len(train_loader))
        train_losses_G.append(avg_loss_G)
        train_losses_D.append(avg_loss_D)
        print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss_G: {avg_loss_G:.4f}, Training Loss_D: {avg_loss_D:.4f}")

        # Validation
        generator.eval()
        discriminator.eval()
        val_loss_G = 0.0
        val_loss_D = 0.0
        val_corr_G = 0.0
        with torch.no_grad():
            for i, (inputs, targets) in enumerate(valid_loader):
                inputs = inputs.to(device)
                targets = targets.to(device)
                fake_image = generator(inputs)
                real_input = torch.cat((inputs, targets), 1)
                output_real = discriminator(real_input)
                label_real = torch.ones_like(output_real, device=device)
                loss_D_real = criterion_GAN(output_real, label_real)
                fake_input = torch.cat((inputs, fake_image), 1)
                output_fake = discriminator(fake_input)
                label_fake = torch.zeros_like(output_fake, device=device)
                loss_D_fake = criterion_GAN(output_fake, label_fake)
                loss_D = (loss_D_real + loss_D_fake) * 0.5
                val_loss_D += loss_D.item()
                loss_G_GAN = criterion_GAN(output_fake, label_real)
                loss_cosine = criterion_spec(fake_image, targets) * lambda_cosine
                loss_G_L2 = criterion_L2(fake_image, targets) * lambda_L2
                loss_G = loss_G_GAN + loss_G_L2 + loss_cosine
                val_loss_G += loss_G.item()
                correlation_per_channel = compute_channel_correlation(fake_image, targets)
                val_corr_G += correlation_per_channel.mean().item()
        val_loss_G /= max(1, len(valid_loader))
        val_loss_D /= max(1, len(valid_loader))
        val_corr_G /= max(1, len(valid_loader))
        val_losses_G.append(val_loss_G)
        val_losses_D.append(val_loss_D)
        val_corr.append(val_corr_G)
        print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss_G: {val_loss_G:.4f}, Validation Loss_D: {val_loss_D:.4f}, validation corr: {val_corr_G:.4f}")

        if val_loss_G < best_val_loss:
            best_val_loss = val_loss_G
            torch.save(generator.state_dict(), os.path.join(MODEL_DIR, f'{save_prefix}_best_generator.pth'))
            torch.save(discriminator.state_dict(), os.path.join(MODEL_DIR, f'{save_prefix}_best_discriminator.pth'))

        if val_corr_G > best_val_corr:
            best_val_corr = val_corr_G
            torch.save(generator.state_dict(), os.path.join(MODEL_DIR, f'{save_prefix}_bestcorr_generator.pth'))
            torch.save(discriminator.state_dict(), os.path.join(MODEL_DIR, f'{save_prefix}_bestcorr_discriminator.pth'))

    # periodic checkpointing (every 50 epochs)
        if (epoch + 1) % 50 == 0:
            torch.save(generator.state_dict(), os.path.join(MODEL_DIR, f'{save_prefix}_generator_epoch{epoch+1}.pth'))
            torch.save(discriminator.state_dict(), os.path.join(MODEL_DIR, f'{save_prefix}_discriminator_epoch{epoch+1}.pth'))

        if epoch == num_epochs - 1:
            torch.save(generator.state_dict(), os.path.join(MODEL_DIR, f'{save_prefix}_generator_lastepoch.pth'))
            torch.save(discriminator.state_dict(), os.path.join(MODEL_DIR, f'{save_prefix}_discriminator_lastepoch.pth'))

    print("Training completed.")
    return generator, discriminator, train_losses_G, train_losses_D, val_losses_G, val_losses_D, val_corr


if __name__ == '__main__':
    # CLI entrypoint: accept image and target file lists (resolved against DATA_DIR when relative)
    parser = argparse.ArgumentParser(description='Train HyperPix2pix model on paired .npy files')
    parser.add_argument('--image-files', nargs='+', help='List of input image .npy files (resolved against DATA_DIR if relative)')
    parser.add_argument('--target-files', nargs='+', help='List of target MALDI .npy files (resolved against DATA_DIR if relative)')
    parser.add_argument('--checkpoint-dir', default=MODEL_DIR, help='Directory to save checkpoints (default: MODEL_DIR)')
    parser.add_argument('--device', default=None, help='Torch device string (e.g., cuda:0). If not set, auto-detect')
    args = parser.parse_args()

    # auto-detect device if not provided
    device = torch.device(args.device) if args.device else (torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))


    raw_image_files = args.image_files if args.image_files else example_image_files
    raw_target_files = args.target_files if args.target_files else example_target_files

    def resolve_paths(paths):
        resolved = []
        for p in paths:
            if os.path.isabs(p):
                resolved.append(p)
            else:
                resolved.append(os.path.join(DATA_DIR, p) if not p.startswith(DATA_DIR) else p)
        return resolved

    image_files = resolve_paths(raw_image_files)
    target_files = resolve_paths(raw_target_files)

    # Build dataset, models, loaders, losses, optimizers and call train(...) directly
    transform = CustomTransform()
    dataset = PairedNPYDataset(image_files, target_files, transform=transform)

    # instantiate models (adjust input_nc to match your data channels)
    generator = HyperPix2pixGenerator(input_nc=4, output_nc=100, ngf=64, num_downs=4, use_dropout=True).to(device)
    discriminator = HyperPix2pixDiscriminator(input_nc=104, ndf=64, n_layers=3).to(device)

    generator.train(); discriminator.train()

    # split (85/15)
    dataset_len = len(dataset)
    train_len = int(0.85 * dataset_len)
    val_len = dataset_len - train_len
    trainset, valset = random_split(dataset, [train_len, val_len], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(dataset=trainset, batch_size=DEFAULTS['batch_size'], shuffle=True, num_workers=4)
    val_loader = DataLoader(dataset=valset, batch_size=DEFAULTS['batch_size'], shuffle=False, num_workers=4)

    # losses and optimizers
    criterion_GAN = nn.BCEWithLogitsLoss()
    criterion_L2 = nn.MSELoss()
    criterion_spec = SAMLoss()

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=DEFAULTS.get('lr_G', 1e-4), betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=DEFAULTS.get('lr_D', 4e-5), betas=(0.5, 0.999))

    scheduler_G = torch.optim.lr_scheduler.StepLR(optimizer_G, step_size=50, gamma=0.5)
    scheduler_D = torch.optim.lr_scheduler.StepLR(optimizer_D, step_size=50, gamma=0.5)

    # default settings
    num_epochs = DEFAULTS['num_epochs']
    pretrain_epochs = DEFAULTS['pretrain_epochs']
    lambda_L2 = DEFAULTS['lambda_L2']
    lambda_cosine = DEFAULTS['lambda_cosine']

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Call the training loop directly
    train(generator=generator,
        discriminator=discriminator,
        train_loader=train_loader,
        valid_loader=val_loader,
        criterion_GAN=criterion_GAN,
        criterion_L2=criterion_L2,
        criterion_spec=criterion_spec,
        optimizer_G=optimizer_G,
        optimizer_D=optimizer_D,
        scheduler_G=scheduler_G,
        scheduler_D=scheduler_D,
        num_epochs=num_epochs,
        pretrain_epochs=pretrain_epochs,
        lambda_L2=lambda_L2,
        lambda_cosine=lambda_cosine,
        device=device,
        save_prefix='srs_hyperpix2pix')
