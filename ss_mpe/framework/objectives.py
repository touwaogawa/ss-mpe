# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

import torch.nn.functional as F
import torch
import math


__all__ = [
    'compute_support_loss',
    'compute_harmonic_loss',
    'compute_sparsity_loss',
    'sample_random_equalization',
    'sample_parabolic_equalization',
    'sample_gaussian_equalization',
    'compute_timbre_loss',
    'compute_geometric_loss',
    'compute_supervised_loss',
    'compute_overtone_loss',
    'compute_smoothness_loss',
    'compute_noise_loss',
    'compute_vibrato_loss',
    'compute_mixture_loss'
]


def compute_support_loss(embeddings, h1_features):
    # Set the weight for positive activations to zero
    pos_weight = torch.tensor(0)

    # Compute support loss as BCE of activations with respect to features (negative activations only)
    support_loss = F.binary_cross_entropy_with_logits(embeddings, h1_features, reduction='none', pos_weight=pos_weight)

    # Sum across frequency bins and average across time and batch
    support_loss = support_loss.sum(-2).mean(-1).mean(-1)

    return support_loss


def compute_harmonic_loss(embeddings, salience):
    # Set the weight for negative activations to zero
    neg_weight = torch.tensor(0)

    # Compute harmonic loss as BCE of activations with respect to salience estimate (positive activations only)
    harmonic_loss = F.binary_cross_entropy_with_logits(-embeddings, (1 - salience), reduction='none', pos_weight=neg_weight)

    # Sum across frequency bins and average across time and batch
    harmonic_loss = harmonic_loss.sum(-2).mean(-1).mean(-1)

    return harmonic_loss


def compute_sparsity_loss(activations):
    # Compute sparsity loss as the L1 norm of the activations
    sparsity_loss = torch.norm(activations, 1, dim=-2)

    # Average loss across time and batch
    sparsity_loss = sparsity_loss.mean(-1).mean(-1)

    return sparsity_loss


def sample_random_equalization(n_bins, batch_size=1, n_points=None, std_dev=0.10, device='cpu'):
    """
    Uniformly sample multiplicative equalization factors and upsample to cover whole frequency spectrum.

    Parameters
    ----------
    n_bins : int
      Final number of frequency bins
    batch_size : int
      Number of curves to sample
    n_points : int or None (optional)
      Number of peaks/troughs to sample
    std_dev : float
      Standard deviation of boost/cut
    device : string
      Device on which to initialize curves

    Returns
    ----------
    curves : Tensor (B x F)
      Sampled equalization curves
    """

    if n_points is None:
        # Default to provided output size
        n_points = n_bins

    # Sample a random equalization curve factor for each sample in batch
    curves = 1 + torch.randn(size=(batch_size, 1, n_points), device=device) * std_dev

    if n_bins != n_points:
        # Upsample equalization curves to number of frequency bins via linear interpolation
        curves = F.interpolate(curves, size=n_bins, mode='linear', align_corners=True)

    # Remove channel dimension
    curves = curves.squeeze(-2)

    return curves


def sample_parabolic_equalization(n_bins, batch_size=1, pointiness=1, device=None):
    """
    Randomly sample parabolic equalization curves covering whole frequency spectrum.

    Parameters
    ----------
    n_bins : int
      Number of frequency bins
    batch_size : int
      Number of curves to sample
    pointiness : float
      Multiplier to shrink parabolic opening
    device : string
      Device on which to initialize curves

    Returns
    ----------
    curves : Tensor (B x F)
      Sampled equalization curves
    """

    # Randomly sample parametric parabolic functions
    alpha, beta = torch.rand(size=(2, batch_size, 1), device=device)
    # Scale parameters to appropriate ranges
    alpha, beta = alpha / (n_bins - 1) ** 2, beta * (n_bins - 1)

    # Create a Tensor of indices for frequency bins of each curve
    idcs = torch.arange(n_bins, device=device).repeat((batch_size, 1))

    # Compute parabolic equalization curves
    curves = 1 - pointiness * alpha * (idcs - beta) ** 2

    return curves


def sample_gaussian_equalization(n_bins, batch_size=1, max_A=0.25, max_std_dev=None, fixed_shape=False, device=None):
    """
    Randomly sample Gaussian equalization curves covering whole frequency spectrum.

    Parameters
    ----------
    n_bins : int
      Number of frequency bins
    batch_size : int
      Number of curves to sample
    max_A : float
      Maximum amplitude of sampled Gaussians
    max_std_dev : float or None (optional)
      Maximum standard deviation of sampled Gaussians
    device : string
      Device on which to initialize curves

    Returns
    ----------
    curves : Tensor (B x F)
      Sampled equalization curves
    """

    if max_std_dev is None:
        # Default to 10% of frequency bins
        max_std_dev = 0.10 * n_bins

    # Randomly sample parametric Gaussian functions
    A, mu, sigma = torch.rand(size=(3, batch_size, 1), device=device)

    if fixed_shape:
        # Amplitude and standard deviation go to maximum values
        A, sigma = A.round(), torch.ones_like(sigma, device=device)

    # Scale parameters to appropriate ranges
    A, mu, sigma = max_A * (A * 2 - 1), mu * (n_bins - 1), sigma * max_std_dev

    # Create a Tensor of indices for frequency bins of each curve
    idcs = torch.arange(n_bins, device=device).repeat((batch_size, 1))

    # Compute Gaussian equalization curves
    curves = 1 + A * torch.exp(-0.5 * (idcs - mu) ** 2 / sigma ** 2)

    return curves


def apply_random_eq(features, hcqt, eq_fn, **eq_kwargs):
    # Obtain dimensionality of features and appropriate device
    (B, H, K, _), device = features.size(), features.device

    # Extract relevant HCQT parameters
    bins_per_octave = hcqt.bins_per_octave

    # Obtain center frequencies (MIDI) associated with each HCQT bin
    midi_freqs = torch.from_numpy(hcqt.midi_freqs).to(device)

    # Infer the number of bins per semitone
    bins_per_semitone = bins_per_octave / 12

    # Determine semitone span of frequency support
    semitone_span = midi_freqs.max() - midi_freqs.min()

    # Determine how many bins are represented across all harmonics
    n_psuedo_bins = (bins_per_semitone * semitone_span).round()

    # Determine how many octaves have been covered
    n_octaves = int(torch.ceil(n_psuedo_bins / bins_per_octave))

    # Perform equalization over full octave
    n_total_bins = n_octaves * bins_per_octave

    # Randomly sample an equalization curve for each sample in batch
    curves = eq_fn(n_total_bins, batch_size=B, device=device, **eq_kwargs)

    # Determine nearest equalization corresponding to each frequency bin
    equalization_bins = bins_per_semitone * (midi_freqs - midi_freqs.min())
    # Round, convert equalization bins to integers, and flatten
    equalization_bins = equalization_bins.round().long().flatten()
    # Obtain indices corresponding to equalization for each sample in the batch
    equalization_idcs = torch.meshgrid(torch.arange(B, device=device), equalization_bins)
    # Obtain the equalization for each sample in the batch
    equalization = curves[equalization_idcs].view(B, H, K, -1)

    # Apply sampled equalization curves to the batch and clamp features
    equalized_features = torch.clip(equalization * features, min=0, max=1)

    return equalized_features


def compute_timbre_loss(model, features, embeddings, eq_fn, detach=False, **eq_kwargs):
    # Perform random equalizations on batch of features
    equalized_features = apply_random_eq(features, model.hcqt, eq_fn, **eq_kwargs)

    # Process equalized features with provided model
    equalization_embeddings = model(equalized_features)[0]

    # Convert both sets of logits to activations (implicit pitch salience)
    original_salience, equalization_salience = torch.sigmoid(embeddings), torch.sigmoid(equalization_embeddings)

    if detach:
        # Treat original activations as fixed targets
        original_salience = original_salience.detach()

    # Compute timbre loss as BCE of embeddings computed from equalized features with respect to original activations
    timbre_loss = F.binary_cross_entropy_with_logits(equalization_embeddings, original_salience, reduction='none')

    # Sum across frequency bins and average across time and batch
    timbre_loss = timbre_loss.sum(-2).mean(-1).mean(-1)

    return timbre_loss


def apply_translation(tensor, shifts, axis=-1, val=None):
    """
    Perform an independent translation on each entry of a tensor.

    Parameters
    ----------
    tensor : Tensor (B x ...)
      Input tensor with at least 2 dimensions
    shifts : Tensor (B)
      Independent translations to perform
    axis : int
      Axis to translate
    val : float or None (optional)
      Value to insert or None to wrap original data

    Returns
    ----------
    translated : Tensor (B x ...)
      Original tensor translated as specified
    """

    # Determine dimensionality and device of input tensor
    dimensionality, device = tensor.size(), tensor.device

    # Copy original tensor
    tensor = tensor.clone()

    if val is not None:
        # Initialize hidden data to replace wrapped elements
        hidden_data = torch.full(dimensionality, val, device=device)
        # Combine original and hidden data to disable wrapping
        tensor = torch.cat([tensor, hidden_data], dim=axis)

    # Translate each entry by specified amount at specified axis
    translated = torch.cat([x.unsqueeze(0).roll(k.item(), axis)
                            for x, k in zip(tensor, shifts)])

    # Trim translated tensor to original dimensionality
    translated = translated.narrow(axis, 0, dimensionality[axis])

    return translated


def apply_distortion(tensor, stretch_factors):
    """
    Perform an independent distortion on each entry of a tensor.

    Parameters
    ----------
    tensor : Tensor (B x C x H x W)
      Input tensor with standard 4 dimensions
    stretch_factors : Tensor (B)
      Independent distortions to perform

    Returns
    ----------
    distorted : Tensor (B x ...)
      Original tensor distorted as specified
    """

    # Initialize list for distortions
    distorted = list()

    # Loop through each entry and scale
    for x, t in zip(tensor, stretch_factors):
        # Stretch entry by specified factor using linear interpolation
        distorted_ = F.interpolate(x, scale_factor=t.item(), mode='linear')

        if t >= 1:
            # Determine starting index to center distortion
            start_idx = (distorted_.size(-1) - x.size(-1)) // 2
            # Center distorted tensor and trim to original width
            distorted_ = distorted_.narrow(-1, start_idx, x.size(-1))
        else:
            # Determine total padding necessary
            pad_t = x.size(-1) - distorted_.size(-1)
            # Distribute padding between both sides
            pad_l, pad_r = pad_t // 2, pad_t - pad_t // 2
            # Pad distorted tensor to match original width
            distorted_ = F.pad(distorted_, (pad_l, pad_r))

        # Append distorted entry to distortion list
        distorted.append(distorted_.unsqueeze(0))

    # Combine all distorted entries
    distorted = torch.cat(distorted)

    return distorted


def compute_geometric_loss(model, features, embeddings, max_shift_v, max_shift_h, max_stretch_factor, detach=False):
    # Determine batch size
    B = features.size(0)

    # Sample a random vertical / horizontal shift for each sample in the batch
    v_shifts = torch.randint(low=-max_shift_v, high=max_shift_v + 1, size=(B,))
    h_shifts = torch.randint(low=-max_shift_h, high=max_shift_h + 1, size=(B,))

    # Compute inverse of maximum stretch factor
    min_stretch_factor = 1 / max_stretch_factor

    # Sample a random stretch factor for each sample in the batch, starting at minimum
    stretch_factors, stretch_factors_ = min_stretch_factor, torch.rand(size=(B,))
    # Split sampled values into piecewise ranges
    neg_perc = 2 * stretch_factors_.clip(max=0.5)
    pos_perc = 2 * (stretch_factors_ - 0.5).relu()
    # Scale stretch factor evenly across effective range
    stretch_factors += neg_perc * (1 - min_stretch_factor)
    stretch_factors += pos_perc * (max_stretch_factor - 1)

    # Apply vertical and horizontal translations, inserting zero at empties
    transformed_features = apply_translation(features, v_shifts, axis=-2, val=0)
    transformed_features = apply_translation(transformed_features, h_shifts, axis=-1, val=0)
    # Apply time distortion, maintaining original dimensionality and padding with zeros
    transformed_features = apply_distortion(transformed_features, stretch_factors)

    # Process transformed features with provided model
    transformation_embeddings = model(transformed_features)[0]

    # Convert logits to activations (implicit pitch salience)
    salience = torch.sigmoid(embeddings).unsqueeze(-3)

    if detach:
        # Treat original activations as fixed targets
        salience = salience.detach()

    # Apply same transformations to activations produced for original features
    transformed_salience = apply_translation(salience, v_shifts, axis=-2, val=0)
    transformed_salience = apply_translation(transformed_salience, h_shifts, axis=-1, val=0)
    transformed_salience = apply_distortion(transformed_salience, stretch_factors)

    # Remove temporarily added channel dimension
    transformed_salience = transformed_salience.squeeze(-3)

    # Compute geometric loss as BCE of embeddings computed from transformed features with respect to transformed activations
    geometric_loss = F.binary_cross_entropy_with_logits(transformation_embeddings, transformed_salience, reduction='none')

    # Sum across frequency bins and average across time and batch
    geometric_loss = geometric_loss.sum(-2).mean(-1).mean(-1)

    return geometric_loss


def compute_supervised_loss(embeddings, ground_truth):
    # Compute supervised loss as BCE of activations with respect to ground-truth
    supervised_loss = F.binary_cross_entropy_with_logits(embeddings, ground_truth, reduction='none')

    # Sum across frequency bins and average across time and batch
    supervised_loss = supervised_loss.sum(-2).mean(-1).mean(-1)

    return supervised_loss


def compute_overtone_loss(activations, bins_per_octave, harmonics=(2, 3, 4, 5), tolerance=1):
    """
    Penalize activations located at overtones of other activations.

    Parameters
    ----------
    activations : Tensor (B x F x T)
      Batch of (implicit) pitch salience activations
    bins_per_octave : int
      Number of frequency bins per octave
    harmonics : tuple of int
      Overtones (multiples of fundamental) to penalize
    tolerance : int
      Number of neighboring bins (each side) considered for the fundamental

    Returns
    ----------
    overtone_loss : Tensor (1)
      Co-activation of fundamentals and their overtones
    """

    # Treat lower activations as fixed so only the overtones are suppressed
    fundamentals = activations.detach()

    if tolerance:
        # Take maximum over neighboring bins to account for rounded offsets
        fundamentals = F.max_pool1d(fundamentals.transpose(-1, -2),
                                    kernel_size=2 * tolerance + 1,
                                    stride=1,
                                    padding=tolerance).transpose(-1, -2)

    # Initialize the loss for each frame
    overtone_loss = 0

    for h in harmonics:
        # Determine the bin offset corresponding to the overtone
        offset = round(bins_per_octave * math.log2(h))

        if offset >= activations.size(-2):
            # Overtone is outside of frequency support
            continue

        # Compute co-activation of each bin with its overtone and sum across frequency bins
        overtone_loss += (fundamentals[..., :-offset, :] * activations[..., offset:, :]).sum(-2)

    # Average loss across time and batch
    overtone_loss = overtone_loss.mean(-1).mean(-1)

    return overtone_loss


def compute_smoothness_loss(activations):
    """
    Penalize differences between the activations of adjacent frames.

    Parameters
    ----------
    activations : Tensor (B x F x T)
      Batch of (implicit) pitch salience activations

    Returns
    ----------
    smoothness_loss : Tensor (1)
      Total variation of activations along time
    """

    # Compute absolute difference between adjacent frames
    smoothness_loss = (activations[..., 1:] - activations[..., :-1]).abs()

    # Sum across frequency bins and average across time and batch
    smoothness_loss = smoothness_loss.sum(-2).mean(-1).mean(-1)

    return smoothness_loss


def extract_features(model, audio):
    """
    Compute input features for audio outside of the computational graph.

    Parameters
    ----------
    model : SS_MPE
      Model with feature extraction module
    audio : Tensor (B x 1 x N)
      Batch of input raw audio

    Returns
    ----------
    features : Tensor (B x H x F x T)
      Batch of HCQT spectral features (dB) scaled to [0, 1]
    """

    with torch.no_grad(), torch.autocast(device_type=audio.device.type, enabled=False):
        # Compute features at full precision
        features = model.get_all_features(audio.float())['db']

    return features


def apply_random_noise(audio, min_snr=10, max_snr=40):
    """
    Add white noise with an independent signal-to-noise ratio to each entry of a batch.

    Parameters
    ----------
    audio : Tensor (B x 1 x N)
      Batch of input raw audio
    min_snr : float
      Minimum signal-to-noise ratio (dB)
    max_snr : float
      Maximum signal-to-noise ratio (dB)

    Returns
    ----------
    noisy_audio : Tensor (B x 1 x N)
      Batch of audio with added noise
    """

    # Determine batch size and appropriate device
    B, device = audio.size(0), audio.device

    # Sample a random signal-to-noise ratio for each sample in the batch
    snrs = min_snr + (max_snr - min_snr) * torch.rand(size=(B, 1, 1), device=device)

    # Compute root-mean-square amplitude of each sample in the batch
    rms = audio.pow(2).mean(-1, keepdim=True).sqrt()

    # Sample white noise and scale to obtain chosen signal-to-noise ratio
    noise = torch.randn_like(audio) * rms * 10 ** (-snrs / 20)

    # Add noise to the original audio
    noisy_audio = audio + noise

    return noisy_audio


def compute_noise_loss(model, audio, embeddings, min_snr=10, max_snr=40, detach=False):
    # Add random noise to batch of audio
    noisy_audio = apply_random_noise(audio, min_snr, max_snr)

    # Process features for noisy audio with provided model
    noise_embeddings = model(extract_features(model, noisy_audio))[0]

    # Convert logits to activations (implicit pitch salience)
    original_salience = torch.sigmoid(embeddings)

    if detach:
        # Treat original activations as fixed targets
        original_salience = original_salience.detach()

    # Compute noise loss as BCE of embeddings computed from noisy audio with respect to original activations
    noise_loss = F.binary_cross_entropy_with_logits(noise_embeddings, original_salience, reduction='none')

    # Sum across frequency bins and average across time and batch
    noise_loss = noise_loss.sum(-2).mean(-1).mean(-1)

    return noise_loss


def apply_vibrato(tensor, depths, rates, phases):
    """
    Perform an independent time-varying frequency shift on each entry of a tensor.

    Parameters
    ----------
    tensor : Tensor (B x C x F x T)
      Input tensor with standard 4 dimensions
    depths : Tensor (B)
      Maximum deviation (bins) of each shift
    rates : Tensor (B)
      Modulation rate (cycles per frame) of each shift
    phases : Tensor (B)
      Initial phase (radians) of each shift

    Returns
    ----------
    modulated : Tensor (B x C x F x T)
      Original tensor shifted along frequency by sinusoidal amount
    """

    # Determine dimensionality and device of input tensor
    (B, C, K, T), device = tensor.size(), tensor.device

    # Create a Tensor of indices for the frames of each entry
    t = torch.arange(T, device=device).unsqueeze(0)

    # Compute frequency shift (bins) for each frame
    shifts = depths.unsqueeze(-1) * torch.sin(2 * math.pi * rates.unsqueeze(-1) * t + phases.unsqueeze(-1))

    # Determine the (fractional) source bin for each frequency bin and frame
    source = torch.arange(K, device=device).view(1, K, 1) - shifts.unsqueeze(-2)

    # Split source bins into lower index and interpolation weight
    lower = source.floor()
    weight = (source - lower).unsqueeze(1).to(tensor.dtype)
    lower = lower.long()

    def gather(idcs):
        # Determine which indices fall within frequency support
        valid = ((idcs >= 0) & (idcs < K)).unsqueeze(1)
        # Broadcast clamped indices across channels
        idcs = idcs.clamp(0, K - 1).unsqueeze(1).expand(B, C, K, T)
        # Obtain values for each index, inserting zero outside of support
        return torch.gather(tensor, -2, idcs) * valid

    # Linearly interpolate between the two nearest bins
    modulated = (1 - weight) * gather(lower) + weight * gather(lower + 1)

    return modulated


def compute_vibrato_loss(model, features, embeddings, max_depth, min_rate, max_rate, detach=False):
    # Determine batch size and appropriate device
    B, device = features.size(0), features.device

    # Sample a random depth, rate, and phase for each sample in the batch
    depths, rates, phases = torch.rand(size=(3, B), device=device)
    # Scale parameters to appropriate ranges
    depths, rates, phases = max_depth * depths, min_rate + (max_rate - min_rate) * rates, 2 * math.pi * phases

    # Apply time-varying frequency shift to batch of features
    modulated_features = apply_vibrato(features, depths, rates, phases)

    # Process modulated features with provided model
    modulation_embeddings = model(modulated_features)[0]

    # Convert logits to activations (implicit pitch salience)
    salience = torch.sigmoid(embeddings).unsqueeze(-3)

    if detach:
        # Treat original activations as fixed targets
        salience = salience.detach()

    # Apply same modulation to activations produced for original features
    modulated_salience = apply_vibrato(salience.float(), depths, rates, phases).squeeze(-3)

    # Compute vibrato loss as BCE of embeddings computed from modulated features with respect to modulated activations
    vibrato_loss = F.binary_cross_entropy_with_logits(modulation_embeddings, modulated_salience, reduction='none')

    # Sum across frequency bins and average across time and batch
    vibrato_loss = vibrato_loss.sum(-2).mean(-1).mean(-1)

    return vibrato_loss


def compute_mixture_loss(model, audio, embeddings, max_sources=3, max_attenuation=12):
    """
    Compute BCE between activations for mixtures of the batch and the
    maximum of the activations produced for the individual sources.

    Parameters
    ----------
    model : SS_MPE
      Model to process mixtures
    audio : Tensor (B x 1 x N)
      Batch of input raw audio
    embeddings : Tensor (B x F x T)
      Batch of logits produced for the original audio
    max_sources : int
      Maximum number of entries to combine in each mixture
    max_attenuation : float
      Maximum attenuation (dB) applied to each additional source

    Returns
    ----------
    mixture_loss : Tensor (1)
      Disagreement between mixture and combined source activations
    """

    # Determine batch size and appropriate device
    B, device = audio.size(0), audio.device

    # Individual source activations serve as fixed targets
    salience = torch.sigmoid(embeddings).detach()

    # Sample the number of sources to combine
    n_sources = torch.randint(low=2, high=max(2, min(max_sources, B)) + 1, size=(1,)).item()

    # Start with original batch
    mixtures, targets = audio.clone(), salience

    for i in range(1, n_sources):
        # Sample a random attenuation for each additional source
        gains = 10 ** (-max_attenuation * torch.rand(size=(B, 1, 1), device=device) / 20)
        # Add another entry of the batch to each mixture
        mixtures = mixtures + gains * audio.roll(i, 0)
        # Any pitch active within a source is active within the mixture
        targets = torch.maximum(targets, salience.roll(i, 0))

    # Process features for mixtures with provided model
    mixture_embeddings = model(extract_features(model, mixtures))[0]

    # Compute mixture loss as BCE of embeddings computed from mixtures with respect to combined activations
    mixture_loss = F.binary_cross_entropy_with_logits(mixture_embeddings, targets, reduction='none')

    # Sum across frequency bins and average across time and batch
    mixture_loss = mixture_loss.sum(-2).mean(-1).mean(-1)

    return mixture_loss
