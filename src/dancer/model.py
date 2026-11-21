"""Dancer reference-guided intrinsic expression and spatial contamination model."""
import math
import pyro
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import kl_divergence
from pyro.distributions import Delta, Dirichlet, Exponential, Gamma, LogNormal, Multinomial, Normal, Poisson, constraints
from scvi.nn import Encoder
from .networks import ConditionalGaussianEncoder, inverse_softplus


class DancerModel(nn.Module):
    def __init__(
        self,
        log_mu: torch.Tensor,
        *,
        program_counts,
        n_cells: int,
        n_batches: int,
        n_neighbors: int,
        median_distance: float,
        n_latent: int,
        n_hidden_reference: int,
        n_hidden_expression: int,
        n_hidden_diffusion: int,
        n_layers: int,
        dropout: float,
        dispersion_floor: float,
        diffusion_epsilon: float,
        alpha_prior_scale: float,
        prior_proportions_rate: float,
        prior_true_amount: float,
        prior_diffusion_amount: float,
        sparsity_diffusion_prior_mean: float,
        background_ratio: float,
        downsample_log_location: float,
        downsample_log_scale: float,
        downsample_counts: bool,
        gamma_sigma_initial: float,
        gamma_sigma_rate: float,
        memory_efficient_loadings: bool = False,
        observation_likelihood_weight: float = 5.0,
    ):
        super().__init__()
        if log_mu.ndim != 2 or min(log_mu.shape) < 1:
            raise ValueError("log_mu must have shape (n_cell_types, n_genes)")

        self.n_cells = int(n_cells)
        self.n_batches = int(n_batches)
        self.n_neighbors = int(n_neighbors)
        self.n_types = int(log_mu.shape[0])
        self.memory_efficient_loadings = bool(memory_efficient_loadings)
        self.n_genes = int(log_mu.shape[1])
        self.n_latent = int(n_latent)
        self.dispersion_floor = float(dispersion_floor)
        self.diffusion_epsilon = float(diffusion_epsilon)
        self.alpha_prior_scale = float(alpha_prior_scale)
        self.downsample_counts = bool(downsample_counts)
        self.observation_likelihood_weight = float(observation_likelihood_weight)
        if not math.isfinite(self.observation_likelihood_weight) or self.observation_likelihood_weight <= 0:
            raise ValueError("observation_likelihood_weight must be finite and positive")

        self.register_buffer("log_mu", log_mu.clone())
        self.raw_loadings = nn.Parameter(
            torch.randn(self.n_types, self.n_genes, self.n_latent) * 0.01
        )
        self.raw_log_gamma = nn.Parameter(torch.zeros(self.n_genes))
        self.raw_sigma_gamma = nn.Parameter(inverse_softplus(float(gamma_sigma_initial)))

        self.reference_encoder = ConditionalGaussianEncoder(
            self.n_genes,
            self.n_types,
            self.n_latent,
            n_hidden=n_hidden_reference,
            n_layers=n_layers,
            dropout=dropout,
        )
        self.expression_encoder = ConditionalGaussianEncoder(
            self.n_genes,
            self.n_types,
            self.n_latent,
            n_hidden=n_hidden_expression,
            n_layers=n_layers,
            dropout=dropout,
        )
        # The construction and numerical settings are copied from ResolVI.
        self.diffusion_encoder = Encoder(
            self.n_genes,
            3,
            n_cat_list=None,
            n_hidden=n_hidden_diffusion,
            n_layers=n_layers,
            dropout_rate=0.0,
            inject_covariates=True,
            use_batch_norm=False,
            use_layer_norm=True,
            var_activation=nn.Softmax(dim=-1),
            var_eps=1e-3,
        )

        self.register_buffer("median_distance", torch.tensor(float(median_distance)))
        self.register_buffer(
            "background_prior_concentration",
            torch.full((self.n_batches, self.n_genes), 5.0, dtype=torch.float32),
        )
        self.register_buffer(
            "prior_proportions_rate", torch.tensor([float(prior_proportions_rate)])
        )
        self.register_buffer(
            "sparsity_diffusion_prior_mean",
            torch.tensor(float(sparsity_diffusion_prior_mean)),
        )
        self.register_buffer(
            "prior_proportions",
            torch.tensor(
                [
                    float(prior_true_amount),
                    float(prior_diffusion_amount),
                    10.0 * float(background_ratio) * float(prior_true_amount) + 1e-3,
                ]
            ),
        )
        self.register_buffer(
            "downsample_log_location", torch.tensor(float(downsample_log_location))
        )
        self.register_buffer(
            "downsample_log_scale", torch.tensor(float(downsample_log_scale))
        )
        self.register_buffer("gamma_sigma_rate", torch.tensor(float(gamma_sigma_rate)))
        counts = torch.as_tensor(program_counts, dtype=torch.long, device=log_mu.device)
        if counts.shape != (self.n_types,) or (counts < 1).any() or (counts > self.n_latent).any():
            raise ValueError("program_counts must contain one count between 1 and n_latent per type")
        self.register_buffer("program_mask", torch.arange(self.n_latent, device=log_mu.device)[None, :] < counts[:, None])
        self.spatial_module = nn.ModuleDict({"expression_encoder": self.expression_encoder, "diffusion_encoder": self.diffusion_encoder, "scalars": nn.ParameterDict({"log_gamma": self.raw_log_gamma, "raw_sigma_gamma": self.raw_sigma_gamma})})

    @property
    def log_gamma(self) -> torch.Tensor:
        return self.raw_log_gamma - self.raw_log_gamma.mean()

    @property
    def gamma(self) -> torch.Tensor:
        return torch.exp(self.log_gamma)

    @property
    def sigma_gamma(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.raw_sigma_gamma) + 1e-6

    @property
    def background(self) -> torch.Tensor:
        return torch.softmax(pyro.param("background_concentration"), dim=-1)

    @property
    def kappa_inverse(self) -> torch.Tensor:
        return pyro.param("px_r_mle")

    def _expanded_labels(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        target_shape = z.shape[:-1]
        expanded = labels
        while expanded.ndim < len(target_shape):
            expanded = expanded.unsqueeze(0)
        return expanded.expand(target_shape)

    def gamma_log_prior(self) -> torch.Tensor:
        sigma = self.sigma_gamma
        return Normal(0.0, sigma).log_prob(self.raw_log_gamma).sum() - self.gamma_sigma_rate * sigma

    def _downsample(self, counts: torch.Tensor) -> torch.Tensor:
        if not (self.training and self.downsample_counts):
            return counts
        target = int(
            LogNormal(self.downsample_log_location, self.downsample_log_scale)
            .sample()
            .item()
        ) + 10
        probabilities = counts / counts.sum(dim=-1, keepdim=True).clamp_min(1.0)
        return Multinomial(total_count=target, probs=probabilities).sample()

    def _eta(self, weights: torch.Tensor, rho_neighbors: torch.Tensor) -> torch.Tensor:
        mixture = (weights.unsqueeze(-1) * rho_neighbors).sum(dim=-2)
        # softmax(log(mixture) + log(gamma)) is exactly gamma*m normalized,
        # while avoiding overflow when gamma is temporarily large.
        return torch.softmax(
            torch.log(mixture.clamp_min(1e-12)) + self.log_gamma,
            dim=-1,
        )

    def _background_by_cell(
        self, profile: torch.Tensor, batch_index: torch.Tensor, expression_shape: torch.Size
    ) -> torch.Tensor:
        """Select the sample axis, preserving Pyro's particle axes.

        The global (sample, gene) event has shape (P, 1, S, G) under
        vectorized particles. The singleton is the unoccupied cell plate,
        not another cell/sample axis. Align it with the local (P, B, G)
        expression after indexing S; profile[batch_index] instead indexes P
        and creates an erroneous (B, P, B, G) observation tensor.
        """
        selected = profile.index_select(-2, batch_index.reshape(-1))
        return selected.reshape(expression_shape)

    @property
    def loadings(self):
        return (self.raw_loadings - self.raw_loadings.mean(dim=1, keepdim=True)) * self.program_mask[:, None, :]

    def reference_named_parameters(self):
        yield from (("reference_encoder." + n, p) for n, p in self.reference_encoder.named_parameters())
        yield "raw_loadings", self.raw_loadings

    def reference_parameters(self):
        return [p for _, p in self.reference_named_parameters()]

    def loading_log_prior(self):
        return self.log_mu.new_zeros(())

    def baseline_penalty(self):
        return self.log_mu.new_zeros(())

    def draw_log_baseline(self, sample_shape=torch.Size(), *, detach_reference):
        return self.log_mu.expand(*sample_shape, self.n_types, self.n_genes)

    def posterior(self, counts, labels, *, reference=False):
        encoder = self.reference_encoder if reference else self.expression_encoder
        mean, logvar = encoder(counts, labels)
        active = self.program_mask[labels]
        mean, logvar = torch.where(active, mean, 0.), torch.where(active, logvar, 0.)
        return Normal(mean, torch.exp(.5 * logvar)).to_event(1), {"gaussian_mean": mean, "gaussian_logvar": logvar}

    def latent_prior(self, counts):
        return Normal(counts.new_zeros((len(counts), self.n_latent)), 1.).to_event(1)

    def state(self, latent):
        return F.softplus(latent)

    def composition(self, latent, labels, *, detach_loadings, log_baseline=None):
        states = self.state(latent).to(self.raw_loadings.dtype)
        flat = states.reshape(-1, self.n_latent)
        loadings = self.loadings.detach() if detach_loadings else self.loadings
        flat_labels = labels.reshape(-1)
        offsets = torch.arange(len(flat) // flat_labels.numel(), device=labels.device) * flat_labels.numel()
        program = states.new_empty((len(flat), self.n_genes))
        for k in torch.unique(flat_labels).tolist():
            indices = torch.nonzero(flat_labels == k, as_tuple=True)[0]
            selected = (offsets[:, None] + indices[None, :]).reshape(-1)
            program.index_copy_(0, selected, F.linear(flat.index_select(0, selected), loadings[k]))
        program = program.reshape(*states.shape[:-1], self.n_genes)
        # Fixed reference means; no spatial gradient reaches reference loadings.
        return torch.softmax(self.log_mu[self._expanded_labels(states, labels)] + program, -1)

    def reference_loss(self, counts, labels, *, kl_weight, n_reference):
        posterior, _ = self.posterior(counts, labels, reference=True)
        latent = posterior.rsample()
        baseline = self.draw_log_baseline(detach_reference=False)
        rho = self.composition(latent, labels, detach_loadings=False, log_baseline=baseline)
        rate = (counts.sum(-1, keepdim=True) * rho).clamp_min(1e-8)
        # Identical Poisson log-score on integers; gamma extension on fractional
        # reference values. This is a reconstruction score, not a continuous pdf.
        reconstruction = (rate - torch.xlogy(counts, rate) + torch.lgamma(counts + 1)).sum(-1).mean()
        latent_kl = kl_divergence(posterior, self.latent_prior(counts)).mean()
        loading_penalty = -self.loading_log_prior() / n_reference
        baseline_penalty = self.baseline_penalty() / n_reference
        loss = reconstruction + kl_weight * latent_kl + loading_penalty + baseline_penalty
        return loss, dict(reference_loss=loss.detach(),
            reference_reconstruction=reconstruction.detach(), reference_kl=latent_kl.detach(),
            loading_prior_penalty=loading_penalty.detach(), baseline_penalty=baseline_penalty.detach())

    def normalized_alpha(self, counts):
        # This is differentiable; use in BOTH training guide and export.
        _, alpha, _ = self.diffusion_encoder(torch.log1p(counts))
        offset = alpha.new_tensor([0.0, self.diffusion_epsilon, 0.0])
        alpha = alpha + offset
        return alpha / alpha.sum(-1, keepdim=True)

    @torch.no_grad()
    def alpha_from_counts(self, counts):
        return self.normalized_alpha(counts)

    def neighbor_compositions(self, counts, labels, center_latent, log_baseline):
        batch, neighbors, genes = counts.shape
        with torch.no_grad():
            posterior, _ = self.posterior(counts.reshape(-1, genes), labels.reshape(-1))
            particle_shape = center_latent.shape[:-2]
            latent = posterior.sample(particle_shape).reshape(
                *particle_shape, batch, neighbors, self.n_latent)
            return self.composition(latent, labels, detach_loadings=True,
                                    log_baseline=log_baseline.detach())

    def model(self, counts, labels, center_indices, neighbor_counts, neighbor_labels,
              distances, batch_index, *, n_obs=None, kl_weight=1.0):
        pyro.module("dancer_spatial_module", self.spatial_module)
        pyro.factor("dancer_gamma_prior", self.gamma_log_prior())
        sparsity = pyro.sample("sparsity_diffusion", Gamma(
            self.prior_proportions_rate,
            self.prior_proportions_rate / self.sparsity_diffusion_prior_mean)).mean()
        background_profile = pyro.sample("per_gene_background", Dirichlet(
            self.background_prior_concentration, validate_args=False).to_event(1))
        conditional_means = sparsity * self.prior_proportions
        true = pyro.sample("true_proportion", Gamma(
            self.prior_proportions_rate, self.prior_proportions_rate / conditional_means[0])).mean()
        background = pyro.sample("background_proportion", Gamma(
            self.prior_proportions_rate, self.prior_proportions_rate / conditional_means[2])).mean()
        diffusion = pyro.sample("diffusion_proportion", Gamma(
            self.prior_proportions_rate, self.prior_proportions_rate / conditional_means[1])).mean()
        distance_scale = pyro.sample("diffuse_scale", Exponential(counts.new_ones(1)).to_event(1))
        with pyro.plate("obs_plate", n_obs or self.n_cells, subsample=center_indices, dim=-1):
            distance_concentration = 30.0 * (
                torch.exp(-torch.clamp(distance_scale * distances / self.median_distance, max=20.0))
                + 1e-3)
            kappa_inverse = pyro.sample("px_r_inv", Exponential(torch.ones_like(counts)).to_event(1))
            weights = pyro.sample("per_neighbor_diffusion", Dirichlet(distance_concentration))
            concentration = torch.stack([true, diffusion, background], -1)
            with pyro.poutine.scale(scale=self.alpha_prior_scale):
                alpha = pyro.sample("mixture_proportions", Dirichlet(concentration))
            with pyro.poutine.scale(scale=kl_weight):
                latent = pyro.sample("spatial_latent", self.latent_prior(counts))
            log_baseline = self.draw_log_baseline(latent.shape[:-2], detach_reference=True)
            rho = self.composition(latent, labels, detach_loadings=True, log_baseline=log_baseline)
            neighbors = self.neighbor_compositions(neighbor_counts, neighbor_labels, latent, log_baseline)
            eta = self._eta(weights, neighbors)
            library = counts.sum(-1, keepdim=True)
            alpha_self, alpha_neighbor, alpha_background = alpha.split(1, -1)
            mean = library * (alpha_self * rho + alpha_neighbor * eta)
            kappa = kappa_inverse.reciprocal()
            intensity = Gamma(kappa, kappa / (mean + 1e-8)).to_event(1).rsample()
            cell_background = self._background_by_cell(background_profile, batch_index, rho.shape)
            background_rate = library * alpha_background * cell_background
            if background_rate.shape != mean.shape:
                raise RuntimeError("Background broadcast does not match (particle, cell, gene)")
            with pyro.poutine.scale(scale=self.observation_likelihood_weight):
                pyro.sample("obs", Poisson(intensity + background_rate + 1e-9).to_event(1), obs=counts)

    def guide(self, counts, labels, center_indices, neighbor_counts, neighbor_labels,
              distances, batch_index, *, n_obs=None, kl_weight=1.0):
        pyro.module("dancer_spatial_module", self.spatial_module)
        sparsity = pyro.param("sparsity_diffusion_est", 20.0 * counts.new_ones(1),
                              constraint=constraints.greater_than(1e-3))
        pyro.sample("sparsity_diffusion", Delta(sparsity))
        background_logits = pyro.param("background_concentration",
                                       lambda: torch.ones_like(self.background_prior_concentration), event_dim=2)
        pyro.sample("per_gene_background", Delta(torch.softmax(background_logits, -1), event_dim=2))
        for name, initial in [("background_proportion", 0.5), ("diffusion_proportion", 3.0),
                              ("true_proportion", 5.0)]:
            value = pyro.param(name + "_est", initial * counts.new_ones(1),
                               constraint=constraints.greater_than(1e-6))
            pyro.sample(name, Delta(value))
        distance_scale = pyro.param("diffuse_scale_est", counts.new_ones(1),
                                    constraint=constraints.greater_than(self.dispersion_floor), event_dim=1)
        pyro.sample("diffuse_scale", Delta(distance_scale, event_dim=1))
        neighbor_logits = pyro.param("per_neighbor_diffusion_map",
                                     lambda: counts.new_zeros((self.n_cells, self.n_neighbors)),
                                     constraint=constraints.interval(-10.0, 10.0), event_dim=1)
        with pyro.plate("obs_plate", n_obs or self.n_cells, subsample=center_indices, dim=-1):
            inverse = pyro.param("px_r_mle", lambda: counts.new_full((self.n_genes,), 0.01),
                                 constraint=constraints.greater_than(self.dispersion_floor), event_dim=1)
            pyro.sample("px_r_inv", Delta(inverse, event_dim=1))
            weights = torch.softmax(neighbor_logits[center_indices] - torch.clamp(
                torch.sqrt(distances / self.median_distance), max=10.0), -1)
            pyro.sample("per_neighbor_diffusion", Delta(weights, event_dim=1))
            with pyro.poutine.scale(scale=self.alpha_prior_scale):
                pyro.sample("mixture_proportions", Delta(self.normalized_alpha(counts), event_dim=1))
            with pyro.poutine.scale(scale=kl_weight):
                posterior, _ = self.posterior(self._downsample(counts), labels)
                pyro.sample("spatial_latent", posterior)
