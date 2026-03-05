# Decoding the Sky: Machine Learning Approaches to Bayesian Modeling of Neutrino Source Populations 
## Population Modeling for AGN Neutrinos under Extreme Sparsity using a Variational Autoencoder (VAE)

## Scientific Motivation

High-energy neutrinos detected by IceCube might well originate from active galactic nuclei (AGN), yet we cannot infer individual AGN neutrino fluxes. We can, however, ask: "What kind of population could have produced what we see?". Essentially, we observe too few neutrinos to identify sources, but not too few to constrain populations. In this hierarchical modeling approach, AGN are viewed as a statistical ensemble and we have population-level parameters, which are thresholds/rules that determine which sources emit neutrinos, as well as source-level parameters which are unique properties of each source. To achieve this, we need to choose appropriate Luminosity Models and tackle Computational Challenges. Direct Bayesian Inference is impossible for high dimensional inputs. To do it properly, we need to integrate over thousands of dimensions, consider every possible AGN configuration and for each one, compute likelihoods. To solve for this, we learn a surrogate for the forward-model outputs (expected rates) to make likelihood evaluation fast.

## What kind of tool is needed?

-> AGN Population Modeling Framework:

![AGN Population Modeling Framework](PopulationModelingFramework.png)

In this framework, each module is validated independently on synthetic data. In essence, a previously intractable inference problem becomes computationally accessible (but not yet astrophysically complete).

## Physical Forward Modeling



### Source Catalog listing AGN Properties 

The analysis uses the **SPIDERS** AGN catalog (SDSS-IV DR16) (`spiders_quasar_bhmass-DR16-v1.fits`), which provides per-source measurements of:

- Redshift ($z$)
- Bolometric luminosity $L_{\mathrm{bol}}$
- Black hole mass $\log M_{\mathrm{BH}}$
- Eddington ratio $\lambda_{\mathrm{Edd}} = L_{\mathrm{bol}} / L_{\mathrm{Edd}}$

A redshift-dependent column selection merges the low-$z$ and high-$z$ sub-samples into a single clean catalog of ~7000 AGN.

### A Rule (Luminosity Model) mapping AGN -> Neutrino Emission



#### Step (Threshold) Model

The simplest model used for a proof-of-principle assumes that only AGN above certain thresholds in black hole mass and accretion rate produce neutrinos:

$$
f_{\nu}\left(z, \lambda_{\mathrm{Edd}}, M_{\mathrm{BH},}L_{\mathrm{bol}}\right)=\xi_{\text{fix}} \cdot \theta\left(\lambda_{\mathrm{Edd}}-\xi_{1,i}\right) \cdot \theta\left(M_{\mathrm{BH}}-\xi_{2,i}\right) \cdot \frac{L_{\mathrm{bol}}}{4 \pi D_L(z)^2}
$$

where $f_{\nu}$ is the neutrino flux at the detector, $z$ is the redshift of the source, and $D_L(z)$ is the luminosity distance at redshift $z$, computed using the Planck18 cosmology. The step model is parameterized by three population-level parameters, out of which two will be learned by a neural network in the subsequent analysis steps. $\xi_{\mathrm{fix}}$ is a fixed normalization parameter that accounts for the overall neutrino production efficiency, detector efficiency, as well as observation time. $\xi_{1,i}$ is the threshold on the Eddington ratio $\lambda_{\mathrm{Edd}}$. This parameter is learned. Only sources with $\lambda_{\mathrm{Edd}} \geq \xi_{1,i}$ contribute to the neutrino luminosity. The second learned parameter is $\xi_{2,i}$, the threshold on the logarithm of the black hole mass $\log M_{\mathrm{BH}}$. Only sources with $\log M_{\mathrm{BH}} \geq \xi_{2,i}$ contribute to the neutrino luminosity.


#### Extended (Complex) Model

A more complex model, utilized to demonstrate dimension reduction (6D+ input -> 3D,2D Latent) adds redshift evolution, a luminosity power law, and configurable step-function signs:

$$
f_\nu(z, L_{\rm bol}, \log M_{\rm BH}, \lambda_{\rm Edd}; \boldsymbol{\xi}) = \xi_{\rm fix}  \theta(\text{sign}(\xi_5) (\lambda_{\rm Edd} - \xi_{1,i}))  \theta(\text{sign}(\xi_6) (\log M_{\rm BH} - \xi_{2,i}))  (1+z)^{\xi_3} \frac{L_{\rm bol}^{\xi_4}}{4\pi D_L(z)^2}
$$

where $f_{\nu}$ is the neutrino flux at the detector, $z$ is the redshift of the source, $\lambda_{\mathrm{Edd}}$ is the Eddington ratio, $M_{\mathrm{BH}}$ is the black hole mass, $L_{\mathrm{bol}}$ is the bolometric luminosity, $D_L(z)$ is the luminosity distance at redshift $z$, computed using the Planck18 cosmology, $\theta(x)$ is the Heaviside step function and $\boldsymbol{\xi} = (\xi_{1,i}, \xi_{2,i}, \xi_3, \xi_4, \xi_5, \xi_6)$ is the vector of population-level parameters.

## Data Generation

For each source \(k\), the observed neutrino count is drawn from a Poisson distribution:

$$
n_k \sim \mathrm{Poisson}\bigl(10^{\mathrm{norm}} \cdot f_\nu^{(k)} + \mathrm{bg}\bigr)
$$

where `norm` controls the overall signal strength and `bg` is a uniform background rate.

## Inference Methods

### Direct Grid Search (Non-VAE)

For the two-parameter step model, the Poisson log-likelihood

$$
\ln\mathcal{L}(\xi_{1,i},\,\xi_{2,i}) = \sum_k \bigl[n_k \ln\lambda_k - \lambda_k\bigr], \qquad \lambda_k = 10^{\mathrm{norm}}\,f_\nu^{(k)} + \mathrm{bg}
$$

is evaluated on a coarse grid followed by a fine grid centered on the coarse maximum. This yields the posterior surface (flat prior), the MLE, and associated test statistics.

### Variational Autoencoder (VAE)

For models with more parameters or when amortized inference is desired, a VAE learns a compressed latent representation \(\mathbf{z}\) of the population-level model parameters:

- **Encoder**: A per-source MLP processes each AGN's features, followed by a top-\(k\) selection and an aggregation network that maps the full catalog to a latent distribution \(q(\mathbf{z} \mid \mathbf{x}) = \mathcal{N}(\mu, \sigma^2)\).
- **Decoder**: Maps a latent sample \(\mathbf{z}\) back to predicted neutrino counts for every source. The decoder is trained so that its output approximates the forward model \(f_\nu\) for the parameters encoded in \(\mathbf{z}\).
- **Architecture**: Fully connected residual blocks (BN → ReLU → Linear → Dropout) throughout.

After training, inference proceeds by encoding the observed data into \(\mathbf{z}\), then evaluating the Poisson log-likelihood of decoder predictions on a grid in latent space to obtain the posterior.

## Statistical Diagnostics

Each model configuration reports:

| Metric | Definition |
|---|---|
| \(\log\mathcal{L}(\mathrm{bkg})\) | Log-likelihood under the background-only (null) model |
| \(\log\mathcal{L}(H_1)\) | Log-likelihood at the MLE under the signal + background model |
| \(\Delta\log\mathcal{L}\) | Difference \(\log\mathcal{L}(H_1) - \log\mathcal{L}(\mathrm{bkg})\) |
| TS | Test statistic \(= 2\,\Delta\log\mathcal{L}\) |
| \(\log_{10}(\mathrm{Odds})\) | Log-base-10 likelihood ratio |
| \(\sigma\) equiv | Gaussian-equivalent significance \(\approx \sqrt{\mathrm{TS}}\) |
| Correlation | Pearson \(r\) between predicted and true per-source signal |
| RMSE | Root mean square error between predicted and true signal |

## Notebooks

| Notebook | Description |
|---|---|
| `catalog.ipynb` | Loads and cleans the SPIDERS catalog; computes derived quantities; produces diagnostic histograms and scatter plots |
| `population_step_nu_lumi.ipynb` | Non-VAE step model: grid-search posterior, likelihood/posterior surface plots, full statistical summary |
| `population_vae_step_nu_lumi.ipynb` | VAE step model: trains the encoder–decoder on the two-parameter step model, evaluates multiple configurations, latent-space visualisation, odds ratio overview |
| `population_vae_complex_nu_lumi_2d_latent.ipynb` | VAE with 2D latent space for the six-parameter extended model |
| `population_vae_complex_nu_lumi_3d_latent.ipynb` | VAE with 3D latent space for the six-parameter extended model |

## Dependencies

- Python 3.10+
- PyTorch
- NumPy, SciPy, Pandas
- Astropy (cosmology calculations, FITS I/O)
- Matplotlib
