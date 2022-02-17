# jacobian

Code to calculate the Jacobian-weighted energy, entropy, and free-energy profiles along a nonlinear latent space coordinate.

Right now, the code is designed to work on a two-dimensional input space with a latent space that is either one- or two-dimensional, 
but the code can be generalized for an input and latent space of arbitrary dimensionality.

**Index of Codes:**

util.py: Python file containing a set of utility functions required to run the Jacobian scripts

jacobian.py: Python file containing the definitions of functions that calculate the Jacobian-corrected free-energy, energy (or enthalpy), and entropy
             along each latent space coordinate. Optionally ouputs plots of the free-energy profiles and projections of the latent variables and
             their cooresponding gradients along the input two-dimensional coordinate space.