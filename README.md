---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.1'
      jupytext_version: 1.1.0
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# HelmholtzExact
A collection of functions to exactly solve the Helmholtz equation for a circular scatterer.


## Dielectric scatteter

The total field must solve 

$$
\left\{\begin{array}{ll}
\Delta u + k^2 u = 0 & \text{in }\mathbb{R^2}\setminus\overline{\mathrm{B}_R(\mathbf{c})}\\
\Delta u + \mathrm{n}k^2u=0 & \text{in }\mathrm{B}_R(\mathbf{c})\\
u^+ - u^- = 0 & \text{on }\partial\mathrm{B}_R(\mathbf{c})\\
\partial_\mathbf{n}u^+ - \partial_\mathbf{n}u^-=0 & \text{on }\partial\mathrm{B}_R(\mathbf{c})\\
\partial_r (u-u_\mathrm{i})-ik(u-u_\mathrm{i})=o\left(\frac{1}{\sqrt{r}}\right) & \text{as }r\to\infty
\end{array}\right.
$$

The scattered field outside the cylinder can be expanded as a combination of Hankel functions:

$$
u(\mathbf{x})-u^\mathrm{i}(\mathbf{x})=\sum_{n=-\infty}^\infty a_n H^1_n\left(k\|\mathbf{x}-\mathbf{c}\|\right)e^{in\theta_\mathbf{c}(\mathbf{x})}
$$

whereas the total field inside the scatterer is given by an expansion in terms of Bessel functions of the first kind:

$$
u(\mathbf{x})=\sum_{n=-\infty}^\infty b_n J_n\left(k\|\mathbf{x}-\mathbf{c}\|\right)e^{in\theta_\mathbf{c}(\mathbf{x})}
$$

