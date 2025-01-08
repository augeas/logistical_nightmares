# logistical_nightmares
Dark Ambient Sonic Abuse of the Logistic Map

## The Prologue:

The [bifurcation diagram](https://en.wikipedia.org/wiki/Bifurcation_diagram) of the [Logistic Map](https://en.wikipedia.org/wiki/Logistic_map) as
[popularised by Robert May](https://www.researchgate.net/publication/237005499_Simple_Mathematical_Models_With_Very_Complicated_Dynamics) somewhat
lacks variety compared to some [other chaotic maps](https://en.wikipedia.org/wiki/List_of_chaotic_maps).

![logistic map bifurfaction diagram](assets/logistic.png)

If we take the recurrance relation

$$\Large{x_{n+1}=rx_{n}}(1-x_{n})$$

we can replace the single coefficient $r$ varied across the horizontal axis with some repeating sequence of two coefficients $A$ and $B$.
(This has been used to generate [Lyapunov fractals](https://en.wikipedia.org/wiki/Lyapunov_fractal).)
For example, the sequence $ABAABBB$, where $A$ and $B$ are described by the points on a circle of radius 0.25 centred on (3.28, 3.28) looks
like:

![a somewhat distressed bifurfaction diagram](assets/ABAABBB.png)

One can treat each histogram column in a bifurcation diagram as a spectrum. By taking
[inverse discrete cosine transforms](https://en.wikipedia.org/wiki/Discrete_cosine_transform#Inverse_transforms)
of the spectra, multiplying by a
[window function](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.windows.blackmanharris.html)
and summing the signals after overlapping them by a [hop-size](https://en.wikipedia.org/wiki/Short-time_Fourier_transform),
an audio loop can be *perpetrated*. We can add two extra coefficients, $C$ and $D$ which also rotate, to produce a sequence
of image that might look and sound something like [this](https://youtu.be/CvHFb1de2Vs?feature=shared).

This has been implemented in Python with the help of [Marimo](https://marimo.io/), [NumPy](https://numpy.org/),
[SciPy](https://scipy.org/), [Pillow](https://pillow.readthedocs.io/en/stable/) [Numba](https://numba.pydata.org/),
[bless it](https://www.oreilly.com/library/view/high-performance-python/9781492055013/) and [FFmpeg](https://www.ffmpeg.org/),
peace be upon it.

## To Gaze Into The Void:

To play with the [Marimo notebook](https://marimo.io/) and generate audio loops and videos for yourself, do the following:

```bash
# Create and start a virtualenv:
mkdir logistic_venv
python3 -m venv logistic_venv
source ./logistic_venv/bin/activate
# Clone the repo:
git clone https://github.com/augeas/logistical_nightmares.git
cd logistical_nightmares
# Get the dependencies:
pip3 install -r requirements.txt
# Finally, run the notebook:
marimo run logistic.py
```

(To mess with the code, try `marimo edit`.)

For the hard-of-terminal, Marimo notebooks can also be run using [web assembly](https://docs.marimo.io/guides/wasm/) in your browser, albeit in a limited fashion.

Press the Big, Green Button:

[![big green button](green_round_button.png)](https://tinyurl.com/logistic-8-1-25)

(However, when doing so you will be limitted to short loops with low audio rates and bit depths, as Marimo's maximum output size when running in web assembly isn't very large.)





